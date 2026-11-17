"""Kernel-enforced program boundary; no model/GPU, only owned test channels."""
import json
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from auriga.core.config import Config
from auriga.runtime.harness import Harness
from auriga.runtime.programs import isolation
from auriga.runtime.programs.runner import ProgramRunner, program_runtime
from auriga.core.storage.store import Store
from program_fixture import program_services
from test_framework_candidates import WaitFixtureRobot
from support.host_capabilities import program_isolation, require


@require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
class ProgramIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.actor = self.root/'actor'
        self.actor.mkdir()
        runtime, packages, read_paths = program_runtime(self.root / 'runtime')
        self.spec = self.root/'isolation.json'
        self.spec.write_text(json.dumps(dict(read_paths=read_paths,write_paths=[str(self.actor),'/dev/null'],tcp_ports=[],deny_socket_creation=True,bounded_program=True,
                                            env=dict(PATH=os.defpath,HOME=str(self.actor),TMPDIR=str(self.actor),PYTHONHOME=str(runtime),PYTHONPATH=str(packages),PYTHONDONTWRITEBYTECODE='1'))))

    def tearDown(self):
        self.temporary.cleanup()

    def run_isolated(self,source,pass_fds=()):
        result = subprocess.run([sys.executable,'-m','auriga.runtime.programs.isolation',str(self.spec),sys.executable,'-S','-c',source],capture_output=True,pass_fds=pass_fds,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        return result.stdout

    def test_actual_program_retains_computation_but_denies_host_and_proc(self):
        store = Store(self.root / 'run', dict(run_id='runtime-fixture', episode_id='ep', env_instance_id='env', env_generation=1))
        harness = SimpleNamespace(store=store, lock=threading.RLock(), cancel=threading.Event(), active_context=None)
        runner = ProgramRunner(program_services(harness))
        hidden = self.root / 'host-secret.txt'
        hidden.write_text('not actor data')
        host_packages = Path(importlib.util.find_spec('numpy').origin).parent.parent.resolve()
        rejected = [str(hidden), str(Path(__file__).resolve()),
                    '/proc/self/environ', '/proc/self/fd', '/proc/self/mem', '/proc/' + str(os.getpid()) + '/mem']
        rejected.extend(str(path) for path in host_packages.glob('__editable*') if path.is_file())
        source = f'''from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import cv2
assert float(np.linalg.norm([3., 4.])) == 5.
assert Image.new('RGB', (2, 2)).size == (2, 2)
assert Rotation.identity().as_matrix().shape == (3, 3)
assert cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), cv2.COLOR_RGB2GRAY).shape == (2, 2)
Path('secret-link').symlink_to({str(hidden)!r})
for name in {rejected!r} + ['secret-link']:
    try:
        Path(name).read_bytes()
    except PermissionError:
        continue
    raise AssertionError('host file readable: ' + name)
print('selected_runtime_and_host_boundary_passed')
assert Path('/proc/self/fd/0').read_bytes() == b''
'''
        try:
            job = runner.run(dict(source=source, timeout_s=20),
                             dict(thread_id='thread', turn_id='turn', tool_call_id='runtime', item_id='item'))
            self.assertEqual(job['status'], 'succeeded', store.resolve_artifact(job['stderr_ref'])[1])
            self.assertIn(b'selected_runtime_and_host_boundary_passed', store.resolve_artifact(job['stdout_ref'])[1])
            policy = json.loads((store.root / 'private' / ('isolation-' + job['job_id'] + '.json')).read_text())
            self.assertNotIn(sys.prefix, policy['read_paths'])
            self.assertNotIn(str(host_packages), policy['read_paths'])
            for directory in ('/usr', '/usr/local', '/usr/bin', '/usr/lib', '/bin', '/lib', '/lib64'):
                self.assertNotIn(directory, policy['read_paths'])
        finally:
            runner.close()
            store.close()

    def test_runtime_rejects_links_into_other_host_content(self):
        stdlib = self.root / 'synthetic-stdlib'
        package = stdlib / 'encodings'
        package.mkdir(parents=True)
        hidden = self.root / 'outside.py'
        hidden.write_text('not runtime')
        (package / 'alias.py').symlink_to(hidden)
        paths = {'stdlib': str(stdlib), 'purelib': str(self.root / 'host-packages')}
        with mock.patch('auriga.runtime.programs.runner.sysconfig.get_path', side_effect=lambda name: paths[name]):
            with self.assertRaisesRegex((ValueError, RuntimeError), 'runtime_.*external_symlink'):
                program_runtime(self.root / 'bad-runtime')

    def test_computation_packages_follow_actual_imports_not_venv_install_destination(self):
        actual_get_path = sysconfig.get_path
        empty_site = self.root / 'empty-venv' / 'site-packages'
        empty_site.mkdir(parents=True)
        with mock.patch('auriga.runtime.programs.runner.sysconfig.get_path', side_effect=lambda name:
                        str(empty_site) if name == 'purelib' else actual_get_path(name)):
            runtime, packages, read_paths = program_runtime(self.root / 'inherited-runtime')
        for name in ('numpy', 'scipy', 'PIL', 'cv2'):
            actual = Path(importlib.util.find_spec(name).origin).parent.resolve()
            self.assertEqual((packages / name).resolve(), actual)
            self.assertIn(str(actual), read_paths)
            self.assertNotIn(str(actual.parent), read_paths)
        policy = json.loads(self.spec.read_text())
        policy['read_paths'] = read_paths
        policy['env'].update(PYTHONHOME=str(runtime), PYTHONPATH=str(packages))
        self.spec.write_text(json.dumps(policy))
        self.run_isolated("import numpy,scipy,cv2; from PIL import Image; assert numpy.linalg.norm([3,4])==5; assert Image.new('RGB',(1,1)).size==(1,1)")

    def test_program_cannot_redirect_host_log_publication_via_workspace_links(self):
        store = Store(self.root / 'log-run', dict(run_id='log-fixture', episode_id='ep', env_instance_id='env', env_generation=1))
        harness = SimpleNamespace(store=store, lock=threading.RLock(), cancel=threading.Event(), active_context=None)
        runner = ProgramRunner(program_services(harness))
        secret = self.root / 'private-secret.txt'
        secret.write_text('SYNTHETIC_HOST_SECRET_MUST_NOT_BE_PUBLISHED')
        # A main actor can replace any child of its writable actor mount. That
        # old programs path must no longer control gateway writes or log reads.
        (store.root / 'actor').mkdir(exist_ok=True)
        (store.root / 'actor' / 'programs').symlink_to(self.root)
        source = f'''from pathlib import Path
for name in ('stdout.txt', 'stderr.txt'):
    Path(name).symlink_to({str(secret)!r})
try:
    Path({str(runner.logs / 'attack-link')!r}).symlink_to({str(secret)!r})
except PermissionError:
    pass
else:
    raise AssertionError('program wrote host-owned log directory')
print('only_program_authored_stdout')
'''
        try:
            job = runner.run(dict(source=source, timeout_s=10),
                             dict(thread_id='thread', turn_id='turn', tool_call_id='log-symlink', item_id='item'))
            self.assertEqual(job['status'], 'succeeded', store.resolve_artifact(job['stderr_ref'])[1])
            stdout = store.resolve_artifact(job['stdout_ref'])[1]
            stderr = store.resolve_artifact(job['stderr_ref'])[1]
            self.assertEqual(stdout.strip(), b'only_program_authored_stdout')
            self.assertEqual(stderr, b'')
            self.assertNotIn(secret.read_bytes(), stdout + stderr)
            self.assertTrue(runner.root.is_relative_to(store.root / 'private'))
            self.assertFalse(any(path.is_dir() and path.name.startswith('program-') for path in self.root.iterdir()))
        finally:
            runner.close()
            store.close()

    def test_threads_and_inherited_channel_work_without_socket_creation(self):
        parent,child = socket.socketpair()
        try:
            parent.settimeout(2)
            parent.sendall(b'ping')
            self.run_isolated(f"import os,threading\nresult=[]\nt=threading.Thread(target=lambda:result.append('thread-ok'))\nt.start();t.join()\nassert result==['thread-ok']\nfd={child.fileno()}\nassert os.read(fd,4)==b'ping'\nos.write(fd,b'pong')\n",(child.fileno(),))
            self.assertEqual(parent.recv(4),b'pong')
        finally:
            parent.close();child.close()

    def test_new_unix_abstract_tcp_and_socketpair_are_denied(self):
        source="""import socket
for family in [socket.AF_UNIX,socket.AF_INET,socket.AF_INET6]:
 try:
  socket.socket(family,socket.SOCK_STREAM)
 except PermissionError:
  continue
 raise AssertionError('program created a socket')
try:
 socket.socketpair()
except PermissionError:
 pass
else:
 raise AssertionError('program created a socketpair')
"""
        self.run_isolated(source)

    def test_connect_cannot_reach_outside_path_or_abstract_namespace(self):
        names = [str(self.root/'outside-private.sock'),'\0auriga-isolation-'+self.root.name]
        for name in names:
            with self.subTest(abstract=name.startswith('\0')):
                listener = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
                candidate = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
                try:
                    listener.bind(name); listener.listen()
                    code = f"import socket\ns=socket.socket(fileno={candidate.fileno()})\ntry:\n s.connect({name!r})\nexcept PermissionError:\n pass\nelse:\n raise AssertionError('outside Unix connect succeeded')\n"
                    self.run_isolated(code,(candidate.fileno(),))
                finally:
                    candidate.close();listener.close()

    def test_fork_signals_cross_process_memory_and_fd_access_are_denied(self):
        source="""import ctypes,errno,os
try:
 child=os.fork()
except PermissionError:
 pass
else:
 if child==0: os._exit(99)
 os.waitpid(child,0)
 raise AssertionError('program forked an extra process')
libc=ctypes.CDLL(None,use_errno=True)
seccomp=ctypes.CDLL('libseccomp.so.2')
seccomp.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
for name,args,expected in [
 ('clone3',[0,0],errno.ENOSYS),
 ('kill',[-1,0],errno.EPERM),('tkill',[-1,0],errno.EPERM),('tgkill',[-1,-1,0],errno.EPERM),
 ('rt_sigqueueinfo',[-1,0,0],errno.EPERM),('rt_tgsigqueueinfo',[-1,-1,0,0],errno.EPERM),
 ('ptrace',[2,-1,0,0],errno.EPERM),('process_vm_readv',[-1,0,0,0,0,0],errno.EPERM),('process_vm_writev',[-1,0,0,0,0,0],errno.EPERM),
 ('pidfd_getfd',[-1,0,0],errno.EPERM),('pidfd_send_signal',[-1,0,0,0],errno.EPERM),
 ('io_uring_setup',[1,0],errno.EPERM)]:
 number=seccomp.seccomp_syscall_resolve_name(name.encode())
 if number<0: continue
 ctypes.set_errno(0)
 result=libc.syscall(ctypes.c_long(number),*(ctypes.c_long(value) for value in args))
 assert result==-1 and ctypes.get_errno()==expected,(name,result,ctypes.get_errno())
"""
        self.run_isolated(source)


class ProgramSandboxConfigTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / 'run', dict(run_id='sandbox-config', episode_id='ep',
            env_instance_id='env', env_generation=1))
        self.addCleanup(self.store.close)
        self.identity = dict(thread_id='thread', turn_id='turn', tool_call_id='program')

    def runner(self, sandbox_enabled):
        config = Config.from_dict({'features': {'sandbox_enabled': sandbox_enabled,
            'observation_profile': 'rgb_proprio_v1'}})
        harness = Harness(self.store, WaitFixtureRobot(self.root), config=config)
        harness.start()
        runner = harness.programs = ProgramRunner(harness.program_services())
        self.addCleanup(runner.close)
        return harness, runner

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_enabled_sandbox_broker_computes_image_from_symlinked_interpreter(self):
        import io
        from PIL import Image

        harness, _ = self.runner(True)
        interpreter = self.root / 'venv' / 'bin' / 'python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        (interpreter.parent.parent / 'pyvenv.cfg').write_text(
            f'home = {Path(sys.executable).parent}\ninclude-system-site-packages = true\n')
        image = Image.new('RGB', (4, 4))
        image.paste((255, 0, 0), (0, 0, 2, 2))
        encoded = io.BytesIO()
        image.save(encoded, format='PNG')
        image_ref = self.store.put_artifact(encoded.getvalue(), 'rgb', 'image/png')
        observer_ref = self.store.put_artifact(b'synthetic private data', 'camera_diagnostic',
            'text/plain', 'observer')
        hidden = self.store.root / 'private' / 'host-only.txt'
        hidden.write_text('synthetic host data')
        source = f'''import io
from pathlib import Path
import numpy as np
from PIL import Image
from auriga_client import AurigaClient
client = AurigaClient()
assert client.call('robot_capabilities')['success'] is True
image = Image.open(io.BytesIO(client.artifact({image_ref!r})))
crop = np.asarray(image.crop((0, 0, 2, 2)))
assert crop.mean(axis=(0, 1)).tolist() == [255., 0., 0.]
assert np.linalg.norm([3., 4.]) == 5.
try:
    client.artifact({observer_ref!r})
except RuntimeError as error:
    assert str(error) == 'artifact_access_scope_denied'
else:
    raise AssertionError('observer artifact readable through broker')
for path in {str(hidden)!r}, {str(self.store.root / 'events.sqlite')!r}, '/proc/self/environ':
    try:
        Path(path).read_bytes()
    except PermissionError:
        continue
    raise AssertionError('private host file readable: ' + path)
print('isolated_broker_image_computation_passed')
'''
        with mock.patch('auriga.runtime.programs.runner.sys.executable', str(interpreter)):
            response = harness.dispatch('program_run', dict(source=source, timeout_s=20), self.identity)
        job = json.loads(response['contentItems'][0]['text'])
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        self.assertIn('isolated_broker_image_computation_passed', job['stdout_text'])
        self.assertEqual(job['calls'], 1)
        self.assertEqual(harness.robot.native_count, 0)
        self.assertTrue(job['stop_confirmed'])

    def test_disabled_sandbox_runs_real_broker_and_review_without_system_dependencies(self):
        harness, runner = self.runner(False)
        self.assertFalse(runner.services.sandbox_enabled)
        outside = self.root / 'ordinary-host-file.txt'
        outside.write_text('synthetic host content')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            source = f'''from pathlib import Path
import resource, socket
import numpy as np
from auriga_client import AurigaClient
assert Path({str(outside)!r}).read_text() == 'synthetic host content'
with socket.create_connection(('127.0.0.1', {listener.getsockname()[1]}), timeout=1):
    pass
assert resource.getrlimit(resource.RLIMIT_AS) == (1024**3, 1024**3)
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (8*1024**2, 8*1024**2)
assert np.linalg.norm([3., 4.]) == 5.
client = AurigaClient()
assert client.call('robot_capabilities')['success'] is True
reply = client.call('robot_wait', {{'native_steps': 1, 'timeout_s': 1,
    'observation_id': {harness.latest['observation_id']!r}}})
assert reply['kind'] == 'action'
assert reply['data']['receipt']['status'] == 'completed'
print('broker_and_review_completed')
'''
            with (mock.patch('auriga.runtime.programs.runner.program_runtime', side_effect=AssertionError('program_runtime called')),
                  mock.patch('auriga.runtime.programs.runner.system_library_path', side_effect=AssertionError('seccomp lookup')),
                  mock.patch('auriga.runtime.programs.runner.runtime_library_paths', side_effect=AssertionError('ldd lookup'))):
                job = runner.run(dict(source=source, timeout_s=20), self.identity)
                self.assertEqual(job['status'], 'awaiting_review', job.get('stderr_text'))
                self.assertEqual(harness.robot.native_count, 0)
                self.assertEqual(job['pending_review']['kind'], 'robot_wait')
                final = runner.run(dict(operation='resume', job_id=job['job_id'],
                    review_id=job['pending_review']['review_id'], decision='approve'),
                    dict(self.identity, tool_call_id='approve'))
            self.assertIsNone(runner.runtime)
            self.assertEqual(final['status'], 'succeeded', final.get('stderr_text'))
            self.assertEqual(final['calls'], 2)
            self.assertEqual(harness.robot.native_count, 1)
            self.assertIn('broker_and_review_completed', final['stdout_text'])
            self.assertTrue(final['stop_confirmed'])

        timed = runner.run(dict(source='import time\ntime.sleep(30)', timeout_s=1),
            dict(self.identity, tool_call_id='timeout'))
        self.assertEqual(timed['status'], 'cancelled')
        self.assertTrue(timed['stop_confirmed'])

    def test_enabled_sandbox_missing_dependency_fails_without_running_source(self):
        _, runner = self.runner(True)
        marker = self.root / 'must-not-run.txt'
        with mock.patch('auriga.runtime.programs.runner.system_library_path', side_effect=RuntimeError('missing-seccomp-fixture')):
            job = runner.run(dict(source=f'from pathlib import Path\nPath({str(marker)!r}).touch()'),
                self.identity)
        self.assertEqual(job['status'], 'failed')
        self.assertIn('missing-seccomp-fixture', job['reason'])
        self.assertTrue(job['stop_confirmed'])
        self.assertFalse(marker.exists())

    def test_enabled_launcher_rejects_landlock_or_seccomp_failure_without_exec(self):
        spec = self.root / 'launcher.json'
        spec.write_text(json.dumps(dict(read_paths=[], write_paths=[], deny_socket_creation=True,
            bounded_program=True, env={'HOME': str(self.root)})))
        for failing in ('restrict', 'restrict_program_syscalls'):
            with self.subTest(failing=failing), \
                    mock.patch.object(sys, 'argv', ['isolation.py', str(spec), sys.executable, '-c', 'pass']), \
                    mock.patch('platform.machine', return_value='x86_64'), \
                    mock.patch.object(isolation, 'restrict') as landlock, \
                    mock.patch.object(isolation, 'restrict_program_syscalls') as seccomp, \
                    mock.patch.object(isolation.os, 'execvpe') as execute:
                (landlock if failing == 'restrict' else seccomp).side_effect = RuntimeError('isolation-unavailable')
                with self.assertRaisesRegex(RuntimeError, 'isolation-unavailable'):
                    isolation.launch()
                execute.assert_not_called()


if __name__=='__main__':
    unittest.main()
