"""Codex namespace and program boundaries; synthetic files, no GPU/model calls."""
from program_fixture import program_services
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codex_cli_bin import bundled_codex_path

from auriga.cortex.session.codex import ACTOR_ISOLATION, CodexSession
from auriga.core.config import Config
from auriga.runtime.programs.runner import ProgramRunner
from auriga.core.storage.store import Store
from support.host_capabilities import bubblewrap_namespaces, program_isolation, require


IDENTITY = dict(run_id="isolation-fixture", episode_id="ep", env_instance_id="env", env_generation=1)


class ActorIsolationTest(unittest.TestCase):
    def test_disabled_sandbox_keeps_private_config_and_uses_host_execution(self):
        from auriga.cortex.context.actor_instructions import instructions
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / 'run', IDENTITY)
            profile = root / 'profile'
            profile.mkdir()
            (profile / 'config.toml').write_text('model_provider="fixture"\n'
                '[model_providers.fixture]\nbase_url="http://127.0.0.1:1/v1"\n')
            config = Config.from_dict({'model': {'codex_home': str(profile)},
                                      'features': {'sandbox_enabled': False}})
            session = CodexSession(store.root, store, lambda _: {}, config=config)
            try:
                with patch('auriga.cortex.session.codex_sandbox.CodexSandbox', side_effect=AssertionError('sandbox used')):
                    session._prepare()
                self.assertIsNone(session.sandbox)
                self.assertEqual(session.command_prefix, [])
                self.assertEqual(session.command[0], str(bundled_codex_path().resolve()))
                self.assertEqual(session.env['HOME'], str(session.actor_dir))
                self.assertEqual(session.env['CODEX_HOME'], str(session.codex_home))
                host_file = root / 'synthetic-host-file'
                host_file.write_text('host-visible')
                result = subprocess.run(session.command_prefix + ['python3', '-c',
                    f'from pathlib import Path; print(Path({str(host_file)!r}).read_text())'],
                    env=session.env, cwd=session.actor_dir, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), 'host-visible')
                text = instructions('observation fixture', 'handoff fixture',
                                    source_root=session.private / 'actor-source')
                self.assertIn(str(session.private / 'actor-source'), text)
                self.assertTrue((session.private / 'actor-source/auriga/runtime/harness.py').is_file())
                self.assertIn('without OS sandbox isolation', text)
                self.assertNotIn('inside your sandbox', text)
                self.assertNotIn('general network access are unavailable', text)
            finally:
                session.close()
                store.close()

    def test_resume_rejects_sandbox_mode_changes_before_starting_processes(self):
        for enabled, previous in ((True, 'disabled'), (False, ACTOR_ISOLATION)):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                store = Store(temporary, IDENTITY)
                session = CodexSession(store.root, store, lambda _: {},
                    config=Config.from_dict({'features': {'sandbox_enabled': enabled}}))
                store.set_state('runtime', 'codex', {'isolation': previous})
                try:
                    with patch.object(session, '_stop_previous_owned_processes') as stop:
                        with self.assertRaisesRegex(ValueError, 'resume_isolation_changed'):
                            session.start([], 'fixture', resume_thread_id='fixture-thread')
                        stop.assert_not_called()
                    with self.assertRaisesRegex(ValueError, 'resume_isolation_changed'):
                        session._prepare()
                finally:
                    session.close()
                    store.close()

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_generated_codex_namespace_allows_workspace_but_denies_host_files_and_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / "run", IDENTITY)
            profile = root / "profile"
            profile.mkdir()
            (profile / "config.toml").write_text('model_provider = "fixture"\n'
                '[model_providers.fixture]\nbase_url = "http://127.0.0.1:1/v1"\n')
            session = CodexSession(store.root, store, lambda params: None,
                                   config=Config.from_dict({"features": {"sandbox_enabled": True},
                                                            "model": {"codex_home": str(profile)}}))
            external_file = root / "outside-run.txt"
            external_file.write_text("synthetic ordinary file")
            output = session.actor_dir / "workspace-output.txt"
            try:
                session._prepare()
                config = tomllib.loads((session.codex_home / "config.toml").read_text())
                self.assertTrue(config["features"]["shell_tool"])
                self.assertTrue(config["features"]["unified_exec"])
                self.assertEqual(config["approval_policy"], "never")
                self.assertEqual(config["sandbox_mode"], "danger-full-access")
                self.assertEqual(session.command[-5:], [str(bundled_codex_path().resolve()),
                    "--no-daemon", "app-server", "--listen", f"unix://{session.socket_path}"])
                self.assertGreater(len(session.command), 5)
                self.assertEqual(session.env["HOME"], str(session.actor_dir))
                self.assertEqual(session.env["CODEX_HOME"], str(session.codex_home))
                self.assertFalse((session.private / "isolation.json").exists())
                permitted = session.actor_dir / 'notes.txt'
                permitted.write_text('synthetic workspace note')
                shortcut = session.actor_dir / 'host-link'
                shortcut.symlink_to(external_file)
                hidden = [external_file, profile / 'config.toml', store.root / 'events.sqlite', shortcut,
                          Path(__file__).resolve().parents[1] / 'auriga' / 'robot' / 'backends' / 'robodojo' / 'robot_worker.py',
                          Path('/harness/auriga/robot/backends/robodojo/robot_worker.py')]
                host_network_namespace = os.readlink('/proc/self/ns/net')
                host_pid_namespace = os.readlink('/proc/self/ns/pid')
                with socket.socket() as local_server:
                    local_server.bind(("127.0.0.1", 0))
                    local_server.listen()
                    source = f"""from pathlib import Path
import os, socket, tempfile
assert os.readlink('/proc/self/ns/net') != {host_network_namespace!r}
assert os.readlink('/proc/self/ns/pid') != {host_pid_namespace!r}
assert Path({str(permitted)!r}).read_text() == 'synthetic workspace note'
Path({str(output)!r}).write_text('synthetic workspace write succeeded')
with tempfile.TemporaryDirectory() as temporary:
    Path(temporary, 'probe').write_text('private tmp is writable')
assert Path('/harness/auriga/runtime/harness.py').read_text()
for path in {list(map(str, hidden))!r}:
    try:
        Path(path).read_bytes()
    except (PermissionError, FileNotFoundError):
        continue
    raise AssertionError('host file readable: ' + path)
try:
    socket.create_connection(('127.0.0.1', {local_server.getsockname()[1]}), timeout=1)
except OSError:
    pass
else:
    raise AssertionError('host network reachable')
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    assert listener.getsockname()[1] > 0
print('codex_namespace_boundary_passed')
"""
                    result = subprocess.run(session.command[:-5] + ['python3', '-c', source], env=session.env,
                                            cwd=session.actor_dir, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                self.assertEqual(result.stdout.strip(), b"codex_namespace_boundary_passed")
                self.assertEqual(output.read_text(), "synthetic workspace write succeeded")
            finally:
                session.close()
                if hasattr(session, "daemon_socket"):
                    session.daemon_socket.with_suffix(".lock").unlink(missing_ok=True)
                store.close()

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_program_channel_denies_observer_but_preserves_actor_derived_internal_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "run", IDENTITY)
            harness = SimpleNamespace(store=store, lock=threading.RLock(), cancel=threading.Event(), active_context=None)
            runner = ProgramRunner(program_services(harness))
            actor_ref = store.put_artifact(b"visible RGB fixture", "rgb", "image/png")
            internal_ref = store.put_artifact(b"actor-derived geometry fixture", "scene_geometry", "text/plain", "internal")
            observer_ref = store.put_artifact(b"synthetic camera truth", "camera_diagnostic", "application/json", "observer")
            _, blob = store.artifact(observer_ref)
            source = f"""from pathlib import Path
import os
from auriga_client import AurigaClient
os.chdir(os.environ['HOME'])
client = AurigaClient()
assert client.artifact({actor_ref!r}) == b'visible RGB fixture'
assert client.artifact({internal_ref!r}) == b'actor-derived geometry fixture'
try:
    client.artifact({observer_ref!r})
except RuntimeError as error:
    assert str(error) == 'artifact_access_scope_denied'
else:
    raise AssertionError('observer artifact readable through broker')
Path('truth-link').symlink_to({str(blob)!r})
for path in [{str(blob)!r}, 'truth-link', {str(store.root / 'events.sqlite')!r}]:
    try:
        Path(path).read_bytes()
    except PermissionError:
        continue
    raise AssertionError('observer data readable through filesystem')
try:
    client.call('observer_evaluate')
except RuntimeError as error:
    assert str(error) == 'program_tool_not_permitted'
else:
    raise AssertionError('observer operation callable')
print('program_boundary_passed')
"""
            try:
                result = runner.run(dict(source=source, timeout_s=10),
                                    dict(thread_id="thread", turn_id="turn", tool_call_id="isolation", item_id="item"))
                self.assertEqual(result["status"], "succeeded", store.resolve_artifact(result["stderr_ref"])[1])
                self.assertEqual(store.resolve_artifact(result["stdout_ref"])[1].strip(), b"program_boundary_passed")
            finally:
                runner.close()
                store.close()


if __name__ == "__main__":
    unittest.main()
