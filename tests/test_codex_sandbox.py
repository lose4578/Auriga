import os
from pathlib import Path
import shutil
import select
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from io import BytesIO

from auriga.cortex.session.codex_sandbox import CodexSandbox, _UnixForwarder
from support.host_capabilities import bubblewrap_namespaces, require


@require(bubblewrap_namespaces, "the native Codex actor sandbox runs real bubblewrap namespaces")
class CodexSandboxTests(unittest.TestCase):
    @staticmethod
    def _python(): return CodexSandbox._find_python_runtime()

    def test_dynamic_host_tools_accept_temporary_path_wrappers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bin_dir = root / "bin"; bin_dir.mkdir()
            real = {name: CodexSandbox._find_host_tool(name) for name in ("bwrap", "sudo", "setpriv")}
            for name, target in real.items():
                if name == "setpriv":
                    shutil.copy2(target, bin_dir / name)
                    continue
                wrapper = bin_dir / name
                wrapper.write_text(f"#!/bin/sh\nexec {target} \"$@\"\n")
                wrapper.chmod(0o755)
            actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            with patch.dict(os.environ, {"PATH": str(bin_dir)}):
                prefix, env = sandbox.prepare()
                self.assertIn(str(bin_dir / "bwrap"), prefix)
                if not sandbox.user_namespace:
                    self.assertIn(str(bin_dir / "sudo"), prefix)
                    self.assertIn("/opt/auriga-bin/setpriv", prefix)
                result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", "pass"],
                                        env={**os.environ, **env}, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            sandbox.close()

    def test_native_socket_alias_is_repaired_only_for_this_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            daemon = ipc / "daemon"; daemon.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            sandbox.prepare()
            native_name = "a" * 64
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(daemon / native_name)); listener.listen(1)
            alias = ipc / "codex.sock"
            alias.symlink_to(Path(f"/tmp/codex-daemon-{os.getuid()}") / native_name)
            try:
                self.assertTrue(sandbox.repair_socket_alias(alias))
                self.assertEqual(alias.readlink(), daemon / native_name)
                client = socket.socket(socket.AF_UNIX)
                try:
                    client.connect(str(sandbox.host_socket_path))
                    accepted, _ = listener.accept(); accepted.close()
                finally:
                    client.close()
            finally:
                listener.close(); sandbox.close()
            self.assertFalse(daemon.exists())

    def test_host_socket_descriptor_is_pinned_across_daemon_parent_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            external = root / "external"; external.mkdir()
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            sandbox.prepare()
            daemon = ipc / "daemon"
            native_name = "b" * 64
            original = socket.socket(socket.AF_UNIX); original.bind(str(daemon / native_name)); original.listen(1)
            malicious = socket.socket(socket.AF_UNIX); malicious.bind(str(external / native_name)); malicious.listen(1)
            alias = ipc / "codex.sock"
            alias.symlink_to(Path(f"/tmp/codex-daemon-{os.getuid()}") / native_name)
            moved = ipc / "daemon-original"
            try:
                self.assertTrue(sandbox.repair_socket_alias(alias))
                daemon.rename(moved)
                daemon.symlink_to(external, target_is_directory=True)
                alias.unlink()
                alias.symlink_to(Path(f"/tmp/codex-daemon-{os.getuid()}") / native_name)
                with self.assertRaisesRegex(ValueError, "socket_directory_changed"):
                    sandbox.repair_socket_alias(alias)
                client = socket.socket(socket.AF_UNIX)
                try:
                    client.connect(str(sandbox.host_socket_path))
                    accepted, _ = original.accept(); accepted.close()
                    self.assertFalse(select.select([malicious], [], [], .1)[0],
                                     "host bridge followed actor-replaced daemon directory")
                finally:
                    client.close()
            finally:
                original.close(); malicious.close(); sandbox.close()
                if daemon.is_symlink(): daemon.unlink()
                if moved.exists(): shutil.rmtree(moved)

    def test_native_prefix_limits_filesystem_and_allows_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            source = root / "snapshot"
            for path in (actor, home, ipc, private, source): path.mkdir(mode=0o700)
            (actor / "actor.txt").write_text("actor-visible")
            (home / "memory.md").write_text("memory-visible")
            (source / "allowed.py").write_text("snapshot-visible")
            secret = root / "secret.txt"
            secret.write_text("host-hidden")
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(),
                                   read_paths=[(source, Path("/harness"))])
            prefix, env = sandbox.prepare()
            code = ("from pathlib import Path; import os; "
                    "assert Path(os.environ['HOME'], 'actor.txt').read_text() == 'actor-visible'; "
                    "assert Path(os.environ['CODEX_HOME'], 'memory.md').read_text() == 'memory-visible'; "
                    "assert Path('/harness/allowed.py').read_text() == 'snapshot-visible'; "
                    "import sysconfig; package_dirs=[Path(sysconfig.get_path(n)) for n in ('purelib','platlib')]; "
                    "assert not any(p.exists() for p in package_dirs), [(str(p),p.exists()) for p in package_dirs]; "
                    "import subprocess; rg=Path('/opt/auriga-bin/rg'); "
                    "assert not rg.exists() or subprocess.run([str(rg), '-n', 'snapshot-visible', '/harness/allowed.py'], "
                    "capture_output=True).returncode == 0; "
                    "import pip, stat; assert pip.__version__; assert stat.S_IMODE(Path('/tmp').stat().st_mode) == 0o1777; "
                    "assert Path(os.environ['PIP_TARGET']).is_dir(); "
                    "import importlib.util; assert importlib.util.find_spec('synthetic_new_package') is None; "
                    "subprocess.run(['python3', '-c', \"import os; from pathlib import Path; "
                    "Path(os.environ['PIP_TARGET'], 'synthetic_new_package.py').write_text('value = 42')\"], check=True); "
                    "import synthetic_new_package; assert synthetic_new_package.value == 42; "
                    "import tempfile; f=tempfile.TemporaryFile(); f.close(); assert Path(os.environ['TMPDIR']).is_dir(); "
                    "status=Path('/proc/self/status').read_text(); "
                    "assert os.geteuid() == os.getuid(); assert 'CapEff:\\t0000000000000000' in status; "
                    "assert 'NoNewPrivs:\\t1' in status; assert set(os.getgroups()).issubset({os.getgid()}); "
                    f"assert not Path({str(secret)!r}).exists(); "
                    f"assert not Path({str(ipc / 'relays' / 'packages')!r}).exists(); "
                    "assert not Path('/var/run/docker.sock').exists(); "
                    f"assert not Path({str(Path.home())!r}).exists(); "
                    "assert not Path('/etc/shadow').exists()")
            try:
                result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", code], env={**os.environ, **env},
                                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
            finally:
                sandbox.close()
            self.assertFalse((ipc / "relays").exists())

    def test_host_environment_is_cleared_before_namespace_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            prefix, env = sandbox.prepare()
            env_tool = shutil.which("env", path=os.defpath)
            self.assertEqual(prefix[:2], [env_tool, "-i"])
            secret_name = "AURIGA_SYNTHETIC_HOST_ENV_SENTINEL"
            secret_value = "synthetic-secret-must-not-cross"
            code = ("from pathlib import Path\n"
                    f"needle={secret_name + '=' + secret_value!r}.encode()\n"
                    "forbidden=False\n"
                    "for f in Path('/proc').glob('*/environ'):\n"
                    " try:\n"
                    "  if needle in f.read_bytes(): forbidden=True\n"
                    " except OSError: pass\n"
                    "assert not forbidden")
            try:
                result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", code],
                                        env={**os.environ, **env, secret_name: secret_value},
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
            finally:
                sandbox.close()

    def test_loopback_port_is_relayed_through_run_owned_unix_socket(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            service = socket.socket(); service.bind(("127.0.0.1", 0)); service.listen()
            port = service.getsockname()[1]
            def respond():
                client, _ = service.accept()
                with client:
                    self.assertEqual(client.recv(4), b"ping")
                    client.sendall(b"pong")
            worker = threading.Thread(target=respond, daemon=True); worker.start()
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), ports=[port], packages=False)
            prefix, env = sandbox.prepare()
            code = ("import socket; s=socket.create_connection(('127.0.0.1', " + str(port) + ")); "
                    "s.sendall(b'ping'); assert s.recv(4)==b'pong'; s.close()")
            try:
                result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", code], env={**os.environ, **env},
                                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                worker.join(2)
            finally:
                sandbox.close(); service.close()

    def test_parent_termination_does_not_leave_native_descendant(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            pid_file = actor / "child.pid"
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            prefix, env = sandbox.prepare()
            marker = "auriga-sandbox-child-" + root.name
            code = (f"import os,time,pathlib; marker={marker!r}; "
                    "pathlib.Path(os.environ['HOME'],'child.pid').write_text(str(os.getpid())); "
                    "time.sleep(60)")
            process = subprocess.Popen(prefix + [str(sandbox.python_runtime), "-c", code],
                                       env={**os.environ, **env}, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                deadline = __import__("time").monotonic() + 5
                while not pid_file.exists() and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(.02)
                self.assertTrue(pid_file.exists(), "sandbox child did not start")
                self.assertTrue(self._marked_processes(marker), "sandbox descendant was not found from host")
                process.terminate(); process.wait(timeout=5)
                deadline = __import__("time").monotonic() + 3
                while self._marked_processes(marker) and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(.02)
                self.assertFalse(self._marked_processes(marker), "sandbox child survived its parent")
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
                sandbox.close()

    def test_owner_listener_accepts_repeated_sandbox_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            prefix, env = sandbox.prepare()
            try:
                for _ in range(4):
                    result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", "pass"],
                                            env={**os.environ, **env}, capture_output=True, text=True,
                                            timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
            finally:
                sandbox.close()

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork for owner-death lifecycle probe")
    def test_owner_sigkill_closes_guardian_and_kills_native_descendant(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            pid_file = actor / "owner-child.pid"
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            prefix, env = sandbox.prepare()
            marker = "auriga-owner-death-child-" + root.name
            code = (f"import time,pathlib; marker={marker!r}; "
                    "pathlib.Path(__import__('os').environ['HOME'],'owner-child.pid').write_text(str(__import__('os').getpid())); "
                    "time.sleep(60)")
            read_fd, write_fd = os.pipe()
            owner_pid = os.fork()
            if owner_pid == 0:
                os.close(read_fd)
                child = subprocess.Popen(prefix + [str(sandbox.python_runtime), "-c", code],
                                         env={**os.environ, **env}, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                deadline = __import__("time").monotonic() + 5
                while not pid_file.exists() and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(.02)
                os.write(write_fd, b"ready" if pid_file.exists() else b"failed")
                os.close(write_fd)
                __import__("time").sleep(60)
                os._exit(0)
            os.close(write_fd)
            try:
                self.assertTrue(select.select([read_fd], [], [], 8)[0], "owner did not start sandbox child")
                self.assertEqual(os.read(read_fd, 16), b"ready")
                os.close(read_fd)
                self.assertTrue(self._marked_processes(marker))
                # The parent process owns this listening FD. The forked owner
                # inherited the listener and accepted connections. Close the
                # test process's inherited copies after killing that owner so
                # EOF reaches the sandbox watchdog.
                os.kill(owner_pid, 9); os.waitpid(owner_pid, 0)
                sandbox.owner_listener.close()
                sandbox.owner_listener = None
                with sandbox.owner_connections_lock:
                    inherited_connections = tuple(sandbox.owner_connections)
                    sandbox.owner_connections.clear()
                for connection in inherited_connections:
                    connection.close()
                deadline = __import__("time").monotonic() + 5
                while self._marked_processes(marker) and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(.02)
                self.assertFalse(self._marked_processes(marker), "native descendant survived owner SIGKILL")
            finally:
                try: os.kill(owner_pid, 9)
                except ProcessLookupError: pass
                try: os.waitpid(owner_pid, 0)
                except ChildProcessError: pass
                try: os.close(read_fd)
                except OSError: pass
                sandbox.close()

    @staticmethod
    def _marked_processes(marker):
        matches = []
        for entry in Path("/proc").glob("[0-9]*"):
            try:
                if marker.encode() in (entry / "cmdline").read_bytes(): matches.append(entry.name)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        return matches


class PackageProxyTests(unittest.TestCase):
    @staticmethod
    def _exchange(request, fake_upstream=None, package_port=20123):
        with tempfile.TemporaryDirectory() as temp:
            relay = _UnixForwarder(Path(temp) / "packages", 0, packages=True, package_port=package_port)
            client, server = socket.socketpair()
            try:
                if fake_upstream is None:
                    worker = threading.Thread(target=PackageProxyTests._serve_test_client,
                                              args=(relay, server), daemon=True)
                else:
                    with patch.object(_UnixForwarder, "_https_request", new=staticmethod(fake_upstream)):
                        worker = threading.Thread(target=PackageProxyTests._serve_test_client,
                                                  args=(relay, server), daemon=True)
                        worker.start()
                        client.sendall(request)
                        response = bytearray()
                        while chunk := client.recv(65536): response.extend(chunk)
                        worker.join(2)
                        return bytes(response)
                worker.start()
                client.sendall(request)
                response = bytearray()
                while chunk := client.recv(65536): response.extend(chunk)
                worker.join(2)
                return bytes(response)
            finally:
                client.close(); server.close()

    @staticmethod
    def _serve_test_client(relay, client):
        try:
            relay._package_connection(client)
        finally:
            client.close()

    class _Response:
        def __init__(self, body, status=200, headers=None):
            self.body, self.status, self.headers = BytesIO(body), status, headers or {}

        def read(self, size=-1): return self.body.read(size)
        def getheader(self, name, default=None): return self.headers.get(name, default)

    class _Connection:
        def close(self): pass

    def test_allowed_simple_index_rewrites_only_file_download_origin(self):
        html = b'<a href="https://files.pythonhosted.org/packages/x/demo-1.0-py3-none-any.whl#sha256=abc">wheel</a>'
        response = self._Response(html, headers={"Content-Type": "text/html; charset=UTF-8"})
        observed = []
        def upstream(host, path, headers):
            observed.append((host, path))
            return self._Connection(), response
        port = 20123
        request = f"GET /simple/Demo_1/ HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        result = self._exchange(request, upstream, package_port=port)
        self.assertIn(b"200 OK", result)
        self.assertIn(f"http://127.0.0.1:{port}/files/packages/x/demo-1.0-py3-none-any.whl".encode(), result)
        self.assertEqual(observed, [("pypi.org", "/simple/demo-1/")])

    def test_allowed_file_fetch_is_streamed_from_fixed_host(self):
        body = b"wheel-bytes"
        response = self._Response(body, headers={"Content-Length": str(len(body)),
                                                  "Content-Type": "application/octet-stream"})
        observed = []
        def upstream(host, path, headers):
            observed.append((host, path))
            return self._Connection(), response
        port = 20123
        request = f"GET /files/packages/ab/demo-1.whl HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        result = self._exchange(request, upstream, package_port=port)
        self.assertIn(b"200 OK", result)
        self.assertTrue(result.endswith(body))
        self.assertEqual(observed, [("files.pythonhosted.org", "/packages/ab/demo-1.whl")])

    def test_host_method_and_path_allowlist_rejections(self):
        port = 20123
        bad_requests = [
            (f"POST /simple/demo/ HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\n\r\n".encode(), b"405"),
            (b"GET /simple/demo/ HTTP/1.1\r\nHost: example.com\r\n\r\n", b"405"),
            (f"GET http://example.com/ HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode(), b"403"),
            (f"GET /files/packages/../../etc/passwd.whl HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode(), b"403"),
            (f"GET /other/path HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode(), b"404"),
        ]
        for request, status in bad_requests:
            with self.subTest(request=request.split(b" ", 1)[0:2]):
                self.assertIn(status, self._exchange(request, package_port=port))

    def test_private_dns_and_redirects_fail_closed(self):
        with patch("auriga.cortex.session.codex_sandbox.socket.getaddrinfo", return_value=[(socket.AF_INET,
                socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]):
            with self.assertRaisesRegex(OSError, "dns_rejected"):
                _UnixForwarder._https_request("pypi.org", "/simple/demo/", {})

        class RawSocket:
            def settimeout(self, *_): pass
            def connect(self, *_): pass
            def close(self): pass
        class SecureContext:
            def wrap_socket(self, raw, *, server_hostname):
                self.server_hostname = server_hostname
                return raw
        class Redirect:
            status = 302
        class Connection:
            def __init__(self, host, timeout): self.host = host
            def request(self, *_args, **_kwargs): pass
            def getresponse(self): return Redirect()
            def close(self): pass
        with (patch("auriga.cortex.session.codex_sandbox.socket.getaddrinfo", return_value=[(socket.AF_INET,
                    socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]),
              patch("auriga.cortex.session.codex_sandbox.socket.socket", return_value=RawSocket()),
              patch("auriga.cortex.session.codex_sandbox.ssl.create_default_context", return_value=SecureContext()),
              patch("auriga.cortex.session.codex_sandbox.http.client.HTTPSConnection", Connection)):
            with self.assertRaisesRegex(OSError, "redirect_rejected"):
                _UnixForwarder._https_request("pypi.org", "/simple/demo/", {})


if __name__ == "__main__": unittest.main()
