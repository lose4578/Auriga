"""AgentSandbox extraction (WP R4-5): Codex golden, mount/forwarder plans, inbound forwarding.

Planning tests stub the host probes (``agent_sandbox_golden_fixture.planning_host``) and run on
every host. Tests that start a real bubblewrap namespace run only where bubblewrap user
namespaces work (Linux) and skip with an explicit reason elsewhere;
``AURIGA_REQUIRE_SANDBOX_TESTS=1`` turns that skip into a failure on a host that must support them.
"""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import agent_sandbox_golden_fixture as golden
import auriga
from auriga.cortex.session import agent_sandbox as agent_sandbox_module
from auriga.cortex.session.agent_sandbox import (
    AgentSandbox, INBOUND_SOCKET_DIR, _SUPERVISOR, _SUPERVISOR_INBOUND, _SUPERVISOR_INBOUND_DEFINITIONS,
    _SUPERVISOR_INBOUND_START, _SUPERVISOR_LAUNCH, _SUPERVISOR_PRELUDE, _SUPERVISOR_RELAYS)
from auriga.cortex.session.codex_sandbox import CodexSandbox
from support.host_capabilities import bubblewrap_namespaces, require

REPO_ROOT = Path(auriga.__file__).resolve().parents[1]
ROBODOJO_DEFAULT_ROOT = Path("~/code/robodojo").expanduser()


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP over a Unix stream socket (what the host-side OpenCode client does with ``unix_socket=``)."""

    def __init__(self, path, timeout=5):
        super().__init__("127.0.0.1", timeout=timeout)
        self.unix_path = str(path)

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.unix_path)
        self.sock = sock


class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"healthy:" + self.path.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def _get(path, url="/global/health"):
    connection = _UnixHTTPConnection(path)
    try:
        connection.request("GET", url)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class CodexGoldenTest(unittest.TestCase):
    """The Codex bwrap argv, environment, relays and IPC lifecycle are byte-identical to 7d5c83b."""
    maxDiff = None

    def test_codex_plan_matches_pre_extraction_golden(self):
        recorded = json.loads(golden.FIXTURE.read_text())
        self.assertEqual(recorded["captured_from"]["revision"], "7d5c83b")
        self.assertEqual(set(recorded["scenarios"]), set(golden.CODEX_SCENARIOS))
        current = json.loads(json.dumps(golden.capture_all()))
        for name, expected in recorded["scenarios"].items():
            with self.subTest(scenario=name):
                for key in expected:
                    self.assertEqual(current[name][key], expected[key], key)
                self.assertEqual(current[name], expected)

    def test_outbound_supervisor_is_the_historical_source(self):
        self.assertEqual(_SUPERVISOR, _SUPERVISOR_PRELUDE + _SUPERVISOR_RELAYS + _SUPERVISOR_LAUNCH)
        self.assertNotIn("inbound", _SUPERVISOR)
        recorded = json.loads(golden.FIXTURE.read_text())["scenarios"]
        for scenario in recorded.values():
            self.assertEqual(scenario["command_prefix"][-1], _SUPERVISOR)

    def test_historical_codex_sandbox_names_stay_patchable(self):
        from auriga.cortex.session import codex, codex_sandbox
        self.assertEqual(codex.ACTOR_ISOLATION, agent_sandbox_module.ACTOR_ISOLATION)
        self.assertEqual(codex.ACTOR_ISOLATION, "bubblewrap_namespaces_package_egress_v1")
        self.assertIs(codex_sandbox._SUPERVISOR, _SUPERVISOR)
        self.assertIs(codex_sandbox._UnixForwarder, agent_sandbox_module._UnixForwarder)
        self.assertTrue(issubclass(CodexSandbox, AgentSandbox))
        for target in ("auriga.cortex.session.codex_sandbox.socket.getaddrinfo",
                       "auriga.cortex.session.codex_sandbox.socket.socket",
                       "auriga.cortex.session.codex_sandbox.ssl.create_default_context",
                       "auriga.cortex.session.codex_sandbox.http.client.HTTPSConnection",
                       "auriga.cortex.session.codex_sandbox.CodexSandbox.prepare",
                       "auriga.cortex.session.codex_sandbox.CodexSandbox"):
            with self.subTest(target=target), mock.patch(target):
                pass

    def test_error_names_keep_the_backend_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            executable = root / "codex"
            executable.write_text("")
            with self.assertRaisesRegex(ValueError, "^codex_sandbox_invalid_relay_port$"):
                CodexSandbox(root / "p", root / "i", root / "a", root / "h", executable, ports=[0], packages=False)
            with self.assertRaisesRegex(ValueError, "^agent_sandbox_invalid_relay_port$"):
                AgentSandbox(root / "p", root / "i", root / "a", ports=[70000], packages=False)
            # The bubblewrap error documented for non-Linux hosts is unchanged for Codex.
            (root / "i").mkdir()
            with mock.patch("shutil.which", return_value=None):
                for sandbox, name in ((CodexSandbox(root / "p", root / "i", root / "a", root / "h", executable,
                                                    packages=False), "codex"),
                                      (AgentSandbox(root / "p", root / "i", root / "a", packages=False), "agent")):
                    with self.subTest(sandbox=name), self.assertRaisesRegex(
                            RuntimeError, f"^{name}_sandbox_bubblewrap_not_found_on_path_or_os_defpath$"):
                        sandbox.prepare()


class AgentSandboxPlanTest(unittest.TestCase):
    """Mount, environment and relay plans for Claude-Code-like and OpenCode-like configurations."""
    maxDiff = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="r45-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.actor, self.ipc, self.private = self.root / "run/actor", self.root / "ipc", self.root / "run/private"
        self.snapshot = self.private / "actor-source"
        for path in (self.actor, self.ipc, self.snapshot / "auriga"):
            path.mkdir(parents=True, mode=0o700)
        (self.snapshot / "auriga/__init__.py").write_text("")

    def _relative(self, mounts):
        return [(golden.normalise(str(source), self.root), golden.normalise(str(target), self.root), readonly)
                for source, target, readonly in mounts]

    @staticmethod
    def _setenv(prefix):
        return {prefix[index + 1]: prefix[index + 2] for index, item in enumerate(prefix) if item == "--setenv"}

    @staticmethod
    def _contains(prefix, *items):
        return any(prefix[index:index + len(items)] == list(items) for index in range(len(prefix)))

    def _assert_no_checkout_or_simulator_mount(self, sandbox):
        for source, _target, _readonly in sandbox.mounts:
            source = Path(source)
            for forbidden in (REPO_ROOT, ROBODOJO_DEFAULT_ROOT):
                self.assertFalse(source == forbidden or forbidden.is_relative_to(source)
                                 or source.is_relative_to(forbidden), (source, forbidden))

    def test_claude_code_like_plan(self):
        config = self.private / "claude-config"
        cli = golden._executable(self.root / "host/claude_agent_sdk/_bundled/claude")
        with golden.planning_host(self.root, AgentSandbox) as host:
            sandbox = AgentSandbox(self.private / "sandbox", self.ipc, self.actor,
                                   state_dirs=[(config, config)],
                                   executables=[(cli, Path("/opt/auriga-bin/claude"))],
                                   env={"CLAUDE_CONFIG_DIR": str(config),
                                        "ANTHROPIC_BASE_URL": "http://127.0.0.1:43001", "IS_SANDBOX": "1"},
                                   ports=[43002, 43001], packages=True,
                                   read_paths=[(self.snapshot, Path("/harness"))])
            try:
                prefix, env = sandbox.prepare()
                self.assertTrue(config.is_dir() and stat.S_IMODE(config.stat().st_mode) == 0o700)
                mounts = self._relative(sandbox.mounts)
                self.assertEqual(mounts[:5], [
                    ("<ROOT>/run/actor", "<ROOT>/run/actor", False),
                    ("<ROOT>/run/private/claude-config", "<ROOT>/run/private/claude-config", False),
                    ("<ROOT>/ipc", "<ROOT>/ipc", False),
                    ("<ROOT>/ipc/relays", "/run/auriga-relays", True),
                    ("<ROOT>/host/claude_agent_sdk/_bundled/claude", "/opt/auriga-bin/claude", True)])
                self.assertEqual(mounts[-2:], [("<ROOT>/host/bin/rg", "/opt/auriga-bin/rg", True),
                                               ("<ROOT>/run/private/actor-source", "/harness", True)])
                self.assertFalse(any("codex" in target for _, target, _ in mounts))
                self.assertFalse(any(target.startswith(INBOUND_SOCKET_DIR) for _, target, _ in mounts))
                self._assert_no_checkout_or_simulator_mount(sandbox)
                self.assertEqual(host.library_calls, [[str(cli), host.tools["rg"]]])
                self.assertTrue(self._contains(prefix, "--ro-bind", str(cli), "/opt/auriga-bin/claude"))
                self.assertTrue(self._contains(prefix, "--bind", str(config), str(config)))
                self.assertTrue(self._contains(prefix, "--chdir", str(self.actor), "--"))
                self.assertNotIn(INBOUND_SOCKET_DIR, prefix)
                self.assertEqual(prefix[-1], _SUPERVISOR)
                self.assertEqual(list(env), ["PATH", "LANG", "TERM", "COLORTERM", "HOME", "CLAUDE_CONFIG_DIR",
                                             "ANTHROPIC_BASE_URL", "IS_SANDBOX", "TMPDIR", "PIP_INDEX_URL",
                                             "PIP_TRUSTED_HOST", "PIP_CONFIG_FILE", "PIP_TARGET", "PYTHONPATH"])
                self.assertEqual(env["PATH"], "/opt/auriga-bin:" + os.defpath)
                self.assertEqual(env["HOME"], str(self.actor))
                self.assertNotIn("CODEX_HOME", env)
                inside = self._setenv(prefix)
                self.assertEqual(inside["AURIGA_RELAY_PORTS"], "43001,43002")
                self.assertEqual(inside["AURIGA_PACKAGE_PROXY_PORT"], "18888")
                self.assertNotIn("AURIGA_INBOUND_PORTS", inside)
                self.assertEqual({key: value for key, value in inside.items() if not key.startswith("AURIGA_")}, env)
                self.assertEqual([(item.path.name, item.port, item.packages) for item in sandbox.forwarders],
                                 [("tcp-43001", 43001, False), ("tcp-43002", 43002, False), ("packages", 0, True)])
                for name in ("tcp-43001", "tcp-43002", "packages"):
                    self.assertTrue(stat.S_ISSOCK((self.ipc / "relays" / name).stat().st_mode))
                self.assertTrue((self.actor / ".python-bootstrap/pip/__init__.py").is_file())
                self.assertTrue((self.actor / ".packages").is_dir())
            finally:
                sandbox.close()
        self.assertEqual(list(self.ipc.iterdir()), [])
        self.assertFalse(sandbox.prepared)

    def test_opencode_like_plan_with_inbound_server_port(self):
        xdg = self.private / "opencode/xdg"
        data, cache, state, config = (xdg / name for name in ("data", "cache", "state", "config"))
        config.mkdir(parents=True, mode=0o700)
        instructions = self.private / "opencode/instructions/auriga.md"
        instructions.parent.mkdir(parents=True, mode=0o700)
        instructions.write_text("instructions")
        cli = golden._executable(self.root / "host/opencode-linux-x64/bin/opencode")
        inbound_dir = self.ipc / "inbound"
        inbound_dir.mkdir(mode=0o700)
        (inbound_dir / "4096.sock").write_text("stale socket from a crashed run")
        with golden.planning_host(self.root, AgentSandbox) as host:
            sandbox = AgentSandbox(self.private / "sandbox", self.ipc, self.actor,
                                   state_dirs=[(data, data), (cache, cache), (state, state)],
                                   executables=[(cli, Path("/opt/auriga-bin/opencode"))],
                                   env={"XDG_DATA_HOME": str(data), "XDG_CACHE_HOME": str(cache),
                                        "XDG_STATE_HOME": str(state), "XDG_CONFIG_HOME": str(config),
                                        "OPENCODE_CONFIG_CONTENT": "{}"},
                                   ports=[43001, 43002], inbound_ports=[4096], packages=True,
                                   read_paths=[(config, config), (instructions, instructions),
                                               (self.snapshot, Path("/harness"))])
            try:
                prefix, env = sandbox.prepare()
                self.assertEqual(sandbox.inbound_socket_path(4096), inbound_dir / "4096.sock")
                self.assertFalse((inbound_dir / "4096.sock").exists(), "stale inbound socket was kept")
                self.assertEqual(stat.S_IMODE(inbound_dir.stat().st_mode), 0o700)
                with self.assertRaisesRegex(ValueError, "agent_sandbox_port_not_inbound"):
                    sandbox.inbound_socket_path(43001)
                mounts = self._relative(sandbox.mounts)
                state_mounts = [(f"<ROOT>/run/private/opencode/xdg/{name}",) * 2 + (False,)
                                for name in ("data", "cache", "state")]
                self.assertEqual(mounts[:8], [
                    ("<ROOT>/run/actor", "<ROOT>/run/actor", False), *state_mounts,
                    ("<ROOT>/ipc", "<ROOT>/ipc", False),
                    ("<ROOT>/ipc/relays", "/run/auriga-relays", True),
                    ("<ROOT>/ipc/inbound", INBOUND_SOCKET_DIR, False),
                    ("<ROOT>/host/opencode-linux-x64/bin/opencode", "/opt/auriga-bin/opencode", True)])
                # Read-only paths follow the state mounts, so the config dir stays read-only.
                self.assertEqual(mounts[-4:], [
                    ("<ROOT>/host/bin/rg", "/opt/auriga-bin/rg", True),
                    ("<ROOT>/run/private/opencode/xdg/config", "<ROOT>/run/private/opencode/xdg/config", True),
                    ("<ROOT>/run/private/opencode/instructions/auriga.md",
                     "<ROOT>/run/private/opencode/instructions/auriga.md", True),
                    ("<ROOT>/run/private/actor-source", "/harness", True)])
                self._assert_no_checkout_or_simulator_mount(sandbox)
                self.assertEqual(host.library_calls, [[str(cli), host.tools["rg"]]])
                self.assertTrue(self._contains(prefix, "--dir", "/run/auriga-relays", "--dir", INBOUND_SOCKET_DIR))
                self.assertTrue(self._contains(prefix, "--bind", str(inbound_dir), INBOUND_SOCKET_DIR))
                self.assertTrue(self._contains(prefix, "--tmpfs", str(self.ipc / "relays"),
                                               "--tmpfs", str(inbound_dir)))
                inside = self._setenv(prefix)
                self.assertEqual(inside["AURIGA_INBOUND_PORTS"], "4096")
                self.assertEqual(inside["AURIGA_RELAY_PORTS"], "43001,43002")
                self.assertNotIn("AURIGA_INBOUND_PORTS", env)
                self.assertEqual(env["XDG_CONFIG_HOME"], str(config))
                self.assertEqual(prefix[-1], _SUPERVISOR_INBOUND)
                self.assertEqual([(item.path.name, item.port) for item in sandbox.forwarders],
                                 [("tcp-43001", 43001), ("tcp-43002", 43002), ("packages", 0)])
                # A socket the in-namespace supervisor left behind is removed on close.
                (inbound_dir / "4096.sock").write_text("bound inside")
            finally:
                sandbox.close()
        self.assertEqual(list(self.ipc.iterdir()), [])

    def test_package_proxy_port_avoids_relay_and_inbound_ports(self):
        sandbox = AgentSandbox(self.private / "sandbox", self.ipc, self.actor, ports=[18889],
                               inbound_ports=[18888], packages=False)
        self.assertEqual(sandbox.package_proxy_port, 18890)

    def test_constructor_validation(self):
        base = (self.private / "sandbox", self.ipc, self.actor)
        cases = [
            (dict(env={"TMPDIR": "/x"}), "agent_sandbox_reserved_environment_name:TMPDIR"),
            (dict(env={"AURIGA_INBOUND_PORTS": "1"}), "agent_sandbox_reserved_environment_name:AURIGA_INBOUND_PORTS"),
            (dict(env={"PYTHONPATH": "/x"}), "agent_sandbox_reserved_environment_name:PYTHONPATH"),
            (dict(env={"BAD-NAME": "x"}), "agent_sandbox_invalid_environment"),
            (dict(env={"NUL": "a\0b"}), "agent_sandbox_invalid_environment"),
            (dict(ports=[4096], inbound_ports=[4096]), "agent_sandbox_inbound_port_conflicts_with_relay_port"),
            (dict(inbound_ports=[0]), "agent_sandbox_invalid_inbound_port"),
            (dict(inbound_ports=[65536]), "agent_sandbox_invalid_inbound_port"),
            (dict(state_dirs=[(self.private / "state", "relative/state")]), "agent_sandbox_invalid_state_mount"),
            (dict(executables=[(golden._executable(self.root / "cli"), "opt/cli")]),
             "agent_sandbox_invalid_executable_mount"),
            (dict(read_paths=[(self.snapshot, "/harness/../etc")]), "agent_sandbox_invalid_read_mount"),
        ]
        for kwargs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                AgentSandbox(*base, packages=False, **kwargs)
        with self.assertRaisesRegex(ValueError, "agent_sandbox_ipc_path_too_long_for_inbound_socket"):
            AgentSandbox(self.private / "sandbox", self.root / ("i" * 100), self.actor, inbound_ports=[4096],
                         packages=False)
        with self.assertRaises(FileNotFoundError):
            AgentSandbox(*base, packages=False, executables=[(self.root / "missing-cli", "/opt/cli")])


class InboundForwardingTest(unittest.TestCase):
    """The supervisor's inbound listener code, run as plain sockets on the host (no namespace)."""

    def setUp(self):
        namespace = {}
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("AURIGA_RELAY_PORTS", "AURIGA_PACKAGE_PROXY_PORT")}
        with mock.patch.dict(os.environ, environment, clear=True):
            # The exact fragments the sandbox runs; with no relay ports configured they only define functions.
            exec(compile("import os,socket,threading\n" + _SUPERVISOR_RELAYS + _SUPERVISOR_INBOUND_DEFINITIONS,
                         "<auriga-sandbox-supervisor>", "exec"), namespace)
        self.start_inbound = namespace["start_inbound"]
        temporary = tempfile.TemporaryDirectory(prefix="r45-", dir="/tmp")  # short AF_UNIX paths
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def _listen(self, ports):
        servers = self.start_inbound(str(self.directory), ports)

        def stop():
            for server in servers:
                try: server.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                server.close()
        self.addCleanup(stop)
        return servers

    def _http_service(self):
        service = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
        threading.Thread(target=service.serve_forever, daemon=True).start()
        self.addCleanup(service.server_close)
        self.addCleanup(service.shutdown)
        return service.server_address[1]

    def test_unix_socket_reaches_loopback_http_server(self):
        port = self._http_service()
        stale = self.directory / f"{port}.sock"
        stale.write_text("stale")
        self._listen([port])
        self.assertTrue(stat.S_ISSOCK(stale.stat().st_mode))
        self.assertEqual(stat.S_IMODE(stale.stat().st_mode), 0o600)
        for url in ("/global/health", "/session"):  # the accept loop keeps serving
            self.assertEqual(_get(stale, url), (200, b"healthy:" + url.encode()))

    def test_refused_inside_connection_closes_the_host_connection(self):
        port = _free_port()
        self._listen([port])
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        try:
            client.connect(str(self.directory / f"{port}.sock"))
            client.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            try:
                self.assertEqual(client.recv(1), b"")
            except ConnectionResetError:
                pass
        finally:
            client.close()

    def test_inbound_supervisor_composition(self):
        self.assertEqual(_SUPERVISOR_INBOUND, _SUPERVISOR_PRELUDE + _SUPERVISOR_RELAYS
                         + _SUPERVISOR_INBOUND_DEFINITIONS + _SUPERVISOR_INBOUND_START + _SUPERVISOR_LAUNCH)
        self.assertIn(f'start_inbound("{INBOUND_SOCKET_DIR}",', _SUPERVISOR_INBOUND_START)
        compile(_SUPERVISOR_INBOUND, "<auriga-sandbox-supervisor>", "exec")




@require(bubblewrap_namespaces, "Claude-Code-like and OpenCode-like agents run in a real AgentSandbox namespace")
class RealBubblewrapAgentSandboxTest(unittest.TestCase):
    """Claude-Code-like and OpenCode-like agents in a real namespace (Linux with bubblewrap only)."""

    def _layout(self, temporary):
        root = Path(temporary).resolve()
        actor, ipc, private = root / "actor", root / "ipc", root / "private"
        snapshot = private / "actor-source"
        for path in (actor, ipc, private, snapshot):
            path.mkdir(parents=True, mode=0o700, exist_ok=True)
        (snapshot / "allowed.py").write_text("snapshot-visible")
        # The agent CLI stand-in: an executable published read-only at a sandbox path.
        cli = golden._executable(root / "cli/agent-cli", '#!/bin/sh\nexec /usr/bin/python3 "$@"\n')
        return root, actor, ipc, private, snapshot, cli

    def test_claude_code_like_agent_reaches_relays_and_sees_only_published_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, actor, ipc, private, snapshot, cli = self._layout(temporary)
            config = private / "agent-config"
            secret = root / "secret.txt"
            secret.write_text("host-hidden")
            relay = socket.socket()
            relay.bind(("127.0.0.1", 0)); relay.listen()
            relay_port, closed_port = relay.getsockname()[1], _free_port()

            def respond():
                client, _ = relay.accept()
                with client:
                    if client.recv(4) == b"ping":
                        client.sendall(b"pong")
            worker = threading.Thread(target=respond, daemon=True)
            worker.start()
            sandbox = AgentSandbox(private / "sandbox", ipc, actor, state_dirs=[(config, config)],
                                   executables=[(cli, "/opt/auriga-bin/agent-cli")],
                                   env={"AGENT_CONFIG_DIR": str(config), "IS_SANDBOX": "1"},
                                   ports=[relay_port], packages=False, read_paths=[(snapshot, "/harness")])
            code = "\n".join([
                "import os, socket", "from pathlib import Path",
                "assert Path('/harness/allowed.py').read_text() == 'snapshot-visible'",
                "Path(os.environ['AGENT_CONFIG_DIR'], 'state.json').write_text('{}')",
                f"assert os.environ['IS_SANDBOX'] == '1' and os.environ['HOME'] == {str(actor)!r}",
                f"assert os.getcwd() == {str(actor)!r}",
                f"for hidden in {[str(secret), str(REPO_ROOT), str(Path.home()), str(ROBODOJO_DEFAULT_ROOT)]!r}:",
                "    assert not Path(hidden).exists(), hidden",
                f"s = socket.create_connection(('127.0.0.1', {relay_port}), timeout=5)",
                "s.sendall(b'ping'); assert s.recv(4) == b'pong'; s.close()",
                f"for address in (('127.0.0.1', {closed_port}), ('192.0.2.1', 9)):",
                "    try: socket.create_connection(address, timeout=2).close()",
                "    except OSError: pass",
                "    else: raise AssertionError(('reachable', address))"])
            try:
                prefix, env = sandbox.prepare()
                result = subprocess.run(prefix + ["/opt/auriga-bin/agent-cli", "-c", code],
                                        env={**os.environ, **env}, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((config / "state.json").is_file())
                worker.join(2)
            finally:
                sandbox.close()
                relay.close()

    def test_opencode_like_server_is_reachable_through_the_inbound_socket_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, actor, ipc, private, snapshot, cli = self._layout(temporary)
            data, config = private / "xdg/data", private / "xdg/config"
            config.mkdir(parents=True, mode=0o700)
            port = _free_port()  # free on the host too, so the host-side refusal below is meaningful
            sandbox = AgentSandbox(private / "sandbox", ipc, actor, state_dirs=[(data, data)],
                                   executables=[(cli, "/opt/auriga-bin/opencode")],
                                   env={"XDG_DATA_HOME": str(data), "XDG_CONFIG_HOME": str(config)},
                                   inbound_ports=[port], packages=False,
                                   read_paths=[(config, config), (snapshot, "/harness")])
            server = "\n".join([
                "import os", "from pathlib import Path",
                "from http.server import BaseHTTPRequestHandler, HTTPServer",
                "try: Path(os.environ['XDG_CONFIG_HOME'], 'opencode.json').write_text('x')",
                "except OSError: pass",
                "else: raise SystemExit('read-only config directory is writable')",
                "Path(os.environ['XDG_DATA_HOME'], 'db').write_text('ok')",
                "class Health(BaseHTTPRequestHandler):",
                "    def do_GET(self):",
                "        body = b'healthy:' + self.path.encode()",
                "        self.send_response(200); self.send_header('Content-Length', str(len(body)))",
                "        self.end_headers(); self.wfile.write(body)",
                "    def log_message(self, *args): pass",
                f"HTTPServer(('127.0.0.1', {port}), Health).serve_forever()"])
            process = None
            try:
                prefix, env = sandbox.prepare()
                process = subprocess.Popen(prefix + ["/opt/auriga-bin/opencode", "-c", server],
                                           env={**os.environ, **env}, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE)
                answer, deadline = None, time.monotonic() + 20
                while answer is None and time.monotonic() < deadline and process.poll() is None:
                    try:
                        answer = _get(sandbox.inbound_socket_path(port))
                    except (OSError, http.client.HTTPException):
                        time.sleep(.05)
                self.assertEqual(answer, (200, b"healthy:/global/health"),
                                 process.stderr.read().decode(errors="replace") if process.poll() is not None else "")
                self.assertEqual((data / "db").read_text(), "ok")
                with self.assertRaises(OSError):
                    socket.create_connection(("127.0.0.1", port), timeout=2).close()
            finally:
                if process is not None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    process.stderr.close()
                sandbox.close()
            self.assertFalse((ipc / "inbound").exists())


if __name__ == "__main__":
    unittest.main()
