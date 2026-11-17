"""OpenCode sandbox launch (WP R4-7, design R4.4): launch/mount plan, client over a unix socket, session wiring.

Pure plan and fake-server tests run everywhere (macOS included), including the launch gate (accepted since R4-8; a
host without bubblewrap fails at start and never runs the server unsandboxed); no bubblewrap is started except in the
one real-namespace test, which needs a host where production ``AgentSandbox.prepare`` can start a bubblewrap
namespace, unprivileged or via its rootful fallback (``bubblewrap_namespaces`` probe), and the pinned binary
(``AURIGA_OPENCODE_EXECUTABLE``); it is skipped with the reason elsewhere. The full boundary (conformance item 28) is
in ``test_agent_opencode_conformance``.
"""
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import agent_sandbox_golden_fixture as golden
import auriga
from support.fake_opencode import PASSWORD, FakeOpenCodeServer
from support.host_capabilities import bubblewrap_namespaces, require
from test_agent_opencode_session_unit import (
    TOOLS, SessionCase, alive, fake_binary, wait_until,
)

from auriga.cortex.agents.isolation import process_tree_snapshot
from auriga.cortex.agents.opencode import config as oc
from auriga.cortex.agents.opencode import session as session_module
from auriga.cortex.agents.opencode.client import OpenCodeClient, OpenCodeTransportError
from auriga.cortex.session.agent_sandbox import (
    ACTOR_ISOLATION, INBOUND_SOCKET_DIR, AgentSandbox,
)

REPO_ROOT = Path(auriga.__file__).resolve().parents[1]
ROBODOJO_ROOT = Path("~/code/robodojo").expanduser()


def build(root, executable, *, sandbox=True, **overrides):
    """``build_launch`` for a run directory with the test seams (no ``--version`` probe, no managed config)."""
    options = dict(executable=str(executable), model="m", wire_api="anthropic_messages",
                   relay_base_url="http://127.0.0.1:43001", instructions_text="Read /harness.", port=4096,
                   password="pw", context_limit_tokens=100000, output_limit_tokens=8000,
                   mcp_url="http://127.0.0.1:43002/mcp", mcp_headers={"Authorization": "Bearer t"},
                   ripgrep="/bin/echo", check_version=False, managed_config_paths=[], sandbox=sandbox)
    options.update(overrides)
    return oc.build_launch(root, **options)


class SandboxLaunchPlanTest(unittest.TestCase):
    """``build_launch(sandbox=True)`` + ``sandbox_spec`` fed to the real ``AgentSandbox`` planner."""
    maxDiff = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="r47-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run_dir, self.ipc = self.root / "run", self.root / "ipc"
        for path in (self.run_dir / "private", self.ipc, self.root / "snapshot/auriga"):
            path.mkdir(parents=True, mode=0o700)
        self.snapshot = self.root / "snapshot"
        self.binary = golden._executable(self.root / "host/opencode-linux-x64/bin/opencode")

    def test_sandbox_launch_has_no_host_paths_for_the_agent(self):
        launch = build(self.run_dir, self.binary)
        self.assertEqual(launch.argv, (oc.SANDBOX_EXECUTABLE, "serve", "--hostname", "127.0.0.1", "--port", "4096"))
        self.assertEqual(launch.executable, os.path.realpath(self.binary))
        self.assertEqual(launch.rg_path, oc.SANDBOX_RIPGREP)
        for name in ("PATH", "HOME", "TMPDIR"):          # the sandbox owns them
            self.assertNotIn(name, launch.env)
        self.assertEqual(launch.env["OPENCODE_SERVER_PASSWORD"], "pw")
        self.assertFalse((launch.dirs["private/opencode/bin"] / "rg").exists())
        plain = build(self.run_dir, self.binary, sandbox=False)
        self.assertEqual(plain.argv[0], str(self.binary))
        self.assertTrue((plain.dirs["private/opencode/bin"] / "rg").is_symlink())
        self.assertIn("PATH", plain.env)

    def test_mount_plan_and_inbound_only_server(self):
        launch = build(self.run_dir, self.binary)
        spec = oc.sandbox_spec(launch, harness_source=self.snapshot)
        xdg = self.run_dir / "private/opencode/xdg"
        self.assertEqual(spec["state_dirs"], [(xdg / name, xdg / name) for name in ("data", "cache", "state")])
        self.assertEqual(spec["executables"], [(os.path.realpath(self.binary), oc.SANDBOX_EXECUTABLE)])
        self.assertEqual(spec["read_paths"], [
            (xdg / "config", xdg / "config"), (self.run_dir / "private/opencode/cfgdir",) * 2,
            (launch.instructions_path, launch.instructions_path), (str(self.snapshot), "/harness")])
        with golden.planning_host(self.root, AgentSandbox) as host:
            sandbox = AgentSandbox(self.run_dir / "private/sandbox", self.ipc, self.run_dir / "actor",
                                   ports=[43001, 43002], inbound_ports=[4096], packages=True, **spec)
            try:
                prefix, env = sandbox.prepare()
                mounts = {(str(target)): (str(source), readonly) for source, target, readonly in sandbox.mounts}
                for name in ("data", "cache", "state"):          # private XDG roots are writable
                    self.assertEqual(mounts[str(xdg / name)], (str(xdg / name), False))
                self.assertEqual(mounts[str(xdg / "config")][1], True)           # config dirs / instructions read-only
                self.assertEqual(mounts[str(self.run_dir / "private/opencode/cfgdir")][1], True)
                self.assertEqual(mounts[str(launch.instructions_path)][1], True)
                self.assertEqual(mounts[oc.SANDBOX_EXECUTABLE], (os.path.realpath(self.binary), True))
                self.assertEqual(mounts[oc.SANDBOX_RIPGREP], (host.tools["rg"], True))
                self.assertEqual(mounts["/harness"], (str(self.snapshot), True))
                self.assertEqual(mounts[INBOUND_SOCKET_DIR], (str(self.ipc / "inbound"), False))
                # Never mounted: the checkout, the user's home, the RoboDojo / simulator tree, the run's other data.
                for source, target, _readonly in sandbox.mounts:
                    for forbidden in (REPO_ROOT, Path.home(), ROBODOJO_ROOT):
                        self.assertFalse(Path(source) == forbidden or forbidden.is_relative_to(source)
                                         or Path(source).is_relative_to(forbidden), (source, forbidden))
                    self.assertNotEqual(Path(source), self.run_dir / "private")
                    self.assertNotEqual(Path(source), self.run_dir)
                # Outbound: relay + MCP (+ the package proxy); inbound: only the server's own socket.
                self.assertEqual([(item.path.name, item.port) for item in sandbox.forwarders],
                                 [("tcp-43001", 43001), ("tcp-43002", 43002), ("packages", 0)])
                self.assertEqual(sandbox.inbound_socket_path(4096), self.ipc / "inbound/4096.sock")
                inside = {prefix[i + 1]: prefix[i + 2] for i, item in enumerate(prefix) if item == "--setenv"}
                self.assertEqual((inside["AURIGA_RELAY_PORTS"], inside["AURIGA_INBOUND_PORTS"]), ("43001,43002", "4096"))
                self.assertIn("--unshare-net", prefix)
                self.assertEqual(env["XDG_DATA_HOME"], str(xdg / "data"))
                self.assertEqual(env["OPENCODE_SERVER_PASSWORD"], "pw")
                self.assertEqual(env["HOME"], str(self.run_dir / "actor"))
                self.assertEqual(env["PATH"], "/opt/auriga-bin:" + os.defpath)
                self.assertEqual(env["XDG_CONFIG_HOME"], str(xdg / "config"))
            finally:
                sandbox.close()
        self.assertEqual(list(self.ipc.iterdir()), [])


class UnixSocketClientTest(unittest.TestCase):
    """``OpenCodeClient(unix_socket=)`` against the scripted fake, served through a unix-to-TCP bridge (what the
    sandbox's inbound supervisor does for the real server)."""

    def setUp(self):
        self.fake = FakeOpenCodeServer().start()
        self.addCleanup(self.fake.close)
        temporary = tempfile.TemporaryDirectory(prefix="r47s-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.path = os.path.join(temporary.name, "4096.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.path)
        self.listener.listen(8)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()
        self.addCleanup(self._close)
        self.host, port = self.fake.url.rsplit(":", 1)[0], int(self.fake.url.rsplit(":", 1)[1])
        self.port = port

    def _close(self):
        self.stop.set()
        self.listener.close()
        self.thread.join(2)

    def _accept(self):
        while not self.stop.is_set():
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            upstream = socket.create_connection(("127.0.0.1", self.port))
            for source, target in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(source, target), daemon=True).start()

    @staticmethod
    def _pump(source, target):
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                target.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (source, target):
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                sock.close()

    def client(self, **keywords):
        # base_url names a port nothing listens on: every byte must travel through the unix socket.
        return OpenCodeClient("http://127.0.0.1:1", PASSWORD, timeout_s=3, unix_socket=self.path, **keywords)

    def test_same_api_over_the_unix_socket(self):
        client = self.client()
        self.assertTrue(client.health()["healthy"])
        session = client.create_session("auriga")
        self.assertTrue(client.session_exists(session))
        self.assertEqual(client.messages(session), [])
        self.assertTrue(client.routes())
        self.assertEqual(client.status(), {})

    def test_authentication_and_error_model_are_unchanged(self):
        wrong = OpenCodeClient("http://127.0.0.1:1", "wrong", timeout_s=3, unix_socket=self.path)
        from auriga.cortex.agents.opencode.client import OpenCodeHttpError
        with self.assertRaises(OpenCodeHttpError) as caught:
            wrong.health()
        self.assertEqual(caught.exception.status, 401)

    def test_event_stream_over_the_unix_socket(self):
        client = self.client()
        stop, frames = threading.Event(), []

        def read():
            for frame in client.event_stream(stop):
                frames.append(frame)
                if len(frames) >= 2:
                    return
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        self.assertTrue(self.fake.wait_subscribers(1))
        self.fake.push("session.idle", {"sessionID": "ses_x"})
        self.fake.push("session.idle", {"sessionID": "ses_y"})
        thread.join(5)
        stop.set()
        client.close_streams()
        self.assertGreaterEqual(len(frames), 2)

    def test_missing_socket_is_a_transport_error(self):
        client = OpenCodeClient("http://127.0.0.1:1", PASSWORD, timeout_s=1, unix_socket=self.path + ".gone")
        with self.assertRaises(OpenCodeTransportError):
            client.health()

    def test_tcp_remains_the_default(self):
        self.assertTrue(OpenCodeClient(self.fake.url, PASSWORD).health()["healthy"])
        self.assertIsNone(OpenCodeClient(self.fake.url, PASSWORD).unix_socket)


class FakeSandbox:
    """Stands in for ``AgentSandbox`` in session tests: records the plan; the "bubblewrap prefix" is a shim that
    drops the in-namespace executable path and runs the host stand-in binary instead."""
    instances = []

    def __init__(self, private_root, ipc, actor_dir, **keywords):
        self.private_root, self.ipc, self.actor_dir, self.keywords = private_root, Path(ipc), actor_dir, keywords
        self.closed = 0
        self.shim = None
        FakeSandbox.instances.append(self)

    def prepare(self):
        self.ipc.mkdir(parents=True, exist_ok=True)
        host = self.keywords["executables"][0][0]
        shim = Path(self.private_root).parent / "shim.py"
        shim.write_text("import os, sys\nos.execv(sys.argv[1], [sys.argv[1]] + sys.argv[3:])\n")
        env = dict(self.keywords["env"], HOME=str(self.actor_dir), TMPDIR="/tmp")
        return [sys.executable, str(shim), str(host)], env

    def inbound_socket_path(self, port):
        return self.ipc / "inbound" / f"{port}.sock"

    def close(self):
        self.closed += 1


class SandboxSessionTest(SessionCase):
    def setUp(self):
        super().setUp()
        FakeSandbox.instances.clear()
        patch = mock.patch.object(session_module, "AgentSandbox", FakeSandbox)
        patch.start()
        self.addCleanup(patch.stop)
        ipc_base = mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(self.dir)})
        ipc_base.start()
        self.addCleanup(ipc_base.stop)
        os.chmod(self.dir, 0o700)
        self.sockets = []

    def sandbox_config(self, **opencode):
        config = self.config(**opencode)
        return type(config).from_dict({**config.to_dict(), "features": {
            **config.to_dict()["features"], "sandbox_enabled": True}})

    def sandbox_factory(self, base_url, password, unix_socket=None):
        self.sockets.append(unix_socket)
        return self.client_factory(base_url, password)

    def started_in_sandbox(self, resume=None, config=None, **overrides):
        session = self.make(config or self.sandbox_config(), client_factory=self.sandbox_factory, **overrides)
        session.start(TOOLS, "Be careful.", resume)
        return session

    def test_start_launches_through_the_sandbox_and_records_isolation(self):
        session = self.started_in_sandbox()
        sandbox, = FakeSandbox.instances
        port = int(session.launch.argv[-1])
        keywords = sandbox.keywords
        self.assertEqual(session.isolation, ACTOR_ISOLATION)
        self.assertEqual(self.store.read_state("runtime", "agent")["isolation"], ACTOR_ISOLATION)
        self.assertEqual(session.provenance()["isolation"], ACTOR_ISOLATION)
        self.assertEqual(keywords["inbound_ports"], [port])
        relay_port = int(session.model_proxy.base_url.rsplit(":", 1)[1])
        mcp_port = int(session.bridge.url.split("//")[1].split("/")[0].rsplit(":", 1)[1])
        self.assertEqual(sorted(keywords["ports"]), sorted([relay_port, mcp_port]))
        self.assertNotIn(port, keywords["ports"])
        self.assertTrue(keywords["packages"])
        self.assertEqual(sandbox.actor_dir, self.root / "actor")
        self.assertIn(("/harness"), [str(target) for _, target in keywords["read_paths"]])
        self.assertTrue((self.root / "private/actor-source/auriga/__init__.py").is_file())
        self.assertEqual(session.launch.argv[0], oc.SANDBOX_EXECUTABLE)
        # The client goes through the inbound socket of the server's port, nothing else.
        self.assertEqual(self.sockets, [str(sandbox.ipc / "inbound" / f"{port}.sock")])
        self.assertEqual(self.clients[0][0], f"http://127.0.0.1:{port}")
        self.assertEqual(session.state, "idle")

    def test_sandbox_off_is_unchanged(self):
        session = self.started()
        self.assertEqual(session.isolation, "disabled")
        self.assertEqual(FakeSandbox.instances, [])
        self.assertEqual(session.launch.argv[0], self.executable)

    def test_close_removes_the_sandbox_and_kills_the_whole_tree(self):
        executable = fake_binary(self.dir, serve="detached")
        session = self.started_in_sandbox(config=self.sandbox_config(executable=executable))
        sandbox, = FakeSandbox.instances
        ipc = sandbox.ipc
        pid = session.process.pid
        self.assertTrue(wait_until(lambda: len(process_tree_snapshot(pid).members) >= 2), "tree not up")
        children = [item.pid for item in process_tree_snapshot(pid).members if item.pid != pid]
        session.close()
        self.assertEqual(sandbox.closed, 1)
        self.assertIsNone(session.sandbox)
        self.assertFalse(ipc.exists())
        self.assertFalse(alive(pid))
        self.assertTrue(wait_until(lambda: not any(alive(item) for item in children)), "detached child survived")

    def test_failed_start_closes_the_sandbox(self):
        def failing(base_url, password, unix_socket=None):
            raise RuntimeError("client_unavailable")
        session = self.make(self.sandbox_config(), client_factory=failing)
        with self.assertRaisesRegex(RuntimeError, "client_unavailable"):
            session.start(TOOLS, "Be careful.")
        sandbox, = FakeSandbox.instances
        self.assertEqual(sandbox.closed, 1)

    def test_resume_refuses_an_isolation_change_both_ways(self):
        sandboxed = self.started_in_sandbox()
        thread = sandboxed.thread_id
        sandboxed.close()
        self.assertEqual(self.store.read_state("runtime", "agent")["isolation"], ACTOR_ISOLATION)
        plain = self.make()
        with self.assertRaisesRegex(ValueError, "^resume_agent_identity_conflict:isolation"):
            plain.start(TOOLS, "Be careful.", thread)
        self.assertIsNone(plain.process)
        # And back: a recorded unsandboxed run cannot resume sandboxed.
        identity = self.store.read_state("runtime", "agent")
        self.store.set_state("runtime", "agent", dict(identity, isolation="disabled"))
        again = self.make(self.sandbox_config(), client_factory=self.sandbox_factory)
        with self.assertRaisesRegex(ValueError, "^resume_agent_identity_conflict:isolation"):
            again.start(TOOLS, "Be careful.", thread)
        self.assertIsNone(again.process)
        self.assertEqual(len(FakeSandbox.instances), 1)        # refused before any sandbox was built

    def test_instructions_point_at_harness_in_the_sandbox(self):
        from auriga.cortex.context.actor_instructions import instructions
        text = instructions("obs", "handoff", source_root=None)
        self.assertIn("/harness", text)
        self.assertNotIn(str(self.root), text)


class LaunchGateAcceptsTheSandbox(SessionCase):
    """R4-8: the launch gate accepts ``features.sandbox_enabled`` (verified on Linux by conformance item 28); a host
    that cannot build the namespace fails at session start, like Codex, and never runs the server unsandboxed."""

    def sandbox_config(self):
        config = self.config()
        return type(config).from_dict({**config.to_dict(), "features": {
            **config.to_dict()["features"], "sandbox_enabled": True}})

    def test_launch_gate_accepts_the_sandbox(self):
        from auriga.cortex.agents import factory
        with mock.patch.object(factory, "opencode_runtime_problem", return_value=None):
            self.assertIsNone(factory.backend_requirements(self.sandbox_config()))

    def test_a_host_without_bubblewrap_fails_at_start_without_running_the_server(self):
        runtime_dir = Path(tempfile.mkdtemp(prefix="r48o-", dir="/tmp"))     # short: Unix socket path limit
        self.addCleanup(shutil.rmtree, runtime_dir, True)
        os.chmod(runtime_dir, 0o700)
        real = AgentSandbox._find_host_tool
        missing = staticmethod(lambda name: None if name == "bwrap" else real(name))
        with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(runtime_dir)}), \
                mock.patch.object(AgentSandbox, "_find_host_tool", missing):
            session = self.make(self.sandbox_config())
            with self.assertRaisesRegex(RuntimeError, "^agent_sandbox_bubblewrap_not_found_on_path_or_os_defpath$"):
                session.start(TOOLS, "Be careful.")
        self.assertIsNone(session.process, "the server was spawned although the sandbox could not be built")
        self.assertEqual(session.state, "closed")
        self.assertEqual(list((runtime_dir / "auriga").iterdir()), [])     # IPC directory removed


# Real-namespace test (Linux host with a usable bubblewrap namespace, unprivileged or rootful, and the pinned
# binary). Skipped elsewhere.
EXECUTABLE = os.environ.get("AURIGA_OPENCODE_EXECUTABLE")


@require(bubblewrap_namespaces, "OpenCode runs inside the real bubblewrap AgentSandbox")
@unittest.skipUnless(EXECUTABLE, "set AURIGA_OPENCODE_EXECUTABLE to the opencode-ai@1.19.5 platform binary")
class OpenCodeRealSandboxTest(unittest.TestCase):
    def test_tool_turn_inside_the_sandbox(self):
        from support.fake_model_servers import FakeAnthropicServer, text_step, tool_step
        from auriga.core.config import Config
        from auriga.core.storage.store import Store
        key_env = "AURIGA_R47_UPSTREAM_KEY"
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        os.environ.setdefault("XDG_RUNTIME_DIR", "")
        marker = "AURIGA-R47-MARKER"

        def router(request):
            body = request.json if isinstance(request.json, dict) else {}
            if not body.get("tools"):
                return text_step("side-ok")
            has_result = any(isinstance(block, dict) and block.get("type") == "tool_result"
                             for message in body.get("messages") or [] if isinstance(message.get("content"), list)
                             for block in message["content"])
            return text_step("FINAL") if has_result else tool_step("auriga_look", {"x": 1})

        server = FakeAnthropicServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        with mock.patch.dict(os.environ, {key_env: "host-secret"}):
            store = Store(root, dict(run_id="r47", episode_id="ep", env_instance_id="env", env_generation=1))
            self.addCleanup(store.close)
            config = Config.from_dict({
                "agent": {"backend": "opencode", "opencode": {"executable": EXECUTABLE, "upstream": {
                    "wire_api": "anthropic_messages", "base_url": server.base_url, "api_key_env": key_env,
                    "auth_scheme": "x-api-key", "provider": "fake"}}},
                "model": {"name": "claude-r47-model", "effort": "high"},
                "features": {"self_briefing": False, "visual_review_model": None, "sandbox_enabled": True}})
            calls = []

            def handler(params):
                calls.append(params)
                return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one cube"}]}

            session = session_module.OpenCodeSession(root, store, handler, config=config)
            session.tool_parallel_hints = {"look": True}
            self.addCleanup(session.close)
            session.start(TOOLS, "Robot instructions. " + marker)
            self.assertEqual(session.isolation, ACTOR_ISOLATION)
            self.assertIn("--unshare-net", session.process.args)
            port = int(session.launch.argv[-1])
            with self.assertRaises(OSError):            # no host TCP listener for the server
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
            session.run_turn("Look at the scene, then answer.")
            self.assertTrue(session.turn_done.wait(120), "tool turn did not end")
            self.assertEqual(session.state, "completed", session.turn_result)
            self.assertEqual(session.completed_final_answer(), "FINAL")
            self.assertEqual(calls[0]["tool"], "look")
            ipc = session.ipc
            pid = session.process.pid
            session.close()
            self.assertFalse(alive(pid))
            self.assertFalse(ipc.exists())


if __name__ == "__main__":
    unittest.main()
