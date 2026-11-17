"""Claude Code in the generic bubblewrap AgentSandbox (WP R4-6; design rev 4, R4.4).

Planning tests stub the host probes (``agent_sandbox_golden_fixture.planning_host``) and run on every host: they check
the mount and argv plan of a sandboxed launch, the launcher script (executed against a recording ``env`` stand-in),
the recorded isolation, the session wiring, the resume refusal and the launch gate (accepted since R4-8; a host without
bubblewrap fails at start and never runs the CLI unsandboxed). The one test that starts the real pinned CLI in a real
namespace runs only where production ``AgentSandbox.prepare`` can start bubblewrap (unprivileged or rootful fallback)
and skips with an explicit reason elsewhere; ``AURIGA_REQUIRE_SANDBOX_TESTS=1`` turns that skip into a failure on a
host that must support it. The full boundary (conformance item 28) is in ``test_agent_claude_code_conformance``.
"""
import contextlib
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent_sandbox_golden_fixture as golden
import auriga
import test_agent_claude_session_unit as base
from support.fake_claude_sdk import requires_sdk
from support.fake_model_servers import FakeAnthropicServer, text_step, tool_step
from support.host_capabilities import bubblewrap_namespaces, require

from auriga.core.config import Config
from auriga.core.storage.store import Store
from auriga.cortex.agents import factory
from auriga.cortex.context.actor_instructions import instructions as actor_instructions
from auriga.cortex.session.agent_sandbox import ACTOR_ISOLATION, AgentSandbox, _SUPERVISOR

REPO_ROOT = Path(auriga.__file__).resolve().parents[1]
ROBODOJO_DEFAULT_ROOT = Path("~/code/robodojo").expanduser()
REAL_UID, REAL_GID = os.getuid(), os.getgid()
SESSION = "9e7c96f2-1111-4222-8333-444455556666"
FAKE_CLI = "#!/bin/sh\nexit 0\n"
SANDBOX_CONFIG = {**base.CONFIG, "features": {**base.CONFIG["features"], "sandbox_enabled": True}}


@contextlib.contextmanager
def planning(root, *, rg=True):
    """``planning_host`` for ``AgentSandbox`` with the real uid/gid (the run directories belong to this user)."""
    with golden.planning_host(root, AgentSandbox, rg=rg) as host, \
            mock.patch.object(os, "getuid", return_value=REAL_UID), mock.patch.object(os, "getgid", return_value=REAL_GID):
        yield host


def setenv_of(prefix):
    return {prefix[index + 1]: prefix[index + 2] for index, item in enumerate(prefix) if item == "--setenv"}


def assert_no_checkout_home_or_simulator(test, sandbox):
    """No mount source or target is the checkout, the user's home, RoboDojo, or anything inside or around them."""
    forbidden = (REPO_ROOT, Path.home(), ROBODOJO_DEFAULT_ROOT)
    for source, target, _readonly in sandbox.mounts:
        for path in (Path(source), Path(target)):
            for item in forbidden:
                test.assertFalse(path == item or item.is_relative_to(path) or path.is_relative_to(item), (path, item))
            test.assertNotIn("robodojo", str(path).lower())


@requires_sdk
class SandboxedLaunchPlan(unittest.TestCase):
    """``build_launch(sandbox=SandboxPlan(...))``: mounts, argv, environment, recorded isolation."""
    maxDiff = None

    def setUp(self):
        from auriga.cortex.agents.claude_code import options
        self.options = options
        temporary = tempfile.TemporaryDirectory(prefix="r46-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run_dir, self.ipc = self.root / "run", self.root / "ipc"
        self.snapshot = self.run_dir / "private" / "actor-source"
        for path in (self.run_dir, self.ipc, self.snapshot / "auriga"):
            path.mkdir(parents=True, mode=0o700)
        (self.snapshot / "auriga" / "__init__.py").write_text("")
        self.cli = golden._executable(self.root / "host/claude_agent_sdk/_bundled/claude", FAKE_CLI)
        self.runtime = options.ClaudeRuntime(sdk_version=options.SDK_VERSION, cli_version=options.CLI_VERSION,
                                             executable=os.fspath(self.cli), sha256="0" * 64)
        self.plan = options.SandboxPlan(ipc=self.ipc, ports=(43002, 43001),
                                        read_paths=((self.snapshot, options.HARNESS_SOURCE_MOUNT),))

    def build(self, **overrides):
        arguments = dict(model="claude-test-1", effort="high", thread_id=SESSION, instructions="Be careful.\n",
                         relay_base_url="http://127.0.0.1:43001", mcp_url="http://127.0.0.1:43002/mcp",
                         mcp_headers={"Authorization": "Bearer tok"}, runtime=self.runtime, sandbox=self.plan)
        arguments.update(overrides)
        return self.options.build_launch(self.run_dir, **arguments)

    def test_mount_and_argv_plan(self):
        with planning(self.root) as host:
            launch = self.build()
            try:
                sandbox = launch.sandbox
                self.assertIsInstance(sandbox, AgentSandbox)
                prefix = sandbox.command_prefix
                private, actor = self.run_dir / "private", self.run_dir / "actor"
                config, tmp, run = private / "claude-config", private / "tmp", private / "run"
                mounts = {(str(target), readonly): str(source) for source, target, readonly in sandbox.mounts}
                # Read-write: workspace, run-private config / tmp / run directories and the IPC directory, each at its own path.
                for path in (actor, config, tmp, run, self.ipc):
                    self.assertEqual(mounts.pop((str(path), False)), str(path))
                self.assertEqual(mounts.pop(("/run/auriga-relays", True)), str(self.ipc / "relays"))
                # Read-only: the REAL CLI (not the launcher), the instructions file, the published Harness source, rg.
                self.assertEqual(mounts.pop((str(self.cli), True)), str(self.cli))
                self.assertEqual(mounts.pop((str(launch.instructions_file), True)), str(launch.instructions_file))
                self.assertEqual(mounts.pop(("/harness", True)), str(self.snapshot))
                self.assertEqual(mounts.pop(("/opt/auriga-bin/rg", True)), host.tools["rg"])
                self.assertFalse([key for key in mounts if not key[1]], mounts)   # nothing else is writable
                for (target, _), source in mounts.items():   # the rest is the published system runtime
                    self.assertTrue(target.startswith(("/usr", "/lib", "/etc/ssl")) or source.startswith(str(self.root / "host")),
                                    (target, source))
                assert_no_checkout_home_or_simulator(self, sandbox)
                self.assertNotIn(str(launch.launcher), [str(source) for source, _, _ in sandbox.mounts])
                # bwrap argv: namespaces, no host network, no host environment, workspace as cwd.
                for flag in ("--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-user", "--clearenv",
                             "--die-with-parent", "--new-session"):
                    self.assertIn(flag, prefix)
                self.assertEqual(prefix[:2], [host.tools["env"], "-i"])
                self.assertEqual(prefix[prefix.index("--chdir") + 1], str(actor))
                self.assertEqual(prefix[-1], _SUPERVISOR)
                # Relay and MCP endpoint are forwarded (same loopback ports inside); the PyPI proxy is added.
                inside = setenv_of(prefix)
                self.assertEqual(inside["AURIGA_RELAY_PORTS"], "43001,43002")
                self.assertEqual(sorted((item.path.name, item.port) for item in sandbox.forwarders),
                                 [("packages", 0), ("tcp-43001", 43001), ("tcp-43002", 43002)])
                self.assertEqual(inside["ANTHROPIC_BASE_URL"], "http://127.0.0.1:43001")
                self.assertEqual(inside["CLAUDE_CONFIG_DIR"], str(config))
                self.assertEqual(inside["CLAUDE_CODE_TMPDIR"], str(tmp))
                self.assertEqual(inside["XDG_RUNTIME_DIR"], str(run))
                self.assertEqual((inside["HOME"], inside["TMPDIR"]), (str(actor), "/tmp"))   # sandbox-owned
                self.assertEqual(inside["PATH"], "/opt/auriga-bin:" + os.defpath)
                self.assertEqual(inside["ANTHROPIC_API_KEY"], self.options.PLACEHOLDER_API_KEY)
                self.assertNotIn("IS_SANDBOX", inside)   # host uid kept: not root, no bypass-permissions exemption needed
                self.assertEqual(inside["CLAUDE_CODE_DISABLE_WEB_FETCH"], "1")
                # The same clean CLI environment as without the sandbox, minus what the sandbox owns.
                self.assertEqual({key: value for key, value in inside.items() if key in launch.env and key not in
                                  self.options.SANDBOX_OWNED_ENV}, {key: value for key, value in launch.env.items()
                                  if key not in self.options.SANDBOX_OWNED_ENV})
                # SDK options point at the launcher and at paths that exist inside the namespace.
                self.assertEqual(launch.options.cli_path, str(launch.launcher))
                self.assertEqual(launch.options.extra_args["append-system-prompt-file"], str(launch.instructions_file))
                self.assertEqual(launch.options.cwd, actor)
                self.assertEqual(launch.options.env, {})
            finally:
                launch.sandbox.close()

    def test_isolation_is_recorded(self):
        options = self.options
        with planning(self.root):
            launch = self.build()
            try:
                self.assertEqual(launch.isolation, ACTOR_ISOLATION)
                block = options.provenance_block(launch, model="claude-test-1", counting="c", isolation=launch.isolation)
                self.assertEqual(block["isolation"], ACTOR_ISOLATION)
                sandbox = block["native_settings"]["sandbox"]
                self.assertEqual((sandbox["isolation"], sandbox["harness_source"]), (ACTOR_ISOLATION, "/harness"))
                self.assertEqual(block["cli"]["executable"], str(self.cli))   # the real CLI, never the launcher
                self.assertNotIn(str(self.ipc), repr(block))   # no host run layout in the manifest
                self.assertNotIn("43001", repr(block["native_settings"]))
            finally:
                launch.sandbox.close()
        plain = self.options.build_launch(self.run_dir, model="claude-test-1", effort="high", thread_id=SESSION,
                                          instructions="x", relay_base_url="http://127.0.0.1:43001",
                                          mcp_url="http://127.0.0.1:43002/mcp", mcp_headers={}, runtime=self.runtime)
        self.assertEqual((plain.isolation, plain.sandbox), ("disabled", None))
        self.assertNotIn("sandbox", plain.native_settings)
        self.assertIn("exec /usr/bin/env -i", plain.launcher.read_text())   # the unsandboxed launcher is unchanged

    def test_is_sandbox_follows_the_cli_uid_not_the_sandbox(self):
        with planning(self.root), mock.patch.object(self.options, "_running_as_root", return_value=True):
            launch = self.build()
            try:
                self.assertEqual(setenv_of(launch.sandbox.command_prefix)["IS_SANDBOX"], "1")
                self.assertTrue(launch.native_settings["is_sandbox_env"])
            finally:
                launch.sandbox.close()

    def test_launcher_execs_prefix_cli_and_sdk_arguments(self):
        """Run the launcher against a recording ``env`` stand-in: argv order, re-exported SDK variables, no leakage."""
        with planning(self.root) as host:
            golden._executable(Path(host.tools["env"]), '#!/bin/sh\nfor a in "$@"; do printf \'%s\\037\' "$a"; done'
                                                          ' > "$ARGV_OUT"\n')
            launch = self.build()
            try:
                script = launch.launcher.read_text()
                self.assertTrue(script.startswith("#!/bin/sh\n"))
                self.assertEqual(stat.S_IMODE(launch.launcher.stat().st_mode), 0o700)
                self.assertNotIn("host-secret", script)
                out = self.root / "argv"
                completed = subprocess.run(
                    [str(launch.launcher), "--output-format", "stream-json", "a b", "--x=$HOME"],
                    env={"PATH": os.defpath, "ARGV_OUT": str(out), "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
                         "CLAUDE_AGENT_SDK_VERSION": "0.2.180", "GATEWAY_SECRET": "host-only"},
                    capture_output=True, text=True, timeout=20)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                argv = out.read_text().split("\x1f")[:-1]
                separator = argv.index("--")
                self.assertEqual(argv.count("--"), 1)
                self.assertEqual(argv[:2], ["-i", host.tools["bwrap"]])
                inside = setenv_of(argv[:separator])
                # SDK-injected variables that were set reach the namespace; the one that was not set does not appear.
                self.assertEqual((inside["CLAUDE_CODE_ENTRYPOINT"], inside["CLAUDE_AGENT_SDK_VERSION"]),
                                 ("sdk-py", "0.2.180"))
                self.assertNotIn("CLAUDE_CODE_SDK_READS_SESSION_STATE", argv)
                self.assertNotIn("GATEWAY_SECRET", inside)
                self.assertFalse([item for item in argv if "host-only" in item])
                # After the separator: the in-namespace supervisor, then the real CLI with the SDK's arguments verbatim.
                self.assertEqual(argv[separator + 1:], [host.tools["python3"], "-c", _SUPERVISOR, str(self.cli),
                                                         "--output-format", "stream-json", "a b", "--x=$HOME"])
                prefix = launch.sandbox.command_prefix
                head = prefix[1:prefix.index("--")]   # the stand-in is argv[0]; the re-exports follow the bwrap options
                self.assertEqual(argv[:len(head)], head)
                self.assertEqual(len(argv[:separator]) - len(head), 6)   # two --setenv NAME VALUE triples
            finally:
                launch.sandbox.close()

    def test_launcher_writer_validation(self):
        write = self.options.write_sandbox_launcher
        path = self.root / "launcher.sh"
        for prefix in (["env", "-i", "bwrap"], ["env", "--", "--", "x"]):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "^sandbox_prefix_separator_ambiguous$"):
                write(path, prefix, "/cli")
        with self.assertRaisesRegex(ValueError, "^launcher_executable_must_be_absolute$"):
            write(path, ["env", "--", "py"], "claude")
        with self.assertRaisesRegex(ValueError, "^invalid_reexport_name"):
            write(path, ["env", "--", "py"], "/cli", reexport=["A B"])
        link = self.root / "link.sh"
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "^launcher_path_unwritable"):
            write(link, ["env", "--", "py"], "/cli")

    def test_sandbox_closes_when_the_launch_cannot_be_completed(self):
        created = []
        original = AgentSandbox.prepare

        def recording(sandbox):
            created.append(sandbox)
            return original(sandbox)

        with planning(self.root), mock.patch.object(AgentSandbox, "prepare", recording), \
                mock.patch.object(self.options, "write_sandbox_launcher", side_effect=ValueError("boom")):
            with self.assertRaisesRegex(ValueError, "^boom$"):
                self.build()
        sandbox, = created
        self.assertFalse(sandbox.prepared)
        self.assertEqual(list(self.ipc.iterdir()), [])   # relays and owner socket removed

    def test_ipc_directory_is_short_private_and_run_keyed(self):
        with tempfile.TemporaryDirectory(prefix="r46x-", dir="/tmp") as runtime:
            os.chmod(runtime, 0o700)
            with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": runtime}):
                first = self.options.prepare_sandbox_ipc(self.run_dir)
                self.assertEqual(first, self.options.sandbox_ipc_dir(self.run_dir))
                self.assertEqual(first.parent, Path(runtime) / "auriga")
                self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o700)
                self.assertNotEqual(first, self.options.sandbox_ipc_dir(self.root))
                self.assertLess(len(os.fsencode(first / "sandbox-owner.sock")), 107)
                os.chmod(runtime, 0o755)
                with self.assertRaisesRegex(ValueError, "^unsafe_claude_code_runtime_directory$"):
                    self.options.prepare_sandbox_ipc(self.run_dir)
            with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": runtime + "/" + "d" * 100}):
                with self.assertRaisesRegex(ValueError, "^claude_code_ipc_path_too_long"):
                    self.options.sandbox_ipc_dir(self.run_dir)


class SandboxedSessionCase(base.SessionCase):
    """``ClaudeCodeSession`` with ``features.sandbox_enabled`` and the scripted fake SDK client (host probes stubbed)."""

    def setUp(self):
        super().setUp()
        runtime_dir = tempfile.mkdtemp(prefix="r46r-", dir="/tmp")
        self.addCleanup(shutil.rmtree, runtime_dir, True)
        environment = mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": runtime_dir})
        environment.start()
        self.addCleanup(environment.stop)
        host = tempfile.TemporaryDirectory(prefix="r46h-", dir="/tmp")
        self.addCleanup(host.cleanup)
        stack = contextlib.ExitStack()
        self.host = stack.enter_context(planning(Path(host.name).resolve()))
        self.addCleanup(stack.close)   # registered before any session: runs after session.close
        self.config = Config.from_dict(SANDBOX_CONFIG)


class SandboxedSessionTests(SandboxedSessionCase):
    def test_start_records_isolation_and_wires_the_sandbox(self):
        session = self.started()
        self.assertTrue(session.sandbox_enabled)
        self.assertEqual(session.isolation, ACTOR_ISOLATION)
        identity = self.store.read_state("runtime", "agent")
        self.assertEqual(identity["isolation"], ACTOR_ISOLATION)
        launch = session.launch
        self.assertEqual(launch.isolation, ACTOR_ISOLATION)
        sandbox = launch.sandbox
        self.assertTrue(sandbox.prepared)
        # Relay and MCP endpoint of THIS session are the forwarded ports.
        relay_port = int(session.model_proxy.base_url.rsplit(":", 1)[1].split("/")[0])
        mcp_port = int(session.bridge.url.rsplit(":", 1)[1].split("/")[0])
        self.assertEqual(sandbox.ports, tuple(sorted((relay_port, mcp_port))))
        # /harness is the run's published source snapshot, read-only.
        snapshot = self.root / "private" / "actor-source"
        self.assertTrue((snapshot / "auriga" / "runtime" / "harness.py").is_file())
        self.assertIn((snapshot, Path("/harness"), True), [(Path(s), Path(t), r) for s, t, r in sandbox.mounts])
        assert_no_checkout_home_or_simulator(self, sandbox)
        block = session.provenance()
        self.assertEqual(block["isolation"], ACTOR_ISOLATION)
        self.assertEqual(block["cli"]["executable"], self.runtime.executable)
        self.assertEqual(Path(self.client.options.cli_path), launch.launcher)
        ipc = sandbox.ipc
        self.assertTrue(ipc.is_dir())
        report = session.close()
        self.assertEqual(report["backend"]["sandbox"], "closed")
        self.assertFalse(sandbox.prepared)
        self.assertFalse(ipc.exists())   # sockets removed, IPC directory removed with the session
        self.assertTrue(self.store.read_state("runtime", "agent")["clean_shutdown"])

    def test_session_without_sandbox_is_unchanged(self):
        self.config = Config.from_dict(base.CONFIG)
        session = self.started()
        self.assertEqual((session.sandbox_enabled, session.isolation, session.launch.sandbox),
                         (False, "disabled", None))
        self.assertEqual(self.store.read_state("runtime", "agent")["isolation"], "disabled")
        self.assertFalse((self.root / "private" / "actor-source").exists())   # Workbench publishes it, not the session
        self.assertNotIn("sandbox", session.close()["backend"])

    def test_instructions_name_the_mounted_harness_source(self):
        """Workbench passes ``source_root=None`` with the sandbox on: the text names /harness, no host path."""
        text = actor_instructions("obs", "handoff", source_root=None)
        self.assertIn("under /harness", text)
        self.assertIn("inside your sandbox", text)
        self.assertNotIn("without OS sandbox isolation", text)
        outside = actor_instructions("obs", "handoff", source_root=self.root / "private" / "actor-source")
        self.assertNotIn("under /harness", outside)
        self.assertIn("without OS sandbox isolation", outside)
        session = self.make()
        session.start(base.TOOLS, text, None)
        launch_text = session.launch.instructions_file.read_text()
        self.assertIn("under /harness", launch_text)
        self.assertEqual(launch_text, session.launch.instructions_text)
        self.assertNotIn(str(self.root), launch_text)

    def test_resume_across_a_sandbox_mode_change_is_refused(self):
        sandboxed = self.started()
        thread = sandboxed.thread_id
        sandboxed.close()
        (self.root / "private" / "claude-config" / "projects" / "-run-actor").mkdir(parents=True, exist_ok=True)
        (self.root / "private" / "claude-config" / "projects" / "-run-actor" / f"{thread}.jsonl").write_text("{}\n")
        # sandbox -> no sandbox
        self.config = Config.from_dict(base.CONFIG)
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^resume_isolation_changed"):
            session.start(base.TOOLS, "x", thread)
        self.assertEqual(session.state, "closed")
        # same mode resumes
        self.config = Config.from_dict(SANDBOX_CONFIG)
        resumed = self.make()
        resumed.start(base.TOOLS, "x", thread)
        self.assertEqual(resumed.thread_id, thread)
        self.assertEqual(self.store.read_state("runtime", "agent")["isolation"], ACTOR_ISOLATION)
        resumed.close()
        # no sandbox recorded -> sandbox requested
        identity = self.store.read_state("runtime", "agent")
        self.store.set_state("runtime", "agent", dict(identity, isolation="disabled"))
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^resume_isolation_changed"):
            session.start(base.TOOLS, "x", thread)

    def test_a_failed_launch_leaves_no_sandbox_behind(self):
        with mock.patch("auriga.cortex.agents.claude_code.options.write_sandbox_launcher", side_effect=OSError("disk")):
            session = self.make()
            with self.assertRaises(OSError):
                session.start(base.TOOLS, "x", None)
        self.assertEqual(session.state, "closed")
        self.assertEqual(list((Path(os.environ["XDG_RUNTIME_DIR"]) / "auriga").iterdir()), [])


class LaunchGateAcceptsTheSandbox(base.SessionCase):
    """R4-8: the launch gate accepts ``features.sandbox_enabled`` (verified on Linux by conformance item 28); a host
    that cannot build the namespace fails at session start, like Codex, and never runs the CLI unsandboxed."""

    def test_launch_gate_accepts_the_sandbox(self):
        config = Config.from_dict(SANDBOX_CONFIG)
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            self.assertIsNone(factory.backend_requirements(config))

    def test_a_host_without_bubblewrap_fails_at_start_without_running_the_cli(self):
        runtime_dir = tempfile.mkdtemp(prefix="r48r-", dir="/tmp")
        self.addCleanup(shutil.rmtree, runtime_dir, True)
        os.chmod(runtime_dir, 0o700)
        self.config = Config.from_dict(SANDBOX_CONFIG)
        real = AgentSandbox._find_host_tool
        missing = staticmethod(lambda name: None if name == "bwrap" else real(name))
        with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": runtime_dir}), \
                mock.patch.object(AgentSandbox, "_find_host_tool", missing):
            session = self.make()
            with self.assertRaisesRegex(RuntimeError, "^agent_sandbox_bubblewrap_not_found_on_path_or_os_defpath$"):
                session.start(base.TOOLS, "x", None)
        self.assertEqual(session.state, "closed")
        self.assertEqual(self.fake.clients, [], "the CLI was started although the sandbox could not be built")
        self.assertEqual(list((Path(runtime_dir) / "auriga").iterdir()), [])   # IPC directory removed


# ------------------------------------------------------------------------------------- real binary, real namespace

RUNTIME_PROBLEM = factory.claude_code_runtime_problem()
MARKER = "AURIGA-SANDBOX-INSTRUCTIONS-5c1e"
BASH_ID, LOOK_ID = "toolu_sandbox_bash", "toolu_sandbox_look"
LOOK_SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x position in m, default 0"}},
               "additionalProperties": False}
LOOK_TOOLS = [{"type": "function", "name": "look", "description": "Read-only view of the scene.",
               "inputSchema": LOOK_SCHEMA}]


def _system_text(body):
    system = body.get("system")
    if isinstance(system, list):
        return "\n".join(block.get("text", "") for block in system if isinstance(block, dict))
    return system if isinstance(system, str) else ""


def _last_result_ids(body):
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            content = message.get("content")
            return {block.get("tool_use_id") for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_result"} if isinstance(content, list) else set()
    return set()


@require(bubblewrap_namespaces, "the pinned Claude Code CLI runs in a real AgentSandbox namespace")
@unittest.skipIf(RUNTIME_PROBLEM is not None, f"Claude Code runtime unavailable: {RUNTIME_PROBLEM}")
class RealClaudeCodeInNamespace(unittest.TestCase):
    """Real CLI, real bubblewrap, ``FakeAnthropicServer`` upstream, real MCP endpoint.

    One tool turn: the (scripted) model runs a Bash command that probes the namespace, then calls the registry tool
    ``look`` through the forwarded MCP endpoint, then answers. Linux host only (skips elsewhere).
    """

    def test_tool_turn_inside_the_sandbox(self):
        from auriga.cortex.agents.claude_code.session import ClaudeCodeSession
        directory = tempfile.TemporaryDirectory(prefix="r46real-", dir="/tmp")
        self.addCleanup(directory.cleanup)
        outside = Path(directory.name).resolve()
        root = outside / "run"
        root.mkdir()
        secret = outside / "robodojo-secret.txt"
        secret.write_text("simulator source stand-in")
        runtime_dir = tempfile.mkdtemp(prefix="r46rr-", dir="/tmp")
        self.addCleanup(shutil.rmtree, runtime_dir, True)
        probe = ("printf 'HARNESS=%s\\nSECRET=%s\\nHOME_DIR=%s\\n' "
                 "\"$(test -f /harness/auriga/runtime/harness.py && echo yes || echo no)\" "
                 f"\"$(test -e {secret} && echo yes || echo no)\" "
                 f"\"$(test -e {Path.home()} && echo yes || echo no)\"")

        def router(request):
            body = request.json if isinstance(request.json, dict) else {}
            if not (body.get("tools") and MARKER in _system_text(body)):
                return text_step("side-ok")
            ids = _last_result_ids(body)
            if LOOK_ID in ids:
                return text_step("FINAL")
            if BASH_ID in ids:
                return tool_step("mcp__auriga__look", {"x": 1}, id=LOOK_ID)
            return tool_step("Bash", {"command": probe, "description": "probe the sandbox"}, id=BASH_ID)

        server = FakeAnthropicServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        environment = mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": runtime_dir, "AURIGA_SANDBOX_UPSTREAM_KEY": "host-key"})
        environment.start()
        self.addCleanup(environment.stop)
        os.chmod(runtime_dir, 0o700)
        store = Store(root, dict(run_id="sandbox", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "claude_code", "claude_code": {"upstream": {
                "base_url": server.base_url, "api_key_env": "AURIGA_SANDBOX_UPSTREAM_KEY", "auth_scheme": "x-api-key"}}},
            "model": {"name": "claude-sandbox-model", "effort": "low"},
            "features": {"self_briefing": False, "visual_review_model": None, "sandbox_enabled": True}})
        calls = []

        def handler(params):
            calls.append(params)
            return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one red cube"}]}

        factory.backend_requirements(config)   # the launch gate accepts the sandbox (R4-8)
        session = ClaudeCodeSession(root, store, handler, config=config)
        session.tool_parallel_hints = {"look": True}
        self.addCleanup(session.close)
        session.start(LOOK_TOOLS, "Robot instructions. Read /harness. " + MARKER)
        self.assertEqual(session.isolation, ACTOR_ISOLATION)
        self.assertEqual(store.read_state("runtime", "agent")["isolation"], ACTOR_ISOLATION)
        session.run_turn("Probe your sandbox, look at the scene, then answer.")
        self.assertTrue(session.turn_done.wait(180), "sandboxed tool turn did not end")
        self.assertEqual((session.state, session.turn_result["status"]), ("completed", "completed"), session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL")
        params, = calls
        self.assertEqual((params["tool"], params["callId"]), ("look", LOOK_ID))   # via the forwarded MCP endpoint
        # The model's Bash tool ran inside the namespace: the published source is there, the host is not.
        generations = [request for request in server.requests if request.is_generation and MARKER in _system_text(request.json or {})]
        results = " ".join(result.text for result in server.find_tool_results(generations[1]))
        self.assertIn("HARNESS=yes", results)
        self.assertIn("SECRET=no", results)
        self.assertIn("HOME_DIR=no", results)
        report = session.close()
        self.assertTrue(report["clean_shutdown"], report)
        self.assertEqual(report["backend"]["sandbox"], "closed")
        self.assertFalse(session.launch.sandbox.prepared)


if __name__ == "__main__":
    unittest.main()
