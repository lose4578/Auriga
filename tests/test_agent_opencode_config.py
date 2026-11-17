import json
import os
import signal
import socket
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from auriga.cortex.agents import isolation as iso
from auriga.cortex.agents.opencode import config as cfg
from auriga.cortex.agents.opencode.client import OpenCodeClient

EXECUTABLE = os.environ.get("AURIGA_OPENCODE_EXECUTABLE")


def fake_binary(directory: Path, version: str = "1.19.5") -> str:
    path = directory / "opencode"
    path.write_text(f"#!/bin/sh\necho {version}\n")
    path.chmod(0o755)
    return str(path)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.exe = fake_binary(self.root)
        self.rg = self.exe  # any executable file stands in for rg

    def launch(self, **kw):
        args = dict(executable=self.exe, model="claude-sonnet-4-5", wire_api="anthropic_messages",
                    relay_base_url="http://127.0.0.1:9999", instructions_text="INSTR auriga_", port=4567,
                    password="secret-test-password", context_limit_tokens=200000, output_limit_tokens=32000,
                    mcp_url="http://127.0.0.1:8888/mcp", mcp_headers={"Authorization": "Bearer t"},
                    ripgrep=self.rg, managed_config_paths=[], base_env={"ANTHROPIC_API_KEY": "host", "TZ": "UTC"})
        args.update(kw)
        return cfg.build_launch(self.run_dir, **args)

    def test_config_body(self):
        launch = self.launch()
        c = launch.config
        self.assertEqual(json.loads(launch.env["OPENCODE_CONFIG_CONTENT"]), c)
        self.assertEqual((c["model"], c["small_model"]), ("relay/claude-sonnet-4-5",) * 2)
        self.assertEqual(c["enabled_providers"], ["relay"])
        provider = c["provider"]["relay"]
        self.assertEqual(provider["npm"], "@ai-sdk/anthropic")
        self.assertEqual(provider["options"], {"baseURL": "http://127.0.0.1:9999/v1", "apiKey": "auriga-local-relay",
                                               "chunkTimeout": False, "headerTimeout": False})
        model = provider["models"]["claude-sonnet-4-5"]
        self.assertIn("image", model["modalities"]["input"])
        self.assertEqual(model["limit"], {"context": 200000, "output": 32000})
        self.assertEqual(set(model["cost"].values()), {0})
        self.assertNotIn("ask", c["permission"].values())
        self.assertEqual(c["permission"], {
            "*": "allow", "webfetch": "deny", "websearch": "deny", "codesearch": "deny", "question": "deny",
            "external_directory": "allow", "doom_loop": "allow"})
        for tool in ("task", "todowrite", "skill", "lsp"):
            self.assertNotIn(tool, c["permission"])
        self.assertIs(c["agent"]["title"]["disable"], True)
        self.assertIs(c["compaction"]["prune"], False)
        self.assertEqual((c["share"], c["autoupdate"], c["snapshot"]), ("disabled", False, False))
        self.assertNotIn("lsp", c)            # absent = OpenCode default (off in 1.19.5); never written as false/true
        self.assertNotIn("formatter", c)
        self.assertEqual(launch.env["OPENCODE_DISABLE_LSP_DOWNLOAD"], "1")
        self.assertEqual(launch.env["OPENCODE_DISABLE_EXTERNAL_SKILLS"], "1")
        self.assertEqual(launch.env["OPENCODE_DISABLE_CLAUDE_CODE"], "1")
        self.assertEqual(c["instructions"], [str(launch.instructions_path)])
        self.assertEqual(c["mcp"]["auriga"], {"type": "remote", "url": "http://127.0.0.1:8888/mcp",
                                              "headers": {"Authorization": "Bearer t"}, "oauth": False,
                                              "timeout": 600000})

    def test_wires_subagents_variants(self):
        npm = {"openai_responses": "@ai-sdk/openai", "openai_chat": "@ai-sdk/openai-compatible"}
        for wire, package in npm.items():
            self.assertEqual(self.launch(wire_api=wire).config["provider"]["relay"]["npm"], package)
        c = self.launch(subagents=False).config
        self.assertEqual(c["permission"]["task"], "deny")
        c = self.launch(subagents=True, variants={"high": {"reasoningEffort": "high"}}).config
        self.assertNotIn("task", c["permission"])
        self.assertEqual(c["provider"]["relay"]["models"]["claude-sonnet-4-5"]["variants"],
                         {"high": {"reasoningEffort": "high"}})
        with self.assertRaises(cfg.OpenCodeConfigError):
            self.launch(wire_api="bogus")
        self.assertNotIn("mcp", self.launch(mcp_url=None).config)

    def test_files_dirs_env(self):
        launch = self.launch()
        self.assertEqual(launch.argv, (self.exe, "serve", "--hostname", "127.0.0.1", "--port", "4567"))
        self.assertEqual(launch.cwd, self.run_dir.resolve() / "actor")
        text = launch.instructions_path
        self.assertEqual(text.read_text(), "INSTR auriga_")
        self.assertEqual(stat.S_IMODE(text.stat().st_mode), 0o444)
        for name in ("private/opencode/xdg/config/opencode", "private/opencode/cfgdir"):
            self.assertEqual(stat.S_IMODE(launch.dirs[name].stat().st_mode), 0o555)
            # D2: an empty .gitignore is pre-created before the chmod, so OpenCode's own startup write of it
            # (which it does unconditionally unless the file already exists) never happens; under a sandboxed
            # read-only bind mount (EROFS, not tolerated) that write would otherwise kill instance bootstrap.
            self.assertEqual([path.name for path in launch.dirs[name].iterdir()], [".gitignore"])
            self.assertEqual((launch.dirs[name] / ".gitignore").read_text(), "")
        self.assertEqual(stat.S_IMODE(launch.dirs["private/opencode/xdg/data"].stat().st_mode), 0o700)
        bin_dir = launch.dirs["private/opencode/bin"]
        self.assertEqual(os.path.realpath(bin_dir / "rg"), os.path.realpath(self.rg))
        env = launch.env
        self.assertTrue(env["PATH"].startswith(str(bin_dir) + ":"))
        self.assertEqual(env["HOME"], str(launch.cwd))
        self.assertEqual(env["OPENCODE_SERVER_PASSWORD"], "secret-test-password")
        self.assertEqual(env["XDG_CONFIG_HOME"], str(self.run_dir.resolve() / "private/opencode/xdg/config"))
        for flag in ("OPENCODE_PURE", "OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_AUTOUPDATE",
                     "OPENCODE_DISABLE_MODELS_FETCH", "OPENCODE_DISABLE_LSP_DOWNLOAD"):
            self.assertEqual(env[flag], "1")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("OPENCODE_TEST_MANAGED_CONFIG_DIR", env)
        self.assertEqual(env["TZ"], "UTC")

    def test_relaunch_on_same_run_dir(self):
        self.launch()
        again = self.launch(instructions_text="CHANGED")
        self.assertEqual(again.instructions_path.read_text(), "CHANGED")
        self.assertEqual(stat.S_IMODE(again.dirs["private/opencode/cfgdir"].stat().st_mode), 0o555)

    def test_version_mismatch_and_probe(self):
        self.assertEqual(cfg.probe_cli_version(self.exe), "1.19.5")
        (self.root / "x").mkdir()
        with self.assertRaises(iso.VersionMismatch) as ctx:
            self.launch(executable=fake_binary(self.root / "x", "1.19.6"))
        self.assertEqual(str(ctx.exception), "opencode_version_mismatch")
        with self.assertRaises(cfg.OpenCodeConfigError):
            self.launch(executable="relative/opencode")

    def test_probe_cli_version_uses_isolated_home(self):
        # D3: without HOME/XDG_*, OpenCode resolves the invoking user's real passwd home and writes
        # ~/.cache/opencode/bin and ~/.local/share/opencode/repos there. The probe must run with a private,
        # disposable HOME and every XDG_*_HOME instead (mirrors `claude_code_runtime_problem`'s probe sandbox).
        captured = {}

        def fake_run(argv, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, stdout="1.19.5\n", stderr="")

        with mock.patch.object(cfg.subprocess, "run", side_effect=fake_run):
            self.assertEqual(cfg.probe_cli_version(self.exe), "1.19.5")
        env = captured["env"]
        home = env["HOME"]
        self.assertTrue(os.path.isabs(home))
        self.assertNotIn(home, (os.path.expanduser("~"), os.environ.get("HOME", "")))
        for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
            self.assertTrue(env[key].startswith(home + os.sep), f"{key}={env[key]!r} not under {home!r}")
        # The temporary HOME is a disposable directory, discarded once the (mocked) call returns: nothing it
        # might have created survives the probe.
        self.assertFalse(os.path.isdir(home))

    def test_managed_config_check(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(cfg.OpenCodeConfigError) as ctx:
                self.launch(managed_config_paths=[directory])
            self.assertEqual(str(ctx.exception), "opencode_managed_config_present")
        cfg.check_managed_config([str(self.root / "absent")])
        with mock.patch.object(cfg.os.path, "lexists", side_effect=lambda p: p.endswith("/alice/ai.opencode.managed.plist")):
            with self.assertRaises(cfg.OpenCodeConfigError):
                cfg.check_managed_config(user="alice")

    def test_ripgrep_missing(self):
        with self.assertRaises(cfg.OpenCodeConfigError) as ctx:
            cfg.resolve_ripgrep(search_path=str(self.root / "nothing"))
        self.assertEqual(str(ctx.exception), "opencode_ripgrep_missing")
        self.assertEqual(cfg.resolve_ripgrep(self.rg), self.rg)

    def test_health_version(self):
        self.assertEqual(cfg.check_health_version({"healthy": True, "version": "1.19.5"}), "1.19.5")
        for health in ({"healthy": True, "version": "1.19.4"}, {"healthy": True}, None):
            with self.assertRaises(iso.VersionMismatch):
                cfg.check_health_version(health)

    def test_provenance_block(self):
        launch = self.launch()
        block = cfg.provenance_block(launch, route_list_sha256="abc", wire_api="anthropic_messages",
                                     model="claude-sonnet-4-5", effort="high", instructions_sha256="d")
        self.assertEqual(block["cli"], {"version": "1.19.5", "executable": launch.executable, "sha256": launch.sha256})
        self.assertEqual(block["native_model_id"], "claude-sonnet-4-5")
        self.assertEqual(block["route_list_sha256"], "abc")
        self.assertEqual(block["native_effort"], {"variant": "high"})
        reasons = {item["name"]: item["reason"] for item in block["native_tools"]["disabled"]}
        self.assertEqual(reasons, {"webfetch": "online_access", "websearch": "online_access",
                                   "codesearch": "online_access", "question": "waits_for_human"})
        self.assertIsNone(block["native_tools"]["enabled"])
        alone = cfg.provenance_block(launch, route_list_sha256=None, wire_api="anthropic_messages", model="m",
                                     subagents=False, enabled_tools=["write", "bash"])
        self.assertEqual(alone["native_tools"]["enabled"], ["bash", "write"])
        self.assertIn({"name": "task", "reason": "subagents_disabled"}, alone["native_tools"]["disabled"])
        self.assertNotIn("secret-test-password", json.dumps(block))
        self.assertNotIn("Bearer", json.dumps(block))


class ToolIdsTests(unittest.TestCase):
    class Client:
        def __init__(self, tools, rules):
            self.tools, self.rules, self.calls = tools, rules, []

        def tool_list(self, provider_id, model_id):
            self.calls.append((provider_id, model_id))
            return self.tools

        def _request(self, method, path):
            self.calls.append((method, path))
            return [{"name": "build", "permission": self.rules}]

    @staticmethod
    def items(*ids):
        return [{"id": name, "description": "d", "parameters": {}} for name in ids]

    def test_read_effective_tool_ids(self):
        rules = [{"permission": "*", "pattern": "*", "action": "allow"},
                 {"permission": "webfetch", "pattern": "*", "action": "deny"},
                 {"permission": "edit", "pattern": "*", "action": "deny"},
                 {"permission": "bash", "pattern": "rm *", "action": "deny"}]
        client = self.Client(self.items("task", "bash", "webfetch", "write", "read", "invalid"), rules)
        self.assertEqual(cfg.read_effective_tool_ids(client, model="m1"), ["bash", "read", "task"])
        self.assertEqual(client.calls, [("relay", "m1"), ("GET", "/agent")])   # offered list, not /tool/ids
        self.assertEqual(cfg.read_effective_tool_ids(self.Client(self.items("a", "invalid"), []), model="m",
                                                     provider="p"), ["a"])
        for bad in ({"a": 1}, [1], None, [{"name": "x"}]):
            with self.assertRaises(cfg.OpenCodeConfigError):
                cfg.read_effective_tool_ids(self.Client(bad, rules), model="m")
        with self.assertRaises(cfg.OpenCodeConfigError):
            cfg.read_effective_tool_ids(self.Client(self.items("a"), None), model="m")


class RouteListTests(unittest.TestCase):
    def test_checked_in_list(self):
        routes = cfg.load_expected_routes()
        self.assertEqual(len(routes), 188)
        self.assertEqual(routes, sorted(routes))
        for needed in ("GET /global/health", "POST /session", "GET /session/{sessionID}",
                       "POST /session/{sessionID}/prompt_async", "POST /session/{sessionID}/abort",
                       "GET /session/status", "GET /session/{sessionID}/message", "GET /event",
                       "GET /config/providers"):
            self.assertIn(needed, routes)
        data = json.loads(Path(cfg._ROUTES_FILE).read_text())
        self.assertEqual(data["opencode_version"], "1.19.5")

    def test_digest_and_mismatch(self):
        routes = cfg.load_expected_routes()
        digest = cfg.check_route_list(routes)
        self.assertEqual(digest, cfg.route_list_digest(reversed(routes)))
        self.assertEqual(len(digest), 64)
        with self.assertRaises(cfg.OpenCodeConfigError) as ctx:
            cfg.check_route_list(routes[1:] + ["GET /new"])
        self.assertEqual(str(ctx.exception), "opencode_route_list_mismatch")
        self.assertEqual((ctx.exception.added, ctx.exception.removed), (["GET /new"], [routes[0]]))

    def test_package_data_declared(self):
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        self.assertIn('"auriga.cortex.agents.opencode" = ["opencode_routes.json"]', text)


@unittest.skipUnless(EXECUTABLE, "AURIGA_OPENCODE_EXECUTABLE not set (pinned opencode binary)")
class RealBinaryTests(unittest.TestCase):
    def test_serve_with_generated_config(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            launch = cfg.build_launch(
                run_dir, executable=EXECUTABLE, model="claude-sonnet-4-5", wire_api="anthropic_messages",
                relay_base_url="http://127.0.0.1:9", instructions_text="x", port=port, password="pw-real",
                context_limit_tokens=200000, output_limit_tokens=16000)
            log = open(run_dir / "serve.log", "wb")
            process = subprocess.Popen(launch.argv, cwd=launch.cwd, env=launch.env, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                client = OpenCodeClient(f"http://127.0.0.1:{port}", "pw-real", timeout_s=2)
                deadline, health = time.monotonic() + 30, None
                while time.monotonic() < deadline and health is None:
                    self.assertIsNone(process.poll(), (run_dir / "serve.log").read_text()[-500:])
                    try:
                        health = client.health()
                    except OSError:
                        time.sleep(0.05)
                self.assertEqual(cfg.check_health_version(health), "1.19.5")
                digest = cfg.check_route_list(client.routes())
                self.assertEqual(digest, cfg.route_list_digest(cfg.load_expected_routes()))
                self.assertEqual(client.status(), {})
                # D2: the pre-created .gitignore in the (chmod 0555, or EROFS under the sandbox) config dirs means
                # OpenCode's startup Config.ensureGitignore never tries to write it, so instance bootstrap succeeds
                # and this, the first instance-scoped request, answers 200 (not 500) with the tool id list.
                ids_response = client._request("GET", "/experimental/tool/ids")
                self.assertIsInstance(ids_response, list)
                self.assertGreater(len(ids_response), 0)
                offered = {item["id"] for item in client.tool_list("relay", "m")}
                self.assertNotIn("invalid", cfg.read_effective_tool_ids(client, model="m"))
                self.assertLessEqual(set(cfg.read_effective_tool_ids(client, model="m")), offered)
                ids = cfg.read_effective_tool_ids(client, model="m")
                for denied in ("webfetch", "websearch", "codesearch", "question"):
                    self.assertNotIn(denied, ids)
                for kept in ("task", "todowrite", "skill", "bash", "read", "edit"):
                    self.assertIn(kept, ids)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                process.wait(10)
                log.close()
                for name in ("private/opencode/xdg/config/opencode", "private/opencode/cfgdir"):
                    os.chmod(launch.dirs[name], 0o700)


if __name__ == "__main__":
    unittest.main()
