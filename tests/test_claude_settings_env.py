"""agent.claude_code.upstream.source = "claude_settings": the settings file's non-credential ``env`` entries reach the
Claude Code CLI environment (as Claude Code applies settings.json env); credentials stay with the host relay, a name
Auriga sets itself is refused, and only the forwarded NAMES are recorded."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support.fake_claude_sdk import requires_sdk

from auriga.core.config import Config, ConfigError
from auriga.cortex.agents import factory
from auriga.cortex.agents.claude_code import options as opts

TOKEN = "sk-test-claude-settings-env-token"
SECRET_HEADER = "x-provider-secret: header-value-never-recorded"
SESSION = "9e7c96f2-1111-4222-8333-444455556666"
FAKE_CLI = '#!/bin/sh\nif [ "$1" = "--version" ]; then echo "2.1.321 (Claude Code)"; exit 0; fi\nexec /usr/bin/env\n'
QUIET = {"self_briefing": False, "visual_review_model": None}


class SettingsCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def write(self, name, env, **extra):
        path = self.root / name
        path.write_text(json.dumps({"env": env, **extra}))
        return path


class ReadSettingsEnvTests(SettingsCase):
    def test_forwards_provider_entries_never_credentials(self):
        path = self.write("tr.json", {"ANTHROPIC_BASE_URL": "https://gw.example/gateway", "ANTHROPIC_AUTH_TOKEN": TOKEN,
                                      "ANTHROPIC_API_KEY": TOKEN, "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0",
                                      "ANTHROPIC_CUSTOM_HEADERS": SECRET_HEADER})
        self.assertEqual(opts.read_claude_settings_env(path), {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0",
                                                                "ANTHROPIC_CUSTOM_HEADERS": SECRET_HEADER})
        self.assertEqual(opts.read_claude_settings_env(self.write("plain.json", {"ANTHROPIC_AUTH_TOKEN": TOKEN})), {})

    def test_auriga_owned_names_are_refused_naming_file_and_variable(self):
        for name in ("CLAUDE_CONFIG_DIR", "CLAUDE_CODE_MAX_RETRIES", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_MODEL",
                     "PATH", "HOME", "LANG", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
                     "CLAUDE_STREAM_IDLE_TIMEOUT_MS", "IS_SANDBOX", "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS"):
            with self.subTest(name=name):
                path = self.write("owned.json", {"ANTHROPIC_AUTH_TOKEN": TOKEN, name: "value-never-shown"})
                with self.assertRaises(ValueError) as raised:
                    opts.read_claude_settings_env(path)
                message = str(raised.exception)
                self.assertTrue(message.startswith(f"claude_settings_env_reserved:{path}: {name} "), message)
                self.assertNotIn("value-never-shown", message)
                self.assertNotIn(TOKEN, message)

    def test_values_must_be_strings_and_names_shell_names(self):
        for env, code in (({"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": 0}, "claude_settings_env_not_string"),
                          ({"CLAUDE_CODE_MAX_OUTPUT_TOKENS": None}, "claude_settings_env_not_string"),
                          ({"BAD-NAME": "1"}, "claude_settings_env_invalid_name")):
            with self.subTest(code=code):
                with self.assertRaises(ValueError) as raised:
                    opts.read_claude_settings_env(self.write("bad.json", {"ANTHROPIC_AUTH_TOKEN": TOKEN, **env}))
                self.assertTrue(str(raised.exception).startswith(code), str(raised.exception))

    def test_launch_gate_refuses_an_owned_name(self):
        good = self.write("good.json", {"ANTHROPIC_AUTH_TOKEN": TOKEN, "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        bad = self.write("bad.json", {"ANTHROPIC_AUTH_TOKEN": TOKEN, "CLAUDE_CODE_MAX_RETRIES": "3"})
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            for path, expected in ((good, None), (bad, "claude_settings_env_reserved")):
                config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
                    "source": "claude_settings", "settings": str(path)}}}, "model": {"name": "m"}, "features": QUIET})
                if expected is None:
                    self.assertIsNone(factory.backend_requirements(config))
                else:
                    with self.assertRaisesRegex(ConfigError, "^" + expected):
                        factory.backend_requirements(config)

    def test_archived_config_holds_only_the_path(self):
        self.write("claude.local.json", {"ANTHROPIC_AUTH_TOKEN": TOKEN, "ANTHROPIC_CUSTOM_HEADERS": SECRET_HEADER})
        config_path = self.root / "auriga.json"
        config_path.write_text(json.dumps({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": "claude.local.json"}}}}))
        dumped = json.dumps(Config.load(config_path, environ={}).to_dict())
        self.assertNotIn(TOKEN, dumped)
        self.assertNotIn(SECRET_HEADER, dumped)


@requires_sdk
class LaunchForwardingTests(SettingsCase):
    def setUp(self):
        super().setUp()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        cli = self.root / "fake-claude"
        cli.write_text(FAKE_CLI)
        cli.chmod(0o755)
        self.runtime = opts.resolve_runtime(cli, {"PATH": os.defpath})

    def build(self, **overrides):
        arguments = dict(model="kimi-k2.6", effort="high", thread_id=SESSION, instructions="Be careful.\n",
                         relay_base_url="http://127.0.0.1:41000", mcp_url="http://127.0.0.1:41001/mcp",
                         mcp_headers={"Authorization": "Bearer tok"}, runtime=self.runtime)
        arguments.update(overrides)
        return opts.build_launch(self.run_dir, **arguments)

    def test_forwarded_values_reach_env_and_launcher_names_only_in_provenance(self):
        forwarded = {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0", "ANTHROPIC_CUSTOM_HEADERS": SECRET_HEADER}
        launch = self.build(settings_env=forwarded)
        for name, value in forwarded.items():
            self.assertEqual(launch.env[name], value)
        script = launch.launcher.read_text()
        self.assertIn("'CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0'", script)
        self.assertEqual(launch.native_settings["settings_env"],
                         ["ANTHROPIC_CUSTOM_HEADERS", "CLAUDE_CODE_THINKING_DISPLAY_UPDATES"])
        recorded = json.dumps(launch.native_settings)
        self.assertNotIn(SECRET_HEADER, recorded)
        self.assertNotIn("CLAUDE_CODE_THINKING_DISPLAY_UPDATES", json.dumps(launch.native_settings["env"]))
        self.assertEqual(launch.env["ANTHROPIC_API_KEY"], opts.PLACEHOLDER_API_KEY)   # never the provider key
        self.assertEqual(self.build().native_settings["settings_env"], [])

    def test_build_launch_refuses_owned_or_credential_names(self):
        for name in ("CLAUDE_CODE_MAX_RETRIES", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"^claude_settings_env_reserved:{name}$"):
                self.build(settings_env={name: "x"})

    def test_every_variable_auriga_sets_is_owned(self):
        """Drift guard: a new variable in build_launch must be added to AURIGA_OWNED_ENV (or a settings file could
        override it)."""
        with mock.patch.object(opts, "_running_as_root", return_value=True):
            launch = self.build(native_memories=False, instance_id="agent-1",
                                config=opts.ClaudeCodeConfig(auto_compact_window_tokens=200000))
        self.assertEqual(set(launch.env) - opts.AURIGA_OWNED_ENV, set())

    def test_sandbox_launch_carries_forwarded_values_into_the_namespace(self):
        created = {}

        class Sandbox:
            def __init__(self, *args, **keywords):
                created.update(keywords)

            def prepare(self):
                return ["/usr/bin/env", "-i", "/usr/bin/bwrap", "--", "/usr/bin/python3", "-c", "pass"], {}

            def close(self):
                pass

        plan = opts.SandboxPlan(ipc=self.root / "ipc", ports=(41000, 41001))
        with mock.patch.object(opts, "AgentSandbox", Sandbox):
            launch = self.build(settings_env={"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}, sandbox=plan)
        self.assertEqual(created["env"]["CLAUDE_CODE_THINKING_DISPLAY_UPDATES"], "0")
        self.assertIn("CLAUDE_CODE_THINKING_DISPLAY_UPDATES", launch.native_settings["sandbox"]["sandbox_env"])


@requires_sdk
class SessionForwardingTests(SettingsCase):
    def test_session_reads_the_forwarded_env_from_the_settings_file(self):
        from auriga.core.storage.store import Store
        from auriga.cortex.agents.claude_code.session import ClaudeCodeSession
        path = self.write("tr.json", {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9", "ANTHROPIC_AUTH_TOKEN": TOKEN,
                                      "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        run = self.root / "run"
        run.mkdir()
        store = Store(run, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": str(path)}}}, "model": {"name": "m"}, "features": QUIET})
        session = ClaudeCodeSession(run, store, lambda params: None, config=config)
        self.assertEqual(session._settings_env(), {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        explicit = Config.from_dict({"agent": {"backend": "claude_code"}, "model": {"name": "m"}, "features": QUIET})
        self.assertEqual(ClaudeCodeSession(run, store, lambda params: None, config=explicit)._settings_env(), {})


if __name__ == "__main__":
    unittest.main()
