"""agent.claude_code.upstream.source = "claude_settings": endpoint and key come from a Claude Code settings file in
the official format, read by the gateway only; the key never enters the archived config."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auriga.core.config import Config, ConfigError
from auriga.cortex.agents import factory
from auriga.cortex.agents.claude_code.options import read_claude_settings_upstream

TOKEN = "sk-test-claude-settings-token"


class ClaudeSettingsUpstreamTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def test_official_format_bearer_token_and_api_key(self):
        path = self.write("claude.json", {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:3000/",
                                                   "ANTHROPIC_AUTH_TOKEN": TOKEN}, "model": "opus"})
        self.assertEqual(read_claude_settings_upstream(path),
                         ("http://127.0.0.1:3000", {"Authorization": "Bearer " + TOKEN}))
        path = self.write("key.json", {"env": {"ANTHROPIC_API_KEY": TOKEN}})
        self.assertEqual(read_claude_settings_upstream(path), ("https://api.anthropic.com", {"x-api-key": TOKEN}))

    def test_errors_name_the_file_never_the_key(self):
        for name, value, code in (("nokey.json", {"env": {"ANTHROPIC_BASE_URL": "http://x"}}, "claude_settings_missing_key"),
                                  ("noenv.json", {"model": "opus"}, "claude_settings_missing_env")):
            with self.assertRaises(ValueError) as raised:
                read_claude_settings_upstream(self.write(name, value))
            self.assertTrue(str(raised.exception).startswith(code), str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            read_claude_settings_upstream(self.root / "missing.json")
        self.assertTrue(str(raised.exception).startswith("claude_settings_unreadable"))

    def test_config_resolves_relative_path_and_never_archives_the_key(self):
        self.write("claude.local.json", {"env": {"ANTHROPIC_AUTH_TOKEN": TOKEN}})
        config_path = self.write("auriga.json", {"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": "claude.local.json"}}}})
        config = Config.load(config_path, environ={})
        self.assertEqual(Path(config.agent.claude_code.upstream.settings), (self.root / "claude.local.json").resolve())
        self.assertNotIn(TOKEN, json.dumps(config.to_dict()))

    def test_settings_path_is_required(self):
        with self.assertRaises(ConfigError):
            Config.from_dict({"agent": {"claude_code": {"upstream": {"source": "claude_settings"}}}})

    def test_launch_gate_reads_the_file(self):
        good = self.write("good.json", {"env": {"ANTHROPIC_AUTH_TOKEN": TOKEN}})
        bad = self.write("bad.json", {"env": {}})
        quiet = {"self_briefing": False, "visual_review_model": None}
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
                "source": "claude_settings", "settings": str(good)}}}, "model": {"name": "m"}, "features": quiet})
            self.assertIsNone(factory.backend_requirements(config))
            config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
                "source": "claude_settings", "settings": str(bad)}}}, "model": {"name": "m"}, "features": quiet})
            with self.assertRaisesRegex(ConfigError, "^claude_settings_missing_key"):
                factory.backend_requirements(config)

    def test_session_upstream_uses_the_settings_file(self):
        from auriga.cortex.agents.claude_code.session import ClaudeCodeSession
        from auriga.core.storage.store import Store
        path = self.write("claude.json", {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9", "ANTHROPIC_AUTH_TOKEN": TOKEN}})
        run = self.root / "run"
        run.mkdir()
        store = Store(run, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": str(path)}}}, "model": {"name": "m"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        with mock.patch.dict(os.environ, {}, clear=False):
            session = ClaudeCodeSession(run, store, lambda params: None, config=config)
            self.assertEqual(session._upstream(), ("http://127.0.0.1:9", {"Authorization": "Bearer " + TOKEN}))


if __name__ == "__main__":
    unittest.main()
