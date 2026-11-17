"""Provider routing and native configuration checks; no model or GPU calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

from codex_cli_bin import bundled_codex_path
from auriga.cortex.session.codex import CodexSession, MODEL_PROXY_PROVIDER, ACTOR_ISOLATION, _model_provider, _upstream_auth
from auriga.core.contracts import digest
from auriga.core.storage.store import Store
from support.host_capabilities import bubblewrap_namespaces, require


class CodexProviderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        config = self.root / "auriga.json"
        config.write_text(json.dumps({"version": 1, "features": {"sandbox_enabled": True},
                                      "model": {"codex_home": str(self.source)}}))
        self.env = mock.patch.dict(os.environ, {"CODEX_HOME": str(self.source),
            "AURIGA_CONFIG": str(config),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")})
        self.env.start()
        self.base_env = mock.patch.dict(os.environ)
        self.base_env.start()
        os.environ.pop("OPENAI_BASE_URL", None)

    def tearDown(self):
        self.base_env.stop()
        self.env.stop()
        self.temporary.cleanup()

    def auth(self, value):
        (self.source / "auth.json").write_text(json.dumps(value))

    def test_builtin_auth_selects_chatgpt_or_api_upstream(self):
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        name, provider, url = _model_provider({}, self.source)
        self.assertEqual(name, "openai")
        self.assertTrue(provider["requires_openai_auth"])
        self.assertEqual(url, "https://chatgpt.com/backend-api/codex")
        self.auth({"OPENAI_API_KEY": "synthetic"})
        self.assertEqual(_model_provider({}, self.source)[2], "https://api.openai.com/v1")
        self.auth({"tokens": {"access_token": "synthetic"}})
        self.assertEqual(_model_provider({}, self.source)[2], "https://chatgpt.com/backend-api/codex")

    def test_explicit_routes_and_custom_provider_auth_are_preserved(self):
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        config = {"openai_base_url": "https://example.invalid/v1"}
        self.assertEqual(_model_provider(config, self.source)[2], config["openai_base_url"])
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.invalid/v1"}):
            self.assertEqual(_model_provider({}, self.source)[2], "https://env.invalid/v1")
            self.assertEqual(_model_provider(config, self.source)[2], config["openai_base_url"])
        config = {"model_provider": "fixture", "model_providers": {"fixture": {
            "base_url": "https://fixture.invalid/v1", "env_key": "FIXTURE_KEY"}}}
        name, provider, url = _model_provider(config, self.source)
        self.assertEqual((name, url), ("fixture", "https://fixture.invalid/v1"))
        self.assertEqual(provider["env_key"], "FIXTURE_KEY")
        self.assertNotIn("requires_openai_auth", provider)

    def test_relay_auth_is_resolved_without_exposing_secrets_to_native_config(self):
        self.auth({"OPENAI_API_KEY": "host-only-key"})
        self.assertEqual(_upstream_auth({"requires_openai_auth": True}, self.source),
                         {"Authorization": "Bearer host-only-key"})
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "host-token", "account_id": "host-account"}})
        self.assertEqual(_upstream_auth({"requires_openai_auth": True}, self.source),
                         {"Authorization": "Bearer host-token", "ChatGPT-Account-Id": "host-account"})
        with mock.patch.dict(os.environ, {"FIXTURE_KEY": "host-env-secret"}):
            self.assertEqual(_upstream_auth({"env_key": "FIXTURE_KEY"}, self.source),
                             {"Authorization": "Bearer host-env-secret"})

    def test_resume_rejects_actor_symlink_before_host_file_writes(self):
        (self.source / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nbase_url="http://127.0.0.1:1/v1"\n')
        store = Store(self.root / "symlink-run", dict(run_id="symlink", episode_id="e", env_instance_id="i", env_generation=1))
        session = CodexSession(store.root, store, lambda _: {})
        try:
            session.codex_home.mkdir(parents=True)
            target = self.root / "host-file"
            target.write_text("host-only-sentinel")
            (session.codex_home / "models.json").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "sandbox_home_symlink"):
                session._prepare()
            self.assertEqual(target.read_text(), "host-only-sentinel")
        finally:
            session.close()
            store.close()

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_verified_sandbox_home_can_resume_with_native_generated_skills(self):
        (self.source / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nbase_url="http://127.0.0.1:1/v1"\n')
        store = Store(self.root / "resume-run", dict(run_id="resume", episode_id="e", env_instance_id="i", env_generation=1))
        session = CodexSession(store.root, store, lambda _: {})
        try:
            skills = session.codex_home / "skills" / ".system"
            skills.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "sandbox_home_contains_host_extensions"):
                session._prepare()
            store.set_state("runtime", "codex", {"isolation": ACTOR_ISOLATION})
            session._prepare()
            self.assertEqual(json.loads((session.codex_home / "auth.json").read_text()),
                             {"OPENAI_API_KEY": "auriga-local-relay"})
        finally:
            session.close()
            store.close()

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_image_reuse_uses_frozen_settings_not_changed_environment(self):
        (self.source / "config.toml").write_text('model = "gpt-6-astra"\n')
        self.auth({"OPENAI_API_KEY": "synthetic"})
        store = Store(self.root / "run", dict(run_id="synthetic-image-policy", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        with mock.patch.dict(os.environ, {"AURIGA_IMAGE_SETTINGS": '{"reuse_exact_images":false}'}):
            session = CodexSession(self.root / "run", store, lambda value: {})
        relay = mock.MagicMock(base_url="http://127.0.0.1:54321")
        relay.start.return_value = relay
        try:
            with mock.patch.dict(os.environ, {"AURIGA_IMAGE_SETTINGS": '{"reuse_exact_images":true}'}), \
                 mock.patch("auriga.cortex.session.codex.ModelRequestProxy", return_value=relay) as factory:
                session._prepare()
                self.assertIsNone(session.account_control, "API key mode does not need ChatGPT account discovery")
                self.assertFalse(factory.call_args.kwargs["image_reuse"])
                self.assertTrue(factory.call_args.kwargs["image_trim"])
                self.assertEqual(factory.call_args.kwargs["request_projection"], session._present_model_request)
                self.assertEqual(factory.call_args.kwargs["request_context"], session._task_decision_context)
                self.assertFalse(store.read_state("runtime", "model_image_policy")["enabled"])
        finally:
            session.close()
            store.close()

    def test_custom_chatgpt_backend_is_not_silently_rerouted(self):
        (self.source / "config.toml").write_text('chatgpt_base_url = "https://region.example.invalid/backend-api"\n')
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        store = Store(self.root / "run", dict(run_id="synthetic-regional-provider", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        session = CodexSession(store.root, store, lambda value: {})
        try:
            with mock.patch("auriga.cortex.session.codex.ModelRequestProxy") as model, \
                 mock.patch("auriga.cortex.session.codex.AccountControlRelay") as account:
                with self.assertRaisesRegex(ValueError, "account_control_requires_default_chatgpt_backend"):
                    session._prepare()
                model.assert_not_called()
                account.assert_not_called()
        finally:
            session.close()
            store.close()

    def test_account_relay_startup_failure_closes_the_started_model_proxy(self):
        (self.source / "config.toml").write_text('model_provider = "openai"\n')
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        store = Store(self.root / "run", dict(run_id="synthetic-relay-startup", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        session = CodexSession(store.root, store, lambda value: {})
        model, account = mock.MagicMock(), mock.MagicMock()
        model.start.return_value = model
        account.start.side_effect = RuntimeError("fixture-certificate-failure")
        try:
            with mock.patch("auriga.cortex.session.codex.ModelRequestProxy", return_value=model), \
                 mock.patch("auriga.cortex.session.codex.AccountControlRelay", return_value=account):
                with self.assertRaisesRegex(RuntimeError, "fixture-certificate-failure"):
                    session._prepare()
            model.close.assert_called_once()
            account.close.assert_called_once()
            self.assertIsNone(session.model_proxy)
            self.assertIsNone(session.account_control)
        finally:
            session.close()
            store.close()

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_generated_native_config_uses_private_id_but_preserves_source_identity(self):
        (self.source / "config.toml").write_text('model = "gpt-6-astra"\n')
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        store = Store(self.root / "run", dict(run_id="synthetic-provider", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        session = CodexSession(self.root / "run", store, lambda value: {})
        relay = mock.MagicMock(base_url="http://127.0.0.1:54321")
        relay.start.return_value = relay
        account = mock.MagicMock(base_url="https://127.0.0.1:54322/backend-api",
                                 ca_path=session.codex_home / "account-control-ca.pem")
        account.server.server_port = 54322
        account.start.return_value = account
        wrapper = self.root / "codex"
        wrapper.write_text("#!/bin/sh\nexit 99\n")
        wrapper.chmod(0o755)
        try:
            with mock.patch("auriga.cortex.session.codex.ModelRequestProxy", return_value=relay) as factory, \
                 mock.patch("auriga.cortex.session.codex.AccountControlRelay", return_value=account), \
                 mock.patch.dict(os.environ, {"PATH": str(self.root)}):
                self.assertEqual(session._prepare(), "openai")
                self.assertEqual(factory.call_args.args[0], "https://chatgpt.com/backend-api/codex")
                self.assertTrue(factory.call_args.kwargs["image_reuse"])
                self.assertTrue(store.read_state("runtime", "image_settings")["reuse_exact_images"])
                policy = store.read_state("runtime", "model_image_policy")
                self.assertEqual(policy, dict(enabled=True, policy="exact_image_part_current_request_v1",
                                              compaction="unchanged", oversized_trim=True, max_inline_images=60))
                self.assertEqual(factory.call_args.kwargs["max_inline_images"], 60)
                self.assertEqual(factory.call_args.kwargs["transport_recovery"]["max_rounds"], 10)
                self.assertEqual(factory.call_args.kwargs["recovery_gate"](), "recovery_gate_unavailable")
            native = bundled_codex_path().resolve()
            self.assertEqual(session.cli_executable, native)
            self.assertEqual(session.command[-5:], [str(native), "--no-daemon", "app-server", "--listen",
                                                    f"unix://{session.socket_path}"])
            self.assertIn("--unshare-net", session.command)
            self.assertNotIn(str(wrapper), session.command)
            config = tomllib.loads((session.codex_home / "config.toml").read_text())
            self.assertEqual(config["model_provider"], MODEL_PROXY_PROVIDER)
            self.assertNotIn("openai", config["model_providers"])
            self.assertTrue(config["model_providers"][MODEL_PROXY_PROVIDER]["requires_openai_auth"])
            provider = config["model_providers"][MODEL_PROXY_PROVIDER]
            self.assertEqual(provider["request_max_retries"], 0)
            self.assertEqual(provider["stream_max_retries"], 0)
            self.assertEqual(provider["stream_idle_timeout_ms"], 2**63-1)
            self.assertEqual(config["chatgpt_base_url"], account.base_url)
            self.assertFalse((session.private / "isolation.json").exists())
            self.assertEqual(session.env["CODEX_CA_CERTIFICATE"], str(account.ca_path))
            self.assertEqual(session.env["CODEX_HOME"], str(session.codex_home))
            self.assertEqual(config["sandbox_mode"], "danger-full-access")
            self.assertEqual(config["web_search"], "disabled")
            self.assertFalse(config["features"]["network_proxy"])
            self.assertNotIn("synthetic", (session.codex_home / "auth.json").read_text())
            for feature in ("shell_tool", "unified_exec"):
                self.assertTrue(config["features"][feature], feature)
            with self.assertRaisesRegex(ValueError, "resume_usage_identity_conflict"):
                session._prepare((session.model, "different-provider"))
            if native.is_file():
                # features list parses the real pinned native config, without inference.
                result = subprocess.run([native, "features", "list"], env=session.env,
                    cwd=session.actor_dir, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
            store.set_state("runtime", "codex", {"provider": "openai", "tools_digest": digest([]),
                                                  "isolation": ACTOR_ISOLATION})
            with mock.patch.object(session, "_stop_previous_owned_processes"), \
                 mock.patch.object(session, "_prepare", side_effect=RuntimeError("identity-checked")) as prepare:
                with self.assertRaisesRegex(RuntimeError, "identity-checked"):
                    session.start([], "synthetic", resume_thread_id="synthetic-thread")
                prepare.assert_called_once_with((session.model, "openai"))
        finally:
            session.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
