"""agent.* configuration: defaults, omission, round trip, validation and the --agent-backend flag."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from auriga.core.config import (AgentConfig, ClaudeCodeConfig, Config, ConfigError, OpenCodeConfig,
                                UpstreamConfig, cli_overrides, namespace_values)


class AgentConfigTests(unittest.TestCase):
    def test_default_is_codex_and_omitted_from_snapshots(self):
        config = Config()
        self.assertEqual(config.agent, AgentConfig())
        self.assertEqual(config.agent.backend, 'codex')
        self.assertNotIn('agent', config.to_dict())
        self.assertEqual(Config.from_dict({}).to_dict().keys(), config.to_dict().keys())
        self.assertEqual(Config().to_dict(), Config(agent=AgentConfig()).to_dict())
        self.assertEqual(Config.from_dict(config.to_dict()).agent, AgentConfig())

    def test_claude_code_defaults_to_the_anthropic_upstream(self):
        upstream = Config().agent.claude_code.upstream
        self.assertEqual((upstream.wire_api, upstream.base_url, upstream.api_key_env, upstream.auth_scheme, upstream.provider),
                         ('anthropic_messages', 'https://api.anthropic.com', 'ANTHROPIC_API_KEY', 'x-api-key', 'anthropic'))
        self.assertIsNone(Config().agent.claude_code.cli_path)
        self.assertEqual(Config().agent.opencode.context_limit_tokens, 200000)
        self.assertEqual(Config().agent.opencode.output_limit_tokens, 32000)

    def test_partial_agent_document_inherits_backend_defaults(self):
        config = Config.from_dict({'agent': {'backend': 'claude_code',
                                             'claude_code': {'upstream': {'api_key_env': 'MY_KEY'}}}})
        upstream = config.agent.claude_code.upstream
        self.assertEqual(config.agent.backend, 'claude_code')
        self.assertEqual((upstream.api_key_env, upstream.base_url, upstream.auth_scheme), ('MY_KEY', 'https://api.anthropic.com', 'x-api-key'))

    def test_non_default_round_trip(self):
        document = {'agent': {'backend': 'opencode', 'claude_code': {'subagents': True, 'auto_compact_window_tokens': 200000,
                              'context_delivery': 'delta'},
                    'opencode': {'executable': '/opt/opencode/bin/opencode', 'context_limit_tokens': 400000,
                                 'upstream': {'wire_api': 'openai_chat', 'base_url': 'http://127.0.0.1:9000/v1',
                                              'api_key_env': 'UPSTREAM_KEY', 'auth_scheme': 'bearer', 'provider': 'local'}}}}
        config = Config.from_dict(document)
        exported = config.to_dict()
        self.assertEqual(exported['agent']['opencode']['executable'], '/opt/opencode/bin/opencode')
        self.assertEqual(Config.from_dict(exported), config)
        self.assertEqual(Config.from_dict(json.loads(json.dumps(exported))).to_dict(), exported)
        self.assertEqual(config.with_overrides({'agent': {'backend': 'codex'}}).agent.opencode, config.agent.opencode)

    def test_relative_paths_resolve_next_to_the_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'c.json'
            path.write_text(json.dumps({'agent': {'claude_code': {'cli_path': 'bin/claude'}}}))
            config = Config.load(path, environ={})
            self.assertEqual(config.agent.claude_code.cli_path, str((Path(directory) / 'bin/claude').resolve()))

    def test_validation_errors(self):
        for agent, message in (
                ({'backend': 'gemini'}, 'invalid_config_choice:AgentConfig.backend'),
                ({'claude_code': {'context_delivery': 'both'}}, 'invalid_config_choice:ClaudeCodeConfig.context_delivery'),
                ({'claude_code': {'auto_compact_window_tokens': 99999}}, 'config_below_minimum'),
                ({'claude_code': {'auto_compact_window_tokens': 1000001}}, 'config_above_maximum'),
                ({'opencode': {'context_limit_tokens': 0}}, 'config_below_minimum'),
                ({'opencode': {'upstream': {'wire_api': 'grpc'}}}, 'invalid_config_choice:UpstreamConfig.wire_api'),
                ({'opencode': {'upstream': {'auth_scheme': 'basic'}}}, 'invalid_config_choice:UpstreamConfig.auth_scheme'),
                ({'opencode': {'upstream': {'source': 'env'}}}, 'invalid_config_choice:UpstreamConfig.source'),
                ({'claude_code': {'upstream': {'base_url': 'https://user:pw@api.example.com'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'base_url': 'https://api.example.com/v1?key=abc'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'base_url': 'https://api.example.com/#frag'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'base_url': 'ftp://api.example.com'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'base_url': 'api.example.com'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'api_key_env': 'sk-ant-abc123 secret'}}}, 'invalid_upstream_api_key_env'),
                ({'claude_code': {'upstream': {'api_key_env': 'sk-ant-abc'}}}, 'invalid_upstream_api_key_env'),
                ({'claude_code': {'subagents': 'yes'}}, 'invalid_config_type'),
                ({'unknown_key': 1}, 'unknown_config_keys'),
                ({'claude_code': {'upstream': {'api_key': 'secret'}}}, 'unknown_config_keys')):
            with self.subTest(agent=agent), self.assertRaisesRegex(ConfigError, message):
                Config.from_dict({'agent': agent})

    def test_section_constructors_validate_directly(self):
        with self.assertRaises(ConfigError):
            UpstreamConfig(base_url='https://host/path?token=1')
        self.assertEqual(UpstreamConfig(base_url='http://127.0.0.1:8080/v1').api_key_env, None)
        self.assertIsInstance(ClaudeCodeConfig().upstream, UpstreamConfig)
        self.assertEqual(OpenCodeConfig().upstream, UpstreamConfig())

    def test_cli_flag_maps_to_agent_backend_only(self):
        self.assertEqual(cli_overrides({'agent_backend': 'claude_code'}), {'agent': {'backend': 'claude_code'}})
        config = Config.load(None, environ={}, overrides=cli_overrides({'agent_backend': 'opencode'}))
        self.assertEqual(config.agent.backend, 'opencode')
        self.assertEqual(namespace_values(config)['agent_backend'], 'opencode')
        self.assertEqual(namespace_values(Config())['agent_backend'], 'codex')

    def test_gateway_parser_accepts_the_flag_and_rejects_other_values(self):
        from auriga.apps.gateway import cli as gateway
        with patch.object(gateway, 'serve', new_callable=AsyncMock) as serve, \
                patch.object(sys, 'argv', ['gateway', '--agent-backend', 'claude_code']):
            gateway.main()
        self.assertEqual(serve.call_args.args[0].agent_backend, 'claude_code')
        self.assertEqual(serve.call_args.args[0].resolved_config.agent.backend, 'claude_code')
        with patch.object(gateway, 'serve', new_callable=AsyncMock) as serve, \
                patch.object(sys, 'argv', ['gateway']):
            gateway.main()
        self.assertEqual(serve.call_args.args[0].agent_backend, 'codex')
        with patch.object(sys, 'argv', ['gateway', '--agent-backend', 'gemini']), self.assertRaises(SystemExit):
            gateway.main()


if __name__ == '__main__':
    unittest.main()
