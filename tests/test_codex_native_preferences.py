"""Fixed harness Codex preferences, verified without model or GPU calls."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import threading
import tomllib
import unittest
from unittest import mock

from openai_codex.generated.v2_all import ConfigReadResponse, ExperimentalFeatureListResponse

from auriga.cortex.session.codex import CodexSession, MODEL_PROXY_PROVIDER, CODEX_PROTOCOL_VERSION
from auriga.core.config import Config
from auriga.core.storage.store import Store
from support.host_capabilities import bubblewrap_namespaces, require


class CodexNativePreferencesTest(unittest.TestCase):
    enabled_features = (
        "goals", "hooks", "memories", "mentions_v2", "multi_agent",
        "prevent_idle_sleep", "guardian_approval", "multi_agent_v2", "context_management",
        "shell_tool", "unified_exec",
    )
    disabled_features = (
        "apps", "plugins", "browser_use", "browser_use_external",
        "computer_use", "artifact", "image_generation", "network_proxy",
    )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.source_config = self.source / "config.toml"
        self.source_config.write_text('model = "different-local-model"\n'
            'model_reasoning_effort = "low"\nmodel_verbosity = "high"\n'
            'service_tier = "fast"\nsuppress_unstable_features_warning = false\n'
            'model_context_window = 9999\nweb_search = "live"\n'
            'approval_policy = "on-request"\nmodel_provider = "fixture"\n'
            '[model_providers.fixture]\nbase_url = "https://fixture.invalid/v1"\n'
            '[features]\nfast_mode = true\napps = true\nplugins = true\n'
            'shell_tool = true\nmulti_agent = false\nmemories = false\n'
            'goals = false\nhooks = false\n'
            '[features.context_management]\nexperimental_mode = false\n'
            '[mcp_servers.fixture]\ncommand = "fixture-tool"\nargs = ["--flag", "two words"]\n'
            'env_vars = ["AURIGA_NATIVE_TOOL_FIXTURE", "HOME", "CODEX_HOME", "TMPDIR"]\n'
            '[mcp_servers.fixture.env]\nMULTILINE = "line one\\nline two"\n'
            '[plugins."fixture@local"]\nenabled = true\n'
            '[marketplaces.local]\nsource_type = "local"\nsource = "/synthetic/path"\n'
            '[skills]\nconfig = [{path = "/synthetic/skill", enabled = true}]\n')
        self.store = Store(self.root / "run", dict(run_id="native-preferences", episode_id="episode",
            env_instance_id="environment", env_generation=1))
        self.addCleanup(self.store.close)
        config = Config.from_dict({"features": {"sandbox_enabled": True},
                                   "model": {"name": "gpt-6-astra", "effort": "high",
                                             "codex_home": str(self.source)}})
        self.session = CodexSession(self.store.root, self.store, lambda _: {}, config=config)
        self.addCleanup(self.session.close)
        relay = mock.MagicMock(base_url="http://127.0.0.1:54321")
        relay.start.return_value = relay
        self.proxy = mock.patch("auriga.cortex.session.codex.ModelRequestProxy", return_value=relay)
        self.proxy.start()
        self.addCleanup(self.proxy.stop)

    def generated(self):
        return tomllib.loads((self.session.codex_home / "config.toml").read_text())

    def assert_fixed_preferences(self, config):
        self.assertEqual(config["model_verbosity"], "high")
        self.assertEqual(config["service_tier"], "default")
        self.assertTrue(config["suppress_unstable_features_warning"])
        self.assertFalse(config["features"]["fast_mode"])
        for feature in self.enabled_features:
            if feature != "context_management":
                self.assertIs(config["features"][feature], True, feature)
        for feature in self.disabled_features:
            self.assertIs(config["features"][feature], False, feature)
        self.assertEqual(config["features"]["context_management"], {"experimental_mode": True})

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_fixed_preferences_ignore_local_values_and_preserve_actor_configuration(self):
        original = self.source_config.read_bytes()
        self.session._prepare()
        config = self.generated()
        self.assert_fixed_preferences(config)
        self.assertEqual(config["model"], "gpt-6-astra")
        self.assertEqual(config["model_reasoning_effort"], "high")
        self.assertEqual(config["model_provider"], MODEL_PROXY_PROVIDER)
        for key in ('mcp_servers', 'plugins', 'marketplaces', 'skills'):
            self.assertNotIn(key, config)
        self.assertNotIn("model_context_window", config)
        self.assertEqual(config["web_search"], "disabled")
        self.assertEqual(config["approval_policy"], "never")
        self.assertEqual(config["features"]["code_mode"], {"enabled": False})
        self.assertFalse(config["features"]["code_mode_host"])
        catalog_path = Path(config['model_catalog_json'])
        self.assertEqual(catalog_path.parent, self.session.codex_home)
        catalog = json.loads(catalog_path.read_text())
        original_catalog = json.loads((Path(__file__).resolve().parents[1] /
                                      'auriga/cortex/session/codex_models.json').read_text())
        self.assertEqual(catalog, {**original_catalog, 'models': [
            {**model, 'tool_mode': 'direct'} for model in original_catalog['models']]})
        self.assertIn(config['model'], [model['slug'] for model in catalog['models']])
        self.assertEqual(self.source_config.read_bytes(), original)
        # The actual bundled binary parses the private actor config, without inference.
        result = subprocess.run(self.session.command[:-5] + [str(self.session.cli_executable), "features", "list"],
            env=self.session.env, cwd=self.session.actor_dir,
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for feature in self.enabled_features:
            self.assertEqual(next(line.split()[-1] for line in result.stdout.splitlines()
                                  if line.startswith(feature + " ")), "true")
        for feature in ("fast_mode", "code_mode", "code_mode_host", *self.disabled_features):
            self.assertEqual(next(line.split()[-1] for line in result.stdout.splitlines()
                                  if line.startswith(feature + " ")), "false")

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_host_extensions_credentials_and_environment_are_not_copied(self):
        credentials = self.source / '.credentials.json'
        credentials.write_text('{"fixture-mcp": {"access_token": "synthetic-only"}}')
        for relative in ('skills/example/SKILL.md', 'plugins/cache/example/plugin.json'):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture extension')
        (self.source / 'memories').mkdir()
        (self.source / 'memories' / 'memory_summary.md').write_text('host private memory')
        with mock.patch.dict(os.environ, {'AURIGA_NATIVE_TOOL_FIXTURE': 'fixture value'}):
            self.session._prepare()
        self.assertNotIn('AURIGA_NATIVE_TOOL_FIXTURE', self.session.env)
        self.assertEqual(self.session.env['HOME'], str(self.session.actor_dir))
        self.assertEqual(self.session.env['CODEX_HOME'], str(self.session.codex_home))
        self.assertEqual(self.session.env['TMPDIR'], '/tmp')
        copied_credentials = self.session.codex_home / '.credentials.json'
        self.assertFalse(copied_credentials.exists())
        self.assertIn('synthetic-only', credentials.read_text())
        for relative in ('skills/example/SKILL.md', 'plugins/cache/example/plugin.json'):
            copied = self.session.codex_home / relative
            self.assertFalse(copied.exists())
            self.assertEqual((self.source / relative).read_text(), 'fixture extension')
        self.assertFalse((self.session.codex_home / 'memories' / 'memory_summary.md').exists())
        self.assertEqual(json.loads((self.session.codex_home / 'auth.json').read_text()),
                         {'OPENAI_API_KEY': 'auriga-local-relay'})

    def test_dependency_mismatch_rejects_start_before_stopping_owned_processes(self):
        for field in ('CODEX_SDK_VERSION', 'CODEX_CLI_VERSION'):
            with self.subTest(field=field), mock.patch('auriga.cortex.session.codex.' + field, '0.0.0'), \
                    mock.patch.object(self.session, '_stop_previous_owned_processes') as stop:
                with self.assertRaisesRegex(RuntimeError, 'codex_version_mismatch'):
                    self.session.start([], 'autonomous task', resume_thread_id='existing-thread')
                stop.assert_not_called()

    def test_sandbox_startup_failure_does_not_fall_back_to_host_execution(self):
        with mock.patch('auriga.cortex.session.codex_sandbox.CodexSandbox.prepare', side_effect=RuntimeError('sandbox unavailable')), \
                mock.patch('auriga.cortex.session.codex.subprocess.Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'sandbox unavailable'):
                self.session.start([], 'fixture')
            process.assert_not_called()

    def test_protocol_version_matches_declared_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        dependencies = tomllib.loads((root / 'pyproject.toml').read_text())['project']['dependencies']
        self.assertIn('openai-codex==' + CODEX_PROTOCOL_VERSION, dependencies)
        pins = (root / 'requirements.lock.txt').read_text().splitlines()
        self.assertIn('openai-codex==' + CODEX_PROTOCOL_VERSION, pins)
        self.assertIn('openai-codex-cli-bin==' + CODEX_PROTOCOL_VERSION, pins)
        self.assertIn('codex_models.json', tomllib.loads((root / 'pyproject.toml').read_text())
                      ['tool']['setuptools']['package-data']['auriga.cortex.session'])

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_fixed_preferences_also_apply_when_source_omits_them_or_run_is_resumed(self):
        self.source_config.write_text('model_provider = "fixture"\n'
            '[model_providers.fixture]\nbase_url = "https://fixture.invalid/v1"\n')
        self.session._prepare()
        self.assert_fixed_preferences(self.generated())
        self.session._prepare((self.session.model, "fixture"))
        self.assert_fixed_preferences(self.generated())

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_native_memories_can_be_disabled_in_private_run_without_changing_host(self):
        original = self.source_config.read_bytes()
        self.session.config = self.session.config.with_overrides({'model': {'native_memories': False}})
        self.session._prepare()
        generated = self.generated()
        self.assertFalse(generated['features']['memories'])
        self.assertTrue(generated['features']['multi_agent'])
        self.assertTrue(generated['features']['hooks'])
        self.assertEqual(self.source_config.read_bytes(), original)
        self.session._prepare((self.session.model, "fixture"))
        self.assertFalse(self.generated()['features']['memories'])

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_external_model_and_effort_reach_native_config_and_turn(self):
        for model, effort in [("gpt-6.1-sol", "xhigh"), ("gpt-6-luna", "low")]:
            with self.subTest(model=model, effort=effort):
                session = CodexSession(self.root / model, self.store, lambda _: {},
                    model=model, effort=effort, config=self.session.config)
                self.addCleanup(session.close)
                session._prepare()
                config = tomllib.loads((session.codex_home / "config.toml").read_text())
                self.assertEqual(config["model"], model)
                self.assertEqual(config["model_reasoning_effort"], effort)
                # Native memory side threads must not fall back to a built-in model the provider may not serve.
                self.assertEqual(config["memories"], dict(extract_model=model, consolidation_model=model))
                client = mock.MagicMock()
                session.client = client
                session.thread_id = "fixture-thread"
                session.run_turn("fixture prompt")
                client.turn_start.assert_called_once_with("fixture-thread",
                    [{"type": "text", "text": "fixture prompt"}], {"model": model, "effort": effort})
                session.client = None

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_real_app_server_initializes_thread_with_requested_features(self):
        # Use real runtime/relays, a closed localhost fixture upstream, and no turns.
        # This catches feature startup failures that `features list` cannot detect.
        self.proxy.stop()
        self.source_config.write_text('model_provider = "fixture"\n'
            '[model_providers.fixture]\nbase_url = "http://127.0.0.1:1/v1"\n')
        session = CodexSession(self.root / "startup", self.store, lambda _: {},
            model="gpt-6.1-sol", effort="medium", config=self.session.config)
        def close_runtime():
            bridge = session.client._proc if session.client else None
            session.close()
            if bridge:
                for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                    if stream and not stream.closed:
                        stream.close()
        self.addCleanup(close_runtime)
        self.assertTrue(session.start([], "Offline feature startup validation."))
        self.assertEqual(session.state, "idle")
        self.assertIsNone(session.process.poll())
        response = session.client.request("config/read", {"cwd": str(session.actor_dir)},
                                          response_model=ConfigReadResponse)
        config = response.config.model_dump(mode="json")
        self.assert_fixed_preferences(config)
        self.assertEqual(config["model"], "gpt-6.1-sol")
        self.assertEqual(config["model_reasoning_effort"], "medium")
        response = session.client.request("experimentalFeature/list", {"limit": 200},
                                          response_model=ExperimentalFeatureListResponse)
        available = {feature.name: feature.enabled for feature in response.data}
        for feature in ("hooks", "memories", "multi_agent", "multi_agent_v2"):
            self.assertTrue(available[feature], feature)
        for feature in ('shell_tool', 'unified_exec'):
            self.assertTrue(available[feature], feature)
        self.assertFalse(available['network_proxy'])
        self.assertEqual(session.model_proxy.request_count, 0)
        self.assertEqual(session.model_proxy.attempt_count, 0)
        self.assertFalse((session.private / "isolation.json").exists())
        self.assertEqual(self.store.read_state('runtime', 'codex')['isolation'],
                         'bubblewrap_namespaces_package_egress_v1')


class CodexNativeMemoryProtocolTest(unittest.TestCase):
    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_native_shell_and_memory_consolidation_execute_in_private_fixture(self):
        self._check_native_shell_and_memory(sandbox_enabled=True)

    def test_native_shell_memory_and_terminal_execute_without_sandbox(self):
        with mock.patch('auriga.cortex.session.codex_sandbox.CodexSandbox.prepare', side_effect=AssertionError('sandbox used')):
            self._check_native_shell_and_memory(sandbox_enabled=False)

    def _check_native_shell_and_memory(self, *, sandbox_enabled):
        import shlex
        import time

        requests = []
        memory_output_received = threading.Event()
        actor_marker = 'AURIGA_NATIVE_SHELL_READ_WRITE_OK'
        memory_marker = 'AURIGA_NATIVE_MEMORY_READ_WRITE_OK'
        commands = {}

        class ResponsesHandler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                memory = payload.get('client_metadata', {}).get('x-openai-subagent') == 'memory_consolidation'
                role = 'memory' if memory else 'actor'
                call_id = 'fixture-' + role + '-exec'
                outputs = [item for item in payload['input'] if
                           item.get('type') == 'function_call_output' and item.get('call_id') == call_id]
                if outputs:
                    item = dict(type='message', id='fixture-' + role + '-finish', role='assistant',
                        status='completed', content=[dict(type='output_text', text='fixture complete', annotations=[])])
                else:
                    item = dict(type='function_call', id='fixture-' + role + '-call', call_id=call_id,
                        name='exec_command', namespace='functions', status='completed',
                        arguments=json.dumps(dict(cmd=commands[role], yield_time_ms=1000, max_output_tokens=1000)))
                events = [
                    ('response.created', {'response': {'id': 'fixture-response-' + str(len(requests)), 'status': 'in_progress', 'output': []}}),
                    ('response.output_item.added', {'output_index': 0, 'item': item}),
                    ('response.output_item.done', {'output_index': 0, 'item': item}),
                    ('response.completed', {'response': {'id': 'fixture-response-' + str(len(requests)), 'status': 'completed',
                        'output': [item], 'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}}}),
                ]
                body = ''.join('event: ' + event + '\ndata: ' + json.dumps({'type': event, **value})
                               + '\n\n' for event, value in events).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
                if memory and outputs:
                    memory_output_received.set()

        server = ThreadingHTTPServer(('127.0.0.1', 0), ResponsesHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory(prefix='auriga-native-memory-') as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'config.toml').write_text('model_provider="fixture"\n'
                '[model_providers.fixture]\nname="Loopback fixture"\n'
                'base_url="http://127.0.0.1:' + str(server.server_port) + '/v1"\n'
                'wire_api="responses"\nrequires_openai_auth=false\nsupports_websockets=false\n')
            store = Store(root / 'run', dict(run_id='native-memory', episode_id='episode',
                env_instance_id='synthetic', env_generation=1))
            with mock.patch('auriga.core.config.LOCAL_CONFIG_PATH', root / 'absent.local.json'):
                config = Config.load(environ={}, overrides={'model': {'codex_home': str(source)},
                    'features': {'sandbox_enabled': sandbox_enabled}})
            self.assertTrue(config.model.native_memories)
            session = CodexSession(store.root, store, lambda _: {}, config=config)
            memories = session.codex_home / 'memories'
            actor_file = session.actor_dir / 'synthetic-shell.txt'
            commands['actor'] = 'python3 -c ' + shlex.quote(
                f"from pathlib import Path; p=Path({str(actor_file)!r}); "
                f"p.write_text({actor_marker!r}); print(p.read_text())")
            commands['memory'] = 'python3 -c ' + shlex.quote(
                f"from pathlib import Path; p=Path({str(memories)!r}); "
                "print((p/'raw_memories.md').read_text()); "
                "print((p/'phase2_workspace_diff.md').read_text()); "
                "(p/'MEMORY.md').write_text('# Memory\\n\\nNo reusable memories yet.\\n'); "
                "(p/'memory_summary.md').write_text('v1\\n\\n## General Tips\\n\\nNo reusable memories yet.\\n'); "
                f"print({memory_marker!r})")
            jobs = []
            terminal_status = None
            try:
                session.start([], 'Loopback diagnostic for a synthetic task.')
                self.assertEqual(session.sandbox is not None, sandbox_enabled)
                self.assertEqual(store.read_state('runtime', 'codex')['isolation'], session.isolation)
                deadline = time.monotonic() + 25
                session.run_turn('Use exec_command to write and read the synthetic fixture, then finish.')
                self.assertTrue(session.turn_done.wait(max(0, deadline - time.monotonic())))
                self.assertEqual(session.state, 'completed')
                self.assertTrue(memory_output_received.wait(max(0, deadline - time.monotonic())),
                                'Native phase2 did not return an exec_command result')
                database = session.codex_home / 'memories_1.sqlite'
                while time.monotonic() < deadline:
                    if database.exists():
                        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
                            jobs = db.execute('SELECT kind,status,last_error FROM jobs').fetchall()
                    if any(kind == 'memory_consolidate_global' and status in ('done', 'error')
                           for kind, status, _ in jobs):
                        break
                    time.sleep(.05)
                self.assertIn(('memory_consolidate_global', 'done', None), jobs)
                for role, marker in [('actor', actor_marker), ('memory', memory_marker)]:
                    replies = [item for request in requests for item in request['input'] if
                        item.get('type') == 'function_call_output' and item.get('call_id') == 'fixture-' + role + '-exec']
                    self.assertTrue(replies, role)
                    self.assertIn(marker, json.dumps(replies))
                    self.assertIn('Process exited with code 0', json.dumps(replies))
                self.assertEqual(actor_file.read_text(), actor_marker)
                self.assertTrue((memories / 'MEMORY.md').read_text().strip())
                self.assertTrue((memories / 'memory_summary.md').read_text().startswith('v1\n'))
                memory = [request for request in requests if
                    request.get('client_metadata', {}).get('x-openai-subagent') == 'memory_consolidation']
                actor = [request for request in requests if
                    request.get('client_metadata', {}).get('session_id') == session.thread_id]
                self.assertEqual(len(actor), 2)
                self.assertEqual(len(memory), 2)
                self.assertTrue(all(request['client_metadata']['session_id'] != session.thread_id for request in memory))
                self.assertFalse((source / 'auth.json').exists())
                # The real read-only remote TUI shares this completed native
                # thread through a second namespace without another model turn.
                import asyncio
                from auriga.cortex.session.terminal import Terminal

                async def attach_terminal():
                    terminal = Terminal(session.root, session.actor_dir,
                        command_prefix=session.command_prefix + [str(session.cli_executable),
                            '-c', 'model=' + json.dumps(session.model),
                            '-c', 'model_reasoning_effort=' + json.dumps(session.effort)], env=session.env)
                    count = len(requests)
                    try:
                        await terminal.start('unix://' + str(session.socket_path), session.thread_id)
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            data, _ = terminal.read()
                            if b'fixture complete' in data or terminal.process.poll() is not None:
                                break
                            await asyncio.sleep(.05)
                        status = terminal.status()
                        self.assertFalse(status['bootstrap_failed'], data.decode(errors='replace'))
                        self.assertIsNone(status['exit_code'], data.decode(errors='replace'))
                        self.assertIn(b'fixture complete', data)
                        self.assertEqual(len(requests), count)
                        return status
                    finally:
                        await terminal.close()

                terminal_status = asyncio.run(attach_terminal())
                if not sandbox_enabled:
                    thread_id = session.thread_id
                    bridge = session.client._proc
                    session.close()
                    for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                        if stream and not stream.closed:
                            stream.close()
                    self.assertEqual(session.start([], 'Loopback diagnostic for a synthetic task.',
                                                   resume_thread_id=thread_id), thread_id)
                    self.assertEqual(session.isolation, 'disabled')
            finally:
                if evidence := os.environ.get('AURIGA_CODEX_MEMORY_TEST_EVIDENCE'):
                    path = Path(evidence)
                    path.mkdir(parents=True, exist_ok=True)
                    (path / 'requests.json').write_text(json.dumps(requests, indent=2) + '\n')
                    (path / 'summary.json').write_text(json.dumps(dict(
                        runtime_version=CODEX_PROTOCOL_VERSION, actor_thread_id=session.thread_id,
                        requests=[dict(model=request['model'], reasoning=request.get('reasoning'),
                            role=request.get('client_metadata', {}).get('x-openai-subagent', 'actor'),
                            thread_id=request.get('client_metadata', {}).get('session_id')) for request in requests],
                        jobs=jobs, native_memories=True, terminal_status=terminal_status,
                        external_model_calls=0, host_credentials_used=False,
                        files={name: (memories / name).read_text() for name in ('MEMORY.md', 'memory_summary.md')
                               if (memories / name).exists()}), indent=2) + '\n')
                    if (session.private / 'app-server.log').exists():
                        (path / 'app-server.log').write_text((session.private / 'app-server.log').read_text())
                    if (session.root / 'terminal.bin').exists():
                        (path / 'terminal.bin').write_bytes((session.root / 'terminal.bin').read_bytes())
                bridge = session.client._proc if session.client else None
                session.close()
                if bridge:
                    for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                        if stream and not stream.closed:
                            stream.close()
                store.close()


if __name__ == "__main__":
    unittest.main()
