"""AgentSession base contract, identity record, backend factory/gating and the Codex backend's protocol members."""
import json
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from auriga.core.config import Config, ConfigError, ModelConfig
from auriga.core.storage.store import Store
from auriga.cortex.agents import factory
from auriga.cortex.agents.base import (AgentCapabilities, AgentCapabilityError, AgentRejected, AgentSession,
                                       AgentTransportUnknown, STATE_REPLAY, TURN_TERMINAL_STATES, UNCONNECTED_STATES,
                                       model_effort_choices, read_agent_identity)
from auriga.cortex.session.codex import (CODEX_CLI_VERSION, CODEX_SDK_VERSION, DYNAMIC_TOOL_NAMESPACE,
                                         MODEL_COMPLETION_COUNTING, CodexSession)
from support.fake_agent import FakeAgentSession

IDENTITY = dict(run_id='agent-contract', episode_id='ep', env_instance_id='env', env_generation=1)
ALL_EFFORTS = next(item.metadata['choices'] for item in fields(ModelConfig) if item.name == 'effort')


class StoreCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / 'run', dict(IDENTITY))
        self.addCleanup(self.store.close)


class BaseContractTests(StoreCase):
    def test_capabilities_default_off_frozen_and_serialisable(self):
        capabilities = AgentCapabilities()
        self.assertEqual({key for key, value in capabilities.to_dict().items() if value is True}, set())
        self.assertEqual(capabilities.compaction_signal, 'relay')
        with self.assertRaises(FrozenInstanceError):
            capabilities.steer = True
        self.assertEqual(json.loads(json.dumps(capabilities.to_dict())), capabilities.to_dict())

    def test_base_has_no_init_and_class_defaults_only(self):
        self.assertNotIn('__init__', AgentSession.__dict__)
        session = AgentSession.__new__(AgentSession)
        self.assertEqual(vars(session), {})
        for name in ('on_disconnect', 'transport_recovery_gate', 'tool_parallel_safe', 'active_turn_model', 'counting'):
            self.assertIsNone(getattr(session, name), name)
        self.assertEqual((session.backend_kind, session.effort_choices), ('unknown', ()))
        self.assertEqual(session.capabilities, AgentCapabilities())
        # CodexSession.__new__ objects (used by existing tests) see the same defaults plus Codex's class values.
        codex = CodexSession.__new__(CodexSession)
        self.assertIsNone(codex.on_disconnect)
        self.assertEqual((codex.backend_kind, codex.counting), ('codex', MODEL_COMPLETION_COUNTING))

    def test_default_methods(self):
        session = AgentSession.__new__(AgentSession)
        session.store = self.store
        with self.assertRaisesRegex(AgentCapabilityError, '^steer_unsupported_by_backend$'):
            session.steer('operator text')
        with self.assertRaisesRegex(ValueError, 'native_request_expired_or_not_owned'):
            session.reply('native-1', {})
        self.assertIsNone(session.expire_requests('closing'))
        self.store.set_state('native_request', 'native-1', dict(request_id='native-1', status='pending'))
        self.assertEqual(session.requests(), [dict(request_id='native-1', status='pending')])
        self.assertEqual(session.owned_processes(), [])
        self.assertIsNone(session.terminal_attach_spec())
        from auriga.cortex.agents.evidence import RelayEvidenceClient  # default since P1-8 (interim refusal removed)
        helper = session.evidence_agent(model='m')
        self.assertIsInstance(helper, RelayEvidenceClient)
        self.assertEqual((helper.session, helper.model), (session, 'm'))
        self.assertEqual(session.provenance(), dict(kind='unknown', counting=None,
                                                    capabilities=AgentCapabilities().to_dict()))
        for method in ('start', 'run_turn', 'interrupt', 'close'):
            with self.assertRaises(NotImplementedError):
                getattr(session, method)(*([[], 'instructions'] if method == 'start' else ['p'] if method == 'run_turn' else []))
        session.enter_replay('thread-recorded')
        self.assertEqual((session.thread_id, session.state), ('thread-recorded', STATE_REPLAY))

    def test_effort_and_model_apply_to_the_next_turn_with_validation(self):
        session = AgentSession.__new__(AgentSession)
        self.assertEqual(model_effort_choices(), ALL_EFFORTS)
        session.set_effort('ultra')  # no declared choices: every ModelConfig choice
        self.assertEqual(session.effort, 'ultra')
        session.effort_choices = ('low', 'high')
        with self.assertRaisesRegex(ValueError, 'invalid_effort'):
            session.set_effort('ultra')
        session.set_effort('low')
        self.assertEqual(session.effort, 'low')
        for bad in ('', None, 3):
            with self.assertRaisesRegex(ValueError, 'invalid_model'):
                session.set_model(bad)
        session.set_model('model-b')
        self.assertEqual(session.model, 'model-b')

    def test_completed_final_answer_reads_only_completed_final_rows_of_the_current_turn(self):
        session = AgentSession.__new__(AgentSession)
        session.store, session.thread_id, session.turn_id = self.store, 'thread', 'turn'
        session.turn_done, session.state = threading.Event(), 'completed'
        rows = [dict(type='agentMessage', status='completed', phase='commentary', text='working'),
                dict(type='agentMessage', status='completed', phase='final_answer', text='done'),
                dict(type='agentMessage', status='completed', phase='final_answer', text='done', turn_id='other')]
        for index, row in enumerate(rows):
            self.store.set_transcript_item(f'item-{index}', dict(dict(thread_id='thread', turn_id='turn'), **row))
        self.assertIsNone(session.completed_final_answer())  # turn_done not set
        session.turn_done.set()
        self.assertEqual(session.completed_final_answer(), 'done')
        session.state = 'failed'
        self.assertIsNone(session.completed_final_answer())

    def test_errors(self):
        error = AgentRejected('http_400', 'bad request', busy=True)
        self.assertEqual((error.code, error.message, error.busy, str(error)), ('http_400', 'bad request', True,
                                                                                'http_400: bad request'))
        self.assertFalse(AgentRejected(-32602, 'x').busy)
        self.assertIsInstance(AgentCapabilityError('steer_unsupported_by_backend'), ValueError)
        self.assertNotIsInstance(AgentTransportUnknown('lost'), ValueError)
        self.assertEqual(TURN_TERMINAL_STATES, {'completed', 'interrupted', 'failed'})
        self.assertEqual(UNCONNECTED_STATES, {'disconnected', 'unknown', 'closed'})


class FakeSessionContractTests(StoreCase):
    def session(self, **options):
        session = FakeAgentSession(self.root, self.store, lambda params: {}, **options)
        session.on_disconnect = mock.Mock()
        session.start([], 'instructions')
        return session

    def test_admission_rejection_restores_previous_turn(self):
        session = self.session()
        first = session.run_turn('first')
        session.complete(final_text='answer')
        self.assertEqual(session.completed_final_answer(), 'answer')
        session.script_admission('reject', code=409, message='busy', busy=True)
        with self.assertRaises(AgentRejected) as raised:
            session.run_turn('second')
        self.assertTrue(raised.exception.busy)
        self.assertEqual((session.state, session.turn_id, session.turn_result['status']), ('completed', first, 'completed'))
        self.assertTrue(session.turn_done.is_set())
        session.on_disconnect.assert_not_called()

    def test_unknown_admission_disconnects_and_is_not_retryable(self):
        session = self.session()
        session.script_admission('unknown')
        with self.assertRaises(AgentTransportUnknown):
            session.run_turn('turn')
        self.assertEqual(session.state, 'disconnected')
        self.assertTrue(session.turn_result['error']['message'].startswith('upstream_connection_lost: '))
        self.assertTrue(session.turn_done.is_set())
        session.on_disconnect.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'session_not_connected'):
            session.run_turn('retry')

    def test_running_turn_refuses_second_turn_and_disconnect_fails_it(self):
        session = self.session()
        session.run_turn('turn')
        with self.assertRaisesRegex(ValueError, 'turn_already_active'):
            session.run_turn('again')
        session.disconnect()
        self.assertEqual(session.turn_result['status'], 'failed')
        self.assertEqual(read_agent_identity(self.store)['backend'], 'fake_agent')


class IdentityRecordTests(StoreCase):
    CODEX = dict(thread_id='thread-1', app_server_instance_id='appserver-1', app_server_pid=42,
                 socket_uri='unix:///tmp/x/codex.sock', model='m', provider='p', effort='high', sdk_version='0.160.0',
                 cli_version='0.160.0', cli_executable='/x/codex', isolation='disabled', tools_digest='d')

    def test_absent_identity_is_none(self):
        self.assertIsNone(read_agent_identity(self.store))
        self.store.set_state('runtime', 'codex', {})  # empty native record: absent, as resume always treated it
        self.assertIsNone(read_agent_identity(self.store))

    def test_thread_less_codex_record_still_reports_its_instance(self):
        # The model-context epoch keys on the instance even when the record carries no thread.
        self.store.set_state('runtime', 'codex', dict(app_server_instance_id='reconnected'))
        identity = read_agent_identity(self.store)
        self.assertEqual((identity['backend'], identity['instance_id']), ('codex', 'reconnected'))
        self.assertNotIn('thread_id', identity)

    def test_pre_change_archive_synthesises_codex_identity(self):
        self.store.set_state('runtime', 'codex', dict(self.CODEX))
        identity = read_agent_identity(self.store)
        self.assertEqual(identity, dict(backend='codex', wire_api='openai_responses', source='runtime/codex',
                                        thread_id='thread-1', instance_id='appserver-1', pid=42, model='m',
                                        provider='p', effort='high', isolation='disabled', tools_digest='d',
                                        sdk_version='0.160.0', cli_version='0.160.0', cli_executable='/x/codex',
                                        native=dict(socket_uri='unix:///tmp/x/codex.sock')))
        # Minimal historical records (thread only) still resolve the thread.
        self.store.set_state('runtime', 'codex', dict(thread_id='thread-only'))
        self.assertEqual(read_agent_identity(self.store)['thread_id'], 'thread-only')
        self.assertNotIn('instance_id', read_agent_identity(self.store))

    def test_new_archive_returns_runtime_agent(self):
        self.store.set_state('runtime', 'codex', dict(self.CODEX))
        agent = dict(backend='codex', thread_id='thread-1', instance_id='appserver-1', counting='c')
        self.store.set_state('runtime', 'agent', agent)
        self.assertEqual(read_agent_identity(self.store), agent)
        other = dict(backend='claude_code', thread_id='session-uuid', instance_id='agent-1')
        self.store.set_state('runtime', 'agent', other)
        self.assertEqual(read_agent_identity(self.store), other)

    def test_codex_record_wins_over_a_stale_codex_mirror(self):
        # e.g. a forked Codex session rewrote runtime/codex only.
        self.store.set_state('runtime', 'agent', dict(backend='codex', thread_id='thread-0', instance_id='appserver-0'))
        self.store.set_state('runtime', 'codex', dict(self.CODEX))
        self.assertEqual(read_agent_identity(self.store)['instance_id'], 'appserver-1')
        self.assertEqual(read_agent_identity(self.store)['source'], 'runtime/codex')


class FactoryTests(StoreCase):
    def test_backends(self):
        self.assertEqual(factory.BACKENDS, ('codex', 'claude_code', 'opencode'))

    def test_codex_session_gets_the_gateway_arguments(self):
        config = Config.from_dict({'model': {'codex_home': str(self.root / 'codex-home')}})
        planning, card, handler = object(), {'input_per_million': 1.0}, (lambda params: {})
        session = factory.create_session(config, self.root / 'run', self.store, handler, price_card=card,
                                         task_planning=planning)
        self.addCleanup(session.close)
        self.assertIs(type(session), CodexSession)
        self.assertEqual((session.root, session.store, session.tool_handler, session.price_card, session.task_planning,
                          session.config), ((self.root / 'run').resolve(), self.store, handler, card, planning, config))

    def test_other_backends_are_refused_until_implemented(self):
        # claude_code (P2) and opencode (P3) are implemented (their *_session_unit tests cover the factory branches).
        for kind in ('other',):
            with self.subTest(kind=kind), self.assertRaisesRegex(AgentCapabilityError,
                                                                 '^unknown_agent_backend:' + kind + '$'):
                factory.create_session(SimpleNamespace(agent=SimpleNamespace(backend=kind)), self.root, self.store,
                                       None)

    def test_requirements_are_a_no_op_for_codex(self):
        for document in ({}, {'features': {'sandbox_enabled': True, 'lean_projection': True,
                                           'visual_review_model': 'gpt-6.1-sol', 'self_briefing': True},
                              'model': {'effort': 'ultra'}}):
            self.assertIsNone(factory.backend_requirements(Config.from_dict(document)))
        self.assertIsNone(factory.backend_requirements(SimpleNamespace()))  # config without an agent section

    def test_requirements_for_other_backends(self):
        quiet = {'self_briefing': False, 'visual_review_model': None}
        cases = (
            # features.sandbox_enabled is accepted on every backend (R4-8): the next check decides.
            ('claude_code', {'features': {'sandbox_enabled': True, 'lean_projection': True}},
             'lean_projection_requires_codex_backend'),
            ('opencode', {'features': {**quiet, 'sandbox_enabled': True}}, 'opencode_upstream_incomplete'),
            ('opencode', {'features': {**quiet, 'lean_projection': True}}, 'lean_projection_requires_codex_backend'),
            ('opencode', {'features': {'self_briefing': True}}, 'opencode_upstream_incomplete'),
            ('claude_code', {'features': {'self_briefing': False, 'visual_review_model': 'gpt-6.1-sol'}},
             'evidence_upstream_required_for_visual_review_model'),
            ('claude_code', {'features': quiet, 'model': {'effort': 'ultra'}}, 'agent_effort_unsupported:claude_code:ultra'),
            ('opencode', {'features': quiet, 'model': {'effort': 'ultra'}}, 'opencode_upstream_incomplete'),
            # features.terminal on a backend without a native terminal is not an error.
            ('opencode', {'features': {**quiet, 'terminal': True}}, 'opencode_upstream_incomplete'),
        )
        for kind, document, code in cases:
            with self.subTest(kind=kind, document=document):
                config = Config.from_dict({**document, 'agent': {'backend': kind}})
                with self.assertRaises(ConfigError) as raised:
                    factory.backend_requirements(config)
                message = str(raised.exception)
                self.assertTrue(message.startswith(code + ': '), message)
                self.assertNotIn('\n', message)


class CodexProtocolMembersTests(StoreCase):
    def session(self):
        config = Config.from_dict({'model': {'codex_home': str(self.root / 'codex-home')}})
        session = CodexSession(self.root / 'run', self.store, lambda params: {}, config=config)
        self.addCleanup(session.close)
        return session

    def test_capabilities_counting_and_effort_choices(self):
        self.assertEqual(CodexSession.capabilities, AgentCapabilities(
            steer=True, native_requests=True, terminal_attach=True, native_terminal_replay=True, fork=True,
            deferred_tools=True, account_relay=True, native_evidence_agent=True, sandbox=True, service_tier=True,
            native_memories=True, live_effort=True, compaction_signal='native+relay'))
        self.assertEqual(CodexSession.counting, MODEL_COMPLETION_COUNTING)
        self.assertEqual(CodexSession.effort_choices, ALL_EFFORTS)

    def test_terminal_attach_spec_equals_the_previous_workbench_values(self):
        session = self.session()
        self.assertIsNone(session.terminal_attach_spec())  # native CLI not resolved yet
        session.cli_executable, session.command_prefix = Path('/fixture/codex'), ['fixture-sandbox']
        session.env, session.thread_id, session.model, session.effort = {'HOME': '/actor'}, 'thread-1', 'gpt-x', 'low'
        # Exactly what Workbench.start assembled for Terminal(...) and terminal.start(...) before the protocol.
        expected = dict(cwd=session.actor_dir,
                        command_prefix=session.command_prefix + [str(session.cli_executable),
                            '-c', 'model=' + json.dumps(session.model),
                            '-c', 'model_reasoning_effort=' + json.dumps(session.effort)],
                        env=session.env, attach_uri=f"unix://{session.socket_path}", thread_id=session.thread_id)
        self.assertEqual(session.terminal_attach_spec(), expected)

    def test_owned_processes_keep_the_recorded_roles(self):
        session = self.session()
        self.assertEqual(session.owned_processes(), [])
        bridge, server = SimpleNamespace(pid=1), SimpleNamespace(pid=2)
        session.client = SimpleNamespace(_proc=bridge)
        self.assertEqual(session.owned_processes(), [('codex_bridge', bridge)])
        session.process = server
        self.assertEqual(session.owned_processes(), [('codex_bridge', bridge), ('codex', server)])
        session.client = session.process = None  # fixtures only; close() must not signal them
        self.assertEqual(CodexSession.owned_processes(SimpleNamespace(client=None, process=server)), [('codex', server)])

    def test_provenance_block(self):
        session = self.session()
        session.cli_executable, session.provider = Path('/fixture/codex'), 'openai'
        block = session.provenance()
        self.assertEqual(block, dict(
            kind='codex', sdk=CODEX_SDK_VERSION,
            cli=dict(version=CODEX_CLI_VERSION, executable='/fixture/codex', sha256=None),
            wire_api='openai_responses', upstream_provider='openai', model_configured=session.model,
            effort=session.effort, native_effort=dict(model_reasoning_effort=session.effort),
            isolation='disabled', counting=MODEL_COMPLETION_COUNTING,
            tool_transport=dict(kind='codex_dynamic_tools', tool_name_prefix='', deferred_tools='none',
                                deferred_namespace=DYNAMIC_TOOL_NAMESPACE),
            prompt_channel=dict(instructions_channel='developer_instructions', native_base_prompt='preserved'),
            evidence_client='codex_agent', capabilities=CodexSession.capabilities.to_dict()))
        session.deferred_tools = frozenset({'fixture'})
        self.assertEqual(session.provenance()['tool_transport']['deferred_tools'], 'native_namespace')
        self.assertEqual(json.loads(json.dumps(block)), block)

    def test_effort_and_model_setters_keep_attribute_semantics(self):
        session = self.session()
        session.set_effort('ultra')
        session.set_model('gpt-next')
        self.assertEqual((session.effort, session.model), ('ultra', 'gpt-next'))
        self.assertEqual(session.active_turn_model, session.config.model.name)  # admitted turn keeps its model

    def test_evidence_agent_is_the_native_codex_helper(self):
        session = self.session()
        with mock.patch('auriga.cortex.session.codex_evidence_agent.CodexEvidenceAgent') as agent:
            result = session.evidence_agent(model='m', source='s')
        agent.assert_called_once_with(session, model='m', source='s')
        self.assertIs(result, agent.return_value)

    def test_start_writes_runtime_agent_next_to_unchanged_runtime_codex(self):
        session = self.session()
        session.private.mkdir(parents=True, exist_ok=True)
        session.socket_path = self.root / 'fixture.sock'
        session.socket_path.touch()
        session.sandbox = mock.Mock(host_socket_path='/proc/fixture/fd/pinned')
        session.sandbox.repair_socket_alias.return_value = True
        session.command, session.env, session.cli_executable = ['fixture-only'], {}, Path('/fixture/codex')
        session.model_proxy = mock.Mock()
        client = mock.MagicMock()
        client.initialize.return_value.model_dump.return_value = {'fixture': True}
        client.thread_start.return_value = SimpleNamespace(thread=SimpleNamespace(id='fixture-thread'))
        with mock.patch.object(session, '_prepare', return_value='fixture-provider'), \
                mock.patch.object(session, '_stop_previous_owned_processes'), \
                mock.patch('auriga.cortex.session.codex.subprocess.Popen',
                           return_value=SimpleNamespace(pid=77, poll=lambda: 0)), \
                mock.patch('auriga.cortex.session.codex.RecordedClient', return_value=client):
            session.start([], 'fixture instructions')
        codex = self.store.read_state('runtime', 'codex')
        self.assertEqual(set(codex), {'thread_id', 'app_server_instance_id', 'app_server_pid', 'socket_uri', 'model',
                                      'provider', 'effort', 'sdk_version', 'cli_version', 'cli_executable',
                                      'initialization', 'isolation', 'tools_digest'})
        agent = self.store.read_state('runtime', 'agent')
        self.assertEqual(agent['instance_id'], codex['app_server_instance_id'])
        for key, codex_key in (('thread_id', 'thread_id'), ('pid', 'app_server_pid'), ('model', 'model'),
                               ('provider', 'provider'), ('effort', 'effort'), ('isolation', 'isolation'),
                               ('tools_digest', 'tools_digest'), ('sdk_version', 'sdk_version'),
                               ('cli_version', 'cli_version'), ('cli_executable', 'cli_executable')):
            self.assertEqual(agent[key], codex[codex_key], key)
        self.assertEqual((agent['backend'], agent['wire_api'], agent['counting']),
                         ('codex', 'openai_responses', MODEL_COMPLETION_COUNTING))
        self.assertEqual(read_agent_identity(self.store), agent)
        self.assertEqual(len(agent['instructions_digest']), 64)


if __name__ == '__main__':
    unittest.main()
