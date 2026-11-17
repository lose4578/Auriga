"""Actual catalog/invoke/program boundaries, typed evidence and plugin extension."""
from program_fixture import program_services
from copy import deepcopy
from contextlib import closing
import hashlib
import json
import time
import threading
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from auriga.core.config import Config
from auriga.apps.gateway.workbench import Workbench
from auriga.runtime.harness import Harness
from auriga.runtime.programs.program_review import action_evidence, live_observation_id, reviewed_plan
from auriga.runtime.programs.runner import ProgramRunner
from auriga.core.storage.store import Store
from auriga.core.tool_errors import ToolInputError
from auriga.core.tool_output import StructuredText, ToolOutput, ToolResponse, wire_response
from auriga.runtime.tools.builtins import create_registry, output_schemas
from auriga.runtime.tools.definitions import TOOLS, public_definitions
from auriga.runtime.tools.registry import ToolRegistry, ToolSpec, validator_for
from test_completion_guidance import RETIRED_TOOLS
from test_harness import FixtureRobot


class ToolRegistryTests(unittest.TestCase):
    def test_public_descriptions_do_not_recommend_retired_tools(self):
        legacy_compare = deepcopy(next(item for item in TOOLS if item['name'] == 'vision_compare'))
        definitions = public_definitions()
        for definition in definitions:
            for retired in RETIRED_TOOLS:
                with self.subTest(tool=definition['name'], retired=retired):
                    self.assertIsNone(re.search(r'\b' + re.escape(retired) + r'\b',
                                                definition['description']))
        compare = next(item for item in definitions if item['name'] == 'vision_compare')
        self.assertIn('optional vision_review when available', compare['description'])
        def validation_schema(value):
            if isinstance(value, dict):
                return {key: validation_schema(item) for key, item in value.items()
                        if key not in {'description', 'examples'}}
            if isinstance(value, list):
                return [validation_schema(item) for item in value]
            return value
        self.assertEqual(validation_schema(compare['inputSchema']),
                         validation_schema(legacy_compare['inputSchema']))
        self.assertEqual(next(item for item in TOOLS if item['name'] == 'vision_compare'), legacy_compare)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = Store(self.root/'run', dict(run_id='registry', episode_id='ep',
            env_instance_id='env', env_generation=1))
        self.addCleanup(self.store.close)
        self.robot = FixtureRobot(self.root)
        self.harness = Harness(self.store, self.robot)
        self.harness.thread_id = 'thread'
        self.harness.capabilities = {'fixture': True}
        self.calls = []
        self.plugin = ToolSpec(dict(type='function', name='fixture_read', description='Read one fixture value.',
            inputSchema=dict(type='object', properties={'value': {'type': 'integer'}},
                required=['value'], additionalProperties=False)), self.read,
            output_schemas={'data': dict(type='object', properties={'value': {'type': 'integer'}},
                required=['value'], additionalProperties=False, examples=[{'value': 7}])})
        self.harness.tool_registry.register(self.plugin)

    def read(self, runtime, args, correlation):
        self.calls.append(deepcopy(args))
        return ToolOutput({'value': args['value']}).response()

    def archived_results(self, store=None):
        store = self.store if store is None else store
        return [(record, json.loads(store.resolve_artifact(record['raw_ref'],
                    allowed_scopes=('observer',))[1]))
                for record in store.records('tool_result') if record.get('raw_ref')]

    def test_native_function_definition_is_checked_before_registration(self):
        from dataclasses import replace
        bad_fields = [('type', 'namespace'), ('type', 7), ('name', None), ('name', []),
                      ('description', 5), ('description', {}), ('deferLoading', 'true')]
        for field, value in bad_fields:
            definition = dict(self.plugin.definition, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=definition)])
            self.assertEqual(caught.exception.code, 'invalid_tool_definition')
            self.assertEqual(caught.exception.field_path, '/' + field)
        for field in ('type', 'name', 'description', 'inputSchema'):
            definition = deepcopy(self.plugin.definition)
            del definition[field]
            with self.subTest(missing=field), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=definition)])
            self.assertEqual(caught.exception.field_path, '/' + field)
        for schema in ({}, True, False):
            definition = dict(self.plugin.definition, inputSchema=schema, deferLoading=False,
                              extension_metadata={'source_url': 'https://example.test/author'})
            registry = ToolRegistry([replace(self.plugin, definition=definition)])
            self.assertEqual(registry.catalog(self.harness), [definition])
        from jsonschema.exceptions import SchemaError
        for schema in (None, [], {'type': 'integer', 'minimum': 'bad'}):
            with self.subTest(schema=schema), self.assertRaises(SchemaError):
                ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, inputSchema=schema))])

    def test_native_admission_constraints_reject_unsupported_function_declarations(self):
        from dataclasses import replace
        for name in ('', ' ', 'leading ', ' trailing', 'newline\n', 'dot.name', 'slash/name',
                     'unicode_中文', 'control\x00', 'x'*129, 'mcp', 'mcp__plugin'):
            with self.subTest(name=name), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, name=name))])
            self.assertEqual(caught.exception.field_path, '/name')
        for name in ('x'*128, 'legal-tool_0', '_', 'MCP', 'mcp_plugin'):
            registry = ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, name=name))])
            self.assertEqual(registry.names, frozenset({name}))
        for field, value in [('namespace', None), ('namespace', 'plugin'), ('exposeToContext', True),
                             ('exposeToContext', False)]:
            with self.subTest(field=field, value=value), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, **{field: value}))])
            self.assertEqual(caught.exception.field_path, '/' + field)
        for schema in ({'type': 'null'}, {'type': ['null']}):
            with self.subTest(schema=schema), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, inputSchema=schema))])
            self.assertEqual(caught.exception.code, 'unsupported_tool_input_schema')
            self.assertEqual(caught.exception.field_path, '/inputSchema/type')
        registry = ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition,
            inputSchema={'type': ['object', 'null']}))])
        self.assertEqual(registry.names, frozenset({'fixture_read'}))

    def test_all_public_definitions_are_admitted_by_native_app_server_without_inference(self):
        from auriga.cortex.session.codex import CodexSession
        source = self.root / 'native-source'
        source.mkdir()
        (source / 'config.toml').write_text('model_provider = "fixture"\n'
            '[model_providers.fixture]\nbase_url = "http://127.0.0.1:1/v1"\n')
        config = Config.from_dict({'model': {'codex_home': str(source)},
            'features': {'observation_profile': 'rgbd_proprio_calibrated_v1'}})
        runtime = SimpleNamespace(config=config, disabled_tools=(),
            observation_profile=config.features.observation_profile,
            capabilities={'unified_action': {'available': True}}, enable_wait=True,
            vision_compare=True, visual_reviewer=object(), action_policy=object(), programs=object())
        catalog = create_registry().catalog(runtime)
        self.assertEqual(len(catalog), 20)
        session = CodexSession(self.root / 'native-startup', self.store, lambda _: {},
            model='gpt-6.1-sol', effort='high', config=config)
        def close_runtime():
            bridge = session.client._proc if session.client else None
            session.close()
            if bridge:
                for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                    if stream and not stream.closed:
                        stream.close()
        self.addCleanup(close_runtime)
        self.assertTrue(session.start(catalog, 'Tool catalog admission only; no model turn.'))
        self.assertEqual(session.state, 'idle')
        self.assertIsNone(session.process.poll())
        self.assertEqual(session.model_proxy.request_count, 0)
        self.assertEqual(session.model_proxy.attempt_count, 0)

    def test_deferred_publication_keeps_schema_roles_and_availability_authoritative(self):
        from dataclasses import replace
        definition = deepcopy(self.plugin.definition)
        definition['deferLoading'] = True
        definition['inputSchema']['properties']['value'].update(minimum=0, maximum=10,
            description='Integer fixture value from 0 through 10.')
        registry = ToolRegistry([replace(self.plugin, definition=definition)])
        self.assertEqual(registry.catalog(self.harness), [definition])
        self.assertEqual(registry.invoke('fixture_read', {'value': 7}, {},
            runtime=self.harness, role='actor').data, {'value': 7})
        with self.assertRaises(ToolInputError):
            registry.invoke('fixture_read', {'value': 11}, {}, runtime=self.harness, role='actor')
        self.assertEqual(self.calls, [{'value': 7}])
        restricted = ToolRegistry([replace(self.plugin, definition=definition, roles=frozenset({'program'}))])
        self.assertEqual(restricted.catalog(self.harness), [])
        with self.assertRaises(ToolInputError) as caught:
            restricted.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
        self.assertEqual(caught.exception.code, 'tool_not_permitted_for_role')
        self.harness.disabled_tools = {'fixture_read'}
        self.assertEqual(registry.catalog(self.harness), [])
        with self.assertRaises(ToolInputError) as caught:
            registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
        self.assertEqual(caught.exception.code, 'tool_disabled_by_config:fixture_read')

    def test_plugin_input_examples_are_validated_at_registration(self):
        from dataclasses import replace
        definition = deepcopy(self.plugin.definition)
        definition['inputSchema']['examples'] = [{'value': 7}]
        valid = ToolRegistry([replace(self.plugin, definition=definition)])
        self.assertEqual(valid.catalog(self.harness)[0]['inputSchema']['examples'], [{'value': 7}])
        for example in ({'value': 'not an integer'}, {}, {'value': 7, 'extra': True}):
            definition['inputSchema']['examples'] = [example]
            with self.subTest(example=example), self.assertRaisesRegex(ValueError, 'invalid_tool_input_example'):
                ToolRegistry([replace(self.plugin, definition=definition)])

    def test_registered_contract_is_isolated_from_all_inspection_copies(self):
        from dataclasses import replace
        source = replace(self.plugin, roles={'actor', 'trusted_legacy'}, legacy_schema={'type': 'object'})
        registry = ToolRegistry([source])
        catalog = registry.catalog(self.harness)
        manifest = registry.manifest(self.harness)
        copies = [source, registry.spec('fixture_read', role='actor'),
                  registry.validate('fixture_read', {'value': 7}, role='actor'), registry._specs['fixture_read']]
        for copied in copies:
            copied.definition['name'] = 'changed-name'
            copied.definition['inputSchema']['properties']['value']['type'] = 'string'
            copied.output_schemas['data']['required'].append('missing')
            copied.legacy_schema['type'] = 'string'
        source.roles.add('program')
        registry._specs.clear()
        registry._specs['fixture_read'] = source
        edited_catalog = registry.catalog(self.harness)
        edited_catalog[0]['inputSchema'].clear()
        edited_manifest = registry.manifest(self.harness)
        edited_manifest['tools'][0]['output_schemas']['data'].clear()
        exposed_validator = validator_for(catalog[0]['inputSchema'])
        exposed_validator.schema['properties']['value']['type'] = 'string'
        self.assertEqual(registry.catalog(self.harness), catalog)
        self.assertEqual(registry.manifest(self.harness), manifest)
        registry.validate('fixture_read', {'value': 7}, role='actor')
        registry.validate('fixture_read', {'any_legacy_field': True}, role='trusted_legacy')
        with self.assertRaises(ToolInputError):
            registry.validate('fixture_read', {'value': 'text'}, role='actor')
        with self.assertRaises(ToolInputError):
            registry.spec('fixture_read', role='program')
        result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
        self.assertTrue(result.success)
        self.assertEqual(result.data, {'value': 7})

    def test_native_content_union_rejects_bad_parts_and_keeps_valid_neighbors(self):
        from dataclasses import replace
        valid = [dict(type='inputText', text='opaque {"hash":"business"}'),
                 dict(type='inputImage', imageUrl='data:image/png;base64,fixture'),
                 dict(type='inputAudio', audioUrl='data:audio/wav;base64,fixture')]
        for item, path in [(7, ''), (None, ''), ([], ''), ({}, '/type'),
                ({'type': []}, '/type'), ({'type': 'bogus', 'text': 'unsupported'}, '/type'),
                ({'type': 'inputText'}, '/text'), ({'type': 'inputText', 'text': 7}, '/text'),
                ({'type': 'inputText', 'text': True}, '/text'),
                ({'type': 'inputImage', 'imageUrl': 5}, '/imageUrl'),
                ({'type': 'inputImage', 'image_url': 'wrong spelling'}, '/imageUrl'),
                ({'type': 'inputAudio'}, '/audioUrl'), ({'type': 'inputAudio', 'audioUrl': []}, '/audioUrl')]:
            with self.subTest(item=item):
                original = ToolResponse([valid[0], item, *valid[1:]], data={'value': 7})
                registry = ToolRegistry([replace(self.plugin, handler=lambda *args: original)])
                result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
                self.assertIs(result.success, False)
                self.assertIs(result.data, original.data)
                self.assertEqual(result.contract_error['reason'], 'invalid_native_content_item')
                self.assertEqual(result.contract_error['field_path'], '/content/1' + path)
                shown = wire_response(result)
                self.assertEqual(shown['contentItems'][:-1], valid)
                self.assertEqual(json.loads(shown['contentItems'][-1]['text'])['output_contract_error'],
                                 result.contract_error)
        original = ToolResponse([StructuredText({'hash': 'business'}), *valid], data={'value': 7})
        registry = ToolRegistry([replace(self.plugin, handler=lambda *args: original)])
        result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
        self.assertIs(result, original)
        self.assertTrue(result.success)
        self.assertEqual(wire_response(result)['contentItems'][1:], valid)

    def test_other_output_failures_also_remove_illegal_native_parts(self):
        from dataclasses import replace
        for response, reason in [(ToolResponse([7], data={'value': 7}, success=1), 'result_success_not_boolean'),
                (ToolResponse([7], data={'value': 'bad'}), 'result_schema_mismatch'),
                (ToolResponse([7], data={'value': 7}, kind='unexpected'), 'unexpected_result_kind')]:
            registry = ToolRegistry([replace(self.plugin, handler=lambda *args: response)])
            result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
            self.assertEqual(result.contract_error['reason'], reason)
            shown = wire_response(result)
            self.assertIs(shown['success'], False)
            self.assertEqual(len(shown['contentItems']), 1)
            self.assertEqual(shown['contentItems'][0]['type'], 'inputText')

    def test_illegal_native_display_preserves_real_action_and_program_replay(self):
        from auriga.robot.motion.arx_actions import compile_action
        self.harness.start()
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolResponse([dict(type='inputImage', imageUrl=5)],
                kind='action', data=context.execute(action))
        definition = dict(type='function', name='fixture_illegal_native_display', description='Native output regression.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        args = {'observation_id': self.harness.latest['observation_id']}
        correlation = {'tool_call_id': 'illegal-native-once'}
        first = self.harness.invoke(definition['name'], args, correlation, role='actor')
        repeated = self.harness.invoke(definition['name'], args, correlation, role='actor')
        self.assertFalse(first.success)
        self.assertEqual(first.contract_error['reason'], 'invalid_native_content_item')
        self.assertEqual(first.contract_error['phase'], 'after_handler')
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(self.robot.executions, 1)
        self.assertEqual(first.data['receipt'], self.store.read('action_receipt', first.data['request']['action_id']))
        record, archived = next((record, archived) for record, archived in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'illegal-native-once')
        self.assertEqual(record['archival_status'], 'complete')
        self.assertEqual(archived['wire_response']['contentItems'], [dict(type='inputImage', imageUrl=5)])
        self.assertEqual(archived['data'], first.data)
        self.assertEqual(archived['contract_error']['reason'], 'invalid_native_content_item')
        evidence = action_evidence(self.store, definition['name'], args, first, 'illegal-native-once', physical=True)
        self.assertEqual(evidence['status'], 'completed')
        self.assertFalse(evidence['dispatch_success'])
        shown = self.harness.actor_output.present(first)
        self.assertEqual(json.loads(shown['contentItems'][0]['text'])['receipt']['status'], 'completed')
        self.assertTrue(all(part['type'] == 'inputText' for part in shown['contentItems']))
        runner = ProgramRunner(program_services(self.harness))
        self.harness.programs = runner
        self.addCleanup(runner.close)
        request = dict(operation_id='execute-once', tool=definition['name'],
            arguments={'observation_id': self.harness.latest['observation_id']})
        source = ('from auriga_client import AurigaClient\nc = AurigaClient()\nrequest = ' + repr(request) +
                  '\nfirst = c._request(request)\nassert c._request(request) == first\nprint(first["success"])')
        job = runner.run(dict(source=source, allow_direct_actions=True),
            dict(thread_id='thread', turn_id='turn', tool_call_id='illegal-native-program'))
        self.assertEqual(job['status'], 'succeeded')
        self.assertEqual(job['call_summary']['calls'], 1)
        self.assertEqual(job['call_summary']['failed_calls'], 1)
        self.assertEqual(job['action_summary']['status_counts'], {'completed': 1})
        self.assertEqual(self.robot.executions, 2)

    def test_unknown_fields_reject_before_actor_and_program_handler(self):
        arguments = dict(detail='full', unexpected_argument=True)
        with self.assertRaises(ToolInputError) as error:
            self.harness.dispatch('robot_capabilities', arguments, {})
        self.assertEqual(error.exception.field_path, '/unexpected_argument')
        with self.assertRaises(ToolInputError):
            self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='bad',
                tool='fixture_read', arguments={'value': 1, 'extra': True}))
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        result = runner.run(dict(source="from auriga_client import AurigaClient\n"
            "AurigaClient().call('fixture_read', {'value': 1, 'extra': True})"),
            dict(thread_id='thread', turn_id='turn', tool_call_id='program-bad'))
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['call_summary']['failed_calls'], 1)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.robot.executions, 0)

    def test_new_read_tool_advertises_and_invokes_through_both_real_ports(self):
        self.assertIn('fixture_read', {x['name'] for x in self.harness.tool_registry.catalog(self.harness)})
        response = self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='read',
            tool='fixture_read', arguments={'value': 7}))
        self.assertEqual(json.loads(response['contentItems'][0]['text']), {'value': 7})
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        result = runner.run(dict(source="from auriga_client import AurigaClient\n"
            "print(AurigaClient().call('fixture_read', {'value': 11})['success'])"),
            dict(thread_id='thread', turn_id='turn', tool_call_id='program-read'))
        self.assertEqual(result['status'], 'succeeded', result)
        self.assertEqual(result['call_summary']['failed_calls'], 0)
        self.assertEqual(self.calls, [{'value': 7}, {'value': 11}])
        self.assertEqual(self.robot.executions, 0)
        self.harness.disabled_tools = frozenset({'fixture_read'})
        self.assertNotIn('fixture_read', {x['name'] for x in self.harness.tool_registry.catalog(self.harness)})
        with self.assertRaisesRegex(ValueError, 'tool_disabled_by_config'):
            runner._invoke('fixture_read', {'value': 1}, {})

    def test_physical_plugin_cannot_bypass_program_invocation_authorization(self):
        executions = []
        def perform(runtime, args, correlation):
            executions.append(correlation['program_id'])
            return ToolOutput({'receipt': {'status': 'completed', 'executed_native_steps': 2}},
                kind='action').response()
        definition = dict(type='function', name='fixture_physical', description='Fixture physical port.',
            inputSchema=dict(type='object', properties={}, additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        source = "from auriga_client import AurigaClient\nAurigaClient().call('fixture_physical', {})"
        for authorized in (False, True):
            result = runner.run(dict(source=source, allow_direct_actions=authorized),
                dict(thread_id='thread', turn_id='turn', tool_call_id='physical-'+str(authorized)))
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(result['call_summary']['failed_calls'], int(not authorized))
            if authorized:
                self.assertEqual(result['action_summary']['status_counts'], {'completed': 1})
                self.assertEqual(result['action_summary']['executed_native_steps_known'], 2)
            else:
                self.assertEqual(result['action_summary']['action_calls'], 0)
                self.assertEqual(executions, [])
        self.assertEqual(len(executions), 1)

    def test_plugin_ports_preserve_single_writer_admission_and_replay(self):
        self.harness.start()
        from auriga.robot.motion.arx_actions import compile_action
        contexts = []
        def perform(context, args, correlation):
            contexts.append(context)
            self.assertFalse(hasattr(context, 'robot'))
            self.assertFalse(hasattr(context, 'store'))
            action = compile_action('move', dict(observation_id=args['observation_id'],
                arm='left', position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]),
                context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_motion', description='Fixture writer port.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        args = {'observation_id': self.harness.latest['observation_id']}
        self.harness.accepting = False
        denied = self.harness.invoke('fixture_motion', args, {'tool_call_id': 'denied'}, role='actor')
        self.assertEqual(denied.data['receipt']['status'], 'rejected')
        self.assertEqual(self.robot.executions, 0)
        self.harness.accepting = True
        first = self.harness.invoke('fixture_motion', args, {'tool_call_id': 'once'}, role='actor')
        repeated = self.harness.invoke('fixture_motion', args, {'tool_call_id': 'once'}, role='actor')
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(first.data['receipt']['status'], 'completed')
        self.assertEqual(self.robot.executions, 1)
        self.assertEqual(len(self.store.records('action_request')), 2)
        self.assertEqual(len(self.store.records('action_receipt')), 2)
        contexts[-1].identity['episode_id'] = 'local-copy'
        self.assertEqual(self.harness.identity['episode_id'], 'ep')
        read_contexts = []
        def read(context, arguments, correlation):
            read_contexts.append(context)
            return ToolOutput({}).response()
        readonly = deepcopy(definition)
        readonly['name'] = 'fixture_read_port'
        self.harness.tool_registry.register(ToolSpec(readonly, read, output_schemas={'data': {'type': 'object'}}))
        self.harness.invoke('fixture_read_port', args, {}, role='actor')
        self.assertIsNone(read_contexts[0].execute)
        reserved = deepcopy(readonly)
        reserved['name'] = 'scene_custom'
        with self.assertRaisesRegex(ValueError, 'reserved_tool_namespace'):
            self.harness.tool_registry.register(ToolSpec(reserved, read))

    def test_actor_action_status_preserves_request_and_failure_diagnostics_in_both_details(self):
        request = dict(action_id='history-action', based_on_observation_id='before',
            path=dict(waypoints=[dict(position_m=[.123456789, 0, 1])]))
        receipt = dict(status='stopped', error='motor_fault', executed_native_steps=3,
            executed_control_steps=4, stop_confirmation=dict(status='unknown'),
            execution_uncertain=True, future_diagnostic=dict(fault_code=7))
        self.store.save('action_request', request['action_id'], request)
        self.store.save('action_receipt', request['action_id'], receipt)
        self.harness.states[request['action_id']] = deepcopy(receipt)
        for detail in ('summary', 'full'):
            with self.subTest(detail=detail):
                reply = self.harness.handle_tool(dict(threadId='thread', turnId='turn',
                    callId='history-' + detail, tool='robot_action_status',
                    arguments=dict(action_id=request['action_id'], detail=detail)))
                self.assertTrue(reply['success'])
                result = json.loads(reply['contentItems'][0]['text'])
                self.assertEqual(result['request'], request)
                self.assertEqual(result['receipt'], receipt)
                if detail == 'summary':
                    self.assertEqual(result['full_result'], dict(tool='robot_action_status',
                        action_id=request['action_id'], detail='full'))
        self.assertEqual(self.store.read('action_receipt', request['action_id']), receipt)
        self.assertEqual(self.robot.executions, 0)

    def test_program_plugin_evidence_uses_request_action_id_without_changing_receipt(self):
        from auriga.robot.motion.arx_actions import compile_action
        self.harness.start()
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_request_id', description='Physical ID fixture.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        args = dict(observation_id=self.harness.latest['observation_id'])
        job = runner.run(dict(source='from auriga_client import AurigaClient\n'
            'assert AurigaClient().call("fixture_request_id", ' + repr(args) + ')["success"]',
            allow_direct_actions=True), dict(thread_id='thread', turn_id='turn', tool_call_id='request-id'))
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        request = self.store.records('action_request')[0]
        receipt = self.store.read('action_receipt', request['action_id'])
        self.assertNotIn('action_id', receipt)
        evidence = job['action_summary']['recent_actions'][0]
        self.assertEqual(evidence['action_id'], request['action_id'])
        self.assertEqual(evidence['status'], receipt['status'])
        self.assertEqual(self.robot.executions, 1)

    def test_interaction_history_actor_and_real_program_share_evidence_without_execution(self):
        self.harness.start()
        raw = self.robot.observe()
        raw['robot_state'].update(sim_step=raw['sim_step'], step_domain=raw['step_domain'])
        observation = self.harness.publish_observation(raw)
        request = dict(self.harness.identity, action_id='history-interaction',
            based_on_observation_id=observation['observation_id'],
            gripper_events=[dict(arm='left', goal_normalized=0.)], input_digest='a'*64)
        receipt = dict(status='rejected', reason='fixture-rejection', executed_native_steps=0,
                       execution_start_observation_id=None, result_observation_id=None)
        self.store.save('action_request', request['action_id'], request)
        self.harness.states[request['action_id']] = deepcopy(receipt)
        response = self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='interactions',
            tool='history_list_actions', arguments=dict(anchor='events')))
        self.assertTrue(response['success'])
        self.assertEqual([part['type'] for part in response['contentItems']], ['inputText'])
        text = response['contentItems'][0]['text']
        self.assertNotIn('a'*64, text)
        result = json.loads(text)
        self.assertEqual(result['actions'][0]['receipt'], receipt)
        self.assertEqual(next(iter(result['observations'].values()))['status'], 'available')
        self.assertEqual(result['actions'][0]['omitted_actions_before'], 0)
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        source = ('from auriga_client import AurigaClient\n'
                  'r = AurigaClient().call("history_list_actions", {"anchor": "events"})\n'
                  'assert r["success"] and r["kind"] == "data"\n'
                  'b = r["data"]["actions"][0]\n'
                  'assert b["receipt"]["status"] == "rejected"\n'
                  'assert b["receipt"]["executed_native_steps"] == 0\n'
                  'assert b["observation_ids"]["result"] is None\n'
                  'assert "input_digest" not in b["request"]\n'
                  'assert "result_observation_not_recorded" in b["evidence_gaps"]\n')
        job = runner.run(dict(source=source), dict(thread_id='thread', turn_id='turn', tool_call_id='history-program'))
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        self.assertEqual(self.robot.executions, 0)
        self.assertEqual(self.harness.latest, observation)
        self.assertEqual(self.harness.states[request['action_id']], receipt)

    def test_interaction_history_invalid_anchor_rejects_before_handler(self):
        from unittest.mock import patch
        with patch('auriga.core.presentation.history_output.interaction_blocks') as query:
            for arguments in (dict(anchor='grasped'), dict(anchor=True), dict(anchor='all', limit=0)):
                with self.subTest(arguments=arguments), self.assertRaises(ToolInputError):
                    self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId=str(arguments),
                        tool='history_list_actions', arguments=arguments))
            query.assert_not_called()
        self.assertEqual(self.robot.executions, 0)

    def test_wait_image_failures_preserve_actor_and_program_action_evidence(self):
        self.harness.start()
        self.harness.capabilities['bounded_wait'] = dict(max_native_steps=50)
        self.harness.capabilities['path_kinds'].append('hold')
        original_execute = self.robot.execute
        def execute(action, progress_callback=None, cancel_flag=None):
            progress_callback(dict(executed_steps=10, observation=self.robot.observe()))
            receipt = original_execute(action, progress_callback, cancel_flag)
            receipt.update(executed_native_steps=20, completed_native_steps=20)
            return receipt
        self.robot.execute = execute
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        original_content = self.harness.observation_content
        for role in ('actor', 'program'):
            for fault in ('temporal_sample', 'final_frame'):
                with self.subTest(role=role, fault=fault):
                    action_id = role + '-' + fault
                    args = dict(observation_id=self.harness.latest['observation_id'],
                        native_steps=20, action_id=action_id)
                    attempts = []
                    def content(bundle, correlation, *args, **kwargs):
                        temporal = kwargs.get('metadata_delivery') is False
                        attempts.append('temporal_sample' if temporal else 'final_frame')
                        if temporal == (fault == 'temporal_sample'):
                            raise OSError('fixture_image_read_failed')
                        return original_content(bundle, correlation, *args, **kwargs)
                    before = self.robot.executions
                    with patch.object(self.harness, 'observation_content', side_effect=content):
                        if role == 'actor':
                            params = dict(threadId='thread', turnId='turn', callId=action_id,
                                tool='robot_wait', arguments=args)
                            reply = self.harness.handle_tool(params)
                            self.assertEqual(self.harness.handle_tool(params), reply)
                        else:
                            request = dict(operation_id='wait-once', tool='robot_wait', arguments=args)
                            source = ('from auriga_client import AurigaClient\nc = AurigaClient()\nrequest = ' +
                                repr(request) + '\nfirst = c._request(request)\nassert c._request(request) == first\n')
                            job = runner.run(dict(source=source, allow_direct_actions=True),
                                dict(thread_id='thread', turn_id='turn', tool_call_id=action_id))
                            self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
                            evidence = job['action_summary']['recent_actions'][0]
                            self.assertEqual((evidence['action_id'], evidence['status']), (action_id, 'completed'))
                            self.assertEqual(evidence['completed_native_steps'], 20)
                            reply = json.loads(self.store.resolve_artifact(evidence['reply_ref'])[1])
                    self.assertIn(fault, attempts)
                    self.assertFalse(reply['success'])
                    result = json.loads(reply['contentItems'][0]['text'])
                    self.assertEqual(result['receipt']['status'], 'completed')
                    self.assertEqual(result['receipt']['completed_native_steps'], 20)
                    warning = json.loads(reply['contentItems'][-1]['text'])['output_contract_error']
                    self.assertEqual(warning['reason'], 'action_feedback_unavailable')
                    saved = self.store.read('action_receipt', action_id)
                    self.assertEqual(saved['status'], 'completed')
                    self.assertEqual(len(saved['wait_observation_ids']), 1)
                    self.assertEqual(self.robot.executions, before + 1)

    def test_policy_image_failure_retains_receipt_without_repeating_prediction_or_execution(self):
        self.harness.start()
        predictions, executions = [], []
        def infer(*args, **kwargs):
            predictions.append(kwargs)
            return dict(actions=[[0.] * 14])
        def execute(action, **kwargs):
            executions.append(deepcopy(action))
            return dict(status='completed', executed_native_steps=1, completed_native_steps=1,
                terminal_observation=self.robot.observe())
        self.harness.action_policy = SimpleNamespace(infer=infer)
        self.robot.execute = execute
        params = dict(threadId='thread', turnId='turn', callId='policy-feedback', tool='robot_policy_step',
            arguments=dict(observation_id=self.harness.latest['observation_id'], prompt='Fixture adjustment', prefix=1))
        with patch.object(self.harness, 'observation_content', side_effect=OSError('fixture_image_read_failed')):
            reply = self.harness.handle_tool(params)
            self.assertEqual(self.harness.handle_tool(params), reply)
        self.assertFalse(reply['success'])
        result = json.loads(reply['contentItems'][0]['text'])
        self.assertEqual(result['receipt']['status'], 'completed')
        self.assertEqual(result['receipt']['completed_native_steps'], 1)
        self.assertEqual(result['request']['type'], 'JointPrefixAction')
        self.assertEqual((len(predictions), len(executions)), (1, 1))
        self.assertEqual(self.store.read('action_receipt', executions[0]['action_id'])['status'], 'completed')

    def test_availability_matrix_catalog_and_invoke_share_rejection(self):
        registry = create_registry()
        runtime = SimpleNamespace(config=Config(), disabled_tools=(),
            observation_profile='rgb_proprio_calibrated_v1', motion_review=False,
            capabilities={'unified_action': {'available': True}}, enable_wait=True,
            vision_compare=True, visual_reviewer=object(), action_policy=object(), programs=object())
        self.assertIn('robot_action', {x['name'] for x in registry.catalog(runtime)})
        self.assertEqual(registry.spec('robot_action', role='program').effect_for({'mode': 'inspect'}), 'read')
        self.assertEqual(registry.spec('robot_action', role='program').effect_for({'mode': 'execute'}), 'physical')
        cases = [
            ('robot_action', 'config', Config.from_dict({'features': {'motion_execution_mode': 'feedback', 'motion_review': False}}), {}),
            ('robot_action', 'capabilities', {'unified_action': {'available': False}}, {}),
            ('robot_action', 'observation_profile', 'rgb_proprio_v1', {}),
            ('robot_wait', 'enable_wait', False, {'observation_id': 'o', 'native_steps': 1}),
            ('vision_review', 'visual_reviewer', None, {'observation_id': 'o', 'target': 'cup', 'condition': 'retained_with_gripper'}),
            ('geometry_query', 'observation_profile', 'rgb_proprio_v1', {'request': {'kind': 'help'}}),
            ('depth_query', 'observation_profile', 'rgb_proprio_calibrated_v1', {'request': {'kind': 'help'}}),
        ]
        for tool, field, value, args in cases:
            variant = deepcopy(runtime)
            setattr(variant, field, value)
            with self.subTest(tool=tool, field=field):
                self.assertNotIn(tool, {x['name'] for x in registry.catalog(variant)})
                spec = registry.spec(tool, role='actor')
                self.assertIsNotNone(registry.unavailable(spec, variant, args))
                with self.assertRaises(ValueError):
                    registry.invoke(tool, args, {}, runtime=variant, role='actor')
        for name in ('robot_move_ee', 'scene_query', 'program_run', 'robot_policy_step', 'robot_policy_delegate'):
            with self.assertRaises(ValueError):
                registry.spec(name, role='program')
        with self.assertRaises(ToolInputError):
            registry.validate('robot_wait', {'observation_id': 'o', 'native_steps': True}, role='actor')
        with self.assertRaises(ToolInputError):
            registry.validate('robot_action', {'timeout_s': float('nan')}, role='actor')

    def test_plugin_metadata_is_filtered_for_actor_and_program_but_preserved_in_raw_data(self):
        from auriga.core.contracts import digest
        payload = dict(version="a"*64, revision=3, solver_version="s1", hash="payload-hash",
            source_url="https://example.test/business", nested=dict(hash="b"*64, version=4,
                source_url="https://example.test/nested"))
        schema = dict(type="object", properties=dict(version={"type": "string"}, revision={"type": "integer"},
            solver_version={"type": "string"}, hash={"type": "string"}, source_url={"type": "string"},
            nested={"type": "object"}), required=list(payload), additionalProperties=False)
        definition = dict(type="function", name="fixture_named_business_fields", description="Public business fixture.",
            inputSchema=dict(type="object", properties={}, additionalProperties=False))
        image = dict(type="inputImage", imageUrl="data:image/png;base64,AAAA",
            hash="raw-image-hash", source_url="https://example.test/raw-image")
        self.harness.tool_registry.register(ToolSpec(definition,
            lambda *args: ToolOutput(deepcopy(payload), extra_content=(image,)).response(), output_schemas={"data": schema}))
        shown = self.harness.handle_tool(dict(threadId="thread", turnId="turn", callId="named-business",
            tool=definition["name"], arguments={}))
        self.assertEqual(json.loads(shown["contentItems"][0]["text"]),
            dict(revision=3, solver_version="s1", nested=dict(version=4)))
        self.assertEqual(shown['contentItems'][1], dict(type='inputImage', imageUrl=image['imageUrl']))
        with closing(Store(self.store.root, self.store.identity, read_only=True)) as reopened:
            record, saved = next((record, saved) for record, saved in self.archived_results(reopened)
                if record['correlation'].get('tool_call_id') == 'named-business')
            self.assertEqual(record['archival_status'], 'complete')
            self.assertEqual(saved['data'], payload)
            self.assertEqual(json.loads(saved['wire_response']['contentItems'][0]['text']), payload)
            self.assertEqual(saved['wire_response']['contentItems'][1], image)
            self.assertEqual(reopened.artifact(record['raw_ref'])[0]['access_scope'], 'observer')
            with self.assertRaisesRegex(ValueError, 'artifact_access_scope_denied'):
                reopened.resolve_artifact(record['raw_ref'], allowed_scopes=('actor', 'internal'))
        raw = self.harness.invoke(definition["name"], {}, {}, role="actor")
        self.assertEqual(raw.data, payload)
        self.assertEqual(json.loads(wire_response(raw)["contentItems"][0]["text"]), payload)
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+60, call_lock=threading.RLock(),
            job_id="program-business-fields", calls=0, correlation={},
            condition=threading.Condition(threading.RLock()),
            job=dict(job_id="program-business-fields", status="running"), workspace=self.root,
            model_wait_s=0, review_wait_s=0, action_evidence=[])
        call = dict(operation_id="read", tool=definition["name"], arguments={})
        first = runner.dispatch(entry, call)
        repeated = runner.dispatch(entry, call)
        self.assertEqual(first, repeated)
        self.assertEqual(json.loads(first["contentItems"][0]["text"]),
            dict(revision=3, solver_version="s1", nested=dict(version=4)))
        self.assertEqual(first['contentItems'][1], dict(type='inputImage', imageUrl=image['imageUrl']))
        identifier = entry["job_id"] + "-" + digest("read")[:16]
        self.assertEqual(self.store.read_state("program_call", identifier)["result"], first)
        self.assertIs(type(first), dict)
        self.assertEqual(json.loads(json.dumps(first)), first)
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['role'] == 'program' and record['correlation'].get('tool_call_id') == identifier)
        self.assertEqual(saved['data'], payload)
        self.assertEqual(json.loads(saved['wire_response']['contentItems'][0]['text']), payload)
        self.assertEqual(saved['wire_response']['contentItems'][1], image)

    def test_program_replay_projects_historical_wire_without_mutation_or_redispatch(self):
        from auriga.core.contracts import digest
        runner = ProgramRunner(self.harness.program_services())
        self.addCleanup(runner.close)
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+60,
            call_lock=threading.RLock(), job_id='program-' + 'a'*24, calls=0)
        request = dict(operation_id='historic', tool='fixture_read', arguments={'value': 7})
        action_id = 'action-plugin-' + 'b'*24
        payload = dict(action_id=action_id, status='completed', executed_native_steps=7,
            request_sha256='c'*64, nested=dict(hash='d'*64, position_m=[.123456789, 0, 1]))
        wire = dict(success=True, contentItems=[dict(type='inputText', text=json.dumps(payload)),
            dict(type='inputText', text='Archived proof: ' + 'e'*64),
            dict(type='inputImage', imageUrl='data:image/png;base64,AAAA', sha256='f'*64)])
        identifier = entry['job_id'] + '-' + digest(request['operation_id'])[:16]
        saved = dict(program_id=entry['job_id'], operation_id=request['operation_id'],
            input_digest=digest(request), status='completed', result=deepcopy(wire))
        self.store.set_state('program_call', identifier, saved)
        shown = runner.dispatch(entry, deepcopy(request))
        self.assertEqual(runner.dispatch(entry, deepcopy(request)), shown)
        value = json.loads(shown['contentItems'][0]['text'])
        self.assertEqual(value['status'], 'completed')
        self.assertEqual(value['executed_native_steps'], 7)
        self.assertEqual(value['nested'], dict(position_m=[.123456789, 0, 1]))
        self.assertNotIn('request_sha256', value)
        self.assertNotEqual(value['action_id'], action_id)
        self.assertEqual(self.harness.actor_output.restore(value['action_id']), action_id)
        for original in (action_id, 'c'*64, 'd'*64, 'e'*64, 'f'*64):
            self.assertNotIn(original, json.dumps(shown))
        self.assertEqual(shown['contentItems'][2], dict(type='inputImage', imageUrl='data:image/png;base64,AAAA'))
        self.assertEqual(self.store.read_state('program_call', identifier), saved)
        self.assertEqual(entry['calls'], 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.robot.executions, 0)
        with self.assertRaisesRegex(ValueError, 'program_operation_content_conflict'):
            runner.dispatch(entry, dict(request, arguments={'value': 8}))

    def test_real_program_short_references_round_trip_through_action_status_and_artifact(self):
        from auriga.robot.motion.arx_actions import compile_action
        self.harness.start()
        seen_observations = []
        def perform(context, args, correlation):
            seen_observations.append(args['observation_id'])
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1], timeout_s=2),
                context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_alias_motion', description='Alias round trip fixture.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        source = '''import json, re
from auriga_client import AurigaClient
c = AurigaClient()
def data(reply):
    assert reply['success'], reply
    return json.loads(next(item['text'] for item in reply['contentItems'] if item['type'] == 'inputText'))
observed = data(c.call('robot_observe', {'detail': 'full'}))
observation_id = observed.get('observation_id') or observed['bundle']['observation_id']
assert not re.search(r'[0-9a-f]{16,}', observation_id), observation_id
request = {'operation_id': 'move-once', 'tool': 'fixture_alias_motion',
           'arguments': {'observation_id': observation_id}}
first = c._request(request)
assert c._request(request) == first
result = data(first)
action_id = result['request']['action_id']
assert not re.search(r'[0-9a-f]{16,}', action_id), action_id
status = data(c.call('robot_action_status', {'action_id': action_id, 'detail': 'full'}))
assert status['request']['action_id'] == action_id
assert status['receipt']['status'] == 'completed', status
reference = result['receipt']['actual_trace_ref']
assert reference.startswith('artifact:ref_'), reference
trace = json.loads(c.artifact(reference))
assert trace[0]['measured_fixture_state']['arms']['left']['ee_pose']['position_m'] == [0.009, 0, 1]
print(json.dumps({'observation_id': observation_id, 'action_id': action_id, 'trace_ref': reference}))
'''
        job = runner.run(dict(source=source, allow_direct_actions=True, timeout_s=10),
            dict(thread_id='thread', turn_id='turn', tool_call_id='alias-program'))
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        shown = json.loads(job['stdout_text'])
        raw_action = job['action_summary']['recent_actions'][0]['action_id']
        self.assertTrue(raw_action.startswith('action-plugin-'))
        self.assertEqual(self.harness.actor_output.restore(shown['action_id']), raw_action)
        self.assertEqual(self.harness.actor_output.restore(shown['observation_id']), seen_observations[0])
        self.assertEqual(self.store.read('action_request', raw_action)['based_on_observation_id'], seen_observations[0])
        self.assertEqual(self.store.read('action_receipt', raw_action)['status'], 'completed')
        self.assertEqual(job['action_summary']['status_counts'], {'completed': 1})
        self.assertEqual(job['call_summary']['calls'], 3)
        self.assertEqual(job['call_summary']['failed_calls'], 0)
        self.assertEqual(self.robot.executions, 1)
        self.assertEqual(len(seen_observations), 1)

    def test_actor_program_source_reuses_reply_aliases_without_rewriting_checksums_or_numbers(self):
        from auriga.core.contracts import digest
        self.harness.start()
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        observation_id = self.harness.actor_output.present(self.harness.latest['observation_id'])
        reference = self.store.put_artifact(b'numeric fixture\n', 'fixture_text', 'text/plain')
        alias = self.harness.actor_output.present(reference)
        source = ("import json\nfrom auriga_client import AurigaClient\nc = AurigaClient()\n"
            "expected = " + repr(observation_id) + "\n"
            "checksum = " + repr('a'*64) + "\n"
            "unrelated = " + repr('checksum-' + 'b'*64) + "\n"
            "precision = '0.12345678901234567890123456789'\n"
            "reply = c.call('history_view_observation', {'observation_id': expected, 'camera_id': 'fixture_camera'})\n"
            "assert reply['success'], reply\n"
            "value = json.loads(reply['contentItems'][0]['text'])\n"
            "assert value['bundle']['observation_id'] == expected, (expected, value)\n"
            "assert c.artifact(" + repr(alias) + ") == b'numeric fixture\\n'\n"
            "assert len(checksum) == 64 and checksum.count('a') == 64\n"
            "assert unrelated.startswith('checksum-') and unrelated.count('b') == 64\n"
            "assert precision == '0.12345678901234567890123456789'\n"
            "print(json.dumps({'observation_id': expected, 'precision': precision}))\n")
        arguments = dict(source=source, timeout_s=10, detail='full')
        params = dict(threadId='thread', turnId='turn', callId='source-alias-roundtrip',
            tool='program_run', arguments=arguments)
        reply = self.harness.handle_tool(params)
        self.assertTrue(reply['success'])
        shown = json.loads(reply['contentItems'][0]['text'])
        self.assertEqual(shown['status'], 'succeeded', shown.get('stderr_text'))
        job = self.store.states('program')[0]
        actual_source = self.store.resolve_artifact(job['source_ref'])[1]
        self.assertEqual(actual_source.decode(), source)
        self.assertEqual(job['source_sha256'], hashlib.sha256(actual_source).hexdigest())
        original_arguments = self.harness.actor_output.restore(dict(source=source, timeout_s=10))
        self.assertEqual(job['input_digest'], digest(original_arguments))
        self.assertEqual(json.loads(self.store.resolve_artifact(job['stdout_ref'])[1]),
            dict(observation_id=observation_id, precision='0.12345678901234567890123456789'))
        repeated = self.harness.handle_tool(params)
        self.assertEqual(json.loads(repeated['contentItems'][0]['text'])['status'], 'succeeded')
        self.assertEqual(len(self.store.states('program')), 1)
        self.assertEqual(job['calls'], 1)
        self.assertEqual(self.robot.executions, 0)

    def test_raw_reply_is_persisted_before_failed_actor_presentation(self):
        payload = dict(value=7, hash='raw-hash', source_url='https://example.test/raw')
        definition = dict(self.plugin.definition, name='fixture_before_display')
        self.harness.tool_registry.register(ToolSpec(definition,
            lambda *args: ToolOutput(payload).response(), output_schemas={'data': {'type': 'object'}}))
        with patch.object(self.harness.actor_output, 'present', side_effect=ValueError('fixture_display_failed')):
            with self.assertRaisesRegex(ValueError, 'fixture_display_failed'):
                self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='before-display',
                    tool=definition['name'], arguments={'value': 7}))
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'before-display')
        self.assertEqual(record['archival_status'], 'complete')
        self.assertEqual(saved['data'], payload)
        self.assertEqual(json.loads(saved['wire_response']['contentItems'][0]['text']), payload)

    def test_non_json_raw_data_records_archival_gap_and_keeps_serializable_receipt(self):
        data = dict(receipt=dict(status='completed', executed_native_steps=4),
            hash='raw-proof', invalid=object())
        definition = dict(self.plugin.definition, name='fixture_non_json_raw_data')
        self.harness.tool_registry.register(ToolSpec(definition,
            lambda *args: ToolResponse([StructuredText(dict(value=7))], data=data),
            output_schemas={'data': {'type': 'object'}}))
        shown = self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='non-json-archive',
            tool=definition['name'], arguments={'value': 7}))
        self.assertFalse(shown['success'])
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'non-json-archive')
        self.assertEqual(record['archival_status'], 'partial')
        self.assertEqual(saved['data'], dict(receipt=data['receipt'], hash='raw-proof'))
        self.assertIn('/data/invalid', {error['field_path'] for error in saved['archival_errors']})
        self.assertEqual(saved['contract_error']['reason'], 'result_data_not_json')
        self.assertEqual(saved['wire_response'],
            dict(success=True, contentItems=[dict(type='inputText', text='{"value":7}')]))
        self.assertNotIn('object at', json.dumps(saved))

    def test_archive_write_failure_preserves_executed_receipt_and_action_is_not_replayed(self):
        from auriga.robot.motion.arx_actions import compile_action
        self.harness.start()
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_archive_failure_motion', description='Archive failure fixture.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        params = dict(threadId='thread', turnId='turn', callId='archive-motion', tool=definition['name'],
            arguments=dict(observation_id=self.harness.latest['observation_id']))
        original = self.store.put_artifact
        def fail_archive(data, kind, mime_type, access_scope='actor'):
            if kind == 'tool_result':
                raise OSError('fixture_archive_write_failed')
            return original(data, kind, mime_type, access_scope)
        with patch.object(self.store, 'put_artifact', side_effect=fail_archive):
            shown = self.harness.handle_tool(params)
        self.assertFalse(shown['success'])
        self.assertEqual(json.loads(shown['contentItems'][0]['text'])['receipt']['status'], 'completed')
        warning = json.loads(shown['contentItems'][-1]['text'])['output_contract_error']
        self.assertEqual(warning['reason'], 'raw_result_archival_failed')
        self.assertEqual(warning['phase'], 'after_handler')
        self.assertIn('do not automatically replay', warning['suggestion'])
        failed = next(record for record in self.store.records('tool_result')
            if record['correlation'].get('tool_call_id') == 'archive-motion')
        self.assertEqual(failed['archival_status'], 'failed')
        repeated = self.harness.handle_tool(params)
        self.assertTrue(repeated['success'])
        self.assertEqual(json.loads(repeated['contentItems'][0]['text'])['receipt']['status'], 'completed')
        self.assertEqual(self.robot.executions, 1)
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'archive-motion')
        self.assertEqual(saved['data']['receipt']['status'], 'completed')
        self.assertEqual(saved['data']['receipt'],
            self.store.read('action_receipt', saved['data']['request']['action_id']))

    def test_robot_information_source_is_archived_before_its_internal_projection(self):
        raw = dict(robot_id='fixture', available=True, section='all', hash='source-proof',
            source_url='https://example.test/robot', measured_q_rad=[1.23456789012345])
        with patch('auriga.robot.model.robot_information.robot_information', return_value=deepcopy(raw)):
            shown = self.harness.robot_information('all', 'fixture')
        self.assertEqual(shown, dict(robot_id='fixture', available=True, section='all',
            measured_q_rad=[1.23456789012345]))
        with closing(Store(self.store.root, self.store.identity, read_only=True)) as reopened:
            record, saved = next((record, saved) for record, saved in self.archived_results(reopened)
                if record['tool'] == 'robot_information' and record['role'] == 'internal')
            self.assertEqual(record['archival_status'], 'complete')
            self.assertEqual(saved['data'], raw)
            self.assertEqual(json.loads(saved['wire_response']['contentItems'][0]['text']), raw)

    def test_archive_deepcopy_recursion_failure_does_not_escape_or_replay_executed_action(self):
        from auriga.robot.motion.arx_actions import compile_action
        self.harness.start()
        nested = {}
        for _ in range(500):
            nested = dict(child=nested)
        part = dict(type='inputText', text='deep fixture log', extension=nested)
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolResponse([part], kind='action', data=context.execute(action))
        definition = dict(type='function', name='fixture_archive_recursion_motion', description='Deep archive fixture.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        args = dict(observation_id=self.harness.latest['observation_id'])
        correlation = dict(tool_call_id='archive-recursion')
        first = self.harness.invoke(definition['name'], args, correlation, role='actor')
        self.assertEqual(first.data['receipt']['status'], 'completed')
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'archive-recursion')
        self.assertEqual(record['archival_status'], 'partial')
        self.assertEqual(saved['data'], first.data)
        self.assertIn(dict(field_path='/wire_response', reason='not_json_serializable',
            error_type='RecursionError'), saved['archival_errors'])
        self.assertEqual(first.data['receipt'],
            self.store.read('action_receipt', first.data['request']['action_id']))
        evidence = action_evidence(self.store, definition['name'], args, first,
            'archive-recursion', physical=True)
        self.assertEqual(evidence['status'], 'completed')
        self.assertTrue(evidence['dispatch_success'])
        reply = json.loads(self.store.resolve_artifact(evidence['reply_ref'])[1])
        self.assertNotIn('reply_archive_error', evidence)
        self.assertEqual(reply['contentItems'][0]['text'], 'deep fixture log')
        extension = reply['contentItems'][0]['extension']
        for _ in range(500):
            self.assertEqual(list(extension), ['child'])
            extension = extension['child']
        self.assertEqual(extension, {})
        # The projected writer can handle this tree; fail its current archive boundary explicitly.
        with patch.object(self.store, 'put_artifact', side_effect=RecursionError('fixture_reply_archive')) as archive:
            failed_evidence = action_evidence(self.store, definition['name'], args, first,
                'archive-recursion', physical=True)
        archive.assert_called_once()
        self.assertEqual(archive.call_args.args[1], 'program_action_reply')
        self.assertEqual(failed_evidence['status'], 'completed')
        self.assertTrue(failed_evidence['dispatch_success'])
        self.assertEqual(failed_evidence['reply_archive_error']['error_type'], 'RecursionError')
        self.assertNotIn('reply_ref', failed_evidence)
        self.assertEqual(self.robot.executions, 1)
        repeated = self.harness.invoke(definition['name'], args, correlation, role='actor')
        self.assertEqual(repeated.data, first.data)
        self.assertEqual(self.robot.executions, 1)

    def test_non_json_display_does_not_lose_trusted_physical_receipt(self):
        from dataclasses import replace
        business = dict(receipt=dict(status="completed", executed_native_steps=4))
        bad = ToolResponse([StructuredText(dict(value=object()))], data=business, kind="action")
        spec = replace(self.plugin, handler=lambda *args: bad,
            output_schemas=output_schemas("robot_wait"))
        result = ToolRegistry([spec]).invoke("fixture_read", {"value": 7}, {}, runtime=self.harness, role="actor")
        self.assertFalse(result.success)
        self.assertIs(result.data, business)
        self.assertEqual(result.contract_error["reason"], "display_not_json")
        evidence = action_evidence(self.store, "fixture_read", {}, result, "physical", physical=True)
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["executed_native_steps"], 4)
        shown = self.harness.actor_output.present(result)
        self.assertFalse(shown["success"])
        self.assertEqual(json.loads(shown["contentItems"][0]["text"]), business)
        self.assertIn("do not automatically replay", json.loads(shown["contentItems"][-1]["text"])
            ["output_contract_error"]["suggestion"])

    def test_api_version_is_explicit_and_independent_of_input_schema_hash(self):
        from dataclasses import replace
        first = ToolRegistry([self.plugin])
        next_version = ToolRegistry([replace(self.plugin, api_version='2.1')])
        a = first.manifest(self.harness)['tools'][0]
        b = next_version.manifest(self.harness)['tools'][0]
        self.assertEqual(a['api_version'], '1.0')
        self.assertEqual(b['api_version'], '2.1')
        self.assertEqual(a['schema_hash'], b['schema_hash'])
        for invalid in ('', 'latest', '0.1', '1', None):
            with self.subTest(version=invalid), self.assertRaisesRegex(ValueError, 'invalid_tool_api_version'):
                ToolRegistry([replace(self.plugin, api_version=invalid)])

    def test_same_tool_name_with_changed_schema_never_reuses_old_validation(self):
        old = ToolRegistry([self.plugin])
        changed = deepcopy(self.plugin.definition)
        changed['inputSchema']['properties']['value']['minimum'] = 10
        new = ToolRegistry([ToolSpec(changed, self.read, output_schemas=self.plugin.output_schemas)])
        old.validate('fixture_read', {'value': 7}, role='actor')
        with self.assertRaises(ToolInputError) as error:
            new.validate('fixture_read', {'value': 7}, role='actor')
        self.assertEqual(error.exception.field_path, '/value')
        new.validate('fixture_read', {'value': 10}, role='actor')
        with self.assertRaises(ToolInputError):
            self.harness.tool_registry.validate('robot_action', {'timeout_s': 10**400}, role='actor')

    def test_broker_uses_typed_values_even_if_rendered_text_is_forged(self):
        plan = ToolResponse([{'type': 'inputText', 'text': 'not JSON'}],
            kind='plan', data={'status': 'planned', 'plan_id': 'actual-plan'})
        self.assertEqual(reviewed_plan(plan)['plan_id'], 'actual-plan')
        fake_wire = ToolOutput({'status': 'planned', 'plan_id': 'forged-plan'}).response()
        with self.assertRaisesRegex(ValueError, 'missing_plan'):
            reviewed_plan(fake_wire)
        action = ToolResponse([{'type': 'inputText', 'text': '{"receipt":{"status":"completed"}}'}],
            kind='action', data={'receipt': {'status': 'stopped', 'executed_native_steps': 3}})
        evidence = action_evidence(self.store, 'robot_action', {'mode': 'execute'}, action, 'call')
        self.assertEqual(evidence['status'], 'stopped')
        self.assertEqual(evidence['executed_native_steps'], 3)
        unknown = action_evidence(self.store, 'robot_action', {'mode': 'execute'}, wire_response(action), 'wire-only')
        self.assertEqual(unknown['status'], 'unknown')
        observed = ToolResponse([], data={'observation_id': 'actual-observation'}, kind='observation')
        self.assertEqual(live_observation_id('robot_observe', observed), 'actual-observation')
        self.assertIsNone(live_observation_id('robot_observe', wire_response(observed)))
        self.assertEqual(set(json.loads(json.dumps(wire_response(action)))), {'success', 'contentItems'})

    def test_output_declarations_examples_and_manifest_share_one_contract(self):
        from dataclasses import replace
        from jsonschema.exceptions import SchemaError
        with self.assertRaisesRegex(ValueError, 'tool_output_contract_required'):
            ToolRegistry([replace(self.plugin, output_schemas=None)])
        with self.assertRaises(SchemaError):
            ToolRegistry([replace(self.plugin, output_schemas={'data': {'type': 'not-a-type'}})])
        wrong_example = deepcopy(self.plugin.output_schemas)
        wrong_example['data']['examples'] = [{'value': 'seven'}]
        with self.assertRaisesRegex(ValueError, 'invalid_tool_output_example'):
            ToolRegistry([replace(self.plugin, output_schemas=wrong_example)])
        original = ToolRegistry([self.plugin]).manifest(self.harness)['tools'][0]
        changed = deepcopy(self.plugin.output_schemas)
        changed['data']['properties']['value']['minimum'] = 2
        updated = ToolRegistry([replace(self.plugin, output_schemas=changed)]).manifest(self.harness)['tools'][0]
        self.assertEqual(original['result_kinds'], ['data'])
        self.assertEqual(original['output_schemas'], self.plugin.output_schemas)
        self.assertEqual(original['schema_hash'], updated['schema_hash'])
        self.assertNotEqual(original['output_schema_hash'], updated['output_schema_hash'])
        # A malformed new plugin cannot use a wire-shaped reply to opt out.
        legacy = replace(self.plugin, roles=frozenset({'trusted_legacy'}), output_schemas=None)
        legacy_registry = ToolRegistry([legacy])
        self.assertEqual(legacy_registry.manifest(self.harness, role='trusted_legacy')['tools'][0]
            ['output_contract'], 'trusted_legacy_wire_unvalidated')
        with self.assertRaises(ToolInputError):
            legacy_registry.spec('fixture_read', role='actor')

    def test_invalid_read_output_is_reported_after_handler_at_actor_and_program_ports(self):
        from dataclasses import replace
        calls = []
        def malformed(context, arguments, correlation):
            calls.append(arguments)
            return ToolOutput({'value': 'not an integer'}).response()
        spec = replace(self.plugin, definition=dict(self.plugin.definition, name='fixture_bad_output'),
            handler=malformed)
        self.harness.tool_registry.register(spec)
        result = self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='bad-output',
            tool=spec.name, arguments={'value': 7}))
        self.assertFalse(result['success'])
        diagnostic = json.loads(result['contentItems'][-1]['text'])['output_contract_error']
        self.assertEqual(diagnostic['field_path'], '/data/value')
        self.assertEqual(diagnostic['phase'], 'after_handler')
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        job = runner.run(dict(source="from auriga_client import AurigaClient\n"
            "print(AurigaClient().call('fixture_bad_output', {'value': 11})['success'])"),
            dict(thread_id='thread', turn_id='turn', tool_call_id='bad-output-program'))
        self.assertEqual(job['status'], 'succeeded')
        self.assertEqual(job['call_summary']['failed_calls'], 1)
        self.assertEqual(calls, [{'value': 7}, {'value': 11}])
        self.assertEqual(self.robot.executions, 0)

    def test_output_failure_preserves_executed_receipt_and_does_not_replay(self):
        self.harness.start()
        from auriga.robot.motion.arx_actions import compile_action
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_bad_motion_output', description='Post-action regression.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        schema = output_schemas('robot_wait')
        schema['action']['required'].append('missing_output_field')
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical', output_schemas=schema))
        args = {'observation_id': self.harness.latest['observation_id']}
        first = self.harness.invoke(definition['name'], args, {'tool_call_id': 'execute-once'}, role='actor')
        repeated = self.harness.invoke(definition['name'], args, {'tool_call_id': 'execute-once'}, role='actor')
        self.assertFalse(first.success)
        self.assertEqual(first.contract_error['phase'], 'after_handler')
        self.assertEqual(first.kind, 'action')
        self.assertEqual(first.data['receipt']['status'], 'completed')
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(self.robot.executions, 1)
        saved = self.store.read('action_receipt', first.data['request']['action_id'])
        self.assertEqual(first.data['receipt'], saved)
        evidence = action_evidence(self.store, definition['name'], args, first, 'execute-once', physical=True)
        self.assertEqual(evidence['status'], 'completed')
        self.assertFalse(evidence['dispatch_success'])
        self.assertEqual(evidence.get('executed_native_steps'), saved.get('executed_native_steps'))

    def test_executed_action_with_non_json_display_remains_completed_and_replays_once(self):
        from auriga.robot.motion.arx_actions import compile_action
        self.harness.start()
        def perform(context, args, correlation):
            action = compile_action("move", dict(observation_id=args["observation_id"], arm="left",
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolResponse([StructuredText(dict(invalid_display=object()))],
                kind="action", data=context.execute(action))
        definition = dict(type="function", name="fixture_non_json_motion_display", description="Display failure after execution.",
            inputSchema=dict(type="object", properties={"observation_id": {"type": "string"}},
                required=["observation_id"], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect="physical",
            output_schemas=output_schemas("robot_wait")))
        args = {"observation_id": self.harness.latest["observation_id"]}
        first = self.harness.invoke(definition["name"], args, {"tool_call_id": "non-json-once"}, role="actor")
        repeated = self.harness.invoke(definition["name"], args, {"tool_call_id": "non-json-once"}, role="actor")
        self.assertFalse(first.success)
        self.assertEqual(first.contract_error["reason"], "display_not_json")
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(first.data["receipt"]["status"], "completed")
        self.assertEqual(self.robot.executions, 1)
        self.assertEqual(first.data["receipt"], self.store.read("action_receipt", first.data["request"]["action_id"]))
        shown = self.harness.actor_output.present(first)
        self.assertEqual(json.loads(shown["contentItems"][0]["text"])["receipt"]["status"], "completed")
        evidence = action_evidence(self.store, definition["name"], args, first, "non-json-once", physical=True)
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence.get("executed_native_steps"), first.data["receipt"].get("executed_native_steps"))
        runner = ProgramRunner(program_services(self.harness))
        self.harness.programs = runner
        self.addCleanup(runner.close)
        source = ("from auriga_client import AurigaClient\nprint(AurigaClient().call(" +
                  repr(definition["name"]) + ", " + repr({"observation_id": self.harness.latest["observation_id"]}) + ")['success'])")
        job = runner.run(dict(source=source, allow_direct_actions=True),
            dict(thread_id="thread", turn_id="turn", tool_call_id="non-json-program"))
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["action_summary"]["status_counts"], {"completed": 1})
        # FixtureRobot does not report native step counts; retain that unknown.
        self.assertEqual(job["action_summary"]["actions_without_step_count"], 1)
        self.assertEqual(job["call_summary"]["failed_calls"], 1)
        self.assertEqual(self.robot.executions, 2)

    def test_typed_success_requires_boolean_without_coercing_truthy_values(self):
        from dataclasses import replace
        for success in (object(), 1, 0, "true", "false", None):
            with self.subTest(success_type=type(success).__name__, success_value=success if isinstance(success, (str, int)) else None):
                original = ToolResponse([StructuredText({"value": 7})], data={"value": 7}, success=success)
                spec = replace(self.plugin, handler=lambda *args: original)
                result = ToolRegistry([spec]).invoke("fixture_read", {"value": 7}, {}, runtime=self.harness, role="actor")
                self.assertIs(result.success, False)
                self.assertIs(result.data, original.data)
                self.assertEqual(result.kind, "data")
                self.assertEqual(result.contract_error["reason"], "result_success_not_boolean")
                self.assertEqual(result.contract_error["field_path"], "/success")
                self.assertEqual(result.contract_error["phase"], "after_handler")
                self.assertIs(json.loads(json.dumps(wire_response(result)))["success"], False)
        for success in (True, False):
            with self.subTest(valid_success=success):
                original = ToolResponse([StructuredText({"value": 7})], data={"value": 7}, success=success)
                spec = replace(self.plugin, handler=lambda *args: original)
                result = ToolRegistry([spec]).invoke("fixture_read", {"value": 7}, {}, runtime=self.harness, role="actor")
                self.assertIs(result, original)
                self.assertIsNone(result.contract_error)

    def test_executed_action_with_invalid_success_keeps_receipt_and_program_replay(self):
        from auriga.robot.motion.arx_actions import compile_action
        from auriga.core.contracts import digest
        self.harness.start()
        runner = ProgramRunner(program_services(self.harness))
        self.harness.programs = runner
        self.addCleanup(runner.close)
        for index, invalid_success in enumerate((object(), 1, "true")):
            with self.subTest(success_type=type(invalid_success).__name__):
                def perform(context, args, correlation):
                    action = compile_action("move", dict(observation_id=args["observation_id"], arm="left",
                        position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
                    response = ToolOutput(context.execute(action), kind="action", projection="action_result").response()
                    response.success = invalid_success
                    return response
                definition = dict(type="function", name="fixture_invalid_success_" + str(index),
                    description="Invalid success after fixture execution.",
                    inputSchema=dict(type="object", properties={"observation_id": {"type": "string"}},
                        required=["observation_id"], additionalProperties=False))
                self.harness.tool_registry.register(ToolSpec(definition, perform, effect="physical",
                    output_schemas=output_schemas("robot_wait")))
                request = dict(operation_id="execute-once", tool=definition["name"],
                    arguments={"observation_id": self.harness.latest["observation_id"]})
                source = ("from auriga_client import AurigaClient\nc = AurigaClient()\nrequest = " +
                    repr(request) + "\nfirst = c._request(request)\nassert c._request(request) == first\nprint(first['success'])")
                job = runner.run(dict(source=source, allow_direct_actions=True),
                    dict(thread_id="thread", turn_id="turn", tool_call_id="invalid-success-program-" + str(index)))
                self.assertEqual(job["status"], "succeeded")
                self.assertEqual(job["action_summary"]["status_counts"], {"completed": 1})
                self.assertEqual(job["action_summary"]["actions_without_step_count"], 1)
                self.assertEqual(job["call_summary"]["calls"], 1)
                self.assertEqual(job["call_summary"]["failed_calls"], 1)
                self.assertEqual(self.robot.executions, 2 * index + 1)
                call_id = job["job_id"] + "-" + digest("execute-once")[:16]
                record = self.store.read_state("program_call", call_id)
                self.assertEqual(record["output_contract_error"]["reason"], "result_success_not_boolean")
                self.assertEqual(record["output_contract_error"]["field_path"], "/success")
                self.assertIs(record["result"]["success"], False)
                evidence = job["action_summary"]["recent_actions"][0]
                self.assertIs(evidence["dispatch_success"], False)
                reply = json.loads(self.store.resolve_artifact(evidence["reply_ref"])[1])
                self.assertIs(reply["success"], False)
                recorded_action = json.loads(reply["contentItems"][0]["text"])
                original_action_id = self.harness.actor_output.restore(recorded_action['request']['action_id'])
                stored_receipt = self.store.read('action_receipt', original_action_id)
                self.assertEqual(recorded_action["receipt"],
                    self.harness.actor_output.present(dict(receipt=stored_receipt),
                        projection='action_result')['receipt'])
                archived = next(saved for item, saved in self.archived_results()
                    if item['correlation'].get('tool_call_id') == call_id)
                self.assertEqual(archived['data']['receipt'], stored_receipt)
                args = {"observation_id": self.harness.latest["observation_id"]}
                correlation = {"tool_call_id": "invalid-success-actor-" + str(index)}
                first = self.harness.invoke(definition["name"], args, correlation, role="actor")
                repeated = self.harness.invoke(definition["name"], args, correlation, role="actor")
                self.assertIs(first.success, False)
                self.assertEqual(first.kind, "action")
                self.assertEqual(first.data["receipt"]["status"], "completed")
                self.assertEqual(first.data, repeated.data)
                self.assertEqual(first.data["receipt"], self.store.read("action_receipt", first.data["request"]["action_id"]))
                self.assertEqual(self.robot.executions, 2 * index + 2)
                shown = self.harness.actor_output.present(first)
                self.assertIs(shown["success"], False)
                self.assertEqual(json.loads(shown["contentItems"][0]["text"])["receipt"]["status"], "completed")

    def test_wrong_kind_and_wire_only_results_never_create_execution_evidence(self):
        from dataclasses import replace
        for bad in (ToolResponse([], data={'value': 7}, kind='action'),
                    ToolResponse([], data={'receipt': None}, kind='action'),
                    ToolResponse([], data={'receipt': {'status': []}}, kind='action'),
                    ToolResponse([], data={'value': 7}, kind=[]),
                    {'success': True, 'contentItems': None},
                    {'success': True, 'contentItems': [{'type': 'inputText',
                        'text': '{"receipt":{"status":"completed","executed_native_steps":9}}'}]}):
            registry = ToolRegistry([replace(self.plugin, handler=lambda *args: bad)])
            result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
            self.assertFalse(result.success)
            self.assertIsNotNone(result.contract_error)
            evidence = action_evidence(self.store, 'fixture_read', {}, result, 'call', physical=True)
            self.assertEqual(evidence['status'], 'unknown')
            self.assertNotIn('executed_native_steps', evidence)

    def test_malformed_physical_reply_is_unknown_without_losing_broker_accounting(self):
        self.harness.start()
        definition = dict(type='function', name='fixture_missing_receipt', description='Malformed receipt fixture.',
            inputSchema=dict(type='object', properties={}, additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition,
            lambda *args: ToolResponse([], kind='action', data={'receipt': None}),
            effect='physical', output_schemas=output_schemas('robot_wait')))
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        job = runner.run(dict(allow_direct_actions=True, source="from auriga_client import AurigaClient\n"
            "print(AurigaClient().call('fixture_missing_receipt', {})['success'])"),
            dict(thread_id='thread', turn_id='turn', tool_call_id='malformed-physical-program'))
        self.assertEqual(job['status'], 'succeeded')
        self.assertEqual(job['call_summary']['failed_calls'], 1)
        self.assertEqual(job['action_summary']['status_counts'], {'unknown': 1})
        self.assertEqual(job['action_summary']['actions_without_step_count'], 1)
        calls = self.store.states('program_call')
        self.assertEqual(calls[0]['output_contract_error']['phase'], 'after_handler')
        self.assertFalse(calls[0]['result']['success'])

    def test_source_archive_and_catalog_manifest_include_new_packages(self):
        self.robot.close = lambda: None
        config = Config.from_dict({'features': {'disabled_tools': ['fixture_read'],
            'grasp_memory_dir': str(self.root/'memory')}})
        workbench = Workbench(self.root/'workbench', robot=self.robot, config=config,
            tool_specs=[self.plugin], observation_profile='rgb_proprio_v1', motion_review=False)
        self.addCleanup(workbench.store.close)
        self.addCleanup(workbench.run_lock.close)
        if workbench.grasp_memory_store:
            self.addCleanup(workbench.grasp_memory_store.close)
        package = Path(__file__).resolve().parents[1]/'auriga'
        for name in ('runtime/harness.py', 'runtime/tools/registry.py', 'apps/evaluation/report.py'):
            self.assertEqual(workbench.source_hashes[name], hashlib.sha256((package/name).read_bytes()).hexdigest())
        info, data = workbench.store.resolve_artifact(workbench.source_ref, allowed_scopes=("observer",))
        archived = json.loads(data)
        self.assertEqual(archived['runtime/tools/registry.py'], (package/'runtime/tools/registry.py').read_text())
        manifest = workbench.harness.tool_registry.manifest(workbench.harness)
        item = next(x for x in manifest['tools'] if x['name'] == 'fixture_read')
        self.assertFalse(item['available'])
        self.assertEqual(item['unavailable_reason'], 'tool_disabled_by_config:fixture_read')
