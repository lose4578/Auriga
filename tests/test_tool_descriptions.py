"""Published tools carry typed parameters and complete help for both roles."""
from copy import deepcopy
import unittest

from auriga.runtime.tools.builtins import create_registry
from auriga.runtime.tools.definitions import (DEPTH_REQUEST_DESCRIPTIONS,
    GEOMETRY_REQUEST_DESCRIPTIONS, TOOLS, public_definitions)
from auriga.runtime.tools.registry import validator_for
from auriga.runtime.actions.robot_action_schema import ROBOT_ACTION_TOOL


PUBLIC_NAMES = {
    'task_plan', 'robot_observe', 'robot_capabilities', 'robot_information',
    'robot_contact_overlay', 'robot_action_status', 'robot_wait',
    'history_list_actions', 'history_view_observation', 'history_read_trace',
    'history_annotate_action', 'geometry_query', 'program_run', 'program_status',
    'vision_compare', 'vision_review', 'robot_policy_step', 'robot_policy_delegate', 'depth_query',
    'robot_action', 'operation_memory',
}


def schema_nodes(value, path=()):
    """Walk real input schemas, excluding examples and conditional selectors."""
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            if key not in {'description', 'examples', 'default', 'enum', 'const', 'if', 'not'}:
                yield from schema_nodes(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from schema_nodes(child, path + (index,))


def validation_schema(value):
    if isinstance(value, dict):
        return {key: validation_schema(item) for key, item in value.items()
                if key not in {'description', 'examples'}}
    if isinstance(value, list):
        return [validation_schema(item) for item in value]
    return value


class ToolDescriptionTests(unittest.TestCase):
    def test_all_public_input_fields_have_help(self):
        definitions = public_definitions()
        self.assertEqual({item['name'] for item in definitions}, PUBLIC_NAMES)
        for definition in definitions:
            self.assertTrue(definition['description'].strip(), definition['name'])
            for path, node in schema_nodes(definition['inputSchema']):
                for field, field_schema in node.get('properties', {}).items():
                    with self.subTest(tool=definition['name'], path=path, field=field):
                        self.assertTrue(field_schema.get('description', '').strip())

    def test_every_published_example_validates_through_both_allowed_roles(self):
        registry = create_registry()
        for definition in public_definitions():
            schema = definition['inputSchema']
            self.assertTrue(schema.get('examples'), definition['name'])
            for path, node in schema_nodes(schema):
                for example in node.get('examples', []):
                    with self.subTest(tool=definition['name'], path=path, example=example):
                        self.assertEqual(list(validator_for(node).iter_errors(example)), [])
            for example in schema['examples']:
                for role in ('actor', 'program'):
                    actor_spec = registry.spec(definition['name'], role='actor')
                    if role in actor_spec.roles:
                        with self.subTest(tool=definition['name'], role=role, example=example):
                            spec = registry.validate(definition['name'], example, role=role)
                            self.assertEqual(spec.definition, definition)

    def test_task_attention_schema_matches_partial_updates_and_text_limits_for_both_roles(self):
        registry = create_registry()
        entry = dict(concern='Current uncertainty', check='Use the next visible endpoint; adjust if it differs.',
                     when='Before placement')
        ledger = dict(id='retention', expected='The object remains aligned after lift.', status='pending',
                      basis='unknown', next_check='Compare the next task-progress frame.')
        valid = [dict(plan='Brief task understanding and approach.', ledger=[]),
                 dict(plan='Brief plan.', attention=[], ledger=[]), dict(operation='read'),
                 dict(operation='update', plan='Changed understanding.'),
                 dict(operation='update', attention=[]),
                 dict(operation='update', ledger=[ledger]),
                 dict(operation='update', attention=[entry] * 3),
                 dict(operation='update', attention=[dict(entry, check='证' * 1600)])]
        invalid = [{}, dict(plan='Missing the required initial ledger.'), dict(attention=[]), dict(operation='update'),
                   dict(operation='read', attention=[]), dict(operation='read', plan='x'),
                   dict(plan='x', attention=None), dict(plan='x', attention=[dict(entry, when=' ')]),
                   dict(plan='x', attention=[dict(entry, check='x' * 1601)]),
                   dict(plan='x', attention=[dict(entry, extra='unknown')]),
                   dict(operation='update', attention=[entry] * 4)]
        invalid += [dict(plan='x', ledger=[dict(ledger, status='bad')]),
                    dict(plan='x', ledger=[dict(ledger, id=' ')]),
                    dict(plan='x', ledger=[dict(ledger, extra='unknown')]),
                    dict(plan='x', ledger=[ledger] * 4)]
        for role in ('actor', 'program'):
            schema = registry.spec('task_plan', role=role).definition['inputSchema']
            validator = validator_for(schema)
            for arguments in valid:
                with self.subTest(role=role, arguments=arguments):
                    self.assertTrue(validator.is_valid(arguments))
                    registry.validate('task_plan', arguments, role=role)
            for arguments in invalid:
                with self.subTest(role=role, arguments=arguments):
                    self.assertFalse(validator.is_valid(arguments))
            self.assertEqual(schema['properties']['attention']['maxItems'], 3)
            self.assertNotIn('minItems', schema['properties']['attention'])
        definition = registry.spec('task_plan', role='actor').definition
        self.assertEqual(definition, registry.spec('task_plan', role='program').definition)
        self.assertIn('[] clears it', definition['description'])
        self.assertIn('normal task-progress', definition['description'])
        self.assertIn('evidence ledger', definition['description'])
        self.assertIn('one to three short ledger rows', definition['description'])
        self.assertIn('change the contact, route or target mechanism', definition['description'])

    def test_publishing_and_editing_a_catalog_do_not_mutate_source_contracts(self):
        legacy_before = deepcopy(TOOLS)
        action_before = deepcopy(ROBOT_ACTION_TOOL)
        catalog = public_definitions()
        self.assertEqual(TOOLS, legacy_before)
        self.assertEqual(ROBOT_ACTION_TOOL, action_before)
        for definition in catalog:
            definition['description'] = 'local display edit'
            definition['inputSchema'].clear()
        self.assertEqual(TOOLS, legacy_before)
        self.assertEqual(ROBOT_ACTION_TOOL, action_before)
        action = next(item for item in public_definitions() if item['name'] == 'robot_action')
        self.assertEqual(action['inputSchema'], action_before['inputSchema'])
        self.assertTrue(action['description'].startswith(action_before['description']))

    def test_typed_geometry_depth_requests_preserve_distinct_legal_branches(self):
        registry = create_registry()
        geometry = [dict(kind='help'),
            dict(kind='pixel_plane', observation_id='obs', camera_id='head', pixel=[320, 240],
                 plane_origin_m=[0, 0, 2], plane_normal=[0, 0, 1], plane_origin_alt_m=[0, 0, 3]),
            dict(kind='project_geometry', observation_id='obs', camera_id='head', points_m=[[0, 0, 2]]),
            dict(kind='sparse_track', observation_ids=['before', 'after'], camera_id='head',
                 roi=[100.5, 80.5, 400., 320.]),
            dict(kind='alignment', observation_ids=['heldout1', 'heldout2', 'heldout3'], camera_id='head', scene_version=1),
            dict(kind='preview_ee', observation_id='obs', camera_id='head', axis_length_m=.03,
                 action=dict(frame_id='env_origin', mode='absolute', path=dict(waypoints=[
                     dict(left=dict(position_m=[0, 0, 2], quaternion_xyzw=[0, 0, 0, 1]))])))]
        depth = [dict(kind='help'), dict(kind='skill_help'), dict(kind='target_status'),
            dict(kind='region_geometry', target_id='target'),
            dict(kind='grasp_candidates', target_id='target', count=3),
            dict(kind='place_candidates', target_id='target', held_dimensions_m=[.1, .2, .3], grasp_offset_m=[0, 0, .02]),
            dict(kind='refresh_target', target_id='target', observation_id='obs', roi=[0, 0, 1, 1]),
            dict(kind='depth_points', observation_id='obs', camera_id='head', pixels=[[1., 2.]], radius_px=0)]
        for name, requests in [('geometry_query', geometry + depth), ('depth_query', depth)]:
            for role in ('actor', 'program'):
                for request in requests:
                    with self.subTest(tool=name, role=role, request=request):
                        registry.validate(name, dict(request=request), role=role)
                if name == 'geometry_query':
                    # The compatibility route preserves owner validation;
                    # test_depth_integration exercises its real rejection path.
                    with self.assertRaises(ValueError):
                        registry.validate(name, dict(request=dict(kind='unknown_depth_kind')), role=role)
                    continue
                for request in [dict(kind='region_geometry', target_id='target', observation_id='obs'),
                        dict(kind='refresh_target', target_id='target', observation_id='obs', camera_id='head', roi=[0, 0, 4, 4]),
                        dict(kind='depth_points', observation_id='obs', camera_id='head', pixels=[[1, 2]], radius_px=5)]:
                    with self.subTest(tool=name, role=role, invalid=request), self.assertRaises(ValueError):
                        registry.validate(name, dict(request=request), role=role)

    def test_request_types_and_distinct_field_meanings_survive_tool_publication(self):
        catalog = {item['name']: item for item in public_definitions()}
        for name, fields in (('geometry_query', GEOMETRY_REQUEST_DESCRIPTIONS),
                             ('depth_query', DEPTH_REQUEST_DESCRIPTIONS)):
            definition = catalog[name]
            request = definition['inputSchema']['properties']['request']
            self.assertIn("tool description's input field guide", request['description'])
            self.assertNotIn('fields below', request['description'])
            self.assertNotIn('properties below', definition['description'])
            if name == 'geometry_query':
                self.assertIn('Use depth_query for the complete field contract', definition['description'])
                self.assertNotIn(DEPTH_REQUEST_DESCRIPTIONS['depth_tolerance_m'], definition['description'])
            for field, text in fields.items():
                with self.subTest(tool=name, field=field):
                    matches = [node['properties'][field] for _, node in schema_nodes(request)
                               if field in node.get('properties', {})]
                    self.assertTrue(matches)
                    self.assertTrue(any(value.get('description') == text for value in matches))
                    self.assertEqual(definition['description'].count(text), 1)
            for path, node in schema_nodes(request):
                for field, child in node.get('properties', {}).items():
                    with self.subTest(tool=name, path=path, field=field):
                        self.assertIn('type', child)
                        self.assertIn(child['description'], definition['description'])

    def test_all_tool_help_preserves_bounds_defaults_and_branch_logic_without_size_heuristics(self):
        catalog = {item['name']: item for item in public_definitions()}
        help_text = catalog['robot_action']['description']
        for required in ('stages[].id:', 'Maximum characters: 80', 'Maximum characters: 1000',
                         'Unique items: true', 'stages[].angles_deg[1]:',
                         'Minimum (inclusive): -90', 'Maximum (inclusive): 90',
                         'if (kind equals "move"', 'requires id, kind, arm'):
            self.assertIn(required, help_text)
        field_help = catalog['robot_action']['inputSchema']['properties']['stages']['items']['properties']
        for name in ('position_m', 'grasp_position_m', 'goal_normalized'):
            self.assertEqual(help_text.count('stages[].' + name + ': ' + field_help[name]['description']), 1)
        self.assertIn('Maximum characters: 200', catalog['vision_review']['description'])
        observe = catalog['robot_observe']
        self.assertIn('Input field guide', observe['description'])
        for text in catalog['robot_wait']['inputSchema']['properties'].values():
            self.assertIn(text['description'], catalog['robot_wait']['description'])
        for required in ('at most 1 fields', 'otherwise', 'at least one of (requires plan | requires attention | requires ledger)'):
            self.assertIn(required, catalog['task_plan']['description'])
        for required in ('if pixels_a is present, requires pixels_b', 'if pixels_b is present, requires pixels_a'):
            self.assertIn(required, catalog['geometry_query']['description'])
        depth = catalog['depth_query']['description']
        for required in ('exactly one of', 'must not satisfy', 'requires target_id',
                         'Minimum (inclusive): 0.0005', 'Maximum (inclusive): 0.15', 'Default: 0.015'):
            self.assertIn(required, depth)

    def test_core_and_frequently_used_tools_are_loaded_initially(self):
        # robot_policy_delegate is published only when features.wall_oss is on; its description is the
        # delegation gate the actor decides from, so it is loaded initially rather than behind tool search.
        core = {'task_plan', 'robot_observe', 'robot_action', 'program_run', 'operation_memory',
                'geometry_query', 'history_view_observation', 'robot_wait',
                'robot_contact_overlay', 'history_list_actions', 'robot_policy_delegate'}
        for item in public_definitions():
            self.assertEqual(item.get('deferLoading', False), item['name'] not in core, item['name'])

    def test_program_help_explains_complete_text_and_repeated_delivery(self):
        catalog = {item['name']: item for item in public_definitions()}
        for name in ('program_run', 'program_status'):
            help_text = catalog[name]['description']
            for required in ('stdout', 'stderr', 'all available saved log text', 'middle diagnostics',
                             'source_complete', 'program_status', "detail='full'",
                             'client.artifact', 'exact saved bytes', "*_delivery.status='unchanged'", 'Inline image data URLs'):
                with self.subTest(tool=name, required=required):
                    self.assertIn(required, help_text)
            self.assertIn('all saved stdout/stderr text', catalog[name]['inputSchema']['properties']['detail']['description'])
            for required in ('automatically forward', 'inputImage', 'without re-encoding',
                             'visual_feedback', 'previous_tool_call_id'):
                self.assertIn(required, help_text)

    def test_shared_action_shapes_retain_each_fields_distinct_meaning(self):
        action = next(item for item in public_definitions() if item['name'] == 'robot_action')
        properties = action['inputSchema']['properties']
        stage = properties['stages']['items']
        stage_properties = stage['properties']
        self.assertIn('center between fingers', stage_properties['grasp_position_m']['description'])
        self.assertIn('link6/flange', stage_properties['position_m']['description'])
        self.assertNotEqual(stage_properties['grasp_position_m']['description'],
                            stage_properties['position_m']['description'])
        self.assertIn('Move only', stage_properties['gripper_goal_normalized']['description'])
        self.assertIn('Gripper stage only', stage_properties['goal_normalized']['description'])
        move = next(branch['then'] for branch in stage['allOf']
                    if branch['if']['properties']['kind']['const'] == 'move')
        for field in ('grasp_position_m', 'position_m'):
            self.assertEqual(stage_properties[field]['description'],
                             move['properties'][field]['description'])
        boxes = [properties['obstacles']['items'], properties['held_object'],
                 *properties['held_objects']['properties'].values()]
        for box in boxes:
            with self.subTest(box=box):
                self.assertIn('lower', box['properties']['min_m']['description'])
                self.assertIn('upper', box['properties']['max_m']['description'])
                self.assertNotEqual(box['properties']['min_m']['description'],
                                    box['properties']['max_m']['description'])

    def test_geometry_help_identifies_unapplied_scene_proposals(self):
        geometry = next(item for item in public_definitions() if item['name'] == 'geometry_query')
        for required in ('proposed patch', 'without applying it', 'existing saved version'):
            self.assertIn(required, geometry['description'])
        branches = geometry['inputSchema']['properties']['request']['oneOf']
        fields = next(part['properties'] for part in branches if part['properties']['kind']['const'] == 'build_scene')
        self.assertIn('does not apply', fields['base_scene_version']['description'])
        self.assertIn('at least 6 accepted', fields['primitive']['description'])


if __name__ == '__main__':
    unittest.main()
