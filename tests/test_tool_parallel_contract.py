"""ToolSpec.parallel / ToolRegistry.parallel_safe: the registry-owned concurrency contract (no model, no robot)."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from auriga.core.config import Config
from auriga.core.storage.store import Store
from auriga.runtime.harness import Harness
from auriga.runtime.tools.builtins import create_registry, output_schemas
from auriga.runtime.tools.plugins.robotic_view import make_robotic_view_tool
from auriga.runtime.tools.plugins.stereo_guide import make_stereo_guide_tool
from auriga.runtime.tools.registry import ToolRegistry, ToolSpec
from test_harness import FixtureRobot

GOLDEN = Path(__file__).with_name('fixtures') / 'tool_parallel_classification.json'
# Tools that the former codex.py allowlist marked parallel regardless of arguments.
ALWAYS_PARALLEL = frozenset({'robot_information', 'robot_contact_overlay', 'robot_action_status', 'robot_capabilities',
                             'history_list_actions', 'history_view_observation', 'history_read_trace', 'program_status',
                             'vision_compare', 'pose_plan'})
REQUEST_DEPENDENT = frozenset({'geometry_query', 'depth_query', 'robot_observe', 'task_plan', 'operation_memory'})


def full_registry():
    """Every tool the gateway can register: built-ins (pose_skill on) and both optional plugins."""
    return create_registry([make_robotic_view_tool(lambda *args: None), make_stereo_guide_tool(lambda *args: None)],
                           runtime=Config.from_dict({'features': {'pose_skill': True}}))


def fixture_spec(name, parallel):
    return ToolSpec(dict(type='function', name=name, description='Fixture tool.',
                         inputSchema=dict(type='object', additionalProperties=True)),
                    lambda context, arguments, correlation: None, parallel=parallel,
                    output_schemas=output_schemas('robot_wait'))


class GoldenClassificationTest(unittest.TestCase):
    """The table was produced by the pre-refactor RecordedClient._is_parallel_read_tool at c4ff16a."""

    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN.read_text())
        cls.registry = full_registry()

    def test_table_covers_every_registered_tool(self):
        self.assertEqual(sorted(self.registry.names), self.golden['registered_tools'])
        self.assertEqual({row['tool'] for row in self.golden['rows']}, set(self.registry.names))

    def test_registry_reproduces_every_golden_row(self):
        for row in self.golden['rows']:
            with self.subTest(tool=row['tool'], arguments=row['arguments']):
                self.assertIs(self.registry.parallel_safe(row['tool'], row['arguments']), row['parallel'])

    def test_retired_allowlist_names_are_not_registered_and_are_ordered(self):
        retired = self.golden['retired_names_formerly_parallel']
        self.assertIn('scene_query', retired)
        for name in retired:
            with self.subTest(tool=name):
                self.assertNotIn(name, self.registry.names)
                self.assertFalse(self.registry.parallel_safe(name, {}))

    def test_declarations_match_the_former_allowlist_shape(self):
        for name in self.registry.names:
            spec = self.registry.spec(name, role='actor')
            with self.subTest(tool=name):
                if name in ALWAYS_PARALLEL:
                    self.assertIs(spec.parallel, True)
                elif name in REQUEST_DEPENDENT:
                    self.assertTrue(callable(spec.parallel))
                else:
                    self.assertIs(spec.parallel, False)

    def test_classification_does_not_use_the_read_effect(self):
        # robot_observe is a read, but a fresh capture replaces harness.latest.
        spec = self.registry.spec('robot_observe', role='actor')
        self.assertEqual(spec.effect_for({}), 'read')
        self.assertFalse(self.registry.parallel_safe('robot_observe', {}))
        self.assertTrue(self.registry.parallel_safe('robot_observe', {'observation_id': 'obs-1', 'crop': {}}))


class ParallelSafeEdgeCasesTest(unittest.TestCase):
    def setUp(self):
        self.registry = full_registry()

    def test_unknown_or_non_string_tool_is_ordered(self):
        for name in ('not_registered', None, 7, ['robot_information'], {'tool': 'robot_information'}):
            with self.subTest(name=name):
                self.assertFalse(self.registry.parallel_safe(name, {}))

    def test_non_object_arguments_are_ordered(self):
        for arguments in (None, [], 'request', 3, ['observation_id']):
            for name in ('robot_information', 'geometry_query', 'robot_observe'):
                with self.subTest(tool=name, arguments=arguments):
                    self.assertFalse(self.registry.parallel_safe(name, arguments))

    def test_unhashable_request_kind_is_ordered_without_raising(self):
        for name, arguments in (('depth_query', {'request': {'kind': ['depth_points']}}),
                                ('operation_memory', {'request': {'kind': {'help': True}}})):
            with self.subTest(tool=name):
                self.assertFalse(self.registry.parallel_safe(name, arguments))

    def test_failing_or_non_boolean_classifier_is_ordered(self):
        def broken(arguments):
            raise RuntimeError('fixture classifier failure')
        registry = ToolRegistry([fixture_spec('fixture_broken', broken),
                                 fixture_spec('fixture_truthy', lambda arguments: 'yes')])
        self.assertFalse(registry.parallel_safe('fixture_broken', {}))
        # parallel_for normalises a callable's verdict to bool.
        self.assertTrue(registry.parallel_safe('fixture_truthy', {}))
        self.assertIs(registry.spec('fixture_truthy', role='actor').parallel_for({}), True)

    def test_parallel_contract_is_validated_at_registration(self):
        self.assertIs(fixture_spec('fixture_default', False).parallel, False)
        self.assertIs(ToolSpec(dict(type='function', name='fixture_plain', description='Fixture.',
                                    inputSchema={'type': 'object'}), None).parallel, False)
        for invalid in ('yes', 1, None, {'parallel': True}):
            with self.subTest(parallel=invalid), self.assertRaisesRegex(ValueError, 'invalid_tool_parallel:fixture_bad'):
                ToolRegistry([fixture_spec('fixture_bad', invalid)])

    def test_inspection_copies_keep_the_contract(self):
        registry = ToolRegistry([fixture_spec('fixture_read', True)])
        self.assertIs(registry.spec('fixture_read', role='program').parallel, True)
        self.assertIs(registry._specs['fixture_read'].parallel, True)
        self.assertIs(replace(registry.spec('fixture_read', role='actor'), parallel=False).parallel, False)
        self.assertTrue(registry.parallel_safe('fixture_read', {}))


class ManifestParallelFieldTest(unittest.TestCase):
    def test_manifest_reports_constant_and_request_dependent_contracts(self):
        registry = ToolRegistry([fixture_spec('fixture_parallel', True), fixture_spec('fixture_ordered', False),
                                 fixture_spec('fixture_dynamic', lambda arguments: bool(arguments.get('read')))])
        runtime = SimpleNamespace(disabled_tools=frozenset({'fixture_ordered'}))
        tools = {item['name']: item for item in registry.manifest(runtime)['tools']}
        self.assertIs(tools['fixture_parallel']['parallel'], True)
        self.assertIs(tools['fixture_ordered']['parallel'], False)
        self.assertEqual(tools['fixture_dynamic']['parallel'], 'request_dependent')
        # The field is reported for unavailable tools too, like effect.
        self.assertFalse(tools['fixture_ordered']['available'])
        json.dumps(registry.manifest(runtime))

    def test_harness_catalog_manifest_declares_builtin_concurrency(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = Store(root / 'run', dict(run_id='parallel-manifest', episode_id='ep', env_instance_id='env',
                                         env_generation=1))
        self.addCleanup(store.close)
        harness = Harness(store, FixtureRobot(root))
        manifest = harness.tool_registry.manifest(harness)
        tools = {item['name']: item for item in manifest['tools']}
        self.assertTrue(tools)
        for name, item in tools.items():
            with self.subTest(tool=name):
                expected = (True if name in ALWAYS_PARALLEL else
                            'request_dependent' if name in REQUEST_DEPENDENT else False)
                self.assertEqual(item['parallel'], expected)
                self.assertIs(type(item['parallel']), str if expected == 'request_dependent' else bool)


if __name__ == '__main__':
    unittest.main()
