"""Queue case keys, frozen settings and schema drift for agent_backend; library and report condition fields."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auriga.apps.evaluation.report import summarize
from auriga.apps.library.experiment_comparisons import ExperimentGroups
from auriga.apps.library.experiments import Catalog, atomic_json, register_run
from auriga.core.config import Config
from auriga.core.storage.store import Store
from scripts import benchmark_queue as queue


class QueueAgentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.checkout = self.root / 'checkout'
        (self.checkout / 'auriga/apps/gateway').mkdir(parents=True)
        (self.checkout / 'auriga/core').mkdir(parents=True)
        (self.checkout / 'auriga/apps/gateway/cli.py').write_text("parser.add_argument('--config')\nConfig.load()\n")
        self.case = dict(cwd=str(self.checkout), run_dir=str(self.root / 'run'), task='stack_bowls',
                         seed=0, layout_id=0, title='agent', tag=[])

    def test_case_keys_become_cli_flags(self):
        command = queue.command({**self.case, 'agent_backend': 'claude_code', 'model': 'claude-x', 'effort': 'medium'}, 1)
        for flag, value in (('--agent-backend', 'claude_code'), ('--model', 'claude-x'), ('--effort', 'medium')):
            self.assertEqual(command[command.index(flag) + 1], value)
        plain = queue.command(self.case, 1)
        for flag in ('--agent-backend', '--model', '--effort'):
            self.assertNotIn(flag, plain)

    def test_invalid_case_values_are_rejected(self):
        for update in ({'agent_backend': 'gemini'}, {'agent_backend': ''}, {'agent_backend': None}, {'model': ''},
                       {'effort': 'extreme'}, {'effort': 3}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                queue.command({**self.case, **update}, 0)

    def test_frozen_settings_stay_unchanged_for_codex_and_record_env_name_otherwise(self):
        self.assertEqual(queue.agent_settings(Config(), 'codex'), {'agent_backend': 'codex'})
        settings = queue.agent_settings(Config(), 'claude_code')
        self.assertEqual(settings, {'agent_backend': 'claude_code', 'agent_upstream_api_key_env': 'ANTHROPIC_API_KEY'})
        config = Config.from_dict({'agent': {'backend': 'opencode', 'opencode': {'upstream': {'api_key_env': 'OC_KEY'}}}})
        self.assertEqual(queue.agent_settings(config, 'opencode')['agent_upstream_api_key_env'], 'OC_KEY')
        self.assertNotIn('agent', queue.configured_manifest({'cases': [self.case]}, Config())['auriga_config'])

    def _old_schema_checkout(self):
        source = (Path(__file__).parents[1] / 'auriga/core/config.py').read_text()
        old = source.replace('class AgentConfig(Section):', 'class RetiredSectionConfig(Section):').replace(
            '    agent: AgentConfig = field(default_factory=AgentConfig)\n', '')
        self.assertNotIn('agent: AgentConfig', old)
        (self.checkout / 'auriga/core/config.py').write_text(old)

    def test_old_checkout_schema_rejects_agent_backend_case_without_downgrading(self):
        self._old_schema_checkout()
        manifest = queue.configured_manifest({'cases': [{**self.case, 'agent_backend': 'codex'}]}, Config())
        output = self.root / 'out'
        with patch.object(queue, 'sample_gpus') as sample, patch.object(queue.subprocess, 'Popen') as launch, \
                self.assertRaisesRegex(ValueError, 'unified_config_incompatible:agent'):
            queue.run_queue(manifest, output)
        sample.assert_not_called()
        launch.assert_not_called()
        self.assertFalse(output.exists())

    def test_old_checkout_schema_rejects_non_default_agent_section(self):
        self._old_schema_checkout()
        config = Config.from_dict({'agent': {'backend': 'claude_code'}})
        manifest = queue.configured_manifest({'cases': [self.case]}, config)
        with patch.object(queue, 'sample_gpus'), self.assertRaisesRegex(ValueError, 'unified_config_incompatible:agent'):
            queue.run_queue(manifest, self.root / 'out2')

    def test_codex_default_projection_ignores_old_checkout_without_agent(self):
        self._old_schema_checkout()
        schema = queue.runtime_config_schema(self.checkout)
        document = Config().to_dict()
        self.assertEqual(queue.project_config_for_runtimes(document, [schema]), document)

    def test_new_checkout_schema_declares_agent_config(self):
        schema = queue.runtime_config_schema(Path(__file__).parents[1])
        self.assertIn('AgentConfig', schema)
        projected = queue.project_config_for_runtimes(Config.from_dict({'agent': {'backend': 'opencode'}}).to_dict(), [schema])
        self.assertEqual(projected['agent']['backend'], 'opencode')


class ConditionFieldTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / 'benchmark' / 'panel' / 'manifest.json'
        self.row = dict(case_id='pick__standard__g0__l0', task='pick', variant='standard', eval_seed=0, layout_id=0)

    def groups(self, config, case_extra=None):
        run = self.root / 'benchmark' / 'panel' / 'runs' / 'one'
        register_run(run, category='benchmark', task='pick', run_id='one')
        atomic_json(run / 'summary.json', dict(normalized_score=1, task_success=True, valid_episode=True,
                                               scored_episode=True, status='completed'))
        case = dict(run_dir=str(run), cwd='/source/x', task='pick', seed=0, layout_id=0, tag=['cand', self.row['case_id']],
                    **(case_extra or {}))
        manifest = dict(benchmark_group='t', method='paired', protocol_cases=[self.row], cases=[case],
                        experiment_panels={'cand': [self.row['case_id']]})
        if config is not None:
            manifest['auriga_config'] = config
        atomic_json(self.path, manifest)
        catalog = Catalog(self.root)
        catalog.scan()
        return {g['condition']: g for g in ExperimentGroups(self.root).snapshot(catalog)['groups']}['cand']

    def test_old_archive_without_agent_section_reads_as_codex(self):
        self.assertEqual(self.groups({'model': {'name': 'm'}})['conditions']['agent_backend'], 'codex')
        self.assertEqual(self.groups(None)['conditions']['agent_backend'], 'codex')

    def test_group_reads_configured_backend_and_case_override(self):
        config = Config.from_dict({'agent': {'backend': 'claude_code'}}).to_dict()
        self.assertEqual(self.groups(config)['conditions']['agent_backend'], 'claude_code')
        self.assertEqual(self.groups(config, {'agent_backend': 'opencode'})['conditions']['agent_backend'], 'opencode')
        self.assertEqual(self.groups(Config().to_dict(), {'agent_backend': 'opencode'})['conditions']['agent_backend'], 'opencode')


class ReportConditionTests(unittest.TestCase):
    def report(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Store(root, dict(run_id='r', episode_id='e', env_instance_id='i', env_generation=1)).close()
            (root / 'manifest.json').write_text(json.dumps(dict(run_id='r', **manifest)))
            (root / 'summary.json').write_text(json.dumps(dict(status='unknown')))
            return summarize(root)['conditions']

    def test_old_manifest_is_codex_with_no_cli_record(self):
        conditions = self.report(dict(codex_sdk='0.160.0'))
        self.assertEqual((conditions['agent_backend'], conditions['agent_cli'], conditions['codex_sdk']), ('codex', None, '0.160.0'))

    def test_manifest_block_supplies_backend_and_cli(self):
        cli = dict(version='1.19.5', executable='/x/opencode', sha256='ab')
        conditions = self.report(dict(agent_backend=dict(kind='opencode', cli=cli)))
        self.assertEqual((conditions['agent_backend'], conditions['agent_cli']), ('opencode', '1.19.5'))
        before = self.report(dict(codex_sdk='0.160.0', codex_cli='0.160.0'))
        after = self.report(dict(codex_sdk='0.160.0', codex_cli='0.160.0', agent_backend=dict(
            kind='codex', cli=dict(version='0.160.0', executable='/machine/specific/codex', sha256=None))))
        self.assertEqual((before['agent_backend'], before['agent_cli']), (after['agent_backend'], after['agent_cli']))
        conditions = self.report(dict(agent_backend=dict(kind='claude_code', sdk='0.2.180')))
        self.assertEqual((conditions['agent_backend'], conditions['agent_cli']), ('claude_code', None))


if __name__ == '__main__':
    unittest.main()
