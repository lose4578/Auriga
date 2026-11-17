"""Harness and evaluation-queue callers of the agent identity and backend gating."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_dual_arm_interface as base
from auriga.core.config import Config, ConfigError
from auriga.cortex.agents import factory
from auriga.runtime.harness import Harness
from scripts import benchmark_queue as queue


class HarnessIdentityTests(unittest.TestCase):
    def setUp(self):
        base.DualArmHarnessTests.setUp(self)
        self.h.thread_id = 'thread'

    def test_context_epoch_instance_is_unchanged_for_codex_archives(self):
        self.assertIsNone(self.h._model_context_epoch()[3])
        self.store.set_state('runtime', 'codex', dict(thread_id='thread', app_server_instance_id='appserver-1'))
        self.assertEqual(self.h._model_context_epoch()[3], 'appserver-1')
        self.store.set_state('runtime', 'agent', dict(backend='codex', thread_id='thread', instance_id='appserver-1'))
        self.assertEqual(self.h._model_context_epoch()[3], 'appserver-1')
        # A reconnect rewrites runtime/codex; the epoch changes even before the mirror does.
        self.store.set_state('runtime', 'codex', dict(thread_id='thread', app_server_instance_id='appserver-2'))
        self.assertEqual(self.h._model_context_epoch()[3], 'appserver-2')

    def test_context_epoch_reads_other_backends_identity(self):
        self.store.set_state('runtime', 'agent', dict(backend='claude_code', thread_id='thread', instance_id='agent-1'))
        self.assertEqual(self.h._model_context_epoch()[3], 'agent-1')

    def test_tool_call_id_source_defaults_to_native_and_records_the_transport_value(self):
        for call_id, extra, expected in (('native', {}, 'native_callId'),
                                         ('minted', {'callIdSource': 'host_minted'}, 'host_minted')):
            with self.subTest(source=expected):
                self.h.handle_tool(dict(threadId='thread', turnId='turn', callId=call_id, tool='robot_capabilities',
                                        arguments=dict(detail='summary'), **extra))
                self.assertEqual(self.store.read('tool_call', call_id)['tool_call_id_source'], expected)

    def test_delivery_channel_defaults_to_codex_tool_and_follows_the_harness_attribute(self):
        self.assertEqual(Harness.delivery_channel, 'codex_tool')

        def delivered_channels():
            return [record['delivery_channel'] for record in self.store.records('delivery')]
        self.h.observation_content(self.h.latest, dict(tool_call_id='a'))
        self.assertEqual(delivered_channels()[-1], 'codex_tool')
        self.h.delivery_channel = 'agent_tool'
        self.h.observation_content(self.h.latest, dict(tool_call_id='b'))
        self.assertEqual(delivered_channels()[-1], 'agent_tool')
        self.h.observation_content(self.h.latest, dict(tool_call_id='c', program_id='program-1'))
        self.assertEqual(delivered_channels()[-1], 'program_fd')


class CheckpointIdentityTests(unittest.TestCase):
    def test_slice_drops_the_parent_agent_identity_so_the_fork_reports_its_own(self):
        from auriga.apps.evaluation.checkpoint import _truncate_sqlite_store
        from auriga.core.storage.store import Store
        from auriga.cortex.agents.base import read_agent_identity
        identity = dict(run_id='r1', episode_id='ep1', env_instance_id='env1', env_generation=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'slice'
            store = Store(root, identity)
            store.set_state('runtime', 'codex', dict(thread_id='parent-thread', app_server_instance_id='parent-a'))
            store.set_state('runtime', 'agent', dict(backend='codex', thread_id='parent-thread',
                                                     instance_id='parent-a', pid=11, instructions_digest='p'))
            store.event('observation.published', {}, {'observation_id': 'obs-1', 'tool_call_id': 'call-cut'})
            store.event('observation.published', {}, {'observation_id': 'obs-2'})
            store.close()
            self.assertTrue(_truncate_sqlite_store(root / 'events.sqlite', until_call_id='call-cut')['truncated'])
            store = Store(root, identity)
            try:
                self.assertIsNone(store.read_state('runtime', 'agent'))
                self.assertEqual(store.read_state('runtime', 'codex')['thread_id'], 'parent-thread')
                # ForkedCodexSession.start_forked rewrites only runtime/codex; the identity follows the fork.
                store.set_state('runtime', 'codex', dict(thread_id='fork-thread', app_server_instance_id='fork-a',
                                                         forked_from_thread_id='parent-thread'))
                forked = read_agent_identity(store)
                self.assertEqual((forked['thread_id'], forked['instance_id'], forked['source']),
                                 ('fork-thread', 'fork-a', 'runtime/codex'))
            finally:
                store.close()


class QueueBackendGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.checkout = self.root / 'checkout'
        (self.checkout / 'auriga/apps/gateway').mkdir(parents=True)
        (self.checkout / 'auriga/core').mkdir(parents=True)
        (self.checkout / 'auriga/apps/gateway/cli.py').write_text("parser.add_argument('--config')\nConfig.load()\n")
        (self.checkout / 'auriga/core/config.py').write_text(
            (Path(__file__).parents[1] / 'auriga/core/config.py').read_text())
        self.case = dict(cwd=str(self.checkout), run_dir=str(self.root / 'run'), task='stack_bowls',
                         seed=0, layout_id=0, title='agent', tag=[])

    def run_queue(self, manifest, output):
        with patch.object(queue, 'sample_gpus') as sample, patch.object(queue.subprocess, 'Popen') as launch:
            try:
                queue.run_queue(manifest, output)
            finally:
                sample.assert_not_called()
                launch.assert_not_called()
                self.assertFalse(output.exists())

    def test_case_backend_is_refused_before_any_gpu_or_gateway(self):
        manifest = queue.configured_manifest({'cases': [{**self.case, 'agent_backend': 'claude_code'}]}, Config())
        # claude_code is implemented; an unavailable pinned runtime must still refuse before any GPU or gateway.
        with patch.object(factory, 'claude_code_runtime_problem', return_value='claude_code_sdk_missing'), \
                self.assertRaisesRegex(ConfigError, 'agent_backend_unavailable:claude_code: claude_code_sdk_missing'):
            self.run_queue(manifest, self.root / 'out')
        quiet = Config.from_dict({'features': {'self_briefing': False}})
        manifest = queue.configured_manifest({'cases': [{**self.case, 'agent_backend': 'opencode'}]}, quiet)
        with self.assertRaisesRegex(ConfigError, 'opencode_upstream_incomplete'):
            self.run_queue(manifest, self.root / 'out2')

    def test_configured_backend_is_refused_with_the_case_effort(self):
        config = Config.from_dict({'agent': {'backend': 'claude_code'}, 'features': {'self_briefing': False}})
        manifest = queue.configured_manifest({'cases': [{**self.case, 'effort': 'ultra'}]}, config)
        with self.assertRaisesRegex(ConfigError, 'agent_effort_unsupported:claude_code:ultra'):
            self.run_queue(manifest, self.root / 'out')

    def test_launch_config_matches_the_command_validation(self):
        config = queue.case_launch_config({**self.case, 'agent_backend': 'opencode', 'effort': 'low',
                                           'experiment_condition': 'arm-a', 'tag': 'single'}, 3)
        self.assertEqual((config.agent.backend, config.model.effort, config.run.gpu, config.run.benchmark,
                          list(config.run.tag)), ('opencode', 'low', 3, True, ['single']))
        # Codex cases keep the unchanged command (no gating outcome for Codex).
        self.assertIn('--benchmark', queue.command(self.case, 0))


class GatewayLaunchGateTests(unittest.TestCase):
    def test_gateway_refuses_a_backend_that_cannot_run_before_creating_the_run(self):
        from auriga.apps.gateway.workbench import Workbench
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            # sandbox_enabled is accepted on every backend (R4-8); lean_projection still needs Codex.
            config = Config.from_dict({'agent': {'backend': 'claude_code'},
                                       'features': {'sandbox_enabled': True, 'lean_projection': True}})
            with self.assertRaisesRegex(ConfigError, '^lean_projection_requires_codex_backend'):
                Workbench(root, robot=object(), config=config)
            self.assertFalse(root.exists())

    def test_serve_refuses_before_registering_a_run_directory(self):
        import asyncio
        from types import SimpleNamespace
        from auriga.apps.gateway import server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'experiments'
            config = Config.from_dict({'agent': {'backend': 'opencode'}, 'features': {'self_briefing': False}})
            args = SimpleNamespace(run_dir=None, resume=False, replay=False, price_card=None, experiment_root=str(root),
                                   category='diagnostic', task='fixture', title='fixture', tag=[], resolved_config=config)
            with patch.object(server, 'Workbench') as workbench, \
                    self.assertRaisesRegex(ConfigError, '^opencode_upstream_incomplete'):
                asyncio.run(server.serve(args))
            workbench.assert_not_called()
            self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
