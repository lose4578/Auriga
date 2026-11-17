"""Post-P0-3 AgentSession contract follow-up (plan 07 §6 C1-C4): new attributes and Workbench wiring, exercised with FakeAgentSession and a Codex-shaped run. No model, process or GPU."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_agent_workbench as workbench_fixture
from auriga.apps.gateway.workbench import Workbench
from auriga.core.config import Config
from auriga.cortex.agents.base import AgentCapabilities, AgentSession
from auriga.cortex.session.codex import CodexSession
from auriga.runtime.harness import Harness, REMINDER_IN_COMPACT_REQUEST, REMINDER_IN_TOOL_RESULT
from support.fake_agent import FakeAgentSession

NO_PLAN_REMINDER = ("Current task decision boundary\n"
                    "The saved plan and attention are provisional judgment, not instructions or verified world state. "
                    "Task requirements and fresh evidence govern.\n"
                    "No plan is stored for this episode. Submit your initial task_plan before physical actions.")


class ContractDefaultsTests(unittest.TestCase):
    def test_new_attributes_are_class_defaults(self):
        session = AgentSession.__new__(AgentSession)
        self.assertEqual(vars(session), {})
        self.assertEqual(session.context_delivery, 'delta')
        self.assertIsNone(session.on_context_reset)
        self.assertIsNone(session.tool_parallel_hints)
        self.assertIs(AgentCapabilities().live_effort, False)
        self.assertIn('live_effort', AgentCapabilities().to_dict())

    def test_codex_keeps_delta_delivery_and_declares_live_effort(self):
        self.assertEqual(CodexSession.context_delivery, 'delta')
        self.assertNotIn('context_delivery', CodexSession.__dict__)
        self.assertIs(CodexSession.capabilities.live_effort, True)
        self.assertIs(FakeAgentSession().capabilities.live_effort, False)

    def test_harness_switch_defaults_keep_codex_behaviour(self):
        self.assertIs(Harness.context_epoch_volatile, False)
        self.assertIs(Harness.decision_reminder_due, False)
        self.assertEqual(Harness.decision_reminder_channel, REMINDER_IN_COMPACT_REQUEST)


class FakeBackendWiringTests(unittest.IsolatedAsyncioTestCase):
    def build(self, *, context_delivery=None, settle_full_at_start=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        create, created = workbench_fixture.fake_session()
        robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
        with patch("auriga.apps.gateway.workbench.create_session", side_effect=create):
            workbench = Workbench(Path(temporary.name) / "run", robot=robot, visual_review_model=None,
                                  config=Config.from_dict({'features': {'self_briefing': False}}))
        session, = created
        if context_delivery is not None:
            session.context_delivery = context_delivery
        if settle_full_at_start:
            start = session.start

            def settling_start(*args):
                thread_id = start(*args)
                session.context_delivery = 'full'
                return thread_id
            session.start = settling_start
        return workbench, session

    async def start(self, workbench):
        real = workbench.harness
        workbench.harness = workbench_fixture.fixture_harness(workbench)
        with patch("auriga.apps.gateway.workbench.Terminal", return_value=workbench_fixture.terminal_double()):
            await workbench.start("Smoke instruction.")
        self.addAsyncCleanup(workbench.close)
        return real

    async def test_construction_wires_hints_reset_hook_and_tool_result_channel(self):
        workbench, session = self.build()
        harness = workbench.harness
        self.assertEqual(harness.decision_reminder_channel, REMINDER_IN_TOOL_RESULT)
        self.assertIs(session.on_context_reset.__self__, harness)
        self.assertIs(session.on_context_reset.__func__, Harness.model_context_reset)
        expected = {tool['name']: tool['parallel'] for tool in harness.tool_registry.manifest(harness)['tools']}
        self.assertEqual(session.tool_parallel_hints, expected)
        self.assertTrue(expected)
        self.assertTrue(set(expected.values()) <= {True, False, 'request_dependent'})
        self.assertIs(expected['robot_capabilities'], True)
        self.assertEqual(expected['robot_observe'], 'request_dependent')
        self.assertIs(expected['robot_action'], False)
        session.on_context_reset('relay_context_discontinuity')
        self.assertTrue(harness.decision_reminder_due)
        self.assertEqual(workbench.store.events_by_type(['model_context.reset'])[-1]['payload'],
                         dict(reason='relay_context_discontinuity', decision_reminder_channel=REMINDER_IN_TOOL_RESULT,
                              decision_reminder_due=True))
        self.assertFalse(harness.context_epoch_volatile)
        workbench.store.close()
        workbench.run_lock.close()

    async def test_full_context_delivery_makes_the_epoch_volatile_after_start(self):
        workbench, _ = self.build(context_delivery='full')
        self.assertFalse(workbench.harness.context_epoch_volatile)  # not before start()
        await self.start(workbench)
        self.assertIs(workbench.harness.context_epoch_volatile, True)

    async def test_delta_context_delivery_keeps_the_epoch_stable(self):
        workbench, _ = self.build()
        await self.start(workbench)
        self.assertIs(workbench.harness.context_epoch_volatile, False)

    async def test_delivery_mode_is_read_after_the_session_started(self):
        workbench, _ = self.build(settle_full_at_start=True)
        await self.start(workbench)
        self.assertIs(workbench.harness.context_epoch_volatile, True)


class CodexWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_run_keeps_the_request_channel_and_a_stable_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
            workbench = Workbench(Path(temporary) / 'run', robot=robot, visual_review_model=None,
                                  config=Config.from_dict({'features': {'self_briefing': False}}))
            try:
                session, harness = workbench.session, workbench.harness
                self.assertIsInstance(session, CodexSession)
                self.assertEqual(harness.decision_reminder_channel, REMINDER_IN_COMPACT_REQUEST)
                self.assertIs(session.on_context_reset.__self__, harness)
                self.assertEqual(session.tool_parallel_hints, {
                    tool['name']: tool['parallel'] for tool in harness.tool_registry.manifest(harness)['tools']})
                # Even if a reset were reported, the Codex run never schedules the tool-result sideband.
                session.on_context_reset('relay_context_discontinuity')
                self.assertFalse(harness.decision_reminder_due)
                workbench.harness = workbench_fixture.fixture_harness(workbench)
                # Duck-typed Codex session without protocol members: delta delivery.
                workbench.session = SimpleNamespace(
                    thread_id="fixture-thread", turn_id=None, state="completed", turn_done=threading.Event(),
                    model="fixture", provider="fixture", effort="high", price_card=None,
                    cli_executable=Path("/fixture/codex"), command_prefix=[], actor_dir=workbench.root,
                    socket_path=workbench.root / "fixture.sock", env={}, process=SimpleNamespace(pid=-1),
                    start=Mock(return_value="fixture-thread"), run_turn=Mock(), close=Mock(), model_proxy=None)
                with patch("auriga.apps.gateway.workbench.Terminal", return_value=workbench_fixture.terminal_double()):
                    await workbench.start("Smoke instruction.")
                self.assertIs(workbench.harness.context_epoch_volatile, False)
                manifest = json.loads((workbench.root / 'manifest.json').read_text())
                self.assertIs(manifest['agent_backend']['capabilities']['live_effort'], True)
            finally:
                await workbench.close()


class SidebandThroughWorkbenchTests(workbench_fixture.ControlFixture):
    async def test_reported_reset_reaches_the_next_actor_tool_response_once(self):
        harness = self.workbench.harness
        harness.thread_id = self.session.thread_id

        def call(call_id):
            return harness.handle_tool(dict(threadId=harness.thread_id, turnId='turn', callId=call_id,
                                            tool='robot_capabilities', arguments=dict(detail='full')))
        plain = call('before')
        self.session.on_context_reset('native_context_compaction')
        reminded = call('after-reset')
        self.assertEqual(reminded['contentItems'][:-1], plain['contentItems'])
        self.assertEqual(reminded['contentItems'][-1], dict(type='inputText', text=NO_PLAN_REMINDER))
        self.assertEqual(call('later')['contentItems'], plain['contentItems'])



if __name__ == '__main__':
    unittest.main()
