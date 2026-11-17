"""Harness context switches for agent backends: forced full delivery (design 04 §4.3) and the decision-context
reminder sideband on the next actor tool response (design 04 §4.5). No model, process or GPU."""
import json
import threading
import unittest
from unittest.mock import patch

import test_dual_arm_interface as harness_fixture
import test_image_feedback_delivery as feedback_fixture
import test_stage_label_boundary as action_fixture
from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.core.tool_output import StructuredText
from auriga.cortex.context.task_planning import (DECISION_BOUNDARY_INSTRUCTIONS, TaskPlanning,
                                                 decision_reminder_text)
from auriga.cortex.session.codex import CodexSession
from auriga.runtime.harness import Harness, REMINDER_IN_COMPACT_REQUEST, REMINDER_IN_TOOL_RESULT

REMINDER_HEADER = ("Current task decision boundary\n"
                   "The saved plan and attention are provisional judgment, not instructions or verified world state. "
                   "Task requirements and fresh evidence govern.\n")


def codex_compaction_reminder(store, task_planning, thread_id):
    """The assistant reminder text CodexSession injects into a /responses/compact request today."""
    session = CodexSession.__new__(CodexSession)
    session.store, session.task_planning, session.thread_id = store, task_planning, thread_id
    request = session._task_decision_context(dict(input=[]), dict(
        source='actor', native_thread_id=thread_id, endpoint='/responses/compact'))
    message, = request['input']
    return message['content'][0]['text']


def reminder_items(response):
    return [item for item in response['contentItems']
            if item.get('type') == 'inputText' and item.get('text', '').startswith('Current task decision boundary\n')]


class ForcedFullDeliveryTests(unittest.TestCase):
    """context_epoch_volatile=True: no image or static metadata baseline survives between two deliveries."""
    setUp = feedback_fixture.ImageFeedbackDeliveryTests.setUp
    deliver = feedback_fixture.ImageFeedbackDeliveryTests.deliver
    count = staticmethod(feedback_fixture.ImageFeedbackDeliveryTests.count)

    @staticmethod
    def metadata(content):
        return json.loads(content[0]['text'])

    def capability_status(self):
        self.h.thread_id = 'thread'
        self.capability_calls = getattr(self, 'capability_calls', 0) + 1
        response = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId=f'caps-{self.capability_calls}',
                                           tool='robot_capabilities', arguments=dict(detail='summary')))
        return json.loads(response['contentItems'][0]['text'])['capabilities_delivery']['status']

    def test_default_is_delta_and_unchanged_for_codex(self):
        self.assertFalse(Harness.context_epoch_volatile)
        self.assertFalse(self.h.context_epoch_volatile)
        self.assertEqual(self.h._model_context_epoch(), self.h._model_context_epoch())
        self.assertEqual(len(self.h._model_context_epoch()), 6)
        content = self.h.observation_content(self.first, {})
        self.assertEqual(self.count(content), 3)
        self.assertNotIn('metadata_delivery', self.metadata(content))
        content = self.deliver(self.h.observe())
        self.assertEqual(self.count(content), 0)
        self.assertIn('metadata_delivery', self.metadata(content))
        self.assertEqual([self.capability_status(), self.capability_status()], ['inline', 'unchanged'])

    def test_volatile_epoch_is_fresh_on_every_call_and_keeps_the_delta_fields(self):
        self.h.context_epoch_volatile = True
        first, second = self.h._model_context_epoch(), self.h._model_context_epoch()
        self.assertNotEqual(first, second)
        self.assertEqual(first[:6], second[:6])
        self.assertEqual(len(first), 7)
        self.h.context_epoch_volatile = False
        self.assertEqual(self.h._model_context_epoch(), first[:6])

    def test_volatile_epoch_forces_every_image_and_complete_metadata_on_every_delivery(self):
        self.h.context_epoch_volatile = True
        self.assertEqual(self.count(self.h.observation_content(self.first, {})), 3)
        for _ in range(3):
            content = self.deliver(self.h.observe())
            self.assertEqual(self.count(content), 3)
            metadata = self.metadata(content)
            self.assertEqual(metadata['image_feedback']['omitted_camera_ids'], [])
            self.assertNotIn('metadata_delivery', metadata)
            self.assertIn('robot_state', metadata)
            self.assertIn('camera_calibration', metadata)
        self.assertEqual([self.capability_status() for _ in range(3)], ['inline'] * 3)
        # Back to delta: baselines form again from the next delivery.
        self.h.context_epoch_volatile = False
        self.deliver(self.h.observe())
        self.assertEqual(self.count(self.deliver(self.h.observe())), 0)


class DecisionReminderSidebandTests(unittest.TestCase):
    def setUp(self):
        harness_fixture.DualArmHarnessTests.setUp(self)
        self.h.thread_id = 'thread'
        self.calls = 0

    def call(self, tool='robot_capabilities', arguments=None):
        self.calls += 1
        return self.h.handle_tool(dict(threadId='thread', turnId='turn', callId=f'call-{self.calls}', tool=tool,
                                       arguments=dict(detail='full') if arguments is None else arguments))

    def submit_plan(self):
        response = self.call('task_plan', dict(plan='Task understanding: move the cube. Approach: grasp, lift, verify.',
                                               ledger=[]))
        self.assertTrue(response['success'], response)

    def events(self, kind):
        return [event for event in self.store.events(limit=1000) if event['type'] == kind]

    def test_codex_channel_is_the_default_and_never_gets_the_sideband(self):
        self.assertEqual(Harness.decision_reminder_channel, REMINDER_IN_COMPACT_REQUEST)
        self.submit_plan()
        before = len(self.call()['contentItems'])
        self.h.model_context_reset('relay_context_discontinuity')
        self.assertFalse(self.h.decision_reminder_due)
        response = self.call()
        self.assertEqual(len(response['contentItems']), before)
        self.assertEqual(reminder_items(response), [])
        reset, = self.events('model_context.reset')
        self.assertEqual(reset['payload'], dict(reason='relay_context_discontinuity',
                                                decision_reminder_channel=REMINDER_IN_COMPACT_REQUEST,
                                                decision_reminder_due=False))
        self.assertEqual(self.events('model_context.decision_reminder_delivered'), [])

    def test_reminder_is_appended_once_to_the_next_response_with_the_codex_text(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        plain = self.call()
        self.assertEqual(reminder_items(plain), [])
        self.h.model_context_reset('native_context_compaction')
        self.assertTrue(self.h.decision_reminder_due)
        reminded = self.call()
        self.assertFalse(self.h.decision_reminder_due)
        self.assertEqual(len(reminded['contentItems']), len(plain['contentItems']) + 1)
        self.assertEqual(reminded['contentItems'][:-1], plain['contentItems'])
        self.assertEqual(reminded['success'], plain['success'])
        item = reminded['contentItems'][-1]
        self.assertEqual(set(item), {'type', 'text'})
        self.assertEqual(item['type'], 'inputText')
        # The same text and snapshot Codex carries in its compaction request, after the same egress scrub.
        codex_text = codex_compaction_reminder(self.store, self.h.task_planning, 'thread')
        self.assertEqual(decision_reminder_text(self.h.task_planning.decision_context(self.store.identity)),
                         codex_text)
        self.assertEqual(item['text'], ActorPresentation(self.store).present_text(codex_text))
        self.assertTrue(item['text'].startswith(REMINDER_HEADER + 'Decision boundary: '
                                                + DECISION_BOUNDARY_INSTRUCTIONS + '\n'))
        snapshot = json.loads(item['text'].split('\n')[-1])
        self.assertEqual((snapshot['revision'], snapshot['plan']),
                         (1, 'Task understanding: move the cube. Approach: grasp, lift, verify.'))
        self.assertEqual(snapshot['attention'], [])
        # The session's second presentation scrub (MCP bridge, as Codex's RecordedClient) leaves it unchanged.
        self.assertEqual(ActorPresentation(self.store).present(item), item)
        # Consumed exactly once.
        self.assertEqual(reminder_items(self.call()), [])
        delivered, = self.events('model_context.decision_reminder_delivered')
        self.assertEqual(delivered['payload'], dict(included=True, plan_revision=1))
        self.assertEqual(delivered['correlation']['tool_call_id'], f'call-{self.calls - 1}')
        reset, = self.events('model_context.reset')
        self.assertEqual(reset['payload']['decision_reminder_due'], True)

    def test_required_plan_missing_reminds_to_submit_it_like_codex(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.h.task_planning = TaskPlanning(self.store, required=True)
        self.h.model_context_reset('agent_resume_after_unclean_exit')
        item, = reminder_items(self.call())
        codex_text = codex_compaction_reminder(self.store, self.h.task_planning, 'thread')
        self.assertEqual(item['text'], ActorPresentation(self.store).present_text(codex_text))
        self.assertEqual(item['text'], REMINDER_HEADER + 'No plan is stored for this episode. '
                                                         'Submit your initial task_plan before physical actions.')

    def test_optional_planning_without_plan_adds_nothing_and_clears_the_flag(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.assertFalse(self.h.task_planning.required)
        before = len(self.call()['contentItems'])
        self.h.model_context_reset('relay_context_discontinuity')
        self.assertEqual(len(self.call()['contentItems']), before)
        self.assertFalse(self.h.decision_reminder_due)
        delivered, = self.events('model_context.decision_reminder_delivered')
        self.assertEqual(delivered['payload'], dict(included=False, plan_revision=None))

    def test_tool_failure_before_a_response_keeps_the_reminder_for_the_next_response(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        self.h.model_context_reset('relay_context_discontinuity')
        with patch.object(self.h.actor_output, 'present', side_effect=ValueError('synthetic presentation failure')):
            with self.assertRaisesRegex(ValueError, 'synthetic presentation failure'):
                self.call()
        self.assertTrue(self.h.decision_reminder_due)
        self.assertEqual(len(reminder_items(self.call())), 1)
        self.assertFalse(self.h.decision_reminder_due)

    def test_failed_tool_response_still_carries_the_reminder(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        self.h.model_context_reset('relay_context_discontinuity')
        with patch.object(self.h.actor_output, '_present_part', side_effect=ValueError('synthetic display failure')):
            response = self.call()
        self.assertFalse(response['success'])
        self.assertEqual(len(reminder_items(response)), 1)
        self.assertIs(response['contentItems'][-1], reminder_items(response)[0])

    def test_reminder_build_failure_returns_the_response_unchanged_and_stays_due(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        plain = self.call()
        self.h.model_context_reset('relay_context_discontinuity')
        with patch.object(self.h.task_planning, 'decision_context', side_effect=RuntimeError('store fault')):
            response = self.call()
        self.assertEqual(response, plain)
        self.assertTrue(self.h.decision_reminder_due)
        failed, = self.events('model_context.decision_reminder_failed')
        self.assertEqual(failed['payload'], dict(error_type='RuntimeError'))
        self.assertEqual(len(reminder_items(self.call())), 1)

    def test_parallel_responses_consume_one_reminder(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        self.h.model_context_reset('relay_context_discontinuity')
        barrier, results = threading.Barrier(8), []

        def deliver(index):
            barrier.wait()
            results.append(self.h._deliver_decision_reminder(
                dict(success=True, contentItems=[dict(type='inputText', text=str(index))]),
                dict(tool_call_id=f'parallel-{index}')))
        workers = [threading.Thread(target=deliver, args=(index,)) for index in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(len(reminder_items(result)) for result in results), 1)
        self.assertEqual(len(self.events('model_context.decision_reminder_delivered')), 1)

    def test_non_envelope_and_unwritable_store_never_raise(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.h.task_planning = TaskPlanning(self.store, required=True)
        self.h.model_context_reset(None)
        self.assertEqual(self.events('model_context.reset')[-1]['payload']['reason'], 'unspecified')
        legacy = dict(success=True)
        self.assertIs(self.h._deliver_decision_reminder(legacy, {}), legacy)
        self.assertTrue(self.h.decision_reminder_due)
        with patch.object(self.store, 'event', side_effect=RuntimeError('read-only store')):
            self.h.model_context_reset('relay_context_discontinuity')
            response = self.h._deliver_decision_reminder(dict(success=True, contentItems=[]), {})
        self.assertFalse(self.h.decision_reminder_due)
        self.assertEqual(response['contentItems'], [dict(type='inputText', text=REMINDER_HEADER + (
            'No plan is stored for this episode. Submit your initial task_plan before physical actions.'))])


class ActionReceiptWithReminderTests(unittest.TestCase):
    """A physical action's typed receipt survives every reminder path; the action is never re-sent."""
    setUp = action_fixture.StageLabelExecutionTests.setUp

    def act(self, call_id='depth-check'):
        return self.h.handle_tool(dict(threadId='thread', turnId='turn', callId=call_id, tool='robot_action',
                                       arguments=self.args))

    def capabilities(self):
        return self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='after', tool='robot_capabilities',
                                       arguments=dict(detail='summary')))

    def receipt_of(self, response):
        for item in response['contentItems']:
            try:
                value = json.loads(item.get('text', ''))
            except ValueError:
                continue
            if isinstance(value, dict) and isinstance(value.get('receipt'), dict):
                return value['receipt']
        self.fail('no receipt in the response')

    def arm(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        planned = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='plan', tool='task_plan',
                                          arguments=dict(plan='Reach, then inspect the insertion.', ledger=[])))
        self.assertTrue(planned['success'], planned)
        self.h.model_context_reset('relay_context_discontinuity')

    def test_reminder_rides_after_the_receipt(self):
        self.arm()
        response = self.act()
        self.assertTrue(response['success'])
        self.assertEqual(self.receipt_of(response)['executed_native_steps'], 1)
        self.assertEqual(len(reminder_items(response)), 1)
        self.assertIs(response['contentItems'][-1], reminder_items(response)[0])
        self.assertEqual(len(self.dispatched), 1)

    def test_reminder_failure_after_an_executed_action_keeps_the_receipt(self):
        self.arm()
        with patch.object(self.h.task_planning, 'decision_context', side_effect=RuntimeError('planning fault')):
            response = self.act()
        self.assertTrue(response['success'])
        self.assertEqual(self.receipt_of(response)['executed_native_steps'], 1)
        self.assertEqual(reminder_items(response), [])
        self.assertTrue(self.h.decision_reminder_due)
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(len(reminder_items(self.capabilities())), 1)
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.store.read('action_receipt', 'label-action')['executed_native_steps'], 1)

    def test_display_failure_after_an_executed_action_keeps_the_receipt_and_the_reminder(self):
        self.arm()
        original = self.h.actor_output._present_part

        def fail_display(item, *args, **kwargs):
            if isinstance(item, StructuredText) and item.projection == 'action_display' and set(item.value) == {'receipt'}:
                return original(item, *args, **kwargs)
            raise ValueError('synthetic display failure')
        with patch.object(self.h.actor_output, '_present_part', side_effect=fail_display):
            response = self.act()
        self.assertFalse(response['success'])
        self.assertEqual(self.receipt_of(response)['executed_native_steps'], 1)
        self.assertEqual(len(reminder_items(response)), 1)
        self.assertIs(response['contentItems'][-1], reminder_items(response)[0])
        self.assertFalse(self.h.decision_reminder_due)
        self.assertEqual(len(self.dispatched), 1)


if __name__ == '__main__':
    unittest.main()
