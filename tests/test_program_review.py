"""Real isolated processes retain their stack while the main actor reviews."""
from program_fixture import program_services
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from auriga.runtime.programs.runner import ProgramRunner, program_remaining, review_wait
from auriga.runtime.programs.program_result import program_result
from auriga.core.storage.store import Store
from auriga.core.tool_output import ToolOutput, ToolResponse, wire_response
from support.host_capabilities import program_isolation, require


class ProgramReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name)/'run', dict(run_id='review',episode_id='ep',
            env_instance_id='env',env_generation=1))
        self.addCleanup(self.store.close)
        self.calls = []
        self.plan_count = 0
        self.action_count = 0
        self.stale = False
        self.plan_cursor = 0
        self.plan_revision = 0
        self.harness = SimpleNamespace(store=self.store,lock=threading.RLock(),
            cancel=threading.Event(),active_context=None,dispatch=self.dispatch)
        self.runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(self.runner.close)
        self.identity = dict(thread_id='thread',turn_id='start',tool_call_id='start-call')

    @staticmethod
    def response(body):
        kind = 'action' if 'receipt' in body else 'plan' if 'plan_id' in body else 'data'
        return ToolOutput(body, kind=kind).response()

    def dispatch(self, name, args, correlation):
        self.calls.append((name,dict(args),dict(correlation)))
        if name == 'robot_wait':
            self.action_count += 1
            return self.response(dict(receipt=dict(action_id='wait-action',status='completed',
                result_observation_id='obs-after-wait',executed_native_steps=args['native_steps'])))
        if name != 'robot_action':
            return self.response(dict(test=True))
        mode = args.get('mode','preview')
        if mode == 'preview':
            self.plan_count += 1
            result = self.response(dict(status='planned',plan_id='plan-'+str(self.plan_count),
                observation_id=args['observation_id'],trajectory_evidence={'stages':args['stages']}))
            result.content.append(dict(type='inputImage',imageUrl='data:image/png;base64,AA=='))
            return result
        if mode == 'inspect':
            return self.response(dict(status='planned',plan_id=args['plan_id'],
                executed_batch_native_steps=self.plan_cursor,plan_revision=self.plan_revision,
                trajectory_evidence={'inspected':True}))
        self.action_count += 1
        status = 'rejected' if self.stale else 'completed'
        return self.response(dict(plan_id=args.get('plan_id'),receipt=dict(action_id='action-'+str(self.action_count),
            status=status,reason='stale_plan' if self.stale else None,
            result_observation_id=None if self.stale else 'obs-final-'+str(self.action_count),
            executed_native_steps=0 if self.stale else 2,stop_confirmed=True)))

    def start(self, source, **args):
        return self.runner.run(dict(source=source,timeout_s=10,**args),self.identity)

    def resume(self, job, decision='approve', **extra):
        return self.runner.run(dict(operation='resume',job_id=job['job_id'],
            review_id=job['pending_review']['review_id'],decision=decision,**extra),
            dict(self.identity,turn_id='review',tool_call_id='resume-call'))

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_real_pause_preserves_stack_and_stops_all_program_threads(self):
        source = '''import json, threading, time
from pathlib import Path
from auriga_client import AurigaClient
client=AurigaClient()
def heartbeat():
    while True:
        Path('heartbeat').write_text(str(time.monotonic()))
        time.sleep(.005)
threading.Thread(target=heartbeat,daemon=True).start()
retained={'value':73}
result=client.action([{'id':'move','arm':'left','kind':'gripper','goal_normalized':1}],observation_id='obs-start')
assert retained['value']==73
print(json.dumps({'retained':retained,'result':result}))
'''
        began=time.monotonic();job=self.start(source)
        self.assertLess(time.monotonic()-began,3)
        self.assertEqual(job['status'],'awaiting_review')
        self.assertEqual(self.action_count,0)
        entry=self.runner.entries[job['job_id']]
        heartbeat=entry['workspace']/'heartbeat'
        first=heartbeat.read_text();remaining=program_remaining(entry)
        time.sleep(.1)
        self.assertEqual(heartbeat.read_text(),first)
        self.assertAlmostEqual(program_remaining(entry),remaining,places=3)
        self.assertEqual(self.runner.review_content(job['job_id'])[-1]['type'],'inputImage')
        presented=program_result(self.store,job)
        self.assertEqual(presented['resume']['operation'],'resume')
        self.assertNotIn('data:image',json.dumps(presented))
        final=self.resume(job)
        self.assertEqual(final['status'],'succeeded',final.get('stderr_text'))
        self.assertEqual(self.action_count,1)
        self.assertEqual([args['mode'] for name,args,_ in self.calls if name=='robot_action'],['preview','inspect','execute'])
        self.assertEqual(self.calls[-1][1]['expected_cursor'],0)
        self.assertEqual(self.calls[-1][1]['expected_revision'],0)
        self.assertIn('"value": 73',final['stdout_text'])
        self.assertEqual(final['action_summary']['status_counts'],{'completed':1})
        self.assertEqual(final['final_observation_id'],'obs-final-1')
        self.assertGreater(final['review_wait_s'],.09)
        self.assertTrue(final['stop_confirmed'])
        repeated=self.resume(job)
        self.assertEqual(repeated,final)
        self.assertEqual(self.action_count,1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_next_action_requires_new_review_and_stale_plan_is_not_replanned(self):
        job=self.start("from auriga_client import AurigaClient\nc=AurigaClient()\nfor i in range(2):\n c.action([{'id':'s','arm':'left','kind':'gripper','goal_normalized':1}],observation_id='obs-start')")
        second=self.resume(job)
        self.assertEqual(second['status'],'awaiting_review')
        self.assertNotEqual(second['pending_review']['review_id'],job['pending_review']['review_id'])
        self.assertEqual(self.action_count,1)
        replay=self.resume(job)
        self.assertEqual(replay['pending_review']['review_id'],second['pending_review']['review_id'])
        self.assertEqual(self.action_count,1)
        self.stale=True
        final=self.resume(second)
        self.assertEqual(final['action_summary']['status_counts'],{'completed':1,'rejected':1})
        self.assertEqual(self.plan_count,2)
        self.assertIsNone(final['final_observation_id'])

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_plan_advanced_by_another_caller_cannot_consume_previous_approval(self):
        job=self.start("from auriga_client import AurigaClient\nAurigaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')")
        self.plan_cursor=4
        final=self.resume(job)
        self.assertEqual(self.action_count,0)
        self.assertEqual(final['action_summary']['action_calls'],0)
        self.assertEqual(final['call_summary']['recent_errors'][0]['reason'],
                         'program_action_plan_changed_since_review')
        self.assertEqual(self.plan_count,1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_zero_step_checkpoint_revision_invalidates_old_approval(self):
        job=self.start("from auriga_client import AurigaClient\nAurigaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')")
        self.plan_revision=1
        final=self.resume(job)
        self.assertEqual(self.action_count,0)
        self.assertEqual(final['call_summary']['recent_errors'][0]['reason'],
                         'program_action_plan_changed_since_review')

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_direct_authorization_is_only_for_one_program_and_cannot_be_added_on_resume(self):
        source="from auriga_client import AurigaClient\nAurigaClient().action([{'id':'s','arm':'left','kind':'gripper','goal_normalized':1}],observation_id='obs-start')"
        direct=self.start(source,allow_direct_actions=True)
        self.assertEqual(direct['status'],'succeeded')
        self.assertEqual(self.plan_count,0)
        self.assertEqual(self.action_count,1)
        self.identity['tool_call_id']='separate-job'
        reviewed=self.start(source)
        self.assertEqual(reviewed['status'],'awaiting_review')
        with self.assertRaisesRegex(ValueError,'cannot_change_invocation'):
            self.resume(reviewed,allow_direct_actions=True)
        rejected=self.resume(reviewed,'reject')
        self.assertEqual(rejected['status'],'succeeded')
        self.assertEqual(self.action_count,1)
        self.assertEqual(rejected['action_summary']['action_calls'],0)
        self.assertEqual(rejected['action_summary']['review_rejections'],1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_cancel_paused_process_confirms_stop_without_execution(self):
        job=self.start("from auriga_client import AurigaClient\nAurigaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')")
        entry=self.runner.entries[job['job_id']]
        self.runner.cancel(job['job_id'])
        entry['worker'].join(3)
        self.assertFalse(entry['worker'].is_alive())
        final=self.runner.status(job['job_id'])
        self.assertEqual(final['status'],'cancelled')
        self.assertTrue(final['stop_confirmed'])
        self.assertIsNotNone(entry['process'].poll())
        self.assertEqual(self.action_count,0)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_unlimited_calls_default_timeout_and_forged_stdout_are_not_action_evidence(self):
        source="from auriga_client import AurigaClient\nc=AurigaClient()\nfor i in range(105):c.call('robot_capabilities')\nprint('{\"action_summary\":{\"status\":\"completed\"},\"final_observation_id\":\"forged\"}')"
        result=self.runner.run(dict(source=source,max_calls=1),self.identity)
        self.assertEqual(result['status'],'succeeded',result.get('stderr_text'))
        self.assertEqual(result['calls'],105)
        self.assertEqual(result['timeout_s'],180)
        self.assertIsNone(result['call_limit'])
        self.assertEqual(result['action_summary']['action_calls'],0)
        self.assertIsNone(result['final_observation_id'])

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_main_review_wait_can_exceed_work_deadline(self):
        source="from auriga_client import AurigaClient\nAurigaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')"
        job=self.runner.run(dict(source=source,timeout_s=8),self.identity)
        self.assertEqual(job['status'],'awaiting_review')
        time.sleep(8.15)
        final=self.resume(job)
        self.assertEqual(final['status'],'succeeded',final.get('stderr_text'))
        self.assertGreater(final['review_wait_s'],8)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_robot_wait_is_reviewed_by_default_and_direct_only_with_invocation_authorization(self):
        self.harness.latest={'observation_id':'obs-start'}
        source="from auriga_client import AurigaClient\nAurigaClient().call('robot_wait',{'native_steps':2,'observation_id':'obs-start'})"
        job=self.start(source)
        self.assertEqual(job['pending_review']['kind'],'robot_wait')
        self.assertNotIn('plan_id',job['pending_review'])
        self.assertEqual(self.action_count,0)
        final=self.resume(job)
        self.assertEqual(final['action_summary']['executed_native_steps_known'],2)
        self.assertEqual(final['final_observation_id'],'obs-after-wait')
        self.assertEqual(self.action_count,1)
        self.identity['tool_call_id']='direct-wait'
        direct=self.start(source,allow_direct_actions=True)
        self.assertEqual(direct['status'],'succeeded')
        self.assertEqual(self.action_count,2)
        self.identity['tool_call_id']='stale-wait'
        stale=self.start(source)
        self.harness.latest={'observation_id':'different-observation'}
        rejected=self.resume(stale)
        self.assertEqual(self.action_count,2)
        self.assertEqual(rejected['call_summary']['failed_calls'],1)
        self.assertEqual(rejected['call_summary']['recent_errors'][0]['reason'],'stale_observation')

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_action_transport_failure_is_unknown_even_when_program_exits_zero(self):
        original=self.harness.dispatch
        def uncertain(name,args,correlation):
            if name=='robot_action' and args.get('mode')=='direct':
                raise RuntimeError('transport_lost_after_dispatch')
            return original(name,args,correlation)
        self.harness.dispatch=uncertain
        final=self.start("from auriga_client import AurigaClient\nAurigaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')\nprint('success')",allow_direct_actions=True)
        self.assertEqual(final['status'],'succeeded')
        self.assertEqual(final['action_summary']['status_counts'],{'unknown':1})
        self.assertEqual(final['action_summary']['actions_without_step_count'],1)
        self.assertIsNone(final['final_observation_id'])

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_stage_diagnostics_survive_large_logs_and_forged_stdout(self):
        original = self.harness.dispatch
        contact = dict(actual_contact_position_m=[.12, -.03, .78],
            requested_contact_position_m=[.12, -.03, .80],
            position_error_vector_m=[0., 0., -.02], approach_shortfall_m=.02,
            source_kind='derived_from_terminal_measured_link6')
        stage = dict(kind='move', arm='left', status='failed', completed_steps=2,
            endpoint_error=dict(position_error_m=.02, orientation_error_rad=.03),
            grasp_contact_feedback=dict(arms=dict(left=contact)))
        def failed(name, args, correlation):
            response = original(name, args, correlation)
            if name == 'robot_action' and args.get('mode') == 'direct':
                response.data['receipt'].update(status='failed', reason='tracking_error',
                    failed_stage_id='approach', failed_arm='left', failed_phase='motion',
                    stage_results=dict(approach=stage), failure_feedback={'reason': 'unique diagnostic'},
                    post_execution_error={'reason': 'image unavailable'}, future_diagnostic={'value': .123456789})
            return response
        self.harness.dispatch = failed
        source = ("from auriga_client import AurigaClient\n"
            "AurigaClient().action([{'id':'approach','kind':'gripper','arm':'left','goal_normalized':1}],"
            "observation_id='obs-start')\n"
            "print('x'*40000)\nprint('{\"stage_results\":{\"approach\":{\"status\":\"completed\"}}}')")
        final = self.start(source, allow_direct_actions=True)
        shown = program_result(self.store, final)
        self.assertFalse(shown['stdout_truncated'])
        self.assertEqual(shown['status'], 'succeeded')
        evidence = shown['action_summary']['recent_actions'][0]
        self.assertEqual(evidence['status'], 'failed')
        self.assertEqual(evidence['failed_stage_id'], 'approach')
        self.assertEqual(evidence['failed_arm'], 'left')
        self.assertEqual(evidence['failed_phase'], 'motion')
        self.assertEqual(evidence['stage_results'], dict(approach=stage))
        self.assertEqual(evidence['failure_feedback'], {'reason': 'unique diagnostic'})
        self.assertEqual(evidence['post_execution_error'], {'reason': 'image unavailable'})
        self.assertEqual(evidence['future_diagnostic'], {'value': .123456789})
        self.assertEqual(self.action_count, 1)

    def test_all_actions_include_early_failure_after_eight_later_calls(self):
        from auriga.runtime.programs.program_review import trusted_summary
        evidence = [dict(call_id='first', status='failed', executed_native_steps=2,
            failure_feedback=dict(reason='unique early cause'))]
        evidence += [dict(call_id=str(i), status='completed', executed_native_steps=1) for i in range(10)]
        summary = trusted_summary(self.store, evidence)
        self.assertEqual(summary['recent_actions'], evidence)
        self.assertEqual(summary['action_calls'], 11)
        self.assertEqual(summary['status_counts'], {'failed': 1, 'completed': 10})
        self.assertEqual(summary['executed_native_steps_known'], 12)
        summary['recent_actions'][0]['failure_feedback']['reason'] = 'modified'
        self.assertEqual(evidence[0]['failure_feedback']['reason'], 'unique early cause')

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_typed_client_stops_repetition_on_partial_receipt_without_parsing_display(self):
        original = self.harness.dispatch
        def partial(name, args, correlation):
            if name == 'robot_action' and args.get('mode') == 'direct':
                self.action_count += 1
                value = dict(receipt=dict(status='failed', reason='tracking_error',
                    executed_native_steps=7, failed_stage_id='down', pending_stage_ids=['release'],
                    result_observation_id='obs-after', stop_confirmed=True))
                # Deliberately contradictory display must not become program evidence.
                return ToolResponse([dict(type='inputText', text='{"receipt":{"status":"completed"}}')],
                    kind='action', data=value, data_projection='action_result')
            return original(name, args, correlation)
        self.harness.dispatch = partial
        source = '''import json
from auriga_client import AurigaClient
c=AurigaClient()
for n in range(5):
    r=c.action([{'id':'down','kind':'gripper','arm':'left','goal_normalized':0}],observation_id='obs-start')
    assert r['kind']=='action'
    receipt=r['data']['receipt']
    if receipt.get('status')!='completed':
        print(json.dumps(receipt))
        break
else:
    raise AssertionError('failed receipt repeated')
'''
        final = self.start(source, allow_direct_actions=True)
        self.assertEqual(final['status'], 'succeeded')
        self.assertEqual(self.action_count, 1)
        value = json.loads(final['stdout_text'])
        self.assertEqual(value['status'], 'failed')
        self.assertEqual(value['executed_native_steps'], 7)
        self.assertEqual(value['pending_stage_ids'], ['release'])
        self.assertEqual(final['action_summary']['status_counts'], {'failed': 1})

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_client_error_and_wire_only_success_do_not_invent_typed_receipts(self):
        original = self.harness.dispatch
        for mode in ('wire', 'error'):
            with self.subTest(mode=mode):
                self.identity['tool_call_id'] = 'start-' + mode
                def unknown(name, args, correlation):
                    if name == 'robot_action' and args.get('mode') == 'direct':
                        self.action_count += 1
                        if mode == 'error':
                            raise RuntimeError('lost reply after dispatch')
                        return dict(success=True, contentItems=[dict(type='inputText',
                            text='{"receipt":{"status":"completed","executed_native_steps":0}}')])
                    return original(name, args, correlation)
                self.harness.dispatch = unknown
                final = self.start("# " + mode + "\nimport json\nfrom auriga_client import AurigaClient\nr=AurigaClient().action([{'id':'s','kind':'gripper','arm':'left','goal_normalized':0}],observation_id='obs-start')\nprint(json.dumps({'kind':r['kind'],'data':r['data'],'success':r['success']}))", allow_direct_actions=True)
                value = json.loads(final['stdout_text'])
                self.assertEqual(value['kind'], 'error' if mode == 'error' else 'untyped')
                if mode == 'wire':
                    self.assertIsNone(value['data'])
                else:
                    self.assertFalse(value['success'])
                    self.assertIn('lost reply', value['data']['error'])
                self.assertEqual(final['action_summary']['status_counts'], {'unknown': 1})
        self.assertEqual(self.action_count, 2)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_review_and_action_reply_artifacts_follow_producer_projection(self):
        typed = []
        def response(body):
            value = deepcopy(body)
            value.update(schema_version='0.1', run_id='fixture', env_generation=1)
            if 'receipt' in value:
                value['receipt'].update(schema_version='0.1', versions={'worker': 'build'},
                    failure_feedback=dict(schema_version='diagnostic-v1', reason='retained diagnostic',
                                          error_m=0.123456789012345))
                kind, projection = 'action', 'action_result'
            else:
                value['plan_revision'] = 0
                kind, projection = 'plan', 'action_plan'
            value['diagnostics'] = dict(version='business-version', revision=9)
            result = ToolOutput(value, kind=kind, projection=projection).response()
            typed.append((result, deepcopy(value)))
            return result
        self.response = response
        job = self.start("from auriga_client import AurigaClient\nAurigaClient().action([{'id':'s','arm':'left','kind':'gripper','goal_normalized':1}],observation_id='obs-start')")
        self.assertEqual(job['status'], 'awaiting_review')
        preview = json.loads(self.store.resolve_artifact(job['pending_review']['preview_ref'])[1])
        shown_preview = json.loads(preview['contentItems'][0]['text'])
        self.assertEqual(shown_preview['plan_revision'], 0)
        self.assertEqual(preview['contentItems'][1], dict(type='inputImage', imageUrl='data:image/png;base64,AA=='))
        for key in ('schema_version', 'run_id', 'env_generation'):
            self.assertNotIn(key, shown_preview)
        final = self.resume(job)
        self.assertEqual(final['status'], 'succeeded')
        evidence = final['action_summary']['recent_actions'][0]
        reply = json.loads(self.store.resolve_artifact(evidence['reply_ref'])[1])
        shown_action = json.loads(reply['contentItems'][0]['text'])
        for key in ('schema_version', 'run_id', 'env_generation'):
            self.assertNotIn(key, shown_action)
        self.assertNotIn('versions', shown_action['receipt'])
        self.assertEqual(shown_action['receipt']['failure_feedback'],
                         dict(reason='retained diagnostic', error_m=0.123456789012345))
        self.assertEqual(shown_action['receipt']['status'], 'completed')
        self.assertEqual(shown_action['receipt']['executed_native_steps'], 2)
        self.assertEqual(shown_action['diagnostics'], dict(version='business-version', revision=9))
        self.assertEqual(self.action_count, 1)
        for response, original in typed:
            self.assertEqual(response.data, original)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_direct_planning_failure_records_zero_execution_and_keeps_last_observation(self):
        original = self.harness.dispatch
        feedback = dict(candidates=[dict(constraint_witness=dict(
            collision_model_witnesses=[dict(trajectory_sample_index=i) for i in range(12)]))])
        def rejected(name, args, correlation):
            if name == 'robot_observe':
                return ToolResponse([], kind='observation', data={'observation_id': 'obs-start'})
            if name == 'robot_action' and args.get('mode') == 'direct':
                return ToolResponse([], kind='plan', data=dict(status='failed', reason='path_infeasible',
                    failed_stage_id='hold', failed_arm='left', physics_advanced=False, executed_native_steps=0,
                    failure_feedback=feedback, future_planning_diagnostic=dict(reason='different approach needed')))
            return original(name, args, correlation)
        self.harness.dispatch = rejected
        final = self.start("from auriga_client import AurigaClient\nc=AurigaClient()\nc.call('robot_observe')\nc.action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')\nprint('motion completed')", allow_direct_actions=True)
        summary = final['action_summary']
        self.assertEqual(final['status'], 'succeeded')
        self.assertEqual(self.action_count, 0)
        self.assertEqual(summary['status_counts'], {'not_executed': 1})
        self.assertEqual(summary['executed_native_steps_known'], 0)
        self.assertEqual(summary['actions_without_step_count'], 0)
        evidence = summary['recent_actions'][0]
        self.assertEqual(evidence['reason'], 'path_infeasible')
        self.assertEqual(evidence['phase'], 'planning')
        self.assertEqual(evidence['failure_feedback'], feedback)
        self.assertEqual(evidence['future_planning_diagnostic'], dict(reason='different approach needed'))
        self.assertIsNone(evidence['action_id'])
        self.assertEqual(final['final_observation_id'], 'obs-start')

    def test_unproven_zero_execution_stays_unknown(self):
        from auriga.runtime.programs.program_review import action_evidence
        plan = dict(status='failed', reason='path_infeasible', physics_advanced=False, executed_native_steps=0)
        replies = [wire_response(ToolResponse([], kind='plan', data=plan)),
                   ToolResponse([], kind='data', data=plan)]
        invalid = ToolResponse([], kind='plan', data=plan)
        invalid.contract_error = {'code': 'invalid_tool_result'}
        replies.append(invalid)
        for change in ({'physics_advanced': None}, {'physics_advanced': 0},
                       {'executed_native_steps': None}, {'executed_native_steps': False},
                       {'executed_native_steps': 1}, {'status': 'planned'}):
            replies.append(ToolResponse([], kind='plan', data={**plan, **change}))
        for response in replies:
            with self.subTest(response=response):
                evidence = action_evidence(self.store, 'robot_action', {'mode': 'direct'}, response, 'call')
                self.assertEqual(evidence['status'], 'unknown')
                self.assertNotIn('executed_native_steps', evidence)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_settling_steps_and_latest_observation_are_in_trusted_program_summary(self):
        original=self.harness.dispatch
        def settled(name,args,correlation):
            response=original(name,args,correlation)
            if name=='robot_action' and args.get('mode')=='direct':
                body=dict(response.data)
                body.update(current_observation_id='obs-after-settling',
                    settling=dict(status='low_visible_motion',executed_native_steps=3))
                return self.response(body)
            return response
        self.harness.dispatch=settled
        final=self.start("from auriga_client import AurigaClient\nAurigaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')",allow_direct_actions=True)
        summary=final['action_summary']
        self.assertEqual(summary['executed_native_steps_known'],5)
        self.assertEqual(summary['action_native_steps_known'],2)
        self.assertEqual(summary['settling_native_steps_known'],3)
        self.assertEqual(final['final_observation_id'],'obs-after-settling')
        self.assertEqual(summary['recent_actions'][0]['motion_observation_id'],'obs-final-1')

    def test_setup_failure_returns_without_leaving_an_unresumable_job(self):
        with patch('auriga.runtime.programs.runner.socket.socketpair',side_effect=OSError('fixture socket failure')):
            final=self.start('print(1)')
        self.assertEqual(final['status'],'failed')
        self.assertTrue(final['stop_confirmed'])
        self.assertIn('fixture socket failure',final['reason'])

    def test_real_harness_unified_cached_plan_review_and_final_main_image_delivery(self):
        from test_unified_actions import UnifiedActionTests
        fixture=UnifiedActionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        plan=fixture.preview()
        fixture.native_execution()
        runner=fixture.h.programs=ProgramRunner(program_services(fixture.h))
        self.addCleanup(runner.close)
        source=("from auriga_client import AurigaClient\n"
                "AurigaClient().call('robot_action',"+repr(dict(mode='execute',plan_id=plan['plan_id']))+")")
        reviewed=fixture.h.dispatch('program_run',dict(source=source),self.identity)
        job=json.loads(reviewed['contentItems'][0]['text'])
        self.assertEqual(job['status'],'awaiting_review',job)
        self.assertEqual(job['pending_review']['plan_id'],plan['plan_id'])
        pending_status=fixture.h.dispatch('program_status',dict(job_id=job['job_id']),
            dict(self.identity,tool_call_id='pending-status'))
        self.assertEqual(json.loads(pending_status['contentItems'][0]['text'])['status'],'awaiting_review')
        fixture.robot.execute.assert_not_called()
        completed=fixture.h.dispatch('program_run',dict(operation='resume',job_id=job['job_id'],
            review_id=job['pending_review']['review_id'],decision='approve'),
            dict(self.identity,turn_id='resume',tool_call_id='resume'))
        final=json.loads(completed['contentItems'][0]['text'])
        self.assertEqual(final['status'],'succeeded',final)
        self.assertEqual(final['action_summary']['status_counts'],{'completed':1})
        self.assertEqual(final['final_observation_id'],fixture.h.latest['observation_id'])
        self.assertTrue(any(item['type']=='inputImage' for item in completed['contentItems']))
        final_status=fixture.h.dispatch('program_status',dict(job_id=final['job_id'],detail='full'),
            dict(self.identity,tool_call_id='final-status'))
        self.assertEqual([part for part in final_status['contentItems'] if part['type']=='inputImage'],
            [part for part in completed['contentItems'] if part['type']=='inputImage'])
        self.assertEqual(fixture.robot.preview_batch.call_count,1)
        fixture.robot.execute.assert_called_once()
        self.assertEqual(fixture.robot.execute.call_args.args[0]['batch_expected_cursor'],0)
        self.assertEqual(fixture.robot.execute.call_args.args[0]['batch_expected_revision'],0)

    def test_program_implicit_action_deadline_fits_budget_without_changing_explicit_or_reviewed_values(self):
        from test_unified_actions import UnifiedActionTests
        for direct in (False, True):
            for explicit in (False, True):
                with self.subTest(direct=direct, explicit=explicit):
                    fixture = UnifiedActionTests()
                    fixture.setUp()
                    self.addCleanup(fixture.doCleanups)
                    fixture.preview()
                    fixture.robot.preview_batch.reset_mock()
                    fixture.native_execution()
                    runner = fixture.h.programs = ProgramRunner(program_services(fixture.h))
                    self.addCleanup(runner.close)
                    args = dict(fixture.request(), mode='direct')
                    if explicit:
                        args['timeout_s'] = 120
                    source = ("from auriga_client import AurigaClient\n"
                              "AurigaClient().call('robot_action'," + repr(args) + ")")
                    with patch.object(runner, 'admit_action', wraps=runner.admit_action) as admission:
                        job = runner.run(dict(source=source, timeout_s=120,
                            allow_direct_actions=direct), self.identity)
                        if not direct:
                            self.assertEqual(job['status'], 'awaiting_review', job)
                            job = runner.run(dict(operation='resume', job_id=job['job_id'],
                                review_id=job['pending_review']['review_id'], decision='approve'), self.identity)
                        self.assertEqual(job['status'], 'succeeded', job)
                        request = fixture.store.records('action_request')[-1]
                        deadline = request['timeout_s']
                        self.assertEqual(admission.call_args.args[1], deadline)
                        if explicit:
                            self.assertEqual(deadline, 120)
                            fixture.robot.execute.assert_not_called()
                            receipt = fixture.store.records('action_receipt')[-1]
                            self.assertEqual(receipt['reason'], 'program_time_insufficient_for_action')
                        else:
                            self.assertGreater(deadline, 100)
                            self.assertLessEqual(deadline, 114)
                            fixture.robot.execute.assert_called_once()
                            self.assertEqual(fixture.robot.execute.call_args.args[0]['timeout_s'], deadline)
                            if not direct:
                                self.assertEqual(fixture.robot.preview_batch.call_args.args[0]['timeout_s'], deadline)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_implicit_program_deadline_rejects_insufficient_time_before_creating_a_plan(self):
        source = ("from auriga_client import AurigaClient\n"
                  "AurigaClient().call('robot_action', {'observation_id':'obs', 'mode':'direct',"
                  "'stages':[{'id':'close','kind':'gripper','arm':'left','goal_normalized':0.0}]})")
        with patch('auriga.runtime.programs.runner.program_remaining', return_value=6.0):
            job = self.start(source)
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertEqual(job['call_summary']['recent_errors'][0]['reason'], 'program_time_insufficient_for_action')
        self.assertEqual(self.calls, [])

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_read_only_preview_is_available_without_an_action_time_reserve(self):
        source = ("from auriga_client import AurigaClient\n"
                  "AurigaClient().call('robot_action', {'observation_id':'obs', 'mode':'preview',"
                  "'stages':[{'id':'close','kind':'gripper','arm':'left','goal_normalized':0.0}]})")
        with patch('auriga.runtime.programs.runner.program_remaining', return_value=6.0):
            job = self.start(source)
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertEqual(self.plan_count, 1)
        self.assertEqual(self.action_count, 0)
        self.assertEqual(job['call_summary']['recent_errors'], [])

    def test_reviewed_implicit_deadline_is_not_lowered_when_planning_consumes_the_reserve(self):
        from test_unified_actions import UnifiedActionTests
        fixture = UnifiedActionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.preview()
        fixture.native_execution()
        runner = fixture.h.programs = ProgramRunner(program_services(fixture.h))
        self.addCleanup(runner.close)
        args = dict(fixture.request(), mode='direct')
        source = ("from auriga_client import AurigaClient\n"
                  "AurigaClient().call('robot_action'," + repr(args) + ")")
        job = runner.run(dict(source=source, timeout_s=120), self.identity)
        self.assertEqual(job['status'], 'awaiting_review', job)
        deadline = fixture.robot.preview_batch.call_args.args[0]['timeout_s']
        with patch('auriga.runtime.programs.runner.program_remaining', return_value=deadline + 4):
            runner.run(dict(operation='resume', job_id=job['job_id'],
                review_id=job['pending_review']['review_id'], decision='approve'), self.identity)
        self.assertEqual(fixture.store.records('action_request')[-1]['timeout_s'], deadline)
        self.assertEqual(fixture.store.records('action_receipt')[-1]['reason'], 'program_time_insufficient_for_action')
        fixture.robot.execute.assert_not_called()

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_wait_defaults_fit_program_budget_and_review_preserves_explicit_deadlines(self):
        for explicit in (None, 30.):
            with self.subTest(explicit=explicit):
                args = dict(observation_id='obs', native_steps=1)
                if explicit is not None:
                    args['timeout_s'] = explicit
                source = ("from auriga_client import AurigaClient\nclient=AurigaClient()\n"
                          "client.call('robot_wait'," + repr(args) + ")")
                job = self.runner.run(dict(source=source, timeout_s=20),
                    dict(self.identity, tool_call_id='wait-' + str(explicit)))
                self.assertEqual(job['status'], 'awaiting_review', job)
                saved = job['pending_review']['preview']['request']['timeout_s']
                if explicit is None:
                    self.assertGreater(saved, .1)
                    self.assertLessEqual(saved, 14)
                else:
                    self.assertEqual(saved, explicit)
                final = self.resume(job)
                self.assertEqual(final['status'], 'succeeded', final)
                self.assertEqual(self.calls[-1][1]['timeout_s'], saved)

    def test_real_harness_checkpoint_advanced_during_review_rejects_previous_approval(self):
        from test_unified_actions import UnifiedActionTests
        fixture=UnifiedActionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        plan=fixture.preview()
        fixture.native_execution()
        fixture.robot.execute.return_value.update(batch_status='awaiting_observation',batch_cursor=1,batch_revision=1,
            completed_native_steps=1,executed_native_steps=1,pending_stage_ids=['reach'])
        runner=ProgramRunner(program_services(fixture.h))
        self.addCleanup(runner.close)
        source=("from auriga_client import AurigaClient\n"
                "AurigaClient().call('robot_action',"+repr(dict(mode='execute',plan_id=plan['plan_id']))+")")
        job=runner.run(dict(source=source,timeout_s=10),self.identity)
        self.assertEqual(job['status'],'awaiting_review')
        fixture.h.dispatch('robot_action',dict(mode='execute',plan_id=plan['plan_id']),
            dict(self.identity,tool_call_id='another-main-action'))
        final=runner.run(dict(operation='resume',job_id=job['job_id'],
            review_id=job['pending_review']['review_id'],decision='approve'),self.identity)
        self.assertEqual(final['call_summary']['recent_errors'][0]['reason'],
                         'program_action_plan_changed_since_review')
        self.assertEqual(final['action_summary']['action_calls'],0)
        fixture.robot.execute.assert_called_once()
        self.assertEqual(fixture.robot.preview_batch.call_count,1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_legacy_motion_and_nested_authorization_cannot_bypass_review(self):
        source="""from auriga_client import AurigaClient
c=AurigaClient()
for name in ('robot_move_grasp','robot_gripper','robot_policy_step','depth_move','program_run'):
    try:c.call(name,{'allow_direct_actions':True})
    except RuntimeError:pass
    else:raise AssertionError(name)
"""
        final=self.start(source)
        self.assertEqual(final['status'],'succeeded',final.get('stderr_text'))
        self.assertEqual(self.calls,[])
        self.assertEqual(self.action_count,0)

    def test_real_depth_candidate_cannot_bypass_review_through_legacy_depth_move(self):
        from test_depth_integration import DepthIntegrationTest
        fixture=DepthIntegrationTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        harness=fixture.harness()
        target=fixture.query(harness,kind='register_target',
            observation_id=harness.latest['observation_id'],camera_id='head',
            roi=[36,36,24,24],arm='left')
        candidates=fixture.query(harness,kind='grasp_candidates',target_id=target['target_id'])
        candidate=candidates['value']['candidates'][0]
        runner=ProgramRunner(program_services(harness))
        self.addCleanup(runner.close)
        arguments=dict(candidate_id=candidate['candidate_id'],arm='left',stage='approach',timeout_s=1)
        source=("from auriga_client import AurigaClient\n"
                "try: AurigaClient().call('depth_move',"+repr(arguments)+")\n"
                "except RuntimeError as error:\n"
                " assert str(error)=='program_action_requires_robot_action'\n"
                "else: raise AssertionError('legacy depth execution was allowed')\n")
        final=runner.run(dict(source=source,timeout_s=10),self.identity)
        self.assertEqual(final['status'],'succeeded',final.get('stderr_text'))
        self.assertEqual(fixture.robot.executions,0)
        self.assertEqual(final['action_summary']['action_calls'],0)
        self.assertEqual(final['review_decisions'],{})

    def test_review_wait_excludes_only_own_wall_time_and_cancellation_stays_active(self):
        now=[100.0];entry=dict(cancel=threading.Event(),deadline=110.0)
        other=dict(cancel=threading.Event(),deadline=110.0)
        with patch('auriga.runtime.programs.runner.time.monotonic',side_effect=lambda:now[0]):
            with review_wait(entry):
                now[0]=1000.0
                self.assertEqual(program_remaining(entry),10)
                self.assertEqual(program_remaining(other),-890)
            self.assertEqual(program_remaining(entry),10)
            self.assertEqual(entry['review_wait_s'],900)
            entry['cancel'].set()
            with self.assertRaisesRegex(ValueError,'channel_expired'):
                with review_wait(entry):pass


if __name__=='__main__':unittest.main()
