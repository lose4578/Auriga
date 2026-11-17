"""Real isolated subprocess/client checks; synthetic robot, no GPU or model."""
from program_fixture import program_services
import base64
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from auriga.runtime.programs.runner import ProgramRunner, model_wait, program_remaining
from auriga.core.storage.store import Store
from support.host_capabilities import program_isolation, require

class ProgramCheck(unittest.TestCase):
    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_opencv_uses_one_thread_inside_bounded_program(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            runner = ProgramRunner(program_services(SimpleNamespace(store=store)))
            self.addCleanup(runner.close)
            source = '''import cv2, numpy as np, resource
assert cv2.getNumThreads() == 1, cv2.getNumThreads()
assert resource.getrlimit(resource.RLIMIT_AS)[0] == 1024**3
image = np.zeros((480, 640, 3), dtype=np.uint8)
image[240:280, 300:350] = (0, 0, 255)
ok, encoded = cv2.imencode('.png', image)
assert ok
decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
count, labels, stats, centers = cv2.connectedComponentsWithStats((gray > 0).astype(np.uint8))
assert count == 2 and stats[1, cv2.CC_STAT_AREA] == 2000
scores = cv2.matchTemplate(gray, gray[230:290, 290:360], cv2.TM_SQDIFF)
assert cv2.minMaxLoc(scores)[2] == (290, 230)
assert np.allclose(np.linalg.svd(np.eye(3), compute_uv=False), 1)
print('opencv_and_linalg_completed')
'''
            result = runner.run(dict(source=source, timeout_s=30),
                                dict(thread_id='thread', turn_id='turn', tool_call_id='opencv'))
            self.assertEqual(result['status'], 'succeeded', result.get('stderr_text'))
            self.assertIn('opencv_and_linalg_completed', result['stdout_text'])
            self.assertNotIn("Can't spawn new thread", result.get('stderr_text', ''))

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_real_client_cannot_nest_manipulation_or_other_program_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            calls = []
            harness = SimpleNamespace(store=store, lock=threading.RLock(), cancel=threading.Event(), active_context=None)
            def dispatch(name, args, correlation):
                calls.append(name)
                return dict(success=True, contentItems=[])
            harness.dispatch = dispatch
            runner = ProgramRunner(program_services(harness))
            self.addCleanup(runner.close)
            source = '''import json
from auriga_client import AurigaClient
client = AurigaClient()
denied = []
for tool in ("program_run", "program_status", "skill_run", "manipulation_run"):
    try:
        client.call(tool, {"parameters": {"kind": "help"}})
    except RuntimeError as error:
        assert str(error) == "program_tool_not_permitted", str(error)
        denied.append(tool)
    else:
        raise AssertionError("nested program tool was admitted: " + tool)
client.call("robot_capabilities")
print(json.dumps(denied))
'''
            result = runner.run(dict(source=source, timeout_s=5, max_calls=1),
                                dict(thread_id='thread', turn_id='turn', tool_call_id='nested'))
            self.assertEqual(result['status'], 'succeeded', result.get('stderr_text'))
            self.assertEqual(json.loads(result['stdout_text']),
                             ['program_run', 'program_status', 'skill_run', 'manipulation_run'])
            self.assertEqual(calls, ['robot_capabilities'])
            self.assertEqual(result['calls'], 1)

    def test_only_own_sol_wait_pauses_work_clock_and_restores_time_after_failure(self):
        clock = [100.0]
        entry = dict(cancel=threading.Event(), deadline=105.0)
        unrelated = dict(cancel=threading.Event(), deadline=105.0)
        with patch("auriga.runtime.programs.runner.time.monotonic", side_effect=lambda: clock[0]):
            with model_wait(entry):
                clock[0] = 1000.0
                self.assertEqual(program_remaining(entry), 5.0)
                self.assertEqual(program_remaining(unrelated), -895.0)
            self.assertEqual(program_remaining(entry), 5.0)
            self.assertEqual(entry["model_wait_s"], 900.0)
            self.assertIsNone(entry["model_wait_started_at"])
            clock[0] += 2
            self.assertEqual(program_remaining(entry), 3.0)
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with model_wait(entry):
                    clock[0] += 100
                    raise RuntimeError("fixture")
            self.assertEqual(program_remaining(entry), 3.0)
            self.assertEqual(entry["model_wait_s"], 1000.0)
            clock[0] += 4
            with self.assertRaisesRegex(ValueError, "program_channel_expired"):
                with model_wait(entry):
                    self.fail("Expired work budget admitted a model call")

    def test_model_wait_never_ignores_explicit_cancellation(self):
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+10)
        with model_wait(entry):
            entry["cancel"].set()
            self.assertTrue(entry["cancel"].is_set())
        with self.assertRaisesRegex(ValueError, "program_channel_expired"):
            with model_wait(entry):
                self.fail("Cancelled program admitted another call")

    def test_artifact_broker_enforces_profile_on_json_and_geometry_kinds(self):
        from auriga.core.observation_policy import ACTOR_PROFILE, LEGACY_ACTOR_PROFILE
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp)/'run', dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            refs = {
                'calibration': store.put_artifact(b'{"K":[[1,0,0],[0,1,0],[0,0,1]]}', 'camera_calibration', 'application/json'),
                'untyped_calibration': store.put_artifact(b'{"nested":{"K":[[1,0,0],[0,1,0],[0,0,1]]}}', 'computed', 'application/json'),
                'calibrated_overlay': store.put_artifact(b'calibrated projected pixels', 'geometry_overlay', 'image/jpeg'),
                'derived_text': store.put_artifact(b'actor-derived geometry fixture', 'scene_geometry', 'text/plain', 'internal'),
                'nested': store.put_artifact(b'{"nested":{"object_poses":{}}}', 'computed', 'application/json'),
                'typed_geometry': store.put_artifact(b'{"points_m":[[0,0,1]]}', 'scene_job_result', 'application/json', 'internal'),
                'rgb': store.put_artifact(b'original RGB bytes', 'rgb', 'image/jpeg'),
            }
            entry = dict(cancel=threading.Event(), deadline=time.monotonic()+30, call_lock=threading.RLock())
            for profile in (LEGACY_ACTOR_PROFILE, ACTOR_PROFILE):
                runner = ProgramRunner(program_services(SimpleNamespace(store=store, observation_profile=profile)))
                try:
                    for key in ('calibration', 'calibrated_overlay'):
                        if profile == LEGACY_ACTOR_PROFILE:
                            with self.assertRaisesRegex(ValueError, 'calibrated_artifact'):
                                runner.dispatch(entry, {'artifact_ref': refs[key]})
                        else:
                            self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': refs[key]}))
                    if profile == LEGACY_ACTOR_PROFILE:
                        with self.assertRaisesRegex(ValueError, 'privileged_data'):
                            runner.dispatch(entry, {'artifact_ref': refs['untyped_calibration']})
                    else:
                        self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': refs['untyped_calibration']}))
                    for key in ('typed_geometry', 'derived_text'):
                        self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': refs[key]}))
                    with self.assertRaisesRegex(ValueError, 'privileged_data'):
                        runner.dispatch(entry, {'artifact_ref': refs['nested']})
                    self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': refs['rgb']}))
                finally:
                    runner.close()

    def test_artifact_owner_metadata_projection_preserves_execution_and_originals(self):
        from auriga.core.contracts import canonical
        with tempfile.TemporaryDirectory() as temp:
            identity = dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1)
            store = Store(Path(temp) / 'run', identity)
            self.addCleanup(store.close)
            runner = ProgramRunner(program_services(SimpleNamespace(store=store)))
            self.addCleanup(runner.close)
            entry = dict(cancel=threading.Event(), deadline=time.monotonic()+30, call_lock=threading.RLock())
            business = dict(version='operator-selected', revision=7, run_id='business-value')
            stage = dict(schema_version='0.1', **identity, path=dict(position_m=[1.23456789012345]),
                         diagnostics=business)
            request = dict(schema_version='0.1', **identity, action_id='test-action', type='BatchAction',
                           stages=[dict(id='reach', kind='move', action=stage)])
            receipt = dict(schema_version='0.1', **identity, action_id='test-action', status='unknown',
                           execution_uncertain=True, stop_confirmed=False, executed_native_steps=None,
                           batch_revision=9, plan_revision=4,
                           stage_results={'reach': dict(status='partial', position_error_m=0.123456789012345)},
                           diagnostics=business)
            store.save('action_request', 'test-action', request)
            store.save('action_receipt', 'test-action', receipt)
            cases = [
                ('unified_action_result', dict(request=request, receipt=receipt)),
                ('program_action_evidence', [receipt]),
                ('manipulation_receipt', receipt),
                ('candidate_trajectory', dict(schema_version='0.1', identity=identity,
                    plan_revision=4, samples=[business])),
                ('computed', dict(schema_version='domain-format', version='domain-version',
                    request=request, receipt=receipt)),
            ]
            for kind, value in cases:
                with self.subTest(kind=kind):
                    raw = canonical(value).encode()
                    ref = store.put_artifact(raw, kind, 'application/json')
                    before = store.resolve_artifact(ref)
                    reply = runner.dispatch(entry, dict(artifact_ref=ref))
                    shown = json.loads(base64.b64decode(reply['data_base64']))
                    self.assertEqual(store.resolve_artifact(ref), before)
                    if kind == 'computed':
                        self.assertEqual(shown, value)
                    elif kind == 'candidate_trajectory':
                        self.assertEqual(shown, dict(plan_revision=4, samples=[business]))
                    else:
                        record = shown['receipt'] if kind == 'unified_action_result' else (
                            shown[0] if isinstance(shown, list) else shown)
                        for key in ('schema_version', *identity):
                            self.assertNotIn(key, record)
                        self.assertEqual(record['stage_results'], receipt['stage_results'])
                        self.assertEqual(record['diagnostics'], business)
                        self.assertEqual(record['status'], 'unknown')
                        self.assertIsNone(record['executed_native_steps'])
                        self.assertFalse(record['stop_confirmed'])
                        self.assertTrue(record['execution_uncertain'])
                        self.assertEqual((record['batch_revision'], record['plan_revision']), (9, 4))
                        if kind == 'unified_action_result':
                            for owner in (shown['request'], shown['request']['stages'][0]['action']):
                                self.assertFalse(set(owner) & {'schema_version', *identity})
                            self.assertEqual(shown['request']['stages'][0]['action']['diagnostics'], business)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_real_client_budgets_isolation_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'run',dict(run_id='test',episode_id='ep',env_instance_id='env',env_generation=1))
            calls=[]
            harness=SimpleNamespace(store=store,lock=threading.RLock(),cancel=threading.Event(),active_context=None)
            def dispatch(name,args,correlation):
                calls.append((name,correlation))
                return {'success':True,'contentItems':[{'type':'inputText','text':json.dumps({'test':True})}]}
            harness.dispatch=dispatch
            runner=ProgramRunner(program_services(harness))
            identity=dict(thread_id='thread',turn_id='turn',tool_call_id='call',item_id='item')
            try:
                secret=Path(temp)/'observer-secret'; secret.write_text('hidden')
                source="from auriga_client import AurigaClient\nclient=AurigaClient()\nprint(client.call('robot_capabilities'))\ntry:\n open("+repr(str(secret))+").read()\n raise AssertionError('observer leaked')\nexcept PermissionError:\n print('isolation_passed')\n"
                args=dict(source=source,timeout_s=5,max_calls=1)
                result=runner.run(args,identity)
                self.assertEqual(result['status'],'succeeded',store.resolve_artifact(result['stderr_ref'])[1])
                self.assertEqual(result['source_sha256'],__import__('hashlib').sha256(source.encode()).hexdigest())
                self.assertIn(b'isolation_passed',store.resolve_artifact(result['stdout_ref'])[1])
                self.assertEqual(calls[0][1]['item_id'],'item')
                self.assertEqual(runner.run(args,identity),result)
                self.assertEqual(len(calls),1)
                numerical=runner.run(dict(source="import numpy as np\nfrom pathlib import Path\nimport os\nassert Path.cwd() == Path(os.environ['HOME'])\nPath('relative-result.txt').write_text('job-local')\nassert float(np.linalg.norm([3.,4.])) == 5.\nprint('numpy_within_memory_budget')",timeout_s=10),dict(identity,tool_call_id='numerical'))
                self.assertEqual(numerical['status'],'succeeded',store.resolve_artifact(numerical['stderr_ref'])[1])
                self.assertIn(b'numpy_within_memory_budget',store.resolve_artifact(numerical['stdout_ref'])[1])
                guarded=runner.run(dict(source="import socket,os\ntry:\n socket.socket(socket.AF_UNIX)\n raise AssertionError('Unix socket allowed')\nexcept PermissionError: pass\ntry:\n os.fork()\n raise AssertionError('fork allowed')\nexcept PermissionError: pass\ntry:\n os.kill(os.getppid(),0)\n raise AssertionError('cross process signal allowed')\nexcept PermissionError: pass\nprint('network_process_guards_passed')"),dict(identity,tool_call_id='guarded'))
                self.assertEqual(guarded['status'],'succeeded',store.resolve_artifact(guarded['stderr_ref'])[1])
                reused=runner.run(dict(source_ref=result['source_ref'],timeout_s=5,max_calls=1),dict(identity,tool_call_id='reused'))
                self.assertEqual(reused['status'],'succeeded')
                limited=runner.run(dict(source="from auriga_client import AurigaClient\nc=AurigaClient()\nc.call('robot_capabilities')\nc.call('robot_capabilities')",max_calls=1),dict(identity,tool_call_id='limited'))
                self.assertEqual(limited['status'],'succeeded')
                self.assertEqual(limited['calls'],2)
                timed=runner.run(dict(source='import time\ntime.sleep(20)',timeout_s=1),dict(identity,tool_call_id='timed'))
                self.assertEqual(timed['status'],'cancelled')
                self.assertTrue(timed['stop_confirmed'])
                denied=runner.run(dict(source="from auriga_client import AurigaClient\nAurigaClient().call('observer_evaluate')"),dict(identity,tool_call_id='denied'))
                self.assertEqual(denied['status'],'failed')
            finally:
                runner.close();store.close()

if __name__=='__main__': unittest.main()
