"""Program-only visual feedback reaches the actor without another image encode."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import base64
import io
import json
import time
import unittest
from unittest.mock import patch

from PIL import Image

from auriga.runtime.programs.runner import ProgramRunner
from auriga.core.tool_output import ToolOutput
import test_dual_arm_interface as harness_fixture
from support.host_capabilities import program_isolation, require


class ProgramVisualFeedbackTests(unittest.TestCase):
    def setUp(self):
        harness_fixture.DualArmHarnessTests.setUp(self)
        self.h.thread_id = 'visual-feedback'
        self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(lambda: self.h.programs.close())

    def call(self, name, args, identifier):
        wire = self.h.handle_tool(dict(threadId=self.h.thread_id, turnId='turn', callId=identifier,
            tool=name, arguments=args))
        self.assertTrue(wire['success'], wire)
        return json.loads(wire['contentItems'][0]['text']), wire['contentItems']

    @staticmethod
    def images(parts):
        return [part for part in parts if part['type'] == 'inputImage']

    def source(self, tool, args):
        return ('from auriga_client import AurigaClient\n'
                'client = AurigaClient()\n'
                'reply = client.call(' + repr(tool) + ', ' + repr(args) + ')\n'
                "assert reply['success']\nprint('visual feedback requested')\n")

    def test_history_crop_exact_bytes_provenance_retry_full_context_and_restart(self):
        old = self.h.latest
        self.h.observe({'tool_call_id': 'new-observation'})
        args = dict(observation_id=old['observation_id'], camera_id=old['cameras'][0]['camera_id'],
                    roi=[0, 0, 1, 1], scale=2)
        source = self.source('history_view_observation', args)
        job, parts = self.call('program_run', dict(source=source, timeout_s=10), 'start')
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertIsNone(job['final_observation_id'])
        feedback = job['visual_feedback'][0]
        self.assertEqual(feedback['tool'], 'history_view_observation')
        self.assertEqual(feedback['arguments']['roi'], args['roi'])
        raw_job = self.store.states('program')[0]
        saved = json.loads(self.store.resolve_artifact(raw_job['visual_feedback'][0]['reply_ref'])[1])
        self.assertEqual(self.images(parts), self.images(saved['contentItems']))
        self.assertEqual(len(self.images(parts)), 1)
        metadata = json.loads(saved['contentItems'][0]['text'])
        self.assertEqual(metadata['image_view']['source_roi'], args['roi'])
        self.assertEqual(metadata['image_view']['display_scale'], 2)
        self.assertFalse(metadata['observation_is_current'])
        self.assertIn('not a new current observation', json.dumps(parts))
        _, retried = self.call('program_run', dict(source=source, timeout_s=10), 'start')
        self.assertEqual(self.images(retried), self.images(parts))
        polled, compact = self.call('program_status', dict(job_id=job['job_id']), 'poll')
        self.assertEqual(self.images(compact), [])
        self.assertEqual(polled['visual_feedback'][0]['delivery']['previous_tool_call_id'], 'start')
        _, full = self.call('program_status', dict(job_id=job['job_id'], detail='full'), 'full')
        self.assertEqual(self.images(full), self.images(parts))
        self.store.set_state('runtime', 'image_context_epoch', {'epoch': 'reset'})
        _, restored = self.call('program_status', dict(job_id=job['job_id']), 'after-reset')
        self.assertEqual(self.images(restored), self.images(parts))
        self.h.programs.close()
        self.h.programs = ProgramRunner(self.h.program_services())
        _, restarted = self.call('program_status', dict(job_id=job['job_id'], detail='full'), 'after-restart')
        self.assertEqual(self.images(restarted), self.images(parts))
        self.assertEqual(self.robot.executions, 0)

    def test_preview_image_is_forwarded_even_without_printing_reply_or_executing(self):
        buffer = io.BytesIO()
        Image.new('RGB', (7, 5), 'red').save(buffer, format='PNG')
        image = dict(type='inputImage', imageUrl='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
        original = self.h.programs.services.invoke
        def invoke(name, args, correlation):
            if name == 'robot_action':
                self.assertEqual(args['mode'], 'preview')
                return ToolOutput(dict(plan_id='fixture-plan', status='planned',
                    label='candidate trajectory; not execution'), extra_content=(image,), kind='plan').response()
            return original(name, args, correlation)
        self.h.programs.services = replace(self.h.programs.services, invoke=invoke)
        args = dict(mode='preview', observation_id=self.h.latest['observation_id'],
                    stages=[dict(id='open', arm='left', kind='gripper', goal_normalized=1.)])
        job, parts = self.call('program_run', dict(source=self.source('robot_action', args)), 'preview')
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertEqual(self.images(parts), [image])
        self.assertIn('candidate trajectory; not execution', json.dumps(parts))
        self.assertEqual(job['action_summary']['action_calls'], 0)
        self.assertIsNone(job['final_observation_id'])
        self.assertNotIn('data:image', job['stdout_text'])
        multi_source = (
            'from auriga_client import AurigaClient\n'
            'client = AurigaClient()\n'
            'assert client.call("robot_action", ' + repr(args) + ')["success"]\n'
            'assert client.call("robot_action", ' + repr(args) + ')["success"]\n'
        )
        multi_job, multi_parts = self.call('program_run', dict(source=multi_source), 'multi-preview')
        self.assertEqual(len(multi_job['visual_feedback']), 1)
        self.assertEqual(len(self.images(multi_parts)), 1)
        current_obs_id = self.h.latest['observation_id']
        cam_id = self.h.latest['cameras'][0]['camera_id']
        crop_out, crop_parts = self.call('robot_observe', dict(camera_id=cam_id, roi=[0, 0, 1, 1], scale=2), 'roi-crop')
        self.assertEqual(crop_out['bundle']['observation_id'], 'obs_1')
        self.assertEqual(self.h.latest['observation_id'], current_obs_id)
        self.assertEqual(len(self.images(crop_parts)), 1)
        self.h.latest = None
        restored, parts = self.call('program_status', dict(job_id=job['job_id'], detail='full'), 'no-current')
        self.assertEqual(restored['status'], 'succeeded')
        self.assertEqual(self.images(parts), [image])
        header = next(json.loads(part['text']) for part in parts if part['type'] == 'inputText'
                      and 'program_visual_feedback' in part['text'])
        self.assertIsNone(header['current_observation_id_at_delivery'])
        self.assertEqual(self.robot.executions, 0)

    def test_failed_delivery_retries_images_and_authored_stdout_is_not_an_image(self):
        args = dict(observation_id=self.h.latest['observation_id'])
        with patch.object(self.h.actor_output, '_present_part', side_effect=ValueError('display failure')):
            failed = self.h.handle_tool(dict(threadId=self.h.thread_id, turnId='turn', callId='failed',
                tool='program_run', arguments=dict(source=self.source('history_view_observation', args))))
        self.assertFalse(failed['success'])
        job = self.store.states('program')[0]
        _, parts = self.call('program_status', dict(job_id=job['job_id']), 'recovery')
        self.assertTrue(self.images(parts))
        authored, parts = self.call('program_run', dict(source="print('data:image/png;base64,AAAA')"), 'authored')
        self.assertEqual(self.images(parts), [])
        self.assertEqual(authored['visual_feedback'], [])
        self.assertIn('omitted', authored['stdout_text'])
        self.assertEqual(self.robot.executions, 0)

    def test_context_change_during_final_frame_and_missing_visual_preserve_job(self):
        source = self.source('history_view_observation', dict(observation_id=self.h.latest['observation_id']))
        source += "client.call('robot_observe', {})\n"
        job, first = self.call('program_run', dict(source=source), 'with-final')
        self.assertEqual(job['status'], 'succeeded')
        self.assertIn('currentness flags and budgets are as of the source call', json.dumps(first))
        original = self.h.observation_content
        def context_reset(*args, **kwargs):
            parts = original(*args, **kwargs)
            self.store.set_state('runtime', 'image_context_epoch', {'epoch': 'during-final'})
            return parts
        with patch.object(self.h, 'observation_content', side_effect=context_reset):
            _, reset = self.call('program_status', dict(job_id=job['job_id']), 'during-reset')
        self.assertEqual(self.images(reset), self.images(first))
        raw_job = self.store.states('program')[0]
        missing_ref = raw_job['visual_feedback'][0]['reply_ref']
        resolve = self.store.resolve_artifact
        def missing(reference, *args, **kwargs):
            if reference == missing_ref:
                raise ValueError('fixture_missing_visual_blob')
            return resolve(reference, *args, **kwargs)
        with patch.object(self.store, 'resolve_artifact', side_effect=missing), \
                patch.object(self.h, 'observation_content', side_effect=ValueError('fixture_missing_final_frame')):
            degraded, parts = self.call('program_status', dict(job_id=job['job_id']), 'missing')
        self.assertEqual(degraded['status'], 'succeeded')
        self.assertEqual(degraded['action_summary'], job['action_summary'])
        self.assertEqual({error['source'] for error in degraded['visual_feedback_errors']},
                         {'supplementary', 'final_observation'})
        self.assertEqual(degraded['visual_feedback'][0]['delivery']['status'], 'unavailable')
        self.assertEqual(self.images(parts), [])
        _, recovered = self.call('program_status', dict(job_id=job['job_id']), 'blob-restored')
        self.assertEqual(self.images(recovered), self.images(first))
        self.assertEqual(self.robot.executions, 0)

    def test_running_job_publishes_visuals_before_terminal_boundary(self):
        source = self.source('history_view_observation', dict(observation_id=self.h.latest['observation_id']))
        source += "from pathlib import Path\nimport time\nwhile not Path('release').exists(): time.sleep(.01)\n"
        with ThreadPoolExecutor(max_workers=1) as pool:
            started = pool.submit(self.h.programs.run, dict(source=source, timeout_s=10),
                dict(thread_id=self.h.thread_id, turn_id='turn', tool_call_id='running-start'))
            deadline = time.monotonic() + 5
            job = None
            while time.monotonic() < deadline:
                jobs = self.store.states('program')
                job = next((item for item in jobs if item.get('visual_feedback')), None)
                if job:
                    break
                time.sleep(.01)
            try:
                self.assertIsNotNone(job, 'visual call must publish before process completion')
                status, parts = self.call('program_status', dict(job_id=job['job_id']), 'running-status')
                self.assertEqual(status['status'], 'running')
                self.assertTrue(self.images(parts))
            finally:
                for item in self.store.states('program'):
                    (self.h.programs.root / item['job_id'] / 'release').touch()
            self.assertEqual(started.result(timeout=5)['status'], 'succeeded')
        self.assertEqual(self.robot.executions, 0)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_python_environment_initializes_without_reading_run_private_parent(self):
        self.h.config = self.h.config.with_overrides({'features': {'sandbox_enabled': True}})
        self.h.programs.close()
        self.h.programs = ProgramRunner(self.h.program_services())
        private = self.store.root / 'private' / 'not-program-evidence.txt'
        private.parent.mkdir(exist_ok=True)
        private.write_text('fixture-private-data')
        source = ('from pathlib import Path\n'
                  'try:\n Path(' + repr(str(private)) + ").read_text()\n"
                  'except PermissionError:\n print("private parent remains unreadable")\n'
                  'else:\n raise AssertionError("program read private parent")\n')
        job, _ = self.call('program_run', dict(source=source), 'python-environment')
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        self.assertIn('private parent remains unreadable', job['stdout_text'])
        self.assertEqual(self.robot.executions, 0)


if __name__ == '__main__':
    unittest.main()
