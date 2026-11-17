"""Actual native/worker output refreshes inactivity; polling and logs do not."""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from auriga.cortex.session.codex import CodexSession
from auriga.core.config import Config
from auriga.robot.backends.robodojo.robot_process import RobotProcess
from auriga.core.storage.store import Store
from auriga.robot.backends.robodojo.worker_protocol import WIRE_PREFIX
from support.host_capabilities import program_isolation, require


class ActivitySourcesTest(unittest.TestCase):
    def test_native_activity_excludes_empty_deltas_status_and_foreign_threads(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp), dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            session = CodexSession(temp, store, lambda _: {}, config=Config.from_dict({}))
            self.addCleanup(session.close)
            session.thread_id = 'owned'
            session.last_activity_at = 1
            for method, params in [
                ('thread/status/changed', {'threadId': 'owned'}),
                ('item/commandExecution/outputDelta', {'threadId': 'owned', 'delta': ''}),
                ('item/commandExecution/outputDelta', {'threadId': 'other', 'delta': 'output'}),
            ]:
                session._notification(dict(method=method, params=params))
                self.assertEqual(session.last_activity_at, 1)
            for method, extra in [
                ('item/commandExecution/outputDelta', {'delta': 'output'}),
                ('item/reasoning/textDelta', {'delta': 'thinking'}),
                ('item/tool/call', {'callId': 'tool'}),
                ('item/started', {'item': {'id': 'command', 'type': 'commandExecution'}}),
                ('item/completed', {'item': {'id': 'command', 'type': 'commandExecution'}}),
            ]:
                session.last_activity_at = 1
                with patch('auriga.cortex.session.codex.time.monotonic', return_value=20):
                    session._notification(dict(method=method, params={'threadId': 'owned', **extra}))
                self.assertEqual(session.last_activity_at, 20)

    def test_robot_activity_requires_new_pending_worker_milestone(self):
        with tempfile.TemporaryDirectory() as temp:
            robot = RobotProcess(temp)
            robot.last_activity_at = 1
            robot.pending['request'] = (threading.Event(), {}, None)
            progress = dict(id='request', progress=dict(phase='executing', executed_steps=5, sim_step=10))
            def output():
                yield 'native log chatter\n'
                self.assertEqual(robot.last_activity_at, 1)
                yield WIRE_PREFIX + json.dumps(dict(id='stale', result={})) + '\n'
                self.assertEqual(robot.last_activity_at, 1)
                yield WIRE_PREFIX + json.dumps(progress) + '\n'
                self.assertEqual(robot.last_activity_at, 20)
                robot.last_activity_at = 2
                yield WIRE_PREFIX + json.dumps(progress) + '\n'
                self.assertEqual(robot.last_activity_at, 2)
                progress['progress']['executed_steps'] = 10
                yield WIRE_PREFIX + json.dumps(progress) + '\n'
                self.assertEqual(robot.last_activity_at, 20)
                robot.last_activity_at = 3
                yield WIRE_PREFIX + json.dumps(dict(id='request', result={})) + '\n'
                self.assertEqual(robot.last_activity_at, 20)
            robot.process = SimpleNamespace(stdout=output(), poll=lambda: 0)
            robot.log = io.StringIO()
            with patch('auriga.robot.backends.robodojo.robot_process.time.monotonic', return_value=20):
                robot._read()

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_program_output_advances_clock_without_polling_refresh(self):
        from program_fixture import program_services
        from auriga.runtime.programs.runner import ProgramRunner
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            runner = ProgramRunner(program_services(SimpleNamespace(store=store)))
            self.addCleanup(runner.close)
            results = []
            worker = threading.Thread(target=lambda: results.append(runner.run(
                dict(source="import time\nprint('first', flush=True)\ntime.sleep(1)\nprint('second', flush=True)\ntime.sleep(1)", timeout_s=10),
                dict(thread_id='thread', turn_id='turn', tool_call_id='clock'))))
            worker.start()
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    logs = list(runner.logs.glob('*.stdout'))
                    if logs and b'first' in logs[0].read_bytes():
                        break
                    time.sleep(.02)
                else:
                    self.fail('program did not produce first output')
                time.sleep(.05)
                first = runner.last_activity_at
                time.sleep(.1)
                self.assertEqual(runner.last_activity_at, first)
                while time.monotonic() < deadline and b'second' not in logs[0].read_bytes():
                    time.sleep(.02)
                time.sleep(.05)
                self.assertGreater(runner.last_activity_at, first)
            finally:
                worker.join(12)
            self.assertFalse(worker.is_alive())
            self.assertEqual(results[0]['status'], 'succeeded')
