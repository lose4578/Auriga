"""Codex transport takes tool concurrency from the registry contract wired by Workbench (offline; no model/GPU)."""
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from openai_codex.client import CodexConfig

from auriga.apps.gateway.workbench import Workbench
from auriga.core.config import Config
from auriga.core.storage.store import Store
from auriga.cortex.session.codex import CodexSession, RecordedClient
from auriga.runtime.tools.builtins import create_registry
from auriga.runtime.tools.registry import ToolRegistry
from test_harness import FixtureRobot

TIMEOUT = 5


def tool_message(identifier, tool, arguments):
    return dict(id=identifier, method='item/tool/call', params=dict(
        threadId='thread-owned', turnId='turn-owned', callId='call-%s' % identifier, tool=tool, arguments=arguments))


class RecordedClientDispatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root, dict(run_id='parallel-dispatch', episode_id='ep', env_instance_id='env',
                                           env_generation=1))
        self.addCleanup(self.store.close)
        self.lock = threading.Lock()
        self.running, self.peak, self.order = 0, 0, []

    def client(self, handler_gate, **kwargs):
        def tool(params):
            with self.lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
                self.order.append(('start', params['tool']))
            try:
                handler_gate(params)
                return {'success': True, 'contentItems': [{'type': 'inputText', 'text': params['tool']}]}
            finally:
                with self.lock:
                    self.running -= 1
                    self.order.append(('end', params['tool']))
        client = RecordedClient(CodexConfig(cwd=str(self.root)), self.store, tool, lambda message: None, **kwargs)
        self.addCleanup(client.server_pool.shutdown, wait=True, cancel_futures=True)
        self.written = []
        client._write_message = self.written.append
        return client

    def test_default_orders_every_tool_call_even_former_allowlist_reads(self):
        client = self.client(lambda params: time.sleep(0.03))
        self.assertIsNone(client.parallel_safe)
        futures = [client._submit_request(tool_message(index, 'robot_information', {})) for index in range(3)]
        for future in futures:
            future.result(TIMEOUT)
        self.assertEqual(self.peak, 1)
        self.assertEqual([reply['id'] for reply in self.written], [0, 1, 2])

    def test_registry_classifier_overlaps_reads_and_orders_state_changes(self):
        registry = create_registry()
        seen = []

        def classify(tool, arguments):
            seen.append((tool, arguments))
            return registry.parallel_safe(tool, arguments)
        rendezvous = threading.Barrier(2, timeout=TIMEOUT)

        def gate(params):
            if params['tool'] == 'robot_information':
                rendezvous.wait()
        client = self.client(gate, parallel_safe=classify)
        reads = [client._submit_request(tool_message('read-%d' % index, 'robot_information', {'section': 'default'}))
                 for index in range(2)]
        plan = client._submit_request(tool_message('plan', 'task_plan', {'operation': 'update', 'plan': 'x'}))
        for future in (*reads, plan):
            future.result(TIMEOUT)
        self.assertEqual(seen[-1], ('task_plan', {'operation': 'update', 'plan': 'x'}))
        self.assertEqual(self.peak, 2)
        plan_start = self.order.index(('start', 'task_plan'))
        self.assertEqual([event for event in self.order[:plan_start] if event[0] == 'end'],
                         [('end', 'robot_information')] * 2)
        self.assertEqual({reply['id'] for reply in self.written}, {'read-0', 'read-1', 'plan'})

    def test_classifier_failure_never_stops_the_reader(self):
        def classify(tool, arguments):
            raise RuntimeError('fixture classifier failure')
        client = self.client(lambda params: None, parallel_safe=classify)
        messages = iter([tool_message(1, 'robot_information', {}), dict(id=9, result={'still': 'reading'})])
        routed = []

        def read():
            try:
                return next(messages)
            except StopIteration:
                raise EOFError('fixture transport ended')
        client._read_message = read
        client._router = SimpleNamespace(route_response=routed.append, fail_all=lambda error: None)
        client._reader_loop()
        self.assertEqual(routed, [dict(id=9, result={'still': 'reading'})])
        client.server_pool.shutdown(wait=True)
        self.assertEqual([reply['id'] for reply in self.written], [1])

    def test_non_tool_requests_bypass_the_tool_barrier(self):
        release = threading.Event()
        replies = []
        client = self.client(lambda params: release.wait(TIMEOUT))
        client.request_handler = lambda message, **kwargs: replies.append((message['id'], kwargs)) or {}
        blocked = client._submit_request(tool_message('tool', 'task_plan', {'operation': 'update'}))
        other = client._submit_request(dict(id='approval', method='item/fileChange/requestApproval', params={}))
        other.result(TIMEOUT)
        self.assertFalse(blocked.done())
        release.set()
        blocked.result(TIMEOUT)
        self.assertEqual(replies[0], ('approval', {}))


class CodexSessionWiringTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def start_with_fake_client(self, session):
        session.private.mkdir(parents=True, exist_ok=True)
        session.socket_path = self.root / 'fixture.sock'
        session.socket_path.touch()
        session.sandbox = mock.Mock(host_socket_path='/proc/fixture/fd/pinned')
        session.sandbox.repair_socket_alias.return_value = True
        session.command, session.env, session.cli_executable = ['fixture-only'], {}, Path('/fixture/codex')
        session.model_proxy = mock.Mock()
        client = mock.MagicMock()
        client.initialize.return_value.model_dump.return_value = {}
        client.thread_start.return_value = SimpleNamespace(thread=SimpleNamespace(id='fixture-thread'))
        with mock.patch.object(session, '_prepare', return_value='fixture'), \
                mock.patch.object(session, '_stop_previous_owned_processes'), \
                mock.patch('auriga.cortex.session.codex.subprocess.Popen',
                           return_value=SimpleNamespace(pid=1, poll=lambda: 0)), \
                mock.patch('auriga.cortex.session.codex.RecordedClient', return_value=client) as factory:
            session.start([], 'fixture instructions')
        self.addCleanup(session.close)
        return factory

    def test_unwired_session_orders_every_call(self):
        store = Store(self.root / 'run', dict(run_id='unwired', episode_id='ep', env_instance_id='env', env_generation=1))
        self.addCleanup(store.close)
        session = CodexSession(self.root / 'run', store, lambda params: {},
                               config=Config.from_dict({'model': {'codex_home': str(self.root / 'codex-home')}}))
        self.assertIsNone(CodexSession.tool_parallel_safe)
        factory = self.start_with_fake_client(session)
        self.assertIsNone(factory.call_args.kwargs['parallel_safe'])
        self.assertIs(factory.call_args.kwargs['route_notifications'], False)

    def test_workbench_wires_registry_classifier_into_the_native_client(self):
        robot = FixtureRobot(self.root)
        robot.close = lambda: None
        config = Config.from_dict({'features': {'grasp_memory_dir': str(self.root / 'memory')},
                                   'model': {'codex_home': str(self.root / 'codex-home')}})
        workbench = Workbench(self.root / 'workbench', robot=robot, config=config,
                              observation_profile='rgb_proprio_v1', motion_review=False)
        self.addCleanup(workbench.store.close)
        self.addCleanup(workbench.run_lock.close)
        if workbench.grasp_memory_store:
            self.addCleanup(workbench.grasp_memory_store.close)
        wired = workbench.session.tool_parallel_safe
        self.assertIs(wired.__self__, workbench.harness.tool_registry)
        self.assertIs(wired.__func__, ToolRegistry.parallel_safe)
        factory = self.start_with_fake_client(workbench.session)
        passed = factory.call_args.kwargs['parallel_safe']
        self.assertEqual(passed, workbench.harness.tool_registry.parallel_safe)
        self.assertTrue(passed('robot_information', {}))
        self.assertFalse(passed('robot_action', {'mode': 'execute'}))
        self.assertFalse(passed('robot_observe', {}))


if __name__ == '__main__':
    unittest.main()
