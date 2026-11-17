"""OrderedBarrier / ToolDispatcher ordering semantics (pure threads; no model, no robot)."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading
import time
import unittest

from auriga.cortex.agents.dispatch import OrderedBarrier, ToolDispatcher, is_parallel_call

TIMEOUT = 5


class Recorder:
    """Records start/end order and the peak number of concurrently running calls."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.running = 0
        self.peak = 0

    def call(self, name, gate=None, error=None):
        with self.lock:
            self.events.append(('start', name))
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            if gate is not None:
                gate()
            if error is not None:
                raise error
            return name
        finally:
            with self.lock:
                self.running -= 1
                self.events.append(('end', name))

    def index(self, kind, name):
        return self.events.index((kind, name))


class OrderedBarrierTest(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='barrier-test')
        self.addCleanup(self.pool.shutdown, wait=True, cancel_futures=True)
        self.barrier = OrderedBarrier()
        self.recorder = Recorder()

    def schedule(self, parallel, name, **kwargs):
        return self.barrier.schedule(self.pool, parallel, self.recorder.call, name, kwargs.get('gate'),
                                     kwargs.get('error'))

    def test_two_parallel_calls_overlap(self):
        rendezvous = threading.Barrier(2, timeout=TIMEOUT)
        first = self.schedule(True, 'read-1', gate=rendezvous.wait)
        second = self.schedule(True, 'read-2', gate=rendezvous.wait)
        # Each call blocks until the other is running: completion proves overlap.
        self.assertEqual([first.result(TIMEOUT), second.result(TIMEOUT)], ['read-1', 'read-2'])
        self.assertEqual(self.recorder.peak, 2)

    def test_ordered_call_waits_for_every_earlier_parallel_call(self):
        release = threading.Event()
        both_running = threading.Barrier(3, timeout=TIMEOUT)

        def hold():
            both_running.wait()
            release.wait(TIMEOUT)
        reads = [self.schedule(True, 'read-1', gate=hold), self.schedule(True, 'read-2', gate=hold)]
        write = self.schedule(False, 'write')
        both_running.wait()
        time.sleep(0.1)
        self.assertNotIn(('start', 'write'), self.recorder.events)
        release.set()
        self.assertEqual(write.result(TIMEOUT), 'write')
        for read in reads:
            read.result(TIMEOUT)
        start = self.recorder.index('start', 'write')
        self.assertLess(self.recorder.index('end', 'read-1'), start)
        self.assertLess(self.recorder.index('end', 'read-2'), start)

    def test_parallel_call_waits_for_previous_ordered_call(self):
        release = threading.Event()
        started = threading.Event()

        def hold():
            started.set()
            release.wait(TIMEOUT)
        write = self.schedule(False, 'write', gate=hold)
        read = self.schedule(True, 'read')
        self.assertTrue(started.wait(TIMEOUT))
        time.sleep(0.1)
        self.assertNotIn(('start', 'read'), self.recorder.events)
        release.set()
        read.result(TIMEOUT)
        write.result(TIMEOUT)
        self.assertLess(self.recorder.index('end', 'write'), self.recorder.index('start', 'read'))

    def test_arrival_order_is_execution_order_for_ordered_calls(self):
        # Earlier calls sleep longer; without the barrier later calls would finish first on 8 workers.
        names = ['call-%d' % index for index in range(8)]
        futures = [self.schedule(False, name, gate=lambda delay=(8 - index) * 0.01: time.sleep(delay))
                   for index, name in enumerate(names)]
        self.assertEqual([future.result(TIMEOUT) for future in futures], names)
        self.assertEqual(self.recorder.events, [event for name in names for event in (('start', name), ('end', name))])
        self.assertEqual(self.recorder.peak, 1)

    def test_mixed_sequence_respects_both_barriers(self):
        plan = [(False, 'o1'), (True, 'p1'), (True, 'p2'), (False, 'o2'), (True, 'p3'), (False, 'o3')]
        futures = [self.schedule(parallel, name, gate=lambda: time.sleep(0.02)) for parallel, name in plan]
        for future in futures:
            future.result(TIMEOUT)
        before = self.recorder.index
        self.assertLess(before('end', 'o1'), before('start', 'p1'))
        self.assertLess(before('end', 'o1'), before('start', 'p2'))
        self.assertLess(max(before('end', 'p1'), before('end', 'p2')), before('start', 'o2'))
        self.assertLess(before('end', 'o2'), before('start', 'p3'))
        self.assertLess(before('end', 'p3'), before('start', 'o3'))

    def test_exceptions_propagate_without_breaking_the_chain(self):
        failed_write = self.schedule(False, 'write-1', error=ValueError('fixture write failure'))
        failed_read = self.schedule(True, 'read-1', error=RuntimeError('fixture read failure'))
        read = self.schedule(True, 'read-2')
        write = self.schedule(False, 'write-2')
        with self.assertRaisesRegex(ValueError, 'fixture write failure'):
            failed_write.result(TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, 'fixture read failure'):
            failed_read.result(TIMEOUT)
        self.assertEqual(read.result(TIMEOUT), 'read-2')
        self.assertEqual(write.result(TIMEOUT), 'write-2')
        self.assertLess(self.recorder.index('end', 'read-1'), self.recorder.index('start', 'write-2'))

    def test_non_true_classification_is_ordered(self):
        rendezvous = threading.Barrier(2, timeout=0.3)
        first = self.schedule(1, 'call-1', gate=rendezvous.wait)
        second = self.schedule('yes', 'call-2', gate=rendezvous.wait)
        # Ordered calls never overlap, so neither can pass the two-party rendezvous.
        for future in (first, second):
            with self.assertRaises(threading.BrokenBarrierError):
                future.result(TIMEOUT)
        self.assertEqual(self.recorder.peak, 1)

    def test_submit_error_propagates_to_the_caller(self):
        self.pool.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            self.schedule(False, 'late')
        with self.assertRaises(RuntimeError):
            self.schedule(True, 'late-read')


class IsParallelCallTest(unittest.TestCase):
    def test_only_an_explicit_true_verdict_is_parallel(self):
        seen = []

        def classify(tool, arguments):
            seen.append((tool, arguments))
            return tool == 'reader'
        self.assertTrue(is_parallel_call(classify, {'tool': 'reader', 'arguments': {'a': 1}}))
        self.assertFalse(is_parallel_call(classify, {'tool': 'writer', 'arguments': {}}))
        self.assertEqual(seen, [('reader', {'a': 1}), ('writer', {})])
        self.assertFalse(is_parallel_call(None, {'tool': 'reader', 'arguments': {}}))
        for params in (None, [], 'reader'):
            self.assertFalse(is_parallel_call(classify, params))
        self.assertFalse(is_parallel_call(lambda tool, arguments: 1, {'tool': 'reader'}))
        self.assertFalse(is_parallel_call(lambda tool, arguments: 1 / 0, {'tool': 'reader'}))


class ToolDispatcherTest(unittest.TestCase):
    def test_handler_runs_on_named_workers_with_registry_classification(self):
        rendezvous = threading.Barrier(2, timeout=TIMEOUT)
        threads = []

        def handler(params):
            threads.append(threading.current_thread().name)
            if params['tool'] == 'reader':
                rendezvous.wait()
            return dict(tool=params['tool'], arguments=params['arguments'])
        dispatcher = ToolDispatcher(handler, lambda tool, arguments: tool == 'reader', thread_name_prefix='fixture-tools')
        try:
            futures = [dispatcher.submit(dict(tool='reader', arguments={'n': index})) for index in range(2)]
            futures.append(dispatcher.submit(dict(tool='writer', arguments={})))
            self.assertEqual([future.result(TIMEOUT) for future in futures],
                             [dict(tool='reader', arguments={'n': 0}), dict(tool='reader', arguments={'n': 1}),
                              dict(tool='writer', arguments={})])
        finally:
            dispatcher.shutdown()
        self.assertTrue(all(name.startswith('fixture-tools') for name in threads))
        self.assertNotIn(threading.current_thread().name, threads)

    def test_without_classifier_every_call_is_ordered(self):
        recorder = Recorder()
        dispatcher = ToolDispatcher(lambda params: recorder.call(params['tool'], lambda: time.sleep(0.02)), None)
        try:
            futures = [dispatcher.submit(dict(tool='call-%d' % index, arguments={})) for index in range(4)]
            self.assertEqual([future.result(TIMEOUT) for future in futures], ['call-%d' % index for index in range(4)])
        finally:
            dispatcher.shutdown()
        self.assertEqual(recorder.peak, 1)

    def test_classifier_failure_is_ordered_and_handler_error_reaches_the_future(self):
        def classify(tool, arguments):
            raise RuntimeError('fixture classifier failure')

        def handler(params):
            if params['tool'] == 'broken':
                raise ValueError('fixture handler failure')
            return params['tool']
        dispatcher = ToolDispatcher(handler, classify, max_workers=2)
        try:
            broken = dispatcher.submit(dict(tool='broken', arguments={}))
            later = dispatcher.submit(dict(tool='later', arguments={}))
            with self.assertRaisesRegex(ValueError, 'fixture handler failure'):
                broken.result(TIMEOUT)
            self.assertEqual(later.result(TIMEOUT), 'later')
        finally:
            dispatcher.shutdown()

    def test_shutdown_cancels_queued_calls_and_rejects_new_ones(self):
        started, release = threading.Event(), threading.Event()

        def handler(params):
            if params['tool'] == 'running':
                started.set()
                release.wait(TIMEOUT)
            return params['tool']
        dispatcher = ToolDispatcher(handler, None, max_workers=1)
        running = dispatcher.submit(dict(tool='running', arguments={}))
        queued = dispatcher.submit(dict(tool='queued', arguments={}))
        self.assertTrue(started.wait(TIMEOUT))
        dispatcher.shutdown(wait=False, cancel_futures=True)
        self.assertTrue(queued.cancelled())
        with self.assertRaises(CancelledError):
            queued.result(TIMEOUT)
        with self.assertRaises(RuntimeError):
            dispatcher.submit(dict(tool='late', arguments={}))
        release.set()
        self.assertEqual(running.result(TIMEOUT), 'running')
        dispatcher.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
