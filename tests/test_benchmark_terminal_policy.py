"""Official benchmark endings cannot be supplied by actor prose or turn status."""
import asyncio
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from auriga.cortex.session.codex import CodexSession
from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.cortex.supervision.completion_guidance import COMPLETION_GUIDANCE_MODES, HOOK_COMPLETION_GUIDANCE
from auriga.apps.evaluation.episode import BENCHMARK_CONTINUATION, wait_benchmark
from auriga.apps.gateway.workbench import Workbench
from auriga.apps.library.scoring import classify_episode_result
from auriga.runtime.supervision import wait_turn
from auriga.core.storage.store import Store
from support.host_capabilities import program_isolation, require


class BenchmarkTerminalPolicyCheck(unittest.IsolatedAsyncioTestCase):
    def workbench(self, mode="none"):
        done = threading.Event()
        done.set()
        session = SimpleNamespace(state="completed", turn_done=done, turn_id="first-turn", thread_id="same-thread",
            turn_result={"status": "completed", "answer": "I have completed the task successfully."},
            model_limit_reached=threading.Event(), model_proxy=SimpleNamespace(limit_notified=False, request_count=3))
        harness = SimpleNamespace(episode_done=threading.Event(), episode_terminal=None, accepting=True,
            native_action_count=0, started_at=time.monotonic(), last_native_progress_at=time.monotonic(), active_action=None,
            home_check=SimpleNamespace(check=Mock(return_value={"status": "away", "initial_observation_id": None})))
        return SimpleNamespace(session=session, harness=harness, completion_guidance=mode, store=SimpleNamespace(event=Mock()),
            identity={"episode_id": "fixture-episode"},
            current_unknown_actions=lambda: False, has_active_programs=lambda: False, control_lock=asyncio.Lock())

    def paused_program(self, workbench):
        from test_program_review import ProgramReviewTests
        fixture = ProgramReviewTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        job = fixture.start("from auriga_client import AurigaClient\nAurigaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')")
        self.assertEqual(job['status'], 'awaiting_review')
        workbench.harness.programs = fixture.runner
        workbench.has_active_programs = lambda: Workbench.has_active_programs(workbench)
        workbench.program_continuation_state = lambda: Workbench.program_continuation_state(workbench)
        return fixture, job

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    async def test_review_paused_program_gets_continuation_and_can_resume_same_process(self):
        workbench = self.workbench()
        fixture, job = self.paused_program(workbench)
        entry = fixture.runner.entries[job['job_id']]
        self.assertTrue(workbench.has_active_programs())
        self.assertFalse(workbench.program_continuation_state()['blocked'])
        def continue_turn(prompt):
            self.assertIn(job['job_id'], prompt)
            self.assertIn(job['pending_review']['review_id'], prompt)
            self.assertIn("operation='resume'", prompt)
            self.assertEqual(fixture.action_count, 0)
            final = fixture.resume(job)
            self.assertEqual(final['status'], 'succeeded')
            self.assertEqual(fixture.action_count, 1)
            workbench.harness.episode_done.set()
        workbench.session.run_turn = Mock(side_effect=continue_turn)
        self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
        workbench.session.run_turn.assert_called_once()
        entry['worker'].join(1)
        self.assertFalse(workbench.has_active_programs())

    async def test_running_uncertain_or_unconfirmed_program_still_blocks_continuation(self):
        for state in ('running', 'unknown', 'unconfirmed_pause', 'cancel_requested', 'missing_review'):
            with self.subTest(state=state):
                workbench = self.workbench()
                cancel = threading.Event()
                if state == 'cancel_requested': cancel.set()
                job = dict(status=state if state in {'running', 'unknown'} else 'awaiting_review',
                    pending_review=dict(review_id='review', decision='pending'))
                if state == 'missing_review': job.pop('pending_review')
                entry = dict(job_id='program', job=job, cancel=cancel,
                    condition=threading.Condition(), process=SimpleNamespace(poll=lambda: None),
                    process_paused=state != 'unconfirmed_pause')
                from auriga.runtime.programs.runner import ProgramRunner
                runner = ProgramRunner.__new__(ProgramRunner)
                runner.lock, runner.entries = threading.RLock(), {'program': entry}
                runner.store = SimpleNamespace(states=lambda _: [])
                workbench.harness.programs = runner
                workbench.program_continuation_state = lambda: Workbench.program_continuation_state(workbench)
                workbench.session.run_turn = Mock()
                self.assertTrue(workbench.program_continuation_state()['blocked'])
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                workbench.session.run_turn.assert_not_called()

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    async def test_stop_still_cancels_and_reaps_review_paused_program(self):
        workbench = self.workbench()
        fixture, job = self.paused_program(workbench)
        entry = fixture.runner.entries[job['job_id']]
        workbench.control_responses = {}
        workbench.starting = False
        workbench.harness.request_stop = Mock(side_effect=lambda: setattr(workbench.harness, 'accepting', False))
        workbench.session.interrupt = Mock(side_effect=lambda: setattr(workbench.session, 'state', 'interrupted'))
        self.assertTrue(workbench.has_active_programs())
        result = await Workbench.stop(workbench, 'stop-paused-program')
        self.assertEqual(result['status'], 'stopped')
        self.assertFalse(workbench.has_active_programs())
        self.assertIsNotNone(entry['process'].poll())
        self.assertEqual(fixture.action_count, 0)
        final = fixture.runner.status(job['job_id'])
        self.assertEqual(final['status'], 'cancelled')
        self.assertTrue(final['stop_confirmed'])

    async def test_early_final_continues_same_episode_until_official_success_or_failure(self):
        for success in (False, True):
            with self.subTest(native_success=success):
                workbench = self.workbench()
                harness, session = workbench.harness, workbench.session
                before_clock, proxy = harness.started_at, session.model_proxy
                def continue_turn(prompt):
                    self.assertEqual(prompt, "任务还没完成，请继续。")
                    session.turn_id = "second-turn"
                    session.model_proxy.request_count += 1
                    harness.native_action_count = 4
                    harness.episode_terminal = {"ended": True, "success": success,
                        "reason": "native_success" if success else "native_step_limit"}
                    harness.episode_done.set()
                    harness.accepting = False
                session.run_turn = Mock(side_effect=continue_turn)
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                session.run_turn.assert_called_once()
                self.assertEqual(session.thread_id, "same-thread")
                self.assertIs(session.model_proxy, proxy)
                self.assertEqual(session.model_proxy.request_count, 4)
                self.assertEqual(harness.started_at, before_clock)
                self.assertEqual(harness.actor_continuation_count, 1)
                self.assertEqual(harness.wait_stop_reason, "native_episode_ended")

    async def test_repeated_early_finals_do_not_reset_model_budget(self):
        for mode in COMPLETION_GUIDANCE_MODES:
            with self.subTest(mode=mode):
                workbench = self.workbench(mode)
                session, harness = workbench.session, workbench.harness
                before_progress, before_start = harness.last_native_progress_at, harness.started_at
                proxy = session.model_proxy
                def continue_turn(prompt):
                    session.model_proxy.request_count += 1
                    if session.model_proxy.request_count == 5:
                        session.model_proxy.limit_notified = True
                session.run_turn = Mock(side_effect=continue_turn)
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                expected = HOOK_COMPLETION_GUIDANCE if mode in {"hook", "both"} else BENCHMARK_CONTINUATION
                self.assertEqual(session.run_turn.call_count, 2)
                self.assertTrue(all(call.args == (expected,) for call in session.run_turn.call_args_list))
                self.assertIs(session.model_proxy, proxy)
                self.assertEqual(session.model_proxy.request_count, 5)
                self.assertEqual(session.thread_id, "same-thread")
                self.assertEqual(harness.actor_continuation_count, 2)
                self.assertEqual(harness.wait_stop_reason, "model_request_limit")
                self.assertEqual(harness.last_native_progress_at, before_progress)
                self.assertEqual(harness.started_at, before_start)
                self.assertFalse(harness.episode_done.is_set())
                for count, call in enumerate(workbench.store.event.call_args_list, 1):
                    self.assertEqual(call.args[0], "benchmark.actor_turn_continuation")
                    self.assertEqual(call.args[1], dict(count=count, previous_turn_id="first-turn",
                        thread_id="same-thread", reason="actor_turn_completed_before_native_terminal",
                        completion_guidance=mode, trigger="turn.completed", prompt=expected, initial_observation_id=None,
                        continuation_kind="home_reminder" if mode in {"hook", "both"} else "generic",
                        home_check={"status": "away", "initial_observation_id": None} if mode in {"hook", "both"} else None))

    async def test_native_terminal_budget_timeout_and_unknown_never_launch_continuation(self):
        for mode in COMPLETION_GUIDANCE_MODES:
            for condition in ("native_success", "native_failure", "budget", "timeout", "progress_timeout",
                              "unknown", "shutdown", "not_accepting", "active_action", "active_program"):
                with self.subTest(mode=mode, condition=condition):
                    workbench = self.workbench(mode)
                    shutdown = asyncio.Event()
                    workbench.session.run_turn = Mock()
                    if condition.startswith("native_"):
                        workbench.harness.episode_terminal = {"ended": True, "success": condition == "native_success"}
                        workbench.harness.episode_done.set()
                    if condition == "budget": workbench.session.model_limit_reached.set()
                    if condition == "timeout": workbench.harness.started_at -= 1801; workbench.harness.last_native_progress_at -= 1801
                    if condition == "progress_timeout":
                        workbench.harness.native_action_count = 1
                        workbench.harness.started_at -= 1801
                        workbench.harness.last_native_progress_at -= 1801
                    if condition == "unknown": workbench.current_unknown_actions = lambda: True
                    if condition == "shutdown": shutdown.set()
                    if condition == "not_accepting": workbench.harness.accepting = False
                    if condition == "active_action": workbench.harness.active_action = "unfinished"
                    if condition == "active_program": workbench.has_active_programs = lambda: True
                    self.assertFalse(await wait_benchmark(workbench, shutdown))
                    workbench.session.run_turn.assert_not_called()
                    workbench.store.event.assert_not_called()
                    workbench.harness.home_check.check.assert_not_called()

    async def test_turn_failure_or_interruption_never_launches_hook(self):
        for mode in COMPLETION_GUIDANCE_MODES:
            for state in ("failed", "interrupted"):
                with self.subTest(mode=mode, state=state):
                    workbench = self.workbench(mode)
                    workbench.session.state = state
                    workbench.session.run_turn = Mock()
                    self.assertTrue(await wait_benchmark(workbench, asyncio.Event()))
                    workbench.session.run_turn.assert_not_called()
                    workbench.store.event.assert_not_called()

    async def test_hook_uses_completed_turn_status_without_matching_actor_prose(self):
        for answer in ("Done.", "I cannot continue.", "", None):
            with self.subTest(answer=answer):
                workbench = self.workbench("hook")
                workbench.session.turn_result["answer"] = answer
                workbench.session.run_turn = Mock(side_effect=lambda prompt: workbench.harness.episode_done.set())
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                workbench.session.run_turn.assert_called_once_with(HOOK_COMPLETION_GUIDANCE)

    async def test_home_or_unknown_uses_exact_original_continuation(self):
        for mode in ("hook", "both"):
            for status in ("home", "unknown", "unavailable"):
                with self.subTest(mode=mode, status=status):
                    workbench = self.workbench(mode)
                    if status == "unavailable":
                        workbench.harness.home_check = None
                    else:
                        workbench.harness.home_check.check.return_value = {"status": status}
                    workbench.session.run_turn = Mock(side_effect=lambda prompt: workbench.harness.episode_done.set())
                    self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                    workbench.session.run_turn.assert_called_once_with("任务还没完成，请继续。")
                    record = workbench.store.event.call_args.args[1]
                    self.assertEqual(record["continuation_kind"], "generic")
                    self.assertEqual(record["home_check"]["status"], "unknown" if status == "unavailable" else status)

    async def test_hook_rechecks_supervisor_and_unresolved_work_after_control_lock(self):
        for condition in ("native", "budget", "timeout", "unknown", "active_action", "active_program"):
            with self.subTest(condition=condition):
                workbench = self.workbench("hook")
                shutdown = asyncio.Event()
                workbench.session.run_turn = Mock()
                class ContendedLock:
                    async def __aenter__(self):
                        if condition == "native": workbench.harness.episode_done.set()
                        if condition == "budget": workbench.session.model_limit_reached.set()
                        if condition == "timeout": workbench.harness.started_at -= 1801; workbench.harness.last_native_progress_at -= 1801
                        if condition == "unknown": workbench.current_unknown_actions = lambda: True
                        if condition == "active_action": workbench.harness.active_action = "unfinished"
                        if condition == "active_program": workbench.has_active_programs = lambda: True
                    async def __aexit__(self, *arguments):
                        pass
                workbench.control_lock = ContendedLock()
                self.assertFalse(await wait_benchmark(workbench, shutdown))
                workbench.session.run_turn.assert_not_called()
                workbench.store.event.assert_not_called()
                workbench.harness.home_check.check.assert_not_called()

    async def test_dev_exit_after_turn_retains_single_turn_behavior(self):
        workbench = self.workbench()
        workbench.session.run_turn = Mock()
        self.assertTrue(await wait_turn(workbench.session, asyncio.Event(), 10, workbench.harness))
        workbench.session.run_turn.assert_not_called()
        self.assertFalse(workbench.harness.episode_done.is_set())

    async def test_disconnect_after_an_early_final_is_infrastructure_failure(self):
        workbench = self.workbench()
        workbench.session.state = "disconnected"
        workbench.session.run_turn = Mock()
        with self.assertRaisesRegex(RuntimeError, "benchmark_actor_disconnected"):
            await wait_benchmark(workbench, asyncio.Event())
        workbench.session.run_turn.assert_not_called()

    def test_existing_codex_continuation_reuses_thread_and_transport(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = Store(temporary.name, {})
        self.addCleanup(store.close)
        session = CodexSession.__new__(CodexSession)
        session.store, session.presentation = store, ActorPresentation(store)
        session.state, session.thread_id, session.model, session.effort = "completed", "same-thread", "fixture", "high"
        session.projection_lock = threading.Lock()
        session.turn_revision = 0
        session.turn_id, session.turn_result = "previous", None
        session.turn_done = threading.Event()
        session.turn_done.set()
        session.model_proxy = SimpleNamespace(request_count=169)
        session.client = SimpleNamespace(turn_start=Mock(return_value=SimpleNamespace(turn=SimpleNamespace(id="next"))))
        self.assertEqual(session.run_turn("continue"), "next")
        self.assertEqual(session.client.turn_start.call_args.args[0], "same-thread")
        self.assertEqual(session.model_proxy.request_count, 169)
        self.assertFalse(session.turn_done.is_set())

    async def test_pending_response_does_not_exempt_thirty_minutes_of_silence(self):
        h, s = self.workbench().harness, self.workbench().session
        h.started_at = h.last_native_progress_at = time.monotonic() - 1801
        s.model_proxy.pending_requests = True
        s.model_proxy.model_wait_seconds = 100000.0
        self.assertFalse(await wait_turn(s, asyncio.Event(), harness=h))
        self.assertEqual(h.wait_stop_reason, "runtime_inactivity_timeout")

    async def test_planner_gpu_oom_stops_turn_as_unscored_retryable_infrastructure(self):
        from auriga.apps.evaluation.retry import oom_evidence
        from auriga.robot.backends.robodojo.robot_process import GPU_OOM_FAILURE
        w = self.workbench()
        w.session.turn_done.clear()
        w.harness.robot = SimpleNamespace(failure="worker disconnected")
        w.harness.episode_done.set()
        self.assertFalse(await wait_turn(w.session, asyncio.Event(), harness=w.harness))
        self.assertEqual(w.harness.wait_stop_reason, "native_episode_ended")
        w.harness.episode_done.clear()
        w.harness.robot.failure = GPU_OOM_FAILURE
        self.assertFalse(await wait_turn(w.session, asyncio.Event(), harness=w.harness))
        self.assertEqual(w.harness.wait_stop_reason, "robot_gpu_out_of_memory")
        # A planner OOM is not a valid zero: it stays out of the score and qualifies for OOM retry.
        result = classify_episode_result(dict(benchmark=True, termination_reason=w.harness.wait_stop_reason,
            infrastructure_error=GPU_OOM_FAILURE, cleanup_status="completed", valid_episode=True,
            task_success=False, normalized_score=0, native_episode={"ended": False}), "unknown")
        self.assertEqual(result["primary_outcome"], "unscored_infrastructure")
        self.assertIsNone(result["normalized_score"])
        with tempfile.TemporaryDirectory() as directory:
            evidence = oom_evidence({"summary": result, "case": {"run_dir": directory}})
        self.assertEqual(evidence["signature"], "robot_gpu_out_of_memory:")

    def test_transport_recovery_gate_requires_settled_paused_world(self):
        w = self.workbench()
        w.closed = False
        block = lambda: Workbench.transport_recovery_block(w)
        self.assertIsNone(block())
        for change, reason in ((lambda: setattr(w.harness, "active_action", "action-1"), "action_in_progress"),
                               (lambda: setattr(w, "current_unknown_actions", lambda: True), "action_status_unknown"),
                               (lambda: setattr(w, "has_active_programs", lambda: True), "program_active"),
                               (lambda: w.harness.episode_done.set(), "native_episode_ended"),
                               (lambda: setattr(w.harness, "accepting", False), "harness_not_accepting"),
                               (lambda: setattr(w, "closed", True), "harness_not_accepting")):
            with self.subTest(reason=reason):
                change()
                self.assertEqual(block(), reason)
                w = self.workbench()
                w.closed = False

    async def test_inactivity_boundary_and_infrastructure_classification(self):
        w = self.workbench()
        w.harness.started_at = w.harness.last_native_progress_at = 100.0
        with patch("auriga.runtime.supervision.time.monotonic", return_value=1899.999):
            self.assertTrue(await wait_turn(w.session, asyncio.Event(), harness=w.harness))
        with patch("auriga.runtime.supervision.time.monotonic", return_value=1900.0):
            self.assertFalse(await wait_turn(w.session, asyncio.Event(), harness=w.harness))
        result = classify_episode_result(dict(benchmark=True, termination_reason=w.harness.wait_stop_reason,
            infrastructure_error=w.harness.wait_stop_reason, cleanup_status="completed",
            valid_episode=True, task_success=False, normalized_score=0, native_episode={"ended": False}), "stopped")
        self.assertEqual(result["primary_outcome"], "unscored_infrastructure")
        self.assertIsNone(result["task_success"])
        self.assertIsNone(result["normalized_score"])

    async def test_each_activity_source_keeps_a_long_running_case_alive(self):
        for source in ("session", "proxy", "robot", "programs", "native"):
            with self.subTest(source=source):
                w = self.workbench()
                h, s = w.harness, w.session
                h.started_at = h.last_native_progress_at = time.monotonic() - 100000
                h.robot, h.programs = SimpleNamespace(), SimpleNamespace()
                if source == "native":
                    h.last_native_progress_at = time.monotonic()
                else:
                    owner = {"session": s, "proxy": s.model_proxy, "robot": h.robot, "programs": h.programs}[source]
                    owner.last_activity_at = time.monotonic()
                self.assertTrue(await wait_turn(s, asyncio.Event(), harness=h))
                self.assertIsNone(h.wait_stop_reason)
                h.last_native_progress_at = h.started_at
                if source != "native": owner.last_activity_at = time.monotonic() - 1801
                self.assertFalse(await wait_turn(s, asyncio.Event(), harness=h))
                self.assertEqual(h.wait_stop_reason, "runtime_inactivity_timeout")

    async def test_terminal_api_error_after_retry_policy_surfaces_without_continuation(self):
        workbench=self.workbench()
        workbench.session.model_proxy.transport_blocked=True
        workbench.session.model_proxy.blocked_error={'error_type':'provider_error','automatic_retries':12,'retry_exhausted':True}
        workbench.session.run_turn=Mock()
        with self.assertRaisesRegex(RuntimeError,'model_transport_terminal_error'):
            await wait_benchmark(workbench,asyncio.Event())
        workbench.session.run_turn.assert_not_called()

    def test_actor_final_is_not_terminal_and_task_limits_score_failure_from_current_snapshot(self):
        partial = dict(benchmark=True, valid_episode=True, task_success=True, normalized_score=1.,
            native_episode={"ended": False, "success": None}, native_reward=1., native_process_score=100.,
            authority="installed RoboDojo task.run_reward / RewardManager", success_authority="native_final_check")
        cases = [("completed", None, "unscored_interrupted"),
                 ("stopped", "model_request_limit", "scored_failure"),
                 ("stopped", "official_no_action_progress_timeout", "scored_failure")]
        for status, reason, expected in cases:
            with self.subTest(status=status, reason=reason):
                result = classify_episode_result({**partial, "termination_reason": reason}, status)
                self.assertEqual(result["primary_outcome"], expected)
                if expected == "scored_failure":
                    self.assertTrue(result["scored_episode"])
                    self.assertFalse(result["task_success"])
                    self.assertEqual(result["normalized_score"], 1.)
                    self.assertEqual(result["native_process_score"], 100.)
                else:
                    self.assertFalse(result["scored_episode"])
                    self.assertIsNone(result["task_success"])
                    self.assertIsNone(result["normalized_score"])
                    self.assertEqual(result["partial_native_result"]["native_process_score"], 100.)
                    self.assertEqual(result["partial_native_result"]["success_authority"], "native_final_check")
                self.assertEqual(result["termination_authority"], "harness")
        native = {**partial, "native_episode": {"ended": True, "success": True}, "success_authority": "locked_native_terminal"}
        self.assertEqual(classify_episode_result(native, "completed")["primary_outcome"], "scored_success")


if __name__ == "__main__":
    unittest.main()
