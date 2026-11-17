"""Process-level run supervision: register, serve, wait, evaluate and finish one run."""

import asyncio
import json
import os
import time
import signal
from pathlib import Path

from aiohttp import web

from ...core.contracts import canonical
from ...core.presentation.actor_payload import DEFAULT_ACTOR_PAYLOAD_PROFILE
from ..library.experiments import allocate_run, register_run, write_runtime, finish_run
from ...cortex.session.usage import load_price_card
from ..library.scoring import classify_episode_result
from ...runtime.supervision import INACTIVITY_REASON, model_budget_exhausted, robot_gpu_oom, start_with_shutdown, wait_turn
from ..evaluation.episode import wait_benchmark, episode_summary
from ...core.config import Config
from ...cortex.agents.base import session_backend_kind
from ...cortex.agents.factory import backend_requirements
from .workbench import DEFAULT_PROMPT, Workbench, _UNSET


async def serve(args):
    if (args.resume or args.replay) and not args.run_dir:
        raise ValueError("resume_and_replay_require_run_dir")
    resolved = getattr(args, "resolved_config", None)
    if resolved is not None and not args.replay and not args.resume:
        # Refuse an agent backend that cannot run this configuration before a run directory is registered. A resumed
        # run is gated by Workbench against its ARCHIVED configuration (manifest effective_config), which is what
        # the resumed session runs; the invocation's own configuration does not describe it.
        backend_requirements(resolved)
    card_path = getattr(args, "price_card", None)
    price_card = load_price_card(card_path) if card_path and not (args.resume or args.replay) else None
    root = Path(args.run_dir).expanduser().resolve() if args.run_dir else allocate_run(
        args.experiment_root, category=args.category, task=args.task, title=args.title, tags=args.tag, kind="workbench")
    if not args.replay:
        register_run(root, category=args.category, task=args.task, title=args.title, tags=args.tag, kind="workbench")
    workbench = Workbench(root, args.task, args.gpu, args.max_actions, resume=args.resume, replay=args.replay,
                          observation_profile=args.observation_profile, seed=args.seed, layout_id=args.layout_id,
                          price_card=price_card, pi05=getattr(args, "pi05", False), visual_review_model=getattr(args, "visual_review_model", None),
                          actor_payload_profile=getattr(args, "actor_payload_profile", DEFAULT_ACTOR_PAYLOAD_PROFILE),
                          enable_wait=getattr(args, "enable_wait", True), tracking_guard=getattr(args, "tracking_guard", False),
                          joint_replan=getattr(args, "joint_replan", _UNSET),
                          completion_guidance=getattr(args, "completion_guidance", _UNSET),
                          default_task_notes=getattr(args, "default_task_notes", True), vision_compare=getattr(args, "vision_compare", True),
                          motion_review=getattr(args, "motion_review", True), config=getattr(args, "resolved_config", None))
    # Runtime semantics come from the restored snapshot; the viewer port remains
    # a property of this invocation, not of the historical experiment.
    if args.resume:
        for key in ('task', 'seed', 'layout_id', 'gpu', 'max_actions', 'wall_budget_s',
                    'benchmark', 'prompt', 'exit_after_turn', 'task_notes', 'task_notes_source'):
            setattr(args, key, getattr(workbench.config.run, key))
    if args.replay:
        args.benchmark, args.exit_after_turn, args.wall_budget_s = False, False, None
    runner = web.AppRunner(workbench.app())
    done = asyncio.Event()
    heartbeat_task = budget_task = None
    final_status = "stopped"
    infrastructure_error = None
    bound_port = args.port

    async def heartbeat():
        while True:
            write_runtime(root, {**workbench.snapshot(), "endpoint": f"http://127.0.0.1:{bound_port}",
                                 "pid": os.getpid(), "kind": "workbench"})
            await asyncio.sleep(2)

    started_at = time.monotonic()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, done.set)
    try:
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", args.port)
        await site.start()
        bound_port = site._server.sockets[0].getsockname()[1]
        if not args.replay:
            heartbeat_task = asyncio.create_task(heartbeat())
        print(f"Workbench http://127.0.0.1:{bound_port}; run {root}", flush=True)
        try:
            await start_with_shutdown(workbench, done, prompt=args.prompt or DEFAULT_PROMPT, benchmark=args.benchmark,
                                  task_notes=args.task_notes, task_notes_source=args.task_notes_source,
                                  wall_budget_s=args.wall_budget_s)
        except asyncio.TimeoutError:
            await workbench.stop("official_startup_timeout")
            raise RuntimeError("official_startup_timeout") from None
        print(canonical(workbench.snapshot()), flush=True)
        if args.exit_after_turn or args.benchmark:
            finished = (await wait_benchmark(workbench, done) if args.benchmark
                        else await wait_turn(workbench.session, done, args.wall_budget_s, workbench.harness))
            if not finished or workbench.has_active_programs():
                await workbench.stop("actor_finished_with_active_programs" if finished else "shutdown-signal" if done.is_set() else getattr(workbench.harness, "wait_stop_reason", None) or "wall-budget")
            if getattr(workbench.harness, "wait_stop_reason", None) == INACTIVITY_REASON:
                infrastructure_error = INACTIVITY_REASON
            elif robot_gpu_oom(workbench.harness):
                infrastructure_error = workbench.harness.robot.failure
            native_terminal = getattr(workbench.harness, "episode_terminal", None)
            unresolved = (workbench.current_unknown_actions() or workbench.harness.active_action or workbench.has_active_programs()
                          or (not native_terminal and (workbench.session.state == "running" or workbench.status == "unknown")))
            final_status = "unknown" if unresolved else "completed" if native_terminal or (not args.benchmark and finished and workbench.session.state == "completed") else "stopped"
            if args.benchmark and unresolved:
                workbench.store.event("observer.finalization_skipped", {"reason": "motion_or_turn_stop_unconfirmed"})
            elif args.benchmark:
                async with workbench.control_lock:
                    evaluation = await asyncio.to_thread(workbench.observer_evaluate)
                    (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
                    if getattr(workbench, "config", Config()).features.export_video and getattr(workbench, "config", Config()).features.record_video:
                        video = await asyncio.to_thread(workbench.observer_video)
                        (root / "video.json").write_text(json.dumps(video, indent=2))
            print(canonical(dict(turn_result=workbench.session.turn_result)), flush=True)
            done.set()
        else:
            async def budget():
                await asyncio.sleep(args.wall_budget_s)
                await workbench.stop("wall-budget")
                await asyncio.to_thread(workbench.robot.close)
            if not args.replay and args.wall_budget_s is not None:
                budget_task = asyncio.create_task(budget())
            await done.wait()
    except Exception as error:
        final_status = "failed"
        infrastructure_error = str(error)
        workbench.store.event("run.infrastructure_failed", {"reason": infrastructure_error})
        raise
    finally:
        for task in (heartbeat_task, budget_task):
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        try:
            await runner.cleanup()
        finally:
            try:
                try:
                    if not args.replay:
                        native_turn = workbench.session.turn_result or {}
                        # Codex keeps its historical names; other agent backends use the neutral ones.
                        codex_backend = session_backend_kind(workbench.session) == "codex"
                        if native_turn.get("status") == "failed":
                            expected_limit_stop = model_budget_exhausted(workbench.session)
                            if expected_limit_stop:
                                workbench.harness.wait_stop_reason = "model_request_limit"
                                workbench.store.event("codex.expected_budget_stop" if codex_backend
                                                      else "agent.expected_budget_stop", {"turn_error":native_turn.get("error")})
                            elif final_status != "unknown" and not getattr(workbench.harness, "episode_terminal", None):
                                final_status = "failed"
                            if infrastructure_error is None and not expected_limit_stop:
                                error = native_turn.get("error") or {}
                                detail = error.get("message") if isinstance(error, dict) else str(error)
                                infrastructure_error = (("codex_turn_failed: " if codex_backend else "agent_turn_failed: ")
                                                        + str(detail or canonical(error))[:4096])
                        if workbench.current_unknown_actions() or workbench.harness.active_action or workbench.has_active_programs():
                            final_status = "unknown"
                        result_summary = episode_summary(workbench.store, workbench.identity,
                            workbench.session, workbench.harness, benchmark=args.benchmark,
                            final_status=final_status, started_at=started_at,
                            infrastructure_error=infrastructure_error)
                finally:
                    cleanup_report = await workbench.close()
                    if isinstance(cleanup_report, dict):
                        # Summary data is collected before closing the Store, but
                        # resource-release evidence must reflect the actual close.
                        result_summary = locals().get("result_summary", {"task_success": None, "normalized_score": None})
                        result_summary["cleanup"] = cleanup_report
                        robot_cleanup = cleanup_report.get("robot") or {}
                        if (cleanup_report.get("errors") or robot_cleanup.get("process_exit_confirmed") is False
                                or robot_cleanup.get("reader_exit_confirmed") is False):
                            result_summary["cleanup_status"] = "failed"
                            final_status = "unknown"
                        elif robot_cleanup.get("forced_termination"):
                            result_summary["cleanup_status"] = "forced_termination"
                        elif robot_cleanup.get("returncode") not in (None, 0):
                            result_summary["cleanup_status"] = "worker_exit_failed"
                        elif (robot_cleanup and robot_cleanup.get("native_close_acknowledged") is not True
                              and not robot_cleanup.get("not_started")):
                            result_summary["cleanup_status"] = "native_close_unconfirmed"
                        else:
                            result_summary["cleanup_status"] = "completed"
            finally:
                if not args.replay:
                    final_summary = locals().get("result_summary", {"task_success": None, "normalized_score": None})
                    # Resource cleanup may reveal that a nonnative stop was not
                    # confirmed. Reclassify from preserved evaluator evidence.
                    final_summary = classify_episode_result(final_summary, final_status)
                    finish_run(root, status=final_status, summary=final_summary)
