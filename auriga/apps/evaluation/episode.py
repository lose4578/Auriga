"""Benchmark continuation and observer-only final evidence, independent of HTTP.

This module consumes existing runtime owners and the shared scoring policy; it
never supplies evaluator facts to actor continuation or executes a robot action.
"""
import asyncio
import json
import time
from pathlib import Path

from ...core.contracts import IDENTITY, canonical
from ...cortex.supervision.actor_stall import (ACTOR_STALL_REASON, ActorStallTracker, ACTOR_PROGRESS_REASON,
                          ActorProgressTracker, completed_turn_activity)
from ...cortex.supervision.completion_guidance import COMPLETION_GUIDANCE_TRIGGER, hook_completion_guidance
from ...runtime.supervision import model_budget_exhausted, wait_turn
from ..library.scoring import classify_episode_result
from ...cortex.session.usage import account_usage
from ...cortex.agents.base import session_backend_kind

BENCHMARK_CONTINUATION = "任务还没完成，请继续。"


async def wait_benchmark(workbench, shutdown):
    """Continue early finals with bounded recovery; never supply task success."""
    session, harness = workbench.session, workbench.harness
    features = getattr(getattr(workbench, "config", None), "features", None)
    bounded = getattr(features, "bounded_continuation", False)
    evidence_aware = bounded and getattr(features, "recovery_evidence", False)
    stall = ActorProgressTracker(evidence_aware=evidence_aware) if bounded else ActorStallTracker()
    def program_state():
        inspect = getattr(workbench, 'program_continuation_state', None)
        return inspect() if inspect is not None else dict(blocked=workbench.has_active_programs(), pending_reviews=[])
    while True:
        finished = await wait_turn(session, shutdown, harness=harness)
        if finished and session.state == "disconnected":
            raise RuntimeError("benchmark_actor_disconnected")
        if not finished or session.state != "completed":
            return finished
        if harness.episode_done.is_set() or model_budget_exhausted(session):
            continue  # A terminal callback may race the completed-turn event.
        if (shutdown.is_set() or not harness.accepting or workbench.current_unknown_actions()
                or harness.active_action or program_state()['blocked']):
            # Reconciliation, explicit stop and executing or uncertain programs
            # stop continuation. A confirmed review pause needs an actor turn.
            harness.wait_stop_reason = harness.wait_stop_reason or "harness_stop_before_continuation"
            return False
        async with workbench.control_lock:
            if not session.turn_done.is_set() or session.state != "completed":
                continue
            # Recheck native terminal, budgets and elapsed time after waiting for
            # control ownership; a reminder never overrides a supervisor stop.
            if not await wait_turn(session, shutdown, harness=harness):
                return False
            if (shutdown.is_set() or not harness.accepting or workbench.current_unknown_actions()
                    or harness.active_action or program_state()['blocked']):
                harness.wait_stop_reason = "harness_stop_before_continuation"
                return False
            previous_turn = session.turn_id
            context = dict(thread_id=session.thread_id, turn_id=previous_turn,
                           episode_id=workbench.identity.get("episode_id"))
            if bounded:
                activity = completed_turn_activity(workbench.store, **context)
                evidence_args = {}
                if evidence_aware:
                    from ...cortex.supervision.recovery_evidence import completed_turn_evidence
                    try:
                        budget = harness.action_budget()
                    except (AttributeError, TypeError, ValueError):
                        budget = {}
                    remaining = budget.get('native_steps_remaining') if isinstance(budget, dict) else None
                    if type(remaining) is not int or remaining < 0:
                        remaining = None
                    evidence_args['evidence'] = completed_turn_evidence(workbench.store, **context,
                        native_action_count=harness.native_action_count, native_steps_remaining=remaining)
                observation = stall.observe(**context, native_action_count=harness.native_action_count,
                    activity=activity, **evidence_args)
            else:
                final_answer = getattr(session, "completed_final_answer", lambda: None)()
                observation = stall.observe(**context,
                    native_action_count=harness.native_action_count, text=final_answer)
            if observation:
                harness.actor_stall = observation
                if bounded or observation["matched_reason"] or observation["triggered"]:
                    workbench.store.event("benchmark.actor_progress" if bounded else "benchmark.actor_inability",
                        observation, {"thread_id": session.thread_id, "turn_id": previous_turn})
            # Public evidence collection and logging can race native/controller callbacks
            # even on the baseline turn, before a stall could be triggered.
            if harness.episode_done.is_set() or model_budget_exhausted(session):
                continue
            if (shutdown.is_set() or not harness.accepting or workbench.current_unknown_actions()
                    or harness.active_action or program_state()['blocked']):
                harness.wait_stop_reason = "harness_stop_before_continuation"
                return False
            if observation and observation["triggered"]:
                harness.wait_stop_reason = ACTOR_PROGRESS_REASON if bounded else ACTOR_STALL_REASON
                return False
            count = getattr(harness, "actor_continuation_count", 0) + 1
            mode = getattr(workbench, "completion_guidance", "none")
            initial_observation_id = getattr(workbench, "initial_observation_id", None)
            home_check_result = None
            if mode in {"hook", "both"}:
                home_check = getattr(harness, "home_check", None)
                home_check_result = (home_check.check(workbench.identity) if home_check is not None
                                     else {"status": "unknown", "reason": "home_check_unavailable"})
                # Reset/recovery starts a new episode with a new reference.
                initial_observation_id = home_check_result.get("initial_observation_id")
            needs_home = home_check_result is not None and home_check_result["status"] == "away"
            continuation = hook_completion_guidance(initial_observation_id) if needs_home else BENCHMARK_CONTINUATION
            recovery_context = None
            if bounded:
                from ...cortex.supervision.recovery_guidance import recovery_guidance
                recovery_context, recovery_prompt = recovery_guidance(harness, home_check_result,
                    **({'evidence': stall.recovery_context()} if evidence_aware else {}))
                continuation += "\n" + recovery_prompt
            if harness.episode_done.is_set() or model_budget_exhausted(session):
                continue
            programs = program_state()
            if (shutdown.is_set() or not harness.accepting or workbench.current_unknown_actions()
                    or harness.active_action or programs['blocked']):
                harness.wait_stop_reason = "harness_stop_before_continuation"
                return False
            if programs['pending_reviews']:
                continuation += ("\nA program is paused for your action review. Use the pending_review evidence "
                    "to approve or reject with program_run(operation='resume', job_id=..., review_id=..., "
                    "decision='approve' or 'reject'). The broker executes an approved cached plan; do not "
                    "execute it separately. Pending reviews: " + canonical(programs['pending_reviews']))
            harness.actor_continuation_count = count
            workbench.store.event("benchmark.actor_turn_continuation", {
                "count": count, "previous_turn_id": previous_turn,
                "thread_id": session.thread_id, "reason": "actor_turn_completed_before_native_terminal",
                "completion_guidance": mode, "trigger": COMPLETION_GUIDANCE_TRIGGER, "prompt": continuation,
                "initial_observation_id": initial_observation_id,
                "continuation_kind": "home_reminder" if needs_home else "generic",
                "home_check": home_check_result,
                **({"recovery_context": recovery_context} if bounded else {})})
            # run_turn reuses the existing thread and ModelRequestProxy; neither
            # native progress clocks nor cumulative model budget are re-created.
            await asyncio.to_thread(session.run_turn, continuation)


def observer_summary(store, identity, final_status):
    if final_status == "unknown": return {}
    evaluations = store.events_by_type(("observer.evaluated",), limit=1, newest_first=True)
    changes = store.events_by_type(("action.requested", "environment.reset_requested", "environment.recovery_reset"),
                                   limit=1, newest_first=True)
    if not evaluations or (changes and evaluations[0]["server_seq"] <= changes[0]["server_seq"]):
        return {}
    ref = evaluations[0]["payload"].get("result_ref")
    if not ref: return {}
    _, content = store.resolve_artifact(ref, allowed_scopes=("observer",))
    evaluation = json.loads(content)
    if any(evaluation.get(key) != identity[key] for key in IDENTITY): return {}
    native = evaluation.get("result", {})
    return {key: native[key] for key in ("task_success", "normalized_score", "valid_episode",
            "native_reward", "native_process_score", "success_authority", "authority") if key in native}


def episode_summary(store, identity, session, harness, *, benchmark, final_status,
                    started_at, infrastructure_error=None):
    receipts = store.records("action_receipt")
    reviews = []
    cursor = 0
    while page := store.events_by_type(("visual.model_reviewed",), after=cursor):
        reviews.extend(event["payload"] for event in page)
        cursor = page[-1]["server_seq"]
    result = {"action_count": harness.action_count,
              "completed_actions": sum(value.get("status") == "completed" for value in receipts),
              "failed_actions": sum(value.get("status") == "failed" for value in receipts),
              "task_success": None, "normalized_score": None, "thread_id": session.thread_id,
              "codex_turn_status": (session.turn_result or {}).get("status"),
              "agent_turn_status": (session.turn_result or {}).get("status"),
              "agent_backend": session_backend_kind(session),
              "token_usage": store.read_state("runtime", "token_usage"),
              "token_accounting": account_usage(store.read_state("runtime", "token_usage"),
                  model=session.model, provider=session.provider, price_card=session.price_card),
              "tool_calls": len(store.records("tool_call")),
              "elapsed_wall_s": time.monotonic() - started_at,
              "infrastructure_error": infrastructure_error,
              "runtime_protocol": "robodojo-native-v1",
              "benchmark": benchmark,
              "actor_continuation_count": getattr(harness, "actor_continuation_count", 0),
              "actor_stall": getattr(harness, "actor_stall", None),
              "native_episode": store.read_state("runtime", "native_episode"),
              "policy_delegation": store.read_state("runtime", "policy_delegation"),
              "model_requests": store.read_state("runtime", "model_requests"),
              # Provider-reported served model per delivered completion, and the relay-measured usage every backend
              # shares (None = not recorded); configured vs served and native vs relay usage are compared, not equated.
              "served_models": (store.read_state("runtime", "model_requests") or {}).get("served_models"),
              "model_usage_relay": store.read_state("runtime", "model_usage_relay"),
              "model_budget_exhausted": model_budget_exhausted(session),
              "visual_review_usage": [dict(model=value["model"], usage=value.get("usage")) for value in reviews],
              "token_accounting_scope": "main_actor_only; auxiliary visual model usage recorded separately",
              "termination_reason": getattr(harness, "wait_stop_reason", None)}
    result.update(observer_summary(store, identity, final_status))
    return classify_episode_result(result, final_status)

def record_evaluation(store, identity, result):
    evidence = {**identity, "result": result}
    reference = store.put_artifact(canonical(evidence).encode(), "native_evaluation", "application/json", "observer")
    store.event("observer.evaluated", {"result_ref": reference}, source="private_evaluator")
    return {**evidence, "audience": "observer", "result_ref": reference}


def record_video(store, identity, videos):
    result = {**identity, **videos}
    for item in result.get("videos", []):
        item["video_ref"] = store.put_artifact(Path(item["video_path"]).read_bytes(), "native_video", "video/mp4", "observer")
    store.event("observer.video_exported", result, source="private_evaluator")
    return result
