"""Whole-subtask delegation to an external closed-loop policy (WALL-OSS via XPolicyLab).

The actor stays the planner. When it judges a subtask worth handing over it calls
``robot_policy_delegate``; this module then runs the policy's own loop:

    capture native inputs -> infer a chunk -> execute up to replan_steps rows as
    one EEChunkAction through Harness.execute -> repeat

until the step budget is spent, the episode ends, the policy fails or times out,
a chunk is not completed (including a native IK safety stop), the run is
stopping, the native budget is exhausted, or, at a supervised gate, the camera
supervisor (runtime/policy/progress_monitor.py) judges the subtask done or the
progress not as expected. Every chunk is an ordinary recorded
action with its own request, receipt and terminal observation; nothing bypasses
the single physical admission point. Policy inputs (raw RGB and EE state) go only
to the policy bridge. The actor receives a compact receipt plus the final
observation; evaluator state and success are never reported.

Each delegation is logged for gate research: ``policy_delegation_request`` and
``policy_delegation_receipt`` records, and ``policy_delegation.started`` /
``policy_delegation.finished`` events (source ``policy_delegation``) carrying the
correlation (thread, turn, tool call), the arguments, the outcome, steps and the
actor token-usage snapshot at both ends. See docs/design/policy-delegation.md.
"""
import time

from ...core.contracts import canonical, digest
from ...core.presentation.actor_presentation import strip_hashes
from ...core.presentation.feedback_delivery import execution_unconfirmed
from ...core.tool_output import StructuredText, ToolResponse
from ...cortex.context.delegation_gate import call_budget, fixed_budget, requires_instruction, supervised
from ...robot.model.arx_contract import EE_CHUNK_POSE_CONVENTION, FRAME_ID
from .progress_monitor import GptProgressMonitor, SEQUENCE_HISTORY
from .policy_capture import capture_path, capture_summary, discard_policy_capture

# Camera-supervisor stops (runtime/policy/progress_monitor.py); like max_policy_steps they end a delegation as finished.
MONITOR_STOP_REASONS = ("monitor_done", "monitor_takeover")
CAMERA_IDS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

POLICY_IDENTITY = dict(policy="wall-oss-1.0-lite-robodojo", protocol="xpolicylab-ws", adapter="WallX",
                       action_space="native dual-arm EE pose (xyz + qwxyz, env_origin) and gripper opening")


def _request(args, settings, native_remaining=None):
    fixed = call_budget(settings, native_remaining)
    # At a supervised gate the run fixes the budget from the remaining native budget; a supplied value is ignored.
    budget = fixed if fixed is not None else args.get("max_policy_steps",
                                                      min(settings.default_policy_steps, settings.max_policy_steps))
    bound = settings.call_budget_max_steps if fixed_budget(settings.gate) else settings.max_policy_steps
    if type(budget) is not int or not 1 <= budget <= bound:
        raise ValueError(f"invalid_max_policy_steps: expected integer 1..{settings.max_policy_steps}")
    if requires_instruction(settings.gate) and not (isinstance(args.get("instruction"), str) and args["instruction"].strip()):
        raise ValueError("instruction_required: write the policy instruction (native task wording or one object's complete action)")
    return dict(observation_id=args["observation_id"], instruction=args.get("instruction"),
                subtask_note=args.get("subtask_note"), max_policy_steps=budget)


def _stored(harness, delegation_id):
    request = harness.store.read("policy_delegation_request", delegation_id)
    try:
        receipt = harness.store.read("policy_delegation_receipt", delegation_id)
    except KeyError:
        # Started but never finished (for example the gateway restarted mid-loop).
        receipt = dict(delegation_id=delegation_id, status="unknown", stop_reason="chunk_unknown",
                       stop_detail="delegation_has_no_terminal_receipt")
    return _result(harness, request["public"], receipt)


def _result(harness, request, receipt):
    current = harness.latest["observation_id"] if harness.latest else None
    return dict(request=request, receipt=receipt, current_observation_id=current,
                result_observation_is_current=current is not None and receipt.get("final_observation_id") == current)


def _last_action(row):
    if row is None:
        return None
    return {side: dict(position_m=row[side + "_ee_pose"][:3], quaternion_wxyz=row[side + "_ee_pose"][3:],
                       gripper_command=row[side + "_ee_joint_state"][0]) for side in ("left", "right")}


def _stop_from_bridge(error):
    return ("policy_timeout" if getattr(error, "timeout", False) else "policy_error", str(error)[:500])


def run_delegation(harness, args, correlation):
    settings = harness.config.wall_oss
    # A supervised gate sizes the call from the rest of the episode's native budget; the none gate ignores this value.
    request = _request(args, settings, harness.action_budget().get("native_steps_remaining"))
    delegation_id = args.get("delegation_id") or "delegation-" + correlation["tool_call_id"]
    store = harness.store
    with harness.policy_lock:
        try:
            previous = store.read("policy_delegation_request", delegation_id)
        except KeyError:
            previous = None
        if previous is not None:
            if previous["request"] != request:
                raise ValueError("delegation_id_content_conflict")
            return _stored(harness, delegation_id)
        # Admission: nothing is recorded or executed when any of these fail.
        harness.task_planning.require(harness.identity)
        policy = harness.delegate_policy
        if policy is None:
            raise ValueError("wall_oss_not_enabled_for_run")
        if harness.latest is None or request["observation_id"] != harness.latest["observation_id"]:
            raise ValueError("stale_observation")
        with harness.lock:
            if not harness.accepting:
                raise ValueError("run_stopping_or_reconciliation_required")
            if harness.active_action:
                raise ValueError("action_writer_busy")
        if request["instruction"] is not None:
            instruction, source = request["instruction"], "actor"
        else:
            instruction, source = harness.latest.get("task_instruction"), "native_task"
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("native_task_instruction_unavailable: pass instruction explicitly")
        instruction = strip_hashes(instruction).strip()
        if not instruction:
            raise ValueError("policy_instruction_empty_after_scrubbing")
        public = dict(request, delegation_id=delegation_id, instruction=instruction, instruction_source=source,
                      replan_steps=settings.replan_steps, based_on_observation_id=request["observation_id"])
        task_id = harness.config.run.task
        started = time.monotonic()
        store.save("policy_delegation_request", delegation_id, dict(request=request, public=public,
            policy=POLICY_IDENTITY, task_id=task_id, gate=settings.gate, correlation=correlation))
        store.event("policy_delegation.started", dict(delegation_id=delegation_id, arguments=args, request=public,
            gate=settings.gate, task_id=task_id, action_budget=harness.action_budget(),
            token_usage=store.read_state("runtime", "token_usage")), correlation, source="policy_delegation")
        tokens_before = store.read_state("runtime", "token_usage")
        receipt, observer = _closed_loop(harness, policy, settings, public, task_id, correlation)
        receipt["action_budget"] = harness.action_budget()
        store.save("policy_delegation_receipt", delegation_id, receipt)
        elapsed = time.monotonic() - started
        tokens_after = store.read_state("runtime", "token_usage")
        call = gate_log(settings.gate, public, receipt, tokens_before, tokens_after, elapsed, correlation)
        call["monitor"] = monitor_mode(settings)
        call["monitor_summary"] = receipt.get("monitor")
        store.event("policy_delegation.finished", dict(delegation_id=delegation_id, receipt=receipt, **observer,
            elapsed_s=elapsed, token_usage=tokens_after, gate_log=call), correlation, source="policy_delegation")
        _update_episode_summary(store, settings.gate, call)
        return _result(harness, public, receipt)


def _total_tokens(snapshot):
    total = (snapshot or {}).get("total") if isinstance(snapshot, dict) else None
    value = total.get("total_tokens") if isinstance(total, dict) else None
    return value if type(value) is int else None


def gate_log(gate, public, receipt, tokens_before, tokens_after, elapsed_s, correlation):
    """One compact record per delegation for gate analysis (observer only, never shown to the actor)."""
    ik = receipt.get("ik_failed_steps") or {}
    return dict(delegation_id=public["delegation_id"], gate=gate, turn_id=correlation.get("turn_id"),
                tool_call_id=correlation.get("tool_call_id"), instruction=public["instruction"],
                instruction_source=public["instruction_source"], max_policy_steps=public["max_policy_steps"],
                policy_steps_used=receipt.get("policy_steps_used"), inference_calls=receipt.get("inference_calls"),
                status=receipt.get("status"), stop_reason=receipt.get("stop_reason"),
                stop_detail=receipt.get("stop_detail"), ik_failed_steps=dict(ik), ik_failed_total=sum(v for v in ik.values() if type(v) is int),
                native_steps_used_after=(receipt.get("action_budget") or {}).get("native_steps_used"),
                total_tokens_before=_total_tokens(tokens_before), total_tokens_after=_total_tokens(tokens_after),
                token_usage_before=tokens_before, token_usage_after=tokens_after, elapsed_s=round(elapsed_s, 3))


def _update_episode_summary(store, gate, call):
    """Per-episode totals under state runtime/policy_delegation; copied into summary.json at episode end."""
    summary = store.read_state("runtime", "policy_delegation") or dict(
        gate=gate, delegations=0, policy_native_steps=0, ik_failed_steps=0, stop_reasons={}, calls=[])
    summary.setdefault("policy_budget_steps", 0)
    summary["delegations"] += 1
    summary.setdefault("monitor", call.get("monitor", "none"))
    for key in ("monitor_checks", "monitor_errors", "monitor_tokens", "monitor_stops"):
        summary.setdefault(key, 0)
    monitor = call.get("monitor_summary") or {}
    summary["monitor_checks"] += monitor.get("checks", 0)
    summary["monitor_errors"] += monitor.get("errors", 0)
    summary["monitor_tokens"] += monitor.get("total_tokens", 0)
    if call["stop_reason"] in MONITOR_STOP_REASONS:
        summary["monitor_stops"] += 1
    summary["policy_budget_steps"] += call["max_policy_steps"] or 0
    summary["policy_native_steps"] += call["policy_steps_used"] or 0
    summary["ik_failed_steps"] += call["ik_failed_total"]
    reason = call["stop_reason"] or "unknown"
    summary["stop_reasons"][reason] = summary["stop_reasons"].get(reason, 0) + 1
    summary["calls"].append(dict({key: call.get(key) for key in ("delegation_id", "instruction", "policy_steps_used",
        "max_policy_steps", "stop_reason", "stop_detail", "ik_failed_total", "total_tokens_before", "total_tokens_after",
        "native_steps_used_after")}, monitor=monitor or None))
    store.set_state("runtime", "policy_delegation", summary)


def _closed_loop(harness, policy, settings, public, task_id, correlation):
    delegation_id = public["delegation_id"]
    budget, replan = public["max_policy_steps"], settings.replan_steps
    chunk_ids, inferences, observer_chunks = [], [], []
    steps, steps_known, last_row = 0, True, None
    ik = dict(left=0, right=0)
    pending, stop, status = [], None, None
    final_observation = None
    supervisor = _supervisor(harness, settings)
    monitor = dict(mode=monitor_mode(settings), checks=0, errors=0, verdicts=dict(normal=0, done=0, takeover=0),
                   total_tokens=0, streak=0, last=None) if supervisor is not None else None
    start_head = _camera_bytes(harness, harness.latest, ("cam_head",)).get("cam_head") if supervisor else None
    # Head images at the call start and at every check: (policy steps, bytes); the last one is the previous check.
    head_history = [(0, start_head)] if start_head else []
    chunks_since_check = 0
    try:
        try:
            policy.reset(task_id=task_id, context_key=("d" + digest(delegation_id)[:24]))
        except Exception as error:
            stop = _stop_from_bridge(error)
        if stop is None:
            try:
                pending = [harness.policy_observation()]
            except Exception as error:
                stop = ("policy_input_unavailable", str(error)[:500])
        while stop is None:
            if harness.episode_done.is_set():
                stop = ("episode_ended", None)
                break
            if harness.cancel.is_set() or not harness.accepting:
                stop = ("cancelled", "run_stopping")
                break
            remaining = budget - steps
            if remaining <= 0:
                stop = ("max_policy_steps", None)
                break
            native_remaining = harness.action_budget().get("native_steps_remaining")
            if native_remaining == 0:
                stop = ("native_budget_exhausted", None)
                break
            if harness.latest is None:
                stop = ("policy_input_unavailable", "current_observation_unavailable")
                break
            try:
                prediction = policy.infer([capture_path(item) for item in pending], public["instruction"])
            except Exception as error:
                stop = _stop_from_bridge(error)
                break
            finally:
                for item in pending:
                    discard_policy_capture(item)
                pending = []
            index = len(inferences)
            rows = prediction["actions"][:min(replan, remaining)]
            inferences.append(dict(index=index, sim_frame=prediction.get("sim_frame"),
                chunk_length=prediction.get("chunk_length"), executed_rows=len(rows),
                inference_seconds=prediction.get("inference_seconds"),
                memory_frame_ids=prediction.get("memory_frame_ids")))
            prediction_ref = harness.store.put_artifact(canonical(dict(delegation_id=delegation_id,
                inference_index=index, policy=POLICY_IDENTITY, instruction=public["instruction"],
                prediction=prediction)).encode(), "policy_prediction", "application/json", "observer")
            action = dict(schema_version="0.1", **harness.identity, world_kind="observed", type="EEChunkAction",
                action_id=f"{delegation_id}-chunk-{index:03d}",
                based_on_observation_id=harness.latest["observation_id"], frame_id=FRAME_ID,
                pose_convention=EE_CHUNK_POSE_CONVENTION, actions=rows, timeout_s=settings.chunk_timeout_s,
                policy_capture=True, ik_failure_stop_steps=settings.ik_failure_stop_steps,
                policy_provenance=dict(prediction_ref=prediction_ref, delegation_id=delegation_id,
                                       inference_index=index, executed_rows=len(rows), generator=POLICY_IDENTITY["policy"]))
            sink = []
            result = harness.execute(action, correlation, policy_capture=sink)
            chunk_ids.append(action["action_id"])
            receipt = result["receipt"]
            executed = receipt.get("executed_native_steps")
            if type(executed) is int and executed >= 0:
                steps += executed
            elif receipt.get("status") != "rejected":
                steps_known = False
            if executed and executed > 0:
                last_row = rows[min(executed, len(rows)) - 1]
            for side, counts in (receipt.get("ee_chunk_ik") or {}).items():
                if side in ik and type(counts.get("failed_steps")) is int:
                    ik[side] += counts["failed_steps"]
            observer_chunks.append(dict(action_id=action["action_id"], status=receipt.get("status"),
                reason=receipt.get("reason"), executed_native_steps=executed,
                capture=[capture_summary(item) for item in sink]))
            if receipt.get("result_observation_id"):
                final_observation = receipt["result_observation_id"]
            pending = sink
            chunk_status = receipt.get("status")
            if execution_unconfirmed(receipt):
                stop, status = ("chunk_unknown", receipt.get("reason")), "unknown"
                break
            if harness.episode_done.is_set():
                stop = ("episode_ended", None)
                break
            if chunk_status != "completed":
                stop = (("safety_stop", receipt.get("reason")) if receipt.get("reason") == "ik_failure_stop"
                        else ("chunk_" + (chunk_status if chunk_status in ("rejected", "failed", "stopped") else "unknown"),
                              receipt.get("reason")))
                break
            chunks_since_check += 1
            if supervisor is not None and chunks_since_check >= settings.monitor_every_chunks and steps < budget:
                chunks_since_check = 0
                fired = _supervise(harness, supervisor, settings, public, steps, head_history, monitor, correlation)
                if fired is not None:
                    stop = fired
                    break
            if not sink:
                stop = ("policy_input_unavailable", "chunk_returned_no_policy_capture")
                break
    finally:
        for item in pending:
            discard_policy_capture(item)
    reason, detail = stop
    if status is None:
        status = ("finished" if reason in ("max_policy_steps", "episode_ended", "native_budget_exhausted",
                                           *MONITOR_STOP_REASONS) else
                  "stopped" if reason in ("safety_stop", "cancelled", "chunk_stopped") else
                  "unknown" if reason == "chunk_unknown" else
                  "rejected" if not chunk_ids else "failed")
    receipt = dict(delegation_id=delegation_id, status=status, stop_reason=reason, stop_detail=detail,
                   policy_steps_used=steps if steps_known else None, max_policy_steps=budget, replan_steps=replan,
                   inference_calls=len(inferences), chunk_action_ids=chunk_ids, instruction=public["instruction"],
                   instruction_source=public["instruction_source"], subtask_note=public["subtask_note"],
                   final_observation_id=final_observation, ik_failed_steps=ik, last_policy_action=_last_action(last_row))
    if monitor is not None:
        receipt["monitor"] = {key: value for key, value in monitor.items() if key != "streak"}
    return receipt, dict(inferences=inferences, chunks=observer_chunks,
                         policy_client=getattr(policy, "metadata", None))


def monitor_mode(settings):
    """'gpt' when calls at this run's gate run under the camera supervisor, else 'none'."""
    return "gpt" if supervised(settings.gate) else "none"


def _supervisor(harness, settings):
    """The camera supervisor of this run, created once per harness; tests inject delegate_monitor."""
    if monitor_mode(settings) != "gpt":
        return None
    if getattr(harness, "delegate_monitor", None) is None:
        harness.delegate_monitor = GptProgressMonitor.from_settings(settings)
    return harness.delegate_monitor


def _camera_bytes(harness, observation, cameras=CAMERA_IDS):
    """Stored RGB bytes of the requested cameras of an observation bundle (missing cameras are skipped)."""
    out = {}
    for camera in (observation or {}).get("cameras") or []:
        if camera.get("camera_id") in cameras and camera.get("rgb_ref"):
            try:
                out[camera["camera_id"]] = harness.store.resolve_artifact(camera["rgb_ref"],
                                                                         ("actor", "internal", "observer"))[1]
            except Exception:
                continue
    return out


def _supervise(harness, supervisor, settings, public, steps, head_history, monitor, correlation):
    """One supervisor check after a chunk; returns (reason, detail) or None and appends the current head image to
    head_history (the start image, then one per check)."""
    images = _camera_bytes(harness, harness.latest)
    context = dict(task_instruction=(harness.latest or {}).get("task_instruction"), instruction=public["instruction"],
                   note=public.get("subtask_note"), steps=steps)
    # The head camera at the call start and at up to SEQUENCE_HISTORY earlier checks, oldest first.
    shown = head_history[:1] + head_history[1:][-SEQUENCE_HISTORY:]
    if shown:
        images["history"] = [data for _, data in shown]
    context.update(budget=public["max_policy_steps"], history_steps=[step for step, _ in shown])
    if not any(name in images for name in CAMERA_IDS):
        result = dict(verdict=None, error="no_camera_images")
    else:
        try:
            result = supervisor.check(context, images)
        except Exception as error:  # a monitor never ends a call by failing
            result = dict(verdict=None, error=f"{type(error).__name__}: {error}"[:300])
    monitor["checks"] += 1
    usage = result.get("usage") or {}
    total = usage.get("total_tokens")
    if type(total) is int:
        monitor["total_tokens"] += total
    verdict = result.get("verdict")
    if verdict in monitor["verdicts"]:
        monitor["verdicts"][verdict] += 1
        monitor["streak"] = monitor["streak"] + 1 if verdict == "takeover" else 0
    else:
        monitor["errors"] += 1   # counted as normal
        monitor["streak"] = 0
    monitor["last"] = dict(policy_steps=steps, verdict=verdict, confidence=result.get("confidence"),
                           reason=result.get("reason"), error=result.get("error"))
    harness.store.event("policy_delegation.monitor", dict(delegation_id=public["delegation_id"], policy_steps=steps,
        verdict=verdict, confidence=result.get("confidence"), reason=result.get("reason"), error=result.get("error"),
        usage=usage or None, latency_s=result.get("latency_s"), attempts=result.get("attempts"),
        model=result.get("model"), takeover_streak=monitor["streak"], completed=result.get("completed"),
        history_steps=context.get("history_steps")), correlation, source="policy_delegation")
    if images.get("cam_head"):
        head_history.append((steps, images["cam_head"]))
    if verdict == "done":
        return "monitor_done", f"supervisor at policy step {steps}: {result.get('reason') or 'subtask done'}"
    if verdict == "takeover" and monitor["streak"] >= settings.monitor_takeover_consecutive:
        return ("monitor_takeover", f"supervisor judged {monitor['streak']} checks in a row (last at policy step "
                                    f"{steps}) that the policy is not completing the instruction: "
                                    f"{result.get('reason') or ''}")
    return None


def delegation_response(runtime, args, correlation):
    """Tool handler: receipt first, then the final observation (images + measured state)."""
    result = runtime.policy_delegate(dict(args), correlation)
    content = [StructuredText(result, projection="owner_record")]
    observation_id = result["receipt"].get("final_observation_id")
    if observation_id:
        bundle = runtime.store.read("observation", observation_id)
        content += runtime.observation_content(bundle, correlation, structured=True)
    return ToolResponse(content, data=result, kind="delegation", data_projection="owner_record")
