"""Broker-owned program review and execution evidence, never stdout parsing."""
from copy import deepcopy

from ...core.presentation.actor_payload import lean_action_result
from ...core.presentation.actor_presentation import ActorPresentation
from ...core.contracts import canonical
from .program_result import program_result
from ...core.tool_output import ToolResponse


LEGACY_ACTION_TOOLS = frozenset({
    "robot_execute_ee", "robot_move_ee", "robot_move_grasp", "robot_move_dual_grasp",
    "robot_gripper", "robot_execute_plan", "robot_execute_queue",
    "robot_policy_step", "robot_policy_delegate", "depth_move",
})


def reviewed_plan(response):
    value = response.data if isinstance(response, ToolResponse) and response.kind == 'plan' else None
    if not isinstance(value, dict) or not isinstance(value.get('plan_id'), str):
        raise ValueError('program_action_preview_missing_plan')
    if response.get('success') is False or value.get('status') not in {'planned', 'awaiting_observation'}:
        raise ValueError('program_action_preview_not_executable')
    return deepcopy(value)


def review_execution_state(plan):
    """The cached execution segment whose approval the main actor reviewed."""
    return {name: deepcopy(plan.get(name, 0 if name in {"executed_batch_native_steps", "plan_revision"} else None))
            for name in ("plan_id", "status", "trajectory_ref", "last_action_id",
                         "executed_batch_native_steps", "plan_revision")}


def action_evidence(store, tool, arguments, response, call_id, *, physical=None):
    """Capture actual dispatched action replies with their original references."""
    if physical is False or (physical is None and tool != 'robot_wait' and
            (tool != 'robot_action' or arguments.get('mode', 'preview') not in {'execute', 'direct'})):
        return None
    value = response.data if isinstance(response, ToolResponse) and response.kind == "action" else {}
    value = value if isinstance(value, dict) else {}
    receipt = value.get("receipt", {})
    receipt = receipt if isinstance(receipt, dict) else {}
    request = value.get("request", {})
    request = request if isinstance(request, dict) else {}
    status = receipt.get('status')
    result = lean_action_result(dict(request=request, receipt=receipt))["receipt"]
    result.update(call_id=call_id, tool=tool, status=status if isinstance(status, str) else 'unknown',
                  dispatch_success=response.get("success"),
                  action_id=receipt.get("action_id") or value.get("action_id") or request.get("action_id"),
                  plan_id=value.get("plan_id", arguments.get("plan_id")),
                  result_observation_id=value.get("current_observation_id") or receipt.get("result_observation_id"))
    based_on = request.get("based_on_observation_id") or arguments.get("observation_id")
    if isinstance(based_on, str) and based_on:
        result["based_on_observation_id"] = based_on
    if isinstance(value.get("next_action"), dict):
        result["next_action"] = deepcopy(value["next_action"])
    if isinstance(value.get("repair_action"), dict):
        result["repair_action"] = deepcopy(value["repair_action"])
    if status in ("stopped", "failed") and isinstance(value.get("episode_ledger"), dict):
        result["episode_ledger"] = deepcopy(value["episode_ledger"])
    if "repeat_failure_warning" in value:
        result["repeat_failure_warning"] = deepcopy(value["repeat_failure_warning"])
    plan = response.data if isinstance(response, ToolResponse) and response.kind == 'plan' else None
    if (tool == 'robot_action' and arguments.get('mode') == 'direct'
            and isinstance(plan, dict) and response.contract_error is None
            and plan.get('status') == 'failed' and plan.get('physics_advanced') is False
            and type(plan.get('executed_native_steps')) is int and plan['executed_native_steps'] == 0):
        # A direct call can stop during planning without producing an action receipt.
        result.update(status='not_executed', phase='planning', physics_advanced=False,
                      executed_native_steps=0)
        for name, item in plan.items():
            if name not in result:
                result[name] = deepcopy(item)
    if value.get('current_observation_id') and value['current_observation_id'] != receipt.get('result_observation_id'):
        result['motion_observation_id'] = receipt.get('result_observation_id')
    if isinstance(value.get('settling'), dict):
        result['settling'] = deepcopy(value['settling'])
    try:
        result["reply_ref"] = store.put_artifact(canonical(ActorPresentation(store).present_response(response)).encode(),
            "program_action_reply", "application/json")
    except Exception as error:
        # A reply-copy failure must not erase an already established action outcome.
        result["reply_archive_error"] = dict(reason="program_action_reply_archival_failed",
                                            error_type=type(error).__name__)
    return result


def live_observation_id(tool, response):
    if tool != "robot_observe":
        return None
    bundle = response.data if isinstance(response, ToolResponse) and response.kind == 'observation' else None
    if isinstance(bundle, dict) and isinstance(bundle.get('observation_id'), str):
        return bundle['observation_id']
    return None


def trusted_summary(store, evidence):
    statuses = {}
    for item in evidence:
        status = item.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    known_steps = [item["executed_native_steps"] for item in evidence
                   if type(item.get("executed_native_steps")) is int]
    settling = [item["settling"] for item in evidence if isinstance(item.get("settling"), dict)]
    settling_steps = [item["executed_native_steps"] for item in settling
                      if type(item.get("executed_native_steps")) is int]
    result = dict(source_kind="program_broker_action_replies", action_calls=len(evidence),
                  status_counts=statuses, executed_native_steps_known=sum(known_steps)+sum(settling_steps),
                  action_native_steps_known=sum(known_steps), settling_native_steps_known=sum(settling_steps),
                  actions_without_step_count=len(evidence)-len(known_steps),
                  settling_actions_without_step_count=len(settling)-len(settling_steps),
                  recent_actions=deepcopy(evidence), task_success="not_assessed",
                  semantics="Recorded action replies only. Process exit and stdout do not establish motion, contact, retention or task success.")
    if len(evidence) > 8:
        result["all_actions_ref"] = store.put_artifact(canonical(evidence).encode(), "program_action_evidence", "application/json")
    return result
