"""Opt-in numerical action deliveries; full observations stay in the archive."""
import math
from copy import deepcopy

from ...core.action_validation import NATIVE_POLICY_ACTION_TYPES
from ...core.observation_policy import allows_depth


def numeric_summary(result):
    """Actor projection; immutable measurements and program values stay exact."""
    result = deepcopy(result)
    for field in ("schema_version", "run_id", "episode_id", "env_instance_id", "env_generation"):
        result.pop(field, None)
    if "observation_id" in result or "before_observation_id" in result:
        result.pop("observation_ids", None)

    def rounded(value):
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if type(value) is float:
            return round(value, 6)
        return value

    return rounded(result)


def _small_translation(value, bound):
    if not isinstance(value, list) or len(value) != 3:
        return False
    if any(type(number) not in (int, float) or not math.isfinite(number) for number in value):
        return False
    return math.sqrt(sum(number * number for number in value)) <= bound


def compact_action_feedback(harness, result, measurements, correlation):
    """Suppress image delivery only on measured uneventful intermediate moves.

    Opening/closing, uncertain evidence, exceptions, waits and every third action
    keep ordinary RGB delivery. This changes presentation only, never execution.
    """
    if (not harness.depth_event_payload or not allows_depth(harness.observation_profile)
            or correlation.get("program_id")):
        return None
    action, receipt = result["request"], result["receipt"]
    if not harness.latest or receipt.get("result_observation_id") != harness.latest["observation_id"]:
        return None
    if (receipt.get("status") != "completed" or not receipt.get("result_observation_id")
            or action.get("gripper_events") or "hold_native_steps" in action
            or action.get("type") in NATIVE_POLICY_ACTION_TYPES or not measurements):
        return None
    for measurement in measurements:
        if measurement.get("status") != "supported":
            return None
        value = measurement.get("value") or {}
        if value.get("evidence") != "rigid_surface_motion":
            return None
        # A stationary target is informative during an approach; a held target
        # may move in world coordinates but must remain aligned with the gripper.
        stationary = (_small_translation(value.get("world_displacement_m"), .003)
                      and type(value.get("world_rotation_deg")) in (int, float)
                      and abs(value["world_rotation_deg"]) <= 3)
        retained = (_small_translation(value.get("relative_gripper_displacement_m"), .003)
                    and type(value.get("relative_gripper_rotation_deg")) in (int, float)
                    and abs(value["relative_gripper_rotation_deg"]) <= 3)
        if not (stationary or retained):
            return None
    key = "depth_event_delivery_" + str(harness.identity["env_generation"])
    saved = harness.store.read_state("depth_delivery", action["action_id"])
    if saved is not None:
        count = saved["count"]
    else:
        count = harness.store.read_state("runtime", key, 0) + 1
        harness.store.set_state("runtime", key, count)
        harness.store.set_state("depth_delivery", action["action_id"], dict(count=count))
    if count % 3 == 0:
        return None
    summary = dict(action_id=action["action_id"], status=receipt["status"],
                   result_observation_id=receipt["result_observation_id"],
                   depth_derived=True, depth_feedback=measurements,
                   delivery="numeric_checkpoint", task_success="not_assessed")
    for field in ("executed_native_steps", "executed_control_steps", "position_error_m", "orientation_error_rad"):
        if field in receipt:
            summary[field] = receipt[field]
    harness.store.event("depth.numeric_delivery", dict(action_id=action["action_id"],
        observation_id=receipt["result_observation_id"], image_delivery="on_demand", periodic_rgb_every=3), correlation)
    return summary
