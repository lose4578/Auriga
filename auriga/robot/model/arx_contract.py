"""Pure ARX X5 command limits; no worker, planner, or simulator imports."""
import math

from ...core.action_validation import (validate_hold_native_steps, validate_gripper_events,
    validate_action_speed, validate_dual_arm_progression)

FRAME_ID = "env_origin"
TCP_CONFIG_ID = "arx-x5-link6-v1"


def validate_action(action):
    """Validate one native EE or joint-prefix request without any worker access."""
    validate_dual_arm_progression(action)
    validate_action_speed(action)
    wait_steps = validate_hold_native_steps(action)
    if action.get("type") == "JointPrefixAction":
        validate_joint_prefix(action)
        return None, None
    if action.get("type") == "EEChunkAction":
        validate_ee_chunk(action)
        return "both", None
    if action.get("mode") != "absolute" or action.get("frame_id") != FRAME_ID:
        raise ValueError("unsupported_capability: absolute EE in env_origin required")
    if action.get("tcp_config_id") != TCP_CONFIG_ID:
        raise ValueError("unsupported_capability: TCP must be arx-x5-link6-v1")
    if action.get("limits_profile") not in {"robodojo-native-v1", "simulation-slow-v1"}:
        raise ValueError("unsupported_capability: limits_profile must be robodojo-native-v1")
    path = action.get("path", {})
    if not isinstance(path, dict):
        raise ValueError("invalid_action: path must be an object")
    points = path.get("waypoints", [])
    if path.get("kind") == "hold":
        if points:
            raise ValueError("unsupported_capability: hold requires empty waypoints")
    elif path.get("kind") != "backend_free" or not isinstance(points, list) or not 1 <= len(points) <= 8:
        raise ValueError("unsupported_capability: one to eight backend_free waypoints required")
    if path.get("blend_radius_m", 0) != 0:
        raise ValueError("unsupported_capability: blending")
    if path.get("orientation", "backend_free") != "backend_free":
        raise ValueError("unsupported_capability: orientation")
    arm, pose, arms = None, None, None
    for point in points:
        if not isinstance(point, dict) or not point or set(point) - {"left", "right"}:
            raise ValueError("unsupported_capability: one left/right arm or a paired target required")
        if (arms is not None and arms != set(point)
                and action.get("dual_arm_progression") != "independent"):
            raise ValueError("unsupported_capability: all waypoints must use the same arms")
        arms = set(point)
        arm, pose = ("both", point) if len(point) == 2 else next(iter(point.items()))
        for current in point.values():
            if not isinstance(current, dict):
                raise ValueError("invalid_pose: pose must be an object")
            for field, size in (("position_m", 3), ("quaternion_xyzw", 4)):
                value = current.get(field)
                if not isinstance(value, list) or len(value) != size or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value
                ):
                    raise ValueError("invalid_pose: finite position and xyzw quaternion required")
            norm = math.sqrt(sum(x * x for x in current["quaternion_xyzw"]))
            if abs(norm - 1) > 1e-3:
                raise ValueError("invalid_pose: quaternion must be normalized")
            if any(abs(x) > 2 for x in current["position_m"]):
                raise ValueError("out_of_bounds: EE target outside the configured 2 m envelope")
    for name, default, lower, upper in (("timeout_s", 20, 0.1, 300),):
        value = action.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"invalid_action: {name} must be in [{lower}, {upper}]")
    tolerance = path.get("position_tolerance_m", 0.005)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 < tolerance <= 0.10:
        raise ValueError("invalid_action: position_tolerance_m must be in (0, 0.10]")
    ori_tolerance = path.get("orientation_tolerance_rad", 0.1)
    if isinstance(ori_tolerance, bool) or not isinstance(ori_tolerance, (int, float)) or not math.isfinite(ori_tolerance) or not 0 < ori_tolerance <= math.pi:
        raise ValueError("invalid_action: orientation_tolerance_rad must be in (0, pi]")
    events = validate_gripper_events(action)
    if path.get("kind") == "hold" and not events and wait_steps is None:
        raise ValueError("invalid_action: hold requires a standalone gripper event")
    if action.get("dual_arm_progression") == "independent":
        return "both", {side: next(point[side] for point in reversed(points) if side in point)
                        for side in ("left", "right")}
    return arm, pose


def validate_joint_prefix(action):
    validate_dual_arm_progression(action)
    validate_action_speed(action)
    validate_hold_native_steps(action)
    if action.get("type") != "JointPrefixAction":
        raise ValueError("invalid_joint_prefix_type")
    rows = action.get("actions")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 50:
        raise ValueError("joint_prefix_requires_1_to_50_actions")
    for row in rows:
        if not isinstance(row, list) or len(row) != 14 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row
        ):
            raise ValueError("joint_prefix_requires_finite_14d_native_actions")
    timeout = action.get("timeout_s", 20)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0.1 <= timeout <= 300:
        raise ValueError("invalid_joint_prefix_timeout")
    if not isinstance(action.get("policy_provenance"), dict) or not action["policy_provenance"]:
        raise ValueError("joint_prefix_requires_policy_provenance")
    return rows


# Native RoboDojo end-effector action keys (eval_env.take_action infers the "ee"
# action type from *_ee_pose keys and solves IK per arm). Pose layout is
# [x, y, z, qw, qx, qy, qz] in the env_origin frame of the arm's ee link, the
# same convention as the native observation state; the gripper is a normalized
# opening command that the native interface clips to [0, 1].
EE_CHUNK_POSE_KEYS = ("left_ee_pose", "right_ee_pose")
EE_CHUNK_GRIPPER_KEYS = ("left_ee_joint_state", "right_ee_joint_state")
EE_CHUNK_POSE_CONVENTION = "xyz_qwxyz"
EE_CHUNK_MAX_ROWS = 50
EE_CHUNK_ENVELOPE_M = 2.0


def validate_ee_chunk(action):
    """Validate a policy-authored native EE-pose chunk without worker access.

    The same bounds Auriga applies to model-authored EE targets: finite values,
    a normalized quaternion and the configured 2 m position envelope. Rows are
    executed as the policy emitted them; nothing is re-solved or retimed here.
    """
    validate_dual_arm_progression(action)
    validate_action_speed(action)
    validate_hold_native_steps(action)
    if action.get("type") != "EEChunkAction":
        raise ValueError("invalid_ee_chunk_type")
    if action.get("frame_id") != FRAME_ID or action.get("pose_convention") != EE_CHUNK_POSE_CONVENTION:
        raise ValueError("ee_chunk_requires_env_origin_xyz_qwxyz_poses")
    rows = action.get("actions")
    if not isinstance(rows, list) or not 1 <= len(rows) <= EE_CHUNK_MAX_ROWS:
        raise ValueError("ee_chunk_requires_1_to_50_actions")
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(EE_CHUNK_POSE_KEYS + EE_CHUNK_GRIPPER_KEYS):
            raise ValueError("ee_chunk_row_requires_both_arm_poses_and_grippers")
        for key in EE_CHUNK_POSE_KEYS + EE_CHUNK_GRIPPER_KEYS:
            value, size = row[key], 7 if key in EE_CHUNK_POSE_KEYS else 1
            if not isinstance(value, list) or len(value) != size or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value
            ):
                raise ValueError("ee_chunk_requires_finite_native_values")
        for key in EE_CHUNK_POSE_KEYS:
            position, quaternion = row[key][:3], row[key][3:]
            if abs(math.sqrt(sum(x * x for x in quaternion)) - 1) > 1e-3:
                raise ValueError("invalid_pose: quaternion must be normalized")
            if any(abs(x) > EE_CHUNK_ENVELOPE_M for x in position):
                raise ValueError("out_of_bounds: EE target outside the configured 2 m envelope")
    timeout = action.get("timeout_s", 20)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0.1 <= timeout <= 300:
        raise ValueError("invalid_ee_chunk_timeout")
    limit = action.get("ik_failure_stop_steps")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= EE_CHUNK_MAX_ROWS):
        raise ValueError("invalid_ik_failure_stop_steps")
    if type(action.get("policy_capture", False)) is not bool:
        raise ValueError("invalid_policy_capture_flag")
    if not isinstance(action.get("policy_provenance"), dict) or not action["policy_provenance"]:
        raise ValueError("ee_chunk_requires_policy_provenance")
    return rows


def validate_preview_action(action):
    """Preview supports one native segment, with the same execution semantics."""
    # Composite requests use their own planners. Reject their type before the
    # leaf EE validator can mistake the missing single path for malformed EE.
    if action.get("type") in {"BatchAction", "ArmQueueAction"}:
        raise ValueError("dual_arm_motion_preview_unsupported")
    arm, target = validate_action(action)
    if arm == "both":
        raise ValueError("dual_arm_motion_preview_unsupported")
    if (action.get("type") in ("JointPrefixAction", "EEChunkAction") or action.get("hold_native_steps") is not None
            or action.get("path", {}).get("kind") != "backend_free"
            or len(action["path"].get("waypoints", [])) != 1 or arm is None):
        raise ValueError("motion_preview_requires_one_backend_free_waypoint")
    if any(action["path"].get(key) for key in ("constraints", "path_constraints", "constraint_pose")):
        raise ValueError("motion_preview_unsupported_path_constraints")
    return arm, target
