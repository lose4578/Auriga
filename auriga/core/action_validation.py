"""Pure motion data checks and arm projection, with no runtime dependencies."""
import math

MAX_HOLD_NATIVE_STEPS = 50
# Rows authored by an external learned policy in its own native action space.
# Harness speed presets, dual-arm progression and EE path options never retime them.
NATIVE_POLICY_ACTION_TYPES = frozenset({"JointPrefixAction", "EEChunkAction"})


def arm_action(action, arm):
    """Project a validated dual action to its complete own route without mutation.

    Per-arm speed fields override shared fields. Shared jaw speed only applies
    to arms with a jaw event; an explicit per-arm jaw speed still needs an event.
    """
    result = {key: value for key, value in action.items()
              if key not in ("arm_options", "dual_arm_progression")}
    path = action["path"]
    indices = [index for index, point in enumerate(path["waypoints"]) if arm in point]
    result["path"] = {**path, "waypoints": [{arm: path["waypoints"][index][arm]} for index in indices]}
    # Shared timing refers to the common index space. Per-arm timing below
    # already refers to that arm's shorter route and must not be padded.
    for name in ("waypoint_speed_presets", "waypoint_speed_scales"):
        if isinstance(result.get(name), list) and len(result[name]) == len(path["waypoints"]):
            result[name] = [result[name][index] for index in indices]
    events = action.get("gripper_events", [])
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("invalid_gripper: expected a list of event objects")
    result["gripper_events"] = [event for event in events if event.get("arm") == arm]
    if not result["gripper_events"]:
        for field in ("gripper_speed_preset", "gripper_speed_scale"):
            result.pop(field, None)
    options = action.get("arm_options", {}).get(arm, {})
    for preset, numeric in (("speed_preset", "speed_scale"), ("gripper_speed_preset", "gripper_speed_scale"),
                            ("waypoint_speed_presets", "waypoint_speed_scales")):
        if preset in options or numeric in options:
            result.pop(preset, None)
            result.pop(numeric, None)
    result.update(options)
    grasp_input = action.get("input_grasp_pose")
    if isinstance(grasp_input, dict) and arm in grasp_input:
        result["input_grasp_pose"] = grasp_input[arm]
    return result


def finite_vector(value, size, name):
    if not isinstance(value, list) or len(value) != size or any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError(f"invalid_{name}")


def validate_gripper_motion(value):
    """A command ramp over fractions of the final segment's native steps."""
    if not isinstance(value, dict) or set(value) != {"start_fraction", "end_fraction"}:
        raise ValueError("invalid_gripper_motion: requires start_fraction and end_fraction")
    start, end = value["start_fraction"], value["end_fraction"]
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end))
            or not 0 <= start < end <= 1):
        raise ValueError("invalid_gripper_motion: require 0 <= start_fraction < end_fraction <= 1")
    return value


def dual_path_shape(action):
    """Recognize paired routes or independent routes with trailing arm endings.

    Independent indices are local to each arm. Both routes start at index zero;
    after an arm ends it cannot reappear. Missing targets are not hold waypoints.
    """
    path = action.get("path")
    points = path.get("waypoints") if isinstance(path, dict) else None
    if not isinstance(points, list) or not points or not isinstance(points[0], dict):
        return False
    previous = {"left", "right"}
    if set(points[0]) != previous:
        return False
    independent = action.get("dual_arm_progression") == "independent"
    for point in points:
        if not isinstance(point, dict) or not point:
            return False
        current = set(point)
        valid = current <= previous if independent else current == previous
        if not valid:
            return False
        previous = current
    return True


def validate_gripper_events(action, capabilities=None):
    events = action.get("gripper_events", [])
    path = action.get("path", {})
    if not isinstance(path, dict):
        raise ValueError("invalid_action: path must be an object")
    points = path.get("waypoints", [])
    if not isinstance(points, list):
        raise ValueError("invalid_action: waypoints must be a list")
    paired = dual_path_shape(action)
    if paired and capabilities is not None and not capabilities.get("dual_arm", False):
        raise ValueError("unsupported_capability:arms")
    if not isinstance(events, list) or len(events) > (2 if paired else 1):
        raise ValueError("unsupported_capability: at most one gripper event per moving arm")
    seen = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("invalid_gripper: event must be an object")
        phase = event.get("phase")
        fields = {"arm", "phase", "goal_normalized"}
        if phase == "during_motion":
            fields |= {"start_fraction", "end_fraction"}
        if phase not in ("after_motion", "during_motion") or set(event) != fields:
            raise ValueError("unsupported_capability: gripper phase or event fields")
        if event["arm"] not in ("left", "right"):
            raise ValueError("invalid_gripper: arm must be left or right")
        if event["arm"] in seen:
            raise ValueError("invalid_gripper: duplicate arm event")
        seen.add(event["arm"])
        goal = event["goal_normalized"]
        if type(goal) not in (int, float) or not math.isfinite(goal) or not 0 <= goal <= 1:
            raise ValueError("invalid_gripper: goal_normalized must be in [0, 1]")
        if not paired and any(not isinstance(point, dict) or event["arm"] not in point for point in points):
            raise ValueError("invalid_gripper: event must match the moving arm")
        if phase == "during_motion":
            validate_gripper_motion({key: event[key] for key in ("start_fraction", "end_fraction")})
            if path.get("kind") != "backend_free" or not points:
                raise ValueError("invalid_gripper_motion: requires a moving arm")
        if capabilities is not None:
            supported = (phase in capabilities.get("gripper_phases", []) if phase == "during_motion"
                         else capabilities.get("gripper_after_motion", False))
            if not supported:
                raise ValueError("unsupported_capability:gripper_events:" + phase)
    return events


def validate_hold_native_steps(value, maximum=MAX_HOLD_NATIVE_STEPS):
    """A bounded wait advances native time without requesting a pose or jaw change."""
    if "hold_native_steps" not in value:
        return None
    steps = value["hold_native_steps"]
    if type(steps) is not int or not 1 <= steps <= min(maximum, MAX_HOLD_NATIVE_STEPS):
        raise ValueError(f"invalid_action: hold_native_steps must be an integer in [1, {min(maximum, MAX_HOLD_NATIVE_STEPS)}]")
    path = value.get("path", {})
    if (value.get("type", "EEAction") != "EEAction" or not isinstance(path, dict)
            or path.get("kind") != "hold" or path.get("waypoints", []) != []):
        raise ValueError("invalid_action: bounded wait requires EEAction hold with empty waypoints")
    if path.get("blend_radius_m", 0) != 0 or path.get("orientation", "backend_free") != "backend_free":
        raise ValueError("invalid_action: bounded wait cannot include motion path constraints")
    if value.get("gripper_events", []) != []:
        raise ValueError("invalid_action: bounded wait cannot include gripper events")
    return steps


def validate_arm_options(action, allowed_fields, capabilities=None):
    """Per-arm speeds apply to each complete own route, paired or independent."""
    options = action.get("arm_options")
    path = action.get("path")
    if (action.get("type") in NATIVE_POLICY_ACTION_TYPES or "hold_native_steps" in action
            or not isinstance(path, dict) or path.get("kind") != "backend_free"):
        raise ValueError("arm_options_requires_dual_arm_motion")
    if not dual_path_shape(action):
        raise ValueError("arm_options_requires_dual_arm_motion")
    if capabilities is not None and not capabilities.get("dual_arm", False):
        raise ValueError("unsupported_capability:arms")
    if not isinstance(options, dict) or set(options) - {"left", "right"}:
        raise ValueError("invalid_arm_options: expected left/right speed selections")
    for selection in options.values():
        if not isinstance(selection, dict) or set(selection) - set(allowed_fields):
            raise ValueError("invalid_arm_options: unsupported speed selection fields")
    return options


def validate_dual_arm_progression(action, capabilities=None):
    """Explicit synchronization semantics apply only to resampled dual paths."""
    if "dual_arm_progression" not in action:
        return "paired"
    mode = action["dual_arm_progression"]
    if not isinstance(mode, str) or mode not in ("paired", "independent"):
        raise ValueError("invalid_dual_arm_progression: expected paired or independent")
    path = action.get("path")
    if (action.get("type", "EEAction") != "EEAction" or "hold_native_steps" in action
            or not isinstance(path, dict) or path.get("kind") != "backend_free"):
        raise ValueError("dual_arm_progression_requires_dual_arm_motion")
    if not dual_path_shape(action):
        raise ValueError("dual_arm_progression_requires_dual_arm_motion")
    if capabilities is not None:
        support = capabilities.get("dual_arm_progression", {})
        if (not capabilities.get("dual_arm", False) or not isinstance(support, dict)
                or support.get("available") is not True or mode not in support.get("modes", [])):
            raise ValueError("unsupported_capability:dual_arm_progression")
    return mode


def validate_action_speed(value, capabilities=None):
    """Request/waypoint command timing; never silently retime waits or policy rows."""
    if {"speed_preset", "gripper_speed_preset", "waypoint_speed_presets"}.intersection(value):
        raise ValueError("unresolved_action_speed_preset: resolve presets before the numeric action boundary")
    if "arm_options" in value:
        validate_arm_options(value, ("speed_scale", "gripper_speed_scale", "waypoint_speed_scales"), capabilities)
        for arm in ("left", "right"):
            validate_action_speed(arm_action(value, arm), capabilities)
    fields = {"speed_scale": (.25, 2.), "gripper_speed_scale": (.25, 1.)}
    present = set(fields).intersection(value)
    has_waypoint_speeds = "waypoint_speed_scales" in value
    if (present or has_waypoint_speeds) and (value.get("type") in NATIVE_POLICY_ACTION_TYPES or "hold_native_steps" in value):
        raise ValueError("action_speed_unsupported_for_wait_or_joint_prefix")
    if "speed_scale" in present and (not isinstance(value.get("path"), dict)
                                     or value["path"].get("kind") != "backend_free"):
        raise ValueError("speed_scale_requires_motion_path")
    if "gripper_speed_scale" in present and not value.get("gripper_events"):
        raise ValueError("gripper_speed_scale_requires_gripper_event")
    for name in present:
        lower, upper = fields[name]
        scale = value[name]
        if type(scale) not in (int, float) or not math.isfinite(scale) or not lower <= scale <= upper:
            raise ValueError(f"invalid_{name}: expected finite number in [{lower}, {upper}]")
        if scale != 1 and capabilities is not None:
            settings = capabilities.get("action_speed", {})
            if settings.get("available") is not True:
                raise ValueError("unsupported_capability:action_speed")
    if has_waypoint_speeds:
        path = value.get("path")
        if not isinstance(path, dict) or path.get("kind") != "backend_free":
            raise ValueError("waypoint_speed_scales_requires_motion_path")
        scales = value["waypoint_speed_scales"]
        points = path.get("waypoints", [])
        if not isinstance(points, list) or not isinstance(scales, list) or not scales or len(scales) != len(points):
            raise ValueError("invalid_waypoint_speed_scales: expected one numeric entry per waypoint")
        for scale in scales:
            if type(scale) not in (int, float) or not math.isfinite(scale) or not .25 <= scale <= 2.:
                raise ValueError("invalid_waypoint_speed_scales: expected finite numbers in [0.25, 2.0]")
        if capabilities is not None:
            settings = capabilities.get("action_speed", {})
            if (settings.get("available") is not True
                    or settings.get("waypoint_speed_scales", {}).get("available") is not True):
                raise ValueError("unsupported_capability:waypoint_speed_scales")
    return value
