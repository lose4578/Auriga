"""Static registrations; a new ToolSpec requires no dispatcher edits."""
from copy import deepcopy
from functools import partial

from ...perception.depth.depth_runtime import DEPTH_KINDS, DEPTH_TOOLS
from ...core.observation_policy import DEVELOPMENT_TOOLS, allows_depth, has_calibration
from . import handlers
from .definitions import public_definitions
from .output_contracts import CONTRACTS, ENVELOPE, UNCONSTRAINED
from .registry import ToolRegistry, ToolSpec


# Retired tool names remain valid disabled_tools entries so archived configurations still launch.
CONFIG_TOOL_NAMES = frozenset({
    'depth_move', 'grasp_memory', 'grasp_plan', 'manipulation_run', 'operation_verify', 'robot_execute_ee',
    'robot_execute_plan', 'robot_execute_queue', 'robot_gripper', 'robot_inspect_trajectory',
    'robot_move_dual_grasp', 'robot_move_ee', 'robot_move_grasp', 'robot_preview_cost', 'robot_preview_motion',
    'scene_capabilities', 'scene_job_status', 'scene_query', 'scene_snapshot', 'scene_update', 'skill_run',
})


def unavailable(name, runtime, args):
    profile = runtime.observation_profile
    if name in DEVELOPMENT_TOOLS | {'robot_contact_overlay', 'robot_action'} and not has_calibration(profile):
        return 'tool_unavailable_without_permitted_calibration'
    depth_query = name == 'depth_query' or (name == 'geometry_query' and args.get('request', {}).get('kind') in DEPTH_KINDS)
    if name in DEPTH_TOOLS or depth_query:
        if not allows_depth(profile):
            return 'depth_not_enabled_for_run'
        if depth_query and 'depth_query' in runtime.disabled_tools:
            return 'tool_disabled_by_config:depth_query'
    if name == 'robot_action':
        capability = (runtime.capabilities or {}).get('unified_action', {})
        if capability.get('available') is False or runtime.features.motion_execution_mode != 'resampled':
            return 'unified_action_unavailable_for_backend'
    if name == 'robot_wait' and not runtime.enable_wait:
        return 'bounded_wait_not_enabled_for_run'
    if name == 'vision_compare' and not runtime.vision_compare:
        return 'vision_compare_not_enabled_for_run'
    if name in {'vision_review', 'operation_verify'} and not runtime.visual_review:
        return 'visual_review_not_enabled_for_run'
    if name == 'vision_review' and args.get('evidence_mode') == 'operation' and 'operation_verify' in runtime.disabled_tools:
        return 'tool_disabled_by_config:operation_verify'
    if name == 'robot_policy_step' and not runtime.policy:
        return 'policy_not_enabled_for_run'
    if name == 'robot_policy_delegate' and not runtime.delegate_policy:
        return 'wall_oss_not_enabled_for_run'
    if name in {'program_run', 'program_status'} and not runtime.programs:
        return 'program_runner_not_started'
    return None


def action_effect(args):
    mode = args.get('mode', 'preview')
    if mode == 'preview' and args.get('auto_execute_if_clean'):
        return 'physical'
    return {'preview': 'plan', 'inspect': 'read', 'execute': 'physical', 'direct': 'physical', 'execute_if_clean': 'physical'}[mode]


def memory_effect(args):
    request = args.get('request', {})
    if request.get('kind') in {'help', 'query'}:
        return 'read'
    return 'shared_memory' if request.get('scope') == 'shared' else 'record'


def plan_effect(args):
    return 'read' if args.get('operation') == 'read' else 'record'


def geometry_effect(args):
    return 'record' if args.get('request', {}).get('kind') in {'register_target', 'refresh_target', 'forget_target', 'grasp_candidates', 'place_candidates'} else 'read'


# Request-dependent ToolSpec.parallel classifiers (moved verbatim from the former
# codex.py name allowlist). True = mutates no Harness/world state for this request.
def geometry_parallel(args):
    # Only an explicit commit_scene (top level or inside request) is ordered.
    request = args.get('request')
    commit = args.get('commit_scene') or (isinstance(request, dict) and request.get('commit_scene'))
    return not commit


def depth_parallel(args):
    # RGB-D kinds (measurements, candidates, target handles) are ordered.
    request = args.get('request')
    return isinstance(request, dict) and request.get('kind') not in DEPTH_KINDS


def observe_parallel(args):
    # Re-cropping a saved observation captures nothing; a fresh capture replaces harness.latest.
    return args.get('observation_id') is not None and args.get('crop') is not None


def plan_parallel(args):
    return args.get('operation') == 'read'


def memory_parallel(args):
    request = args.get('request')
    return isinstance(request, dict) and request.get('kind') in {'help', 'query'}


def _bind(handler, runtime):
    def invoke(context, arguments, correlation):
        if runtime is None:
            raise ValueError('builtin_tool_runtime_not_bound')
        return handler(runtime, arguments, correlation)
    return invoke


def output_schemas(name):
    # Field contracts describe typed results; they are not a second robot/job
    # state machine. Owners still validate observations, plans and receipts.
    if name in UNCONSTRAINED:
        return dict(data=dict(description='Compatibility: ' + UNCONSTRAINED[name] + '; only JSON serialization is required.'))
    if name in CONTRACTS:
        return deepcopy(CONTRACTS[name])
    return dict(data=deepcopy(ENVELOPE))


def create_registry(extra_specs=(), *, runtime=None):
    from ...perception.external.vggt_height import mode as vggt_height_mode
    from ...perception.external.object_table import enabled as object_table_enabled
    from ...perception.external.scene3d import enabled as scene3d_enabled
    from ...perception.external.obs3d import flags as obs3d_flags, action_overlay as obs3d_action_overlay
    from ...perception.external.pose_skill import enabled as pose_skill_enabled
    selected, object_table, scene3d = vggt_height_mode(runtime), object_table_enabled(runtime), scene3d_enabled(runtime)
    observe3d = obs3d_flags(runtime)
    if any(observe3d):
        observe3d = (*observe3d, obs3d_action_overlay(runtime))
    pose_skill = pose_skill_enabled(runtime)
    config = getattr(runtime, 'config', None)
    if hasattr(config, 'wall_oss'):
        from ...cortex.context.delegation_gate import monitor_steps, published_budget
        gate, budget = config.wall_oss.gate, published_budget(config.wall_oss)
        monitor = (monitor_steps(config.wall_oss), config.wall_oss.monitor_takeover_consecutive,
                   config.wall_oss.call_budget_fraction)
    else:
        gate, budget, monitor = 'none', 400, (64, 1, 0.8)
    definitions = {item['name']: item for item in (public_definitions(vggt_height=selected or False, object_table=object_table,
                                                                      scene3d=scene3d, observe3d=observe3d,
                                                                      pose_skill=pose_skill, wall_oss_gate=gate,
                                                                      wall_oss_call_budget=budget, wall_oss_monitor=monitor)
                   if (selected or object_table or scene3d or any(observe3d))
                   else public_definitions(pose_skill=pose_skill, wall_oss_gate=gate, wall_oss_call_budget=budget,
                                           wall_oss_monitor=monitor))}
    effects = dict(robot_action=action_effect, robot_wait='physical', robot_policy_step='physical',
        robot_policy_delegate='physical',
        program_run='physical', task_plan=plan_effect, operation_memory=memory_effect,
        history_annotate_action='record', depth_query=geometry_effect, geometry_query=geometry_effect)
    parallel = dict(geometry_query=geometry_parallel, depth_query=depth_parallel, robot_observe=observe_parallel,
        task_plan=plan_parallel, operation_memory=memory_parallel, robot_information=True,
        robot_contact_overlay=True, robot_action_status=True, robot_capabilities=True, history_list_actions=True,
        history_view_observation=True, history_read_trace=True, program_status=True, vision_compare=True,
        pose_plan=True)
    specs = []
    for name, definition in definitions.items():
        roles = {'actor', 'program', 'trusted_legacy'}
        if name in {'program_run', 'program_status', 'robot_policy_step', 'robot_policy_delegate'}:
            roles.remove('program')
        legacy_schema = None
        if name == 'task_plan':
            # These trusted adapters already provide their historical field diagnostics.
            legacy_schema = {'type': 'object'}
        specs.append(ToolSpec(definition, _bind(getattr(handlers, name), runtime), frozenset(roles),
            effects.get(name, 'read'), partial(unavailable, name), legacy_schema,
            api_version='1.1' if name == 'robot_action' else '1.0',
            output_schemas=output_schemas(name), parallel=parallel.get(name, False)))
    return ToolRegistry([*specs, *extra_specs])
