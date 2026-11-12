"""One actor action interface for complete cached motion batches and inspection."""
from copy import deepcopy

from ...core.presentation.actor_payload import lean_action_result
from ...core.contracts import IDENTITY, canonical, validate_action
from ...robot.motion.grip_guard import GRIP_GUARD_FIELDS, validate_grip_guard_fields
from ...robot.motion.motion_safety import JOINT_TRAVEL_FIELDS, extract_joint_travel_options
from ...robot.motion.motion_speed import resolve_action_speed
from ...robot.motion.motion_tolerance import STALL_GUARD_FIELDS, TOLERANCE_FIELDS, resolve_stage_tolerance, resolve_stall_guard, validate_stall_guard_fields, validate_tolerance_fields
from ...core.tool_output import ToolOutput, ToolResponse, output_contract_error
from ...robot.motion.action_compilation import compile_action, uses_arx_actions, validate_backend_action
from ...robot.motion.arx_actions import DEFAULT_ACTION_TIMEOUT_S
from ...robot.motion.scene_obstacles import (
    DEFAULT_CLEARANCE_HEIGHT_M,
    OBSTACLE_CLEARANCE_FIELDS,
    STAGE_CLEARANCE_FIELDS,
    resolve_stage_clearance,
)
from .robot_action_schema import ROBOT_ACTION_TOOL, _STAGE_COMMON, _STAGE_SCHEMAS, _validate_stage


def _records(harness):
    if not hasattr(harness, 'unified_action_plans'):
        harness.unified_action_plans = {}
    return harness.unified_action_plans


def _next(harness, kind):
    name = '_unified_'+kind+'_sequence'
    prefix = 'p' if kind == 'plan' else 'a'
    record_kind = 'unified_action_plan' if kind == 'plan' else 'action_request'
    with harness.lock:
        if not hasattr(harness, name):
            setattr(harness, name, harness.store.record_sequence(record_kind, prefix))
        value = getattr(harness, name)+1
        setattr(harness, name, value)
        return prefix+str(value)


def compile_request(harness, args):
    from ...robot.motion.batch_contract import validate_batch_action
    from .plan_continuation import resolve_continuation_args
    args, _ = resolve_continuation_args(harness, args)
    stages = args.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('robot_action_requires_stages')
    original_ids = [stage.get('id') if isinstance(stage, dict) else None for stage in stages]
    if any(not isinstance(identifier, str) or not identifier for identifier in original_ids) or len(set(original_ids)) != len(original_ids):
        raise ValueError('batch_requires_unique_stage_ids')
    validate_tolerance_fields(args)
    validate_stall_guard_fields(args)
    if any(k in args for k in STAGE_CLEARANCE_FIELDS):
        batch_clearance = resolve_stage_clearance(
            {k: args[k] for k in STAGE_CLEARANCE_FIELDS if k in args},
            DEFAULT_CLEARANCE_HEIGHT_M,
        )
    else:
        batch_clearance = None
    output = []
    for stage in stages:
        common = {key: deepcopy(stage[key]) for key in ('id', 'arm', 'kind', 'depends_on') if key in stage}
        fields = {key: value for key,value in stage.items() if key not in common}
        kind, arm = common.get('kind'), common.get('arm')
        if kind == 'observe':
            if set(fields) != {'reason'}:
                raise ValueError('observe_requires_only_reason')
            if not isinstance(fields['reason'], str) or not fields['reason'].strip():
                raise ValueError('batch_observe_requires_reason')
            _validate_stage(stage)
            output.append(dict(common, reason=fields['reason']))
            continue
        if arm not in ('left', 'right'):
            raise ValueError('robot_action_stage_requires_arm')
        if kind not in ('move', 'gripper'):
            raise ValueError('unsupported_robot_action_stage_fields')
        allowed = set(_STAGE_SCHEMAS[kind]['properties']) - set(_STAGE_COMMON) - {'kind'}
        if set(fields)-allowed:
            raise ValueError('unsupported_robot_action_stage_fields')
        jaw_after = kind == 'move' and 'gripper_goal_normalized' in fields and 'gripper_motion' not in fields
        checkpoint_after_gripper = fields.pop('checkpoint_after_gripper', None)
        if checkpoint_after_gripper is not None:
            if not isinstance(checkpoint_after_gripper, bool):
                raise ValueError('invalid_checkpoint_after_gripper')
            if kind == 'move' and not jaw_after and 'gripper_goal_normalized' not in fields:
                _validate_stage(stage)
                raise ValueError('checkpoint_after_gripper_requires_gripper_event')
        grip_guard_fields = {k: fields.pop(k) for k in GRIP_GUARD_FIELDS if k in fields}
        if grip_guard_fields:
            validate_grip_guard_fields(grip_guard_fields)
        joint_travel_fields = {k: fields.pop(k) for k in JOINT_TRAVEL_FIELDS if k in fields}
        if joint_travel_fields:
            extract_joint_travel_options(joint_travel_fields)
        stage_tol_fields = {key: fields.pop(key) for key in TOLERANCE_FIELDS if key in fields}
        stage_guard_fields = {key: fields.pop(key) for key in STALL_GUARD_FIELDS if key in fields}
        stage_clearance_fields = {key: fields.pop(key) for key in STAGE_CLEARANCE_FIELDS if key in fields}
        if kind == 'move':
            resolved_tol = resolve_stage_tolerance(stage_tol_fields, args)
            resolved_guard = resolve_stall_guard(stage_guard_fields, args)
            resolved_clearance = None
            if stage_clearance_fields or batch_clearance is not None:
                resolved_clearance = resolve_stage_clearance(
                    dict(stage_clearance_fields),
                    DEFAULT_CLEARANCE_HEIGHT_M if batch_clearance is None else batch_clearance,
                )
        jaw = {}
        if jaw_after:
            jaw = {key: fields.pop(key) for key in ('gripper_speed_scale', 'gripper_speed_preset') if key in fields}
            jaw['goal_normalized'] = fields.pop('gripper_goal_normalized')
        grasp = 'grasp_position_m' in fields or 'angles_deg' in fields
        if kind == 'move' and (grasp and ('position_m' in fields or 'quaternion_xyzw' in fields)):
            raise ValueError('batch_move_requires_one_pose_convention')
        name = ('grasp' if grasp else 'move') if kind == 'move' else 'gripper'
        try:
            own = compile_action(getattr(harness, "robot", None), name,
                dict(fields, arm=arm, observation_id=args['observation_id'], timeout_s=args.get('timeout_s', DEFAULT_ACTION_TIMEOUT_S)), harness.identity, harness.capabilities)
        except KeyError:
            # Missing public pose/goal fields must be input errors, while valid
            # stages still retain unrelated caller/runtime KeyError behavior.
            _validate_stage(stage)
            raise
        own = resolve_action_speed(own, harness.config.features.default_motion_speed_scale)
        if kind == 'move':
            own['path'].update(resolved_tol, stall_guard_preset=resolved_guard['stall_guard_preset'])
            if resolved_guard['max_tracking_divergence_m'] is not None:
                own['path']['max_tracking_divergence_m'] = resolved_guard['max_tracking_divergence_m']
            if resolved_clearance is not None:
                own['path']['clearance_height_m'] = resolved_clearance
        chk_field = {'checkpoint_after_gripper': checkpoint_after_gripper} if checkpoint_after_gripper is not None else {}
        if jaw_after:
            internal = common['id']+'/motion'
            if internal in original_ids:
                raise ValueError('batch_generated_stage_id_conflict')
            jaw_action = compile_action(getattr(harness, "robot", None), 'gripper', dict(jaw, arm=arm,
                observation_id=args['observation_id'], timeout_s=args.get('timeout_s', DEFAULT_ACTION_TIMEOUT_S)), harness.identity, harness.capabilities)
            jaw_action = resolve_action_speed(jaw_action, harness.config.features.default_motion_speed_scale)
            _validate_stage(stage)
            output.append(dict(common, id=internal, action=own, **joint_travel_fields))
            output.append(dict(id=common['id'], arm=arm, kind='gripper', depends_on=[internal],
                action=jaw_action, **chk_field, **grip_guard_fields))
        else:
            _validate_stage(stage)
            output.append(dict(common, action=own, **chk_field, **grip_guard_fields, **joint_travel_fields))
    action = dict(schema_version='0.1', **harness.identity, world_kind='observed', type='BatchAction',
        action_id=args.get('action_id') or _next(harness, 'action'), based_on_observation_id=args['observation_id'],
        mode='absolute', frame_id=harness.capabilities['frame_id'], tcp_config_id=harness.capabilities['tcp_config_id'],
        limits_profile=harness.capabilities.get('limits_profile', 'simulation-slow-v1'),
        timeout_s=args.get('timeout_s', DEFAULT_ACTION_TIMEOUT_S), stages=output)
    if 'checkpoint_after_gripper' in args:
        if not isinstance(args['checkpoint_after_gripper'], bool):
            raise ValueError('invalid_checkpoint_after_gripper')
        action['checkpoint_after_gripper'] = args['checkpoint_after_gripper']
    for key in (*TOLERANCE_FIELDS, *STALL_GUARD_FIELDS, *GRIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS):
        if key in args:
            action[key] = args[key]
    from ...robot.motion.scene_obstacles import merge_and_persist_obstacles
    args = merge_and_persist_obstacles(harness, args, output)
    proxy_fields = ('obstacles','held_objects','contact_entity_id','initial_contact','placement_contact')
    proxies = {key:deepcopy(args[key]) for key in proxy_fields if key in args}
    if 'held_object' in args:
        if 'held_objects' in args:
            raise ValueError('choose_held_object_or_held_objects')
        box=deepcopy(args['held_object']); arm=box.pop('arm',None)
        owners={stage['arm'] for stage in output if stage['kind']!='observe'}
        if arm is None:
            if len(owners)!=1:
                raise ValueError('dual_batch_held_object_requires_arm')
            arm=next(iter(owners))
        proxies['held_objects']={arm:box}
    if proxies:
        from ...robot.motion.batch_geometry import validate_proxies
        action['review_proxies']=validate_proxies(proxies)
    validate_batch_action(action, harness.capabilities)
    if not uses_arx_actions(harness.capabilities):
        validate_backend_action(harness.robot, action, harness.capabilities)
    return action


def _preview(harness, args, correlation):
    from .inspection import _trajectory_image
    from .plan_continuation import resolve_continuation_args
    resolved_args, continuation_of = resolve_continuation_args(harness, args)
    action = compile_request(harness, resolved_args)
    settle = None
    if 'settle' in resolved_args:
        from .stability import validate_options
        settle = validate_options(resolved_args['settle'])
    with harness.lock:
        if not harness.accepting or harness.active_action:
            raise ValueError('robot_action_preview_requires_idle_active_run')
        observation = harness.current_observation()
        validate_action(action, harness.identity, observation['observation_id'], harness.capabilities)
        validate_backend_action(harness.robot, action, harness.capabilities)
        budget_before = harness.action_budget()
        with harness.robot_lock:
            raw = harness.robot.preview_batch(action)
        from ...robot.motion.motion_failure import archive_collision_witnesses
        archive_collision_witnesses(harness.store, raw, harness.observation_profile)
        if raw.get('status') != 'planned':
            binding = dict(observation_id=observation['observation_id'], sim_step=observation.get('sim_step'),
                step_domain=observation.get('step_domain'), robot_state_ref=observation.get('robot_state_ref'))
            zero_step = (raw.get('physics_advanced') is False
                and type(raw.get('executed_native_steps')) is int and raw['executed_native_steps'] == 0)
            failure = {key: deepcopy(raw[key]) for key in (
                'status', 'reason', 'failed_stage_id', 'failed_arm', 'failure_feedback',
                'motion_safety_checks', 'planned_native_steps', 'stage_costs', 'physics_advanced',
                'executed_native_steps') if key in raw}
            if type(failure.get('planned_native_steps')) is not int:
                costs = {arm.get('nominal_native_steps')
                    for check in raw.get('motion_safety_checks', []) if isinstance(check, dict)
                    for arm in (check.get('arms') or {}).values() if isinstance(arm, dict)
                    and type(arm.get('nominal_native_steps')) is int and arm['nominal_native_steps'] >= 0}
                if len(costs) == 1:
                    failure['planned_native_steps'] = costs.pop()
            failure.update(observation_id=observation['observation_id'], state_binding=binding,
                action_budget=budget_before, zero_step_check_passed=zero_step)
            raw['action_budget'] = deepcopy(budget_before)
            harness.store.event('action.batch_preview', failure, correlation, world_kind='internal',
                sim_step=observation.get('sim_step'), step_domain=observation.get('step_domain'))
            return raw, []
        short = _next(harness, 'plan')
        trajectory = raw.pop('trajectory_data')
        trajectory.update(identity=deepcopy(harness.identity), observation_id=resolved_args['observation_id'], label=short)
        from .grasp_preview_3d import audit_batch_grasp_3d
        grasp_3d_review, grasp_3d_targets = audit_batch_grasp_3d(harness, action, resolved_args['observation_id'])
        if grasp_3d_targets:
            trajectory['grasp_3d_targets'] = grasp_3d_targets
        reference = harness.store.put_artifact(canonical(trajectory).encode(), 'candidate_trajectory', 'application/json')
        metadata, images = _trajectory_image(harness, trajectory, resolved_args.get('camera_id') or 'cam_head')
        summary = {key: deepcopy(value) for key,value in raw.items() if key != 'plan_id'}
        if grasp_3d_review is not None:
            summary['grasp_3d_review'] = grasp_3d_review
        if 'motion_safety_checks' in summary:
            checks = summary.pop('motion_safety_checks')
            safety_ref = harness.store.put_artifact(canonical(checks).encode(), 'batch_motion_safety', 'application/json')
            summary['motion_safety'] = dict(details_ref=safety_ref, checked_stages=len(checks),
                enabled=any(check['enabled'] for check in checks),
                status='reject' if any(check['status']=='reject' for check in checks) else 'pass',
                checks=[dict(stage_id=check['stage_id'], status=check['status'], start_state_source=check['start_state_source']) for check in checks])
        summary.update(plan_id=short, observation_id=resolved_args['observation_id'], trajectory_ref=reference,
            trajectory_overlay=metadata, action_budget=harness.action_budget(),
            executed_batch_native_steps=0, plan_revision=0,
            next_action=dict(mode='execute', plan_id=short, expected_cursor=0, expected_revision=0))
        if continuation_of is not None:
            summary['continuation_of'] = deepcopy(continuation_of)
        if settle is not None:
            summary['settle'] = deepcopy(settle)
            summary['additional_settle_budget'] = dict(max_native_steps=settle['max_native_steps'],
                timing='Only after the complete batch finishes, never at an observation checkpoint.',
                semantics='Additional bounded measured-RGB settling requests; separate from cached motion rows and their cost.')
        from .action_state_ledger import attach_preview_ledger
        summary = attach_preview_ledger(harness, action, summary)
        record = dict(identity=deepcopy(harness.identity), action=action, native_plan_id=raw['plan_id'],
            source_args=deepcopy(resolved_args),
            source_stages=deepcopy(resolved_args['stages']), raw_stages=deepcopy(resolved_args['stages']),
            completed_stage_ids=[],
            remaining_stage_ids=[s['id'] for s in resolved_args['stages']], failed_stage_id=None,
            executable=True, continuation_of=continuation_of,
            summary=summary, trajectory_ref=reference, settle=settle)
        _records(harness)[short] = record
        from ...core.config import motion_thresholds
        cache_limit = int(motion_thresholds(harness)['batch_plan_cache_size'])
        while len(_records(harness)) > cache_limit:
            _records(harness).pop(next(iter(_records(harness))))
        harness.store.save('unified_action_plan', short, record)
        harness.store.event('action.batch_preview', summary, correlation, world_kind='internal')
        return summary, images


def _lookup(harness, identifier):
    record = _records(harness).get(identifier)
    if record is None:
        raise ValueError('robot_action_plan_unknown_or_expired')
    if any(record['identity'].get(key) != harness.identity[key] for key in IDENTITY):
        raise ValueError('stale_environment_identity')
    return record


def _execute(harness, args, correlation):
    identifier = args['plan_id']
    record = _lookup(harness, identifier)
    if not hasattr(harness, '_unified_action_call_ids'):
        harness._unified_action_call_ids = {}
    key = correlation.get('tool_call_id')
    if key and key not in harness._unified_action_call_ids:
        harness._unified_action_call_ids[key] = _next(harness, 'action')
    action_id = args.get('action_id') or (harness._unified_action_call_ids[key] if key else _next(harness, 'action'))
    replay = False
    try:
        previous = harness.store.read('action_request', action_id)
    except KeyError:
        previous = None
    if previous is not None:
        if previous.get('batch_plan_id') != record['native_plan_id']:
            raise ValueError('action_id_content_conflict')
        if 'expected_cursor' in args and previous.get('batch_expected_cursor') != args['expected_cursor']:
            raise ValueError('action_id_content_conflict')
        if 'expected_revision' in args and previous.get('batch_expected_revision') != args['expected_revision']:
            raise ValueError('action_id_content_conflict')
        action, replay = previous, True
    else:
        action = dict(record['action'], action_id=action_id, batch_plan_id=record['native_plan_id'],
            batch_expected_cursor=args.get('expected_cursor',record['summary'].get('executed_batch_native_steps',0)),
            batch_expected_revision=args.get('expected_revision',record['summary'].get('plan_revision',0)),
            based_on_observation_id=harness.current_observation()['observation_id'])
        programs = getattr(harness, 'programs', None)
        if programs is not None and isinstance(correlation, dict) and correlation.get('program_id'):
            entry = getattr(programs, 'entries', {}).get(correlation['program_id'])
            if (isinstance(entry, dict) and entry.get('allow_direct_actions')
                    and action.get('timeout_s') != entry.get('_explicit_timeout')):
                from ..programs.runner import ACTION_RESERVE_S, program_remaining
                available = program_remaining(entry) - ACTION_RESERVE_S - 0.5
                if available >= 0.1 and float(action.get('timeout_s', 0)) > available:
                    action['timeout_s'] = round(available, 2)
        if not uses_arx_actions(harness.capabilities):
            # Cached continuation binds every leaf to the same current observation.
            action['stages'] = [dict(stage, action=dict(stage['action'],
                based_on_observation_id=action['based_on_observation_id'])) if 'action' in stage else dict(stage)
                for stage in action['stages']]
    result = harness.execute(action, correlation)
    receipt = result['receipt']
    from .action_state_ledger import attach_execution_ledger, record_execution_outcome
    from .batch_repair import build_repair_hint
    from .plan_continuation import build_next_action, summarize_stage_progress
    if not replay:
        record_execution_outcome(harness, identifier, action, receipt)
        progress = summarize_stage_progress(record, receipt)
        record['completed_stage_ids'] = progress['completed_stage_ids']
        record['remaining_stage_ids'] = progress['remaining_stage_ids']
        record['failed_stage_id'] = progress['failed_stage_id']
        record['executable'] = receipt.get('batch_status') == 'awaiting_observation'
        record['summary']['status'] = receipt.get('batch_status', receipt['status'])
        record['summary']['last_action_id'] = action_id
        record['summary']['completed_stage_ids'] = progress['completed_stage_ids']
        record['summary']['remaining_stage_ids'] = progress['remaining_stage_ids']
        record['summary']['failed_stage_id'] = progress['failed_stage_id']
        if 'batch_cursor' in receipt:
            record['summary']['executed_batch_native_steps'] = receipt['batch_cursor']
        if 'batch_revision' in receipt:
            record['summary']['plan_revision'] = receipt['batch_revision']
        if receipt.get('batch_status') == 'awaiting_observation':
            record['summary'].pop('repair_action', None)
            record['summary']['checkpoint_observation_id'] = receipt.get('result_observation_id')
            plans = getattr(harness.robot, '_batch_plans', None)
            if isinstance(plans, dict) and record['native_plan_id'] in plans:
                plans[record['native_plan_id']]['expected_observation_id'] = receipt.get('result_observation_id')
        else:
            hint = build_repair_hint(identifier, receipt)
            if hint is not None:
                record['summary']['repair_action'] = hint
            else:
                record['summary'].pop('repair_action', None)
        record['summary']['next_action'] = build_next_action(identifier, record, receipt, result)
    result = attach_execution_ledger(
        harness,
        dict(result, plan_id=identifier, action_id=action_id, next_action=deepcopy(record['summary']['next_action'])),
        record['summary'],
    )
    if 'repair_action' in record['summary']:
        result['repair_action'] = deepcopy(record['summary']['repair_action'])
    if record.get('continuation_of') is not None:
        result['continuation_of'] = deepcopy(record['continuation_of'])
    hook = getattr(harness, 'unified_action_postprocess', None)
    if hook is not None and not replay:
        try:
            updated = hook(result, correlation, settle=record.get('settle'))
            if updated is not None:
                result = updated
        except Exception as error:
            result = dict(result, postprocess=dict(status='unavailable', reason=type(error).__name__,
                action_execution='preserved_in_receipt_do_not_replay'))
        record['summary']['next_action'] = build_next_action(identifier, record, receipt, result)
        result['next_action'] = deepcopy(record['summary']['next_action'])
    try:
        return _action_response(harness, result, action, record, correlation)
    except Exception:
        # Optional feedback cannot discard the already acknowledged execution.
        return output_contract_error(ToolOutput(result, kind="action", projection="action_result").response(),
            "robot_action", "action_feedback_unavailable", "/content")


def _action_response(harness, result, action, record, correlation):
    identifier, action_id = result['plan_id'], result['action_id']
    receipt = result['receipt']
    details = harness.store.put_artifact(canonical(result).encode(), 'unified_action_result', 'application/json')
    program_data = result
    if correlation.get('program_id'):
        program_data = dict(
            result,
            status=receipt.get('status'),
            reason=receipt.get('reason'),
            failed_stage_id=receipt.get('failed_stage_id'),
            failed_arm=receipt.get('failed_arm'),
            observation_id=result.get('current_observation_id') or receipt.get('result_observation_id'),
            result_observation_id=receipt.get('result_observation_id'),
            stages=deepcopy(receipt.get('stage_results', {})),
        )
        presented = program_data
    else:
        projected_receipt = lean_action_result(result)['receipt']
        presented = dict(plan_id=identifier, action_id=action_id,
            receipt=projected_receipt,
            planned_native_steps=record['summary'].get('planned_native_steps'),
            nominal_sim_time_s=record['summary'].get('nominal_sim_time_s'),
            action_budget=harness.action_budget(), details_ref=details)
        presented['next_action'] = deepcopy(record['summary']['next_action'])
        if 'repair_action' in record['summary']:
            presented['repair_action'] = deepcopy(record['summary']['repair_action'])
        presented['stages'] = projected_receipt.pop('stage_results', {})
        for key in ('settling', 'postprocess', 'current_observation_id', 'result_observation_is_current', 'continuation_of', 'repeat_failure_warning'):
            if key in result:
                presented[key] = result[key]
        if projected_receipt.get('batch_status', projected_receipt.get('status')) in ('stopped', 'failed') and 'episode_ledger' in result:
            presented['episode_ledger'] = result['episode_ledger']
    from .strategy_checkpoint import adaptive_action_checkpoint, checkpoint_triggered
    checkpoint = adaptive_action_checkpoint(harness.store, harness.identity)
    if checkpoint_triggered(checkpoint):
        presented = {**presented, "strategy_checkpoint": checkpoint}
    from ...core.presentation.feedback_delivery import action_trajectory_feedback
    trajectory, trajectory_images = action_trajectory_feedback(harness, action, receipt, correlation)
    if trajectory is not None:
        presented = dict(presented, trajectory_feedback=trajectory)
        if trajectory.get('trajectory_feedback_ref'):
            harness.store.set_state('unified_action_feedback',action_id,
                dict(trajectory_ref=trajectory['trajectory_feedback_ref'],identity=deepcopy(harness.identity)))
    content = ToolOutput(presented, extra_content=tuple(trajectory_images),
        projection="action_result" if correlation.get("program_id") else "action_display").content_parts()
    observation = result.get('current_observation_id') or receipt.get('result_observation_id')
    if observation:
        bundle = harness.store.read('observation', observation)
        content += harness.observation_content(bundle, correlation,
            feedback_action=action, feedback_receipt=receipt, structured=True)
        from ...perception.external.obs3d import action_attachments
        content += action_attachments(harness, bundle, correlation, before_observation_id=action.get('based_on_observation_id'))
    return ToolResponse(content, data=program_data, kind="action", data_projection="action_result")


def _preview_downgrade_reasons(summary, args=None, harness=None):
    """检查 preview 产物是否存在风险并返回透明的降级原因列表（空列表表示零风险可直接执行）。"""
    if not isinstance(summary, dict) or summary.get('status') != 'planned':
        return ['preview_not_planned']
    reasons = []
    safety = summary.get('motion_safety')
    if isinstance(safety, dict) and safety.get('status') not in (None, 'pass'):
        reasons.append('motion_safety_reject')
    grasp_review = summary.get('grasp_3d_review')
    if isinstance(grasp_review, dict):
        if (grasp_review.get('status') not in (None, 'aligned')
                or grasp_review.get('warnings')
                or grasp_review.get('risk_flags')):
            reasons.append('grasp_3d_review_warnings')
    geom_review = summary.get('geometry_review')
    if isinstance(geom_review, dict) and geom_review.get('risks'):
        reasons.append('geometry_review_risks')
    # 若批次对某机械臂执行闭爪抓取（goal_normalized <= 0.2），检查该臂在闭爪前是否已处于未张开状态且未先安排开爪阶段
    if isinstance(args, dict) and isinstance(args.get('stages'), list) and harness is not None:
        from ...core.config import motion_thresholds
        from ...robot.motion.grip_guard import _aperture_from_jaw, _extract_jaw
        closed_thresh_m = float(motion_thresholds(harness).get('grip_holding_threshold_mm', 3.0)) / 1000.0
        obs = getattr(harness, 'latest', None) or {}
        live_state = getattr(harness, 'robot_state', None)
        if isinstance(obs, dict):
            grip_facts = obs.get('grip_facts') if isinstance(obs.get('grip_facts'), dict) else {}
            r_state = obs.get('robot_state') if isinstance(obs.get('robot_state'), dict) else {}
            grip_pos = r_state.get('gripper_positions') if isinstance(r_state.get('gripper_positions'), dict) else {}
            opened_in_batch = set()
            for st in args['stages']:
                if not isinstance(st, dict):
                    continue
                arm = st.get('arm')
                if arm not in ('left', 'right'):
                    continue
                goal = st.get('goal_normalized') if st.get('kind') == 'gripper' else st.get('gripper_goal_normalized')
                if isinstance(goal, (int, float)):
                    if float(goal) >= 0.5:
                        opened_in_batch.add(arm)
                    elif float(goal) <= 0.2 and arm not in opened_in_batch:
                        arm_gf = grip_facts.get(arm)
                        ap = arm_gf.get('measured_aperture_m') if isinstance(arm_gf, dict) else None
                        if ap is None:
                            ap = grip_pos.get(arm)
                        if ap is None and isinstance(live_state, dict):
                            ap = _aperture_from_jaw(_extract_jaw(live_state, arm))
                        if isinstance(ap, (int, float)) and float(ap) < closed_thresh_m:
                            reasons.append(f'closed_jaw_before_grasp:{arm}')
                            break
    return reasons


def _is_preview_clean(summary, args=None, harness=None):
    """判断 preview 产物是否零风险（无规划失败、无安全拒绝、无 3D 抓取预警、无碰撞风险且无闭爪下探）。"""
    return len(_preview_downgrade_reasons(summary, args=args, harness=harness)) == 0


def _apply_plan_update(harness, plan_update, args, correlation):
    """在 robot_action 内部顺带原子执行 task_plan 的 submit 或 update，消灭独立 task_plan 回合。"""
    if not isinstance(plan_update, dict) or not plan_update:
        return None
    tp = getattr(harness, 'task_planning', None)
    if tp is None:
        return None
    with harness.lock:
        previous = tp.current(harness.identity)
        op = 'update' if previous is not None else 'submit'
        tp_args = {'operation': op, **deepcopy(plan_update)}
        if 'ledger' in tp_args and isinstance(tp_args['ledger'], list):
            normalized_ledger = []
            seen_ids = set()
            for idx, item in enumerate(tp_args['ledger'][:3]):
                if not isinstance(item, dict):
                    continue
                expected = str(item.get('expected') or item.get('event') or item.get('step') or '').strip()
                if not expected:
                    continue
                lid = str(item.get('id') or f'L{idx + 1}').strip()
                while lid in seen_ids:
                    lid = f'{lid}_{idx + 1}'
                seen_ids.add(lid)
                raw_st = str(item.get('status') or 'pending').strip()
                if raw_st == 'completed':
                    st = 'supported'
                elif raw_st in ('pending', 'supported', 'contradicted', 'unknown'):
                    st = raw_st
                else:
                    st = 'pending'
                basis = str(item.get('basis') or 'unknown').strip() or 'unknown'
                next_check = str(item.get('next_check') or 'After current action observation').strip()
                normalized_ledger.append({
                    'id': lid[:80],
                    'expected': expected[:800],
                    'status': st,
                    'basis': basis[:800],
                    'next_check': next_check[:800],
                })
            tp_args['ledger'] = normalized_ledger
        elif op == 'submit' and 'ledger' not in tp_args:
            tp_args['ledger'] = []
        obs = getattr(harness, 'latest', None)
        if obs is None and hasattr(harness, 'current_observation'):
            try:
                obs = harness.current_observation()
            except Exception:
                obs = None
        if args.get('observation_id'):
            tp_args['observation_id'] = args['observation_id']
        elif isinstance(obs, dict) and obs.get('observation_id'):
            tp_args['observation_id'] = obs['observation_id']
        return tp.call(tp_args, harness.identity, obs, correlation)


def _dispatch(harness, args, correlation=None):
    from jsonschema import validate
    validate(args, ROBOT_ACTION_TOOL['inputSchema'])
    correlation = correlation or {}
    args = dict(args)
    plan_update = args.pop('plan_update', None)
    auto_execute_if_clean = bool(args.pop('auto_execute_if_clean', False))
    mode = args.get('mode', 'preview')
    if mode == 'inspect' and (plan_update is not None or auto_execute_if_clean):
        raise ValueError('robot_action_inspect_has_command_fields')
    if (
        mode == 'execute_if_clean'
        and 'plan_id' in args
        and not any(k in args for k in ('stages', 'source_plan_id', 'skip_failed_stage'))
    ):
        mode = 'execute'
        args['mode'] = 'execute'
    plan_update_receipt = _apply_plan_update(harness, plan_update, args, correlation) if plan_update is not None else None
    if mode in ('preview', 'direct', 'execute_if_clean'):
        if set(args)-{'mode', 'observation_id', 'action_id', 'plan_id', 'skip_failed_stage',
                'stages', 'source_plan_id', 'start_from_stage_id', 'stage_overrides',
                'checkpoint_after_gripper', 'timeout_s', 'camera_id', 'detail', 'settle',
                'obstacles','held_object','held_objects','contact_entity_id','initial_contact','placement_contact',
                *TOLERANCE_FIELDS, *STALL_GUARD_FIELDS, *GRIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS,
                *OBSTACLE_CLEARANCE_FIELDS}:
            raise ValueError('robot_action_preview_has_execute_or_inspect_fields')
        if 'observation_id' not in args:
            raise ValueError('robot_action_requires_observation_id')
        if 'plan_id' in args:
            from .batch_repair import build_repaired_args
            args = build_repaired_args(harness, args, _lookup)
        summary, images = _preview(harness, args, correlation)
        if plan_update_receipt is not None and isinstance(summary, dict):
            summary['plan_update_receipt'] = plan_update_receipt
        if mode == 'direct':
            if summary.get('status') == 'planned':
                return _execute(harness, dict(plan_id=summary['plan_id'], **({'action_id': args['action_id']} if 'action_id' in args else {})), correlation)
            summary = dict(summary, receipt=dict(status=summary.get('status', 'rejected'),
                reason=summary.get('reason'), failed_stage_id=summary.get('failed_stage_id'),
                failed_arm=summary.get('failed_arm'), physics_advanced=False, executed_native_steps=0))
        elif mode == 'execute_if_clean' or auto_execute_if_clean:
            downgrade_reasons = _preview_downgrade_reasons(summary, args=args, harness=harness)
            if not downgrade_reasons:
                return _execute(harness, dict(plan_id=summary['plan_id'], **({'action_id': args['action_id']} if 'action_id' in args else {})), correlation)
            if isinstance(summary, dict) and summary.get('status') == 'planned':
                summary['downgraded_from_execute_if_clean'] = True
                summary['downgrade_reasons'] = downgrade_reasons
        return ToolOutput(summary, extra_content=tuple(images), kind="plan", projection="action_plan").response()
    if mode == 'execute':
        if 'plan_id' not in args:
            raise ValueError('robot_action_requires_plan_id')
        # 容忍调用方在确认执行缓存计划时顺带传入不修改动作阶段的上下文元字段（如 observation_id、timeout_s）
        for benign_key in ('observation_id', 'timeout_s', 'camera_id', 'detail'):
            args.pop(benign_key, None)
        if set(args)-{'mode', 'plan_id', 'action_id', 'expected_cursor', 'expected_revision'}:
            raise ValueError('cached_batch_commands_cannot_be_changed')
        return _execute(harness, args, correlation)
    if set(args)-{'mode', 'plan_id', 'action_id','trajectory_ref','path_kind','camera_id', 'detail', 'native_steps', 'start_native_step', 'max_samples'}:
        raise ValueError('robot_action_inspect_has_command_fields')
    if sum(key in args for key in ('plan_id','action_id','trajectory_ref')) != 1:
        raise ValueError('inspect_requires_one_plan_action_or_trajectory_reference')
    from .inspection import inspect_trajectory
    inspection = {key: value for key,value in args.items() if key not in ('mode', 'plan_id','action_id')}
    prefix=[]
    if 'plan_id' in args:
        record = _lookup(harness, args['plan_id'])
        inspection['trajectory_ref']=record['trajectory_ref']
        prefix=ToolOutput(deepcopy(record['summary']), projection='owner_record').content_parts()
    elif 'action_id' in args:
        action=harness.store.read('action_request',args['action_id'])
        if any(action.get(key)!=harness.identity[key] for key in IDENTITY):
            raise ValueError('stale_environment_identity')
        stored=harness.store.read_state('unified_action_feedback',args['action_id'])
        if stored is None:
            from .trajectory_feedback import build_action_trajectory_feedback
            feedback,_=build_action_trajectory_feedback(harness,action,harness.store.read('action_receipt',args['action_id']),correlation,diagnostic=True)
            if not feedback.get('trajectory_feedback_ref'):
                return ToolOutput(dict(action_id=args['action_id'],available=False,trajectory_feedback=feedback)).response()
            stored=dict(trajectory_ref=feedback['trajectory_feedback_ref'],identity=deepcopy(harness.identity))
            harness.store.set_state('unified_action_feedback',args['action_id'],stored)
        inspection['trajectory_ref']=stored['trajectory_ref']
        inspection.setdefault('path_kind','measured')
    content = inspect_trajectory(harness, inspection, correlation, cached_action=True, structured=True)
    return ToolResponse(prefix+content, data=deepcopy(record["summary"]) if "plan_id" in args else None,
        kind="plan" if "plan_id" in args else "data", data_projection="action_plan")


def dispatch(harness, args, correlation=None):
    """Replay the same tool call without planning, execution or postprocessing."""
    from ...core.contracts import digest
    correlation = correlation or {}
    key = (correlation.get('program_id'), correlation.get('tool_call_id'), args.get('mode', 'preview'))
    if key[1]:
        if not hasattr(harness, '_unified_tool_results'):
            harness._unified_tool_results = {}
        prior = harness._unified_tool_results.get(key)
        if prior is not None:
            if prior[0] != digest(args):
                raise ValueError('robot_action_tool_call_content_conflict')
            return deepcopy(prior[1])
    result = _dispatch(harness, args, correlation)
    if key[1]:
        harness._unified_tool_results[key] = (digest(args), deepcopy(result))
        while len(harness._unified_tool_results) > 128:
            harness._unified_tool_results.pop(next(iter(harness._unified_tool_results)))
    return result
