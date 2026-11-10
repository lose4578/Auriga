"""Compile complete two-arm stage graphs, then execute only the reviewed rows.

Future starting states are commanded endpoints, never future measurements.
Checkpoints stop the clock and retain the exact unexecuted suffix. A changed
state invalidates that suffix; execution never invokes a planner or resampler.
"""
from copy import deepcopy
import math
import time
import uuid

import numpy as np

from .batch_contract import ARMS, validate_batch_action
from ...core.config import motion_thresholds
from .contact_diagnostics import build_contact_block_summary
from ...core.contracts import digest
from .grip_guard import (
    GRIP_GUARD_FIELDS,
    _aperture_from_jaw,
    _extract_jaw,
    compute_grip_facts,
    evaluate_grip_guard,
    resolve_grip_guard,
)
from .motion_safety import JOINT_TRAVEL_FIELDS, extract_joint_travel_options
from .motion_tolerance import (
    STALL_GUARD_FIELDS,
    TOLERANCE_FIELDS,
    compute_endpoint_residual,
    resolve_stage_tolerance,
    resolve_stall_guard,
)


def motion_digest(action):
    payload = {key: action.get(key) for key in ('stages', 'frame_id', 'tcp_config_id', 'limits_profile', 'review_proxies', 'checkpoint_after_gripper')}
    for key in (*TOLERANCE_FIELDS, *STALL_GUARD_FIELDS, *GRIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS):
        if key in action:
            payload[key] = action[key]
    return digest(payload)


def _arm_grip_state(adapter, state, arm):
    """提取用于计算 grip_facts 的单臂状态，优先使用真实夹爪物理测量开度。"""
    q = list(state['arms'][arm].get('q_rad', [0.0] * 6))
    jaw = _extract_jaw(state, adapter=adapter, arm=arm)
    if jaw is not None:
        return [*q[:6], float(jaw)]
    arm_state = deepcopy(state['arms'][arm])
    if 'gripper' not in arm_state and hasattr(adapter, 'last_dispatched_goals'):
        goal = adapter.last_dispatched_goals.get(arm)
        if goal is not None:
            return [*q[:6], float(goal)]
    return arm_state


def _jaw_rows(adapter, stage, held, initial):
    action, arm = stage['action'], stage['arm']
    offset = 0 if arm == 'left' else 7
    goal = action['gripper_events'][0]['goal_normalized']
    scale = action.get('gripper_speed_scale', 1.)
    dwell = max(1, math.ceil(abs(goal-held[offset+6]) / adapter.eef_gripper_dwell_step))
    values = []
    if scale < 1:
        count = math.ceil(abs(goal-initial) / (adapter.eef_gripper_dwell_step*scale))
        values += [initial+(goal-initial)*i/count for i in range(1, count+1)]
    values += [goal]*dwell
    rows = []
    for value in values:
        row = list(held)
        row[offset+6] = value
        rows.append(row)
    return rows


def _evidence(adapter, before, action, rows, contexts):
    from .trajectory_evidence import build_trajectory_evidence, compact_trajectory_evidence
    compiled = {arm: (np.array([row[(0 if arm == 'left' else 7):][:6] for row in rows]), rows, [], None)
                for arm in ARMS}
    # A checkpoint-only batch has a valid initial frame and no command samples.
    if not rows:
        rows_for_evidence = [adapter._held_row(before)]
        compiled = {arm: (np.array([before['arms'][arm]['q_rad']]), rows_for_evidence, [], None) for arm in ARMS}
    _, full = build_trajectory_evidence(adapter, before, compiled, 'independent', action)
    if not rows:
        full['samples'] = full['samples'][:1]
    for sample, context in zip(full['samples'][1:], contexts):
        for arm in ARMS:
            sample['arms'][arm].update(context[arm])
    full.update(execution_binding='exact_cached_rows; no replanning or resampling',
        jaw_scope='exact cached jaw commands; measured starting aperture checked before dispatch',
        plan_scope='complete batch including checkpoint suffix; future states are planned, not measured')
    compact = compact_trajectory_evidence(full)
    compact['limits'] = [text.replace('execution replans.', 'execution uses these exact cached commands or stops.')
                         for text in compact['limits']]
    compact['details'] = 'robot_action mode=inspect reads the original complete cached trajectory and geometry.'
    return compact, full


def _populate_planned_positions(adapter, before, records, full):
    """为每个 move 阶段生成每步规划末端位置，用于执行期 stall_guard 实时偏差对比。"""
    samples = full.get('samples') or []
    if full.get('geometry', {}).get('status') == 'available' and len(samples) > 1:
        for record in records.values():
            if record['kind'] != 'move':
                continue
            arm = record['arm']
            record['planned_positions_m'] = [
                list(samples[step + 1]['arms'][arm]['link6_pose']['position_m'])
                for step in range(record['start'], record['end'])
            ]
        return

    model = getattr(adapter, 'robot_model', None)
    fk_fn = None
    bases = {}
    ee_links = {}
    if isinstance(model, dict) and model.get('joints'):
        try:
            from ..model.geometry import array, fk, transform
            from ..model.roboprobe_pose import quat_to_matrix
            fk_fn = fk
            for arm in ARMS:
                robot = adapter.robot.get_robot_by_arm_name(arm + '_arm')
                ee_link = getattr(robot, 'ee_link_name', 'link6')
                ee_links[arm] = ee_link
                state = before['arms'][arm]
                names = list(state.get('joint_names') or robot.arm_joints_name)
                values = list(state['q_rad'])
                base_ee = fk({**model, 'T_world_base': np.eye(4)},
                    {'joint_names': names, 'q_rad': values, 'source_kind': 'measurement'})[ee_link]
                pose = state['ee_pose']
                q = array(pose['quaternion_xyzw'], (4,))
                world_ee = np.eye(4)
                world_ee[:3, :3] = quat_to_matrix(q[[3, 0, 1, 2]])
                world_ee[:3, 3] = array(pose['position_m'], (3,))
                bases[arm] = transform(world_ee @ np.linalg.inv(base_ee))
        except Exception:
            fk_fn = None

    start_poses = {arm: list(before['arms'][arm]['ee_pose']['position_m']) for arm in ARMS}
    for _, record in sorted(records.items(), key=lambda item: item[1]['start']):
        if record['kind'] != 'move':
            continue
        arm = record['arm']
        offset = 0 if arm == 'left' else 7
        target_pos = np.asarray(record['target']['position_m'], dtype=float)
        start_pos = np.asarray(start_poses[arm], dtype=float)
        positions = None
        if fk_fn is not None and arm in bases:
            try:
                robot = adapter.robot.get_robot_by_arm_name(arm + '_arm')
                names = list(before['arms'][arm].get('joint_names') or robot.arm_joints_name)
                cand = [
                    fk_fn({**model, 'T_world_base': bases[arm]},
                          {'joint_names': names, 'q_rad': row[offset:offset + 6], 'source_kind': 'estimate'})[ee_links[arm]][:3, 3].tolist()
                    for row in record['rows']
                ]
                if cand and float(np.linalg.norm(np.asarray(cand[-1]) - target_pos)) <= 0.02:
                    positions = cand
            except Exception:
                positions = None
        if positions is None:
            count = len(record['rows'])
            positions = [
                (start_pos + (target_pos - start_pos) * ((i + 1) / count)).tolist()
                for i in range(count)
            ]
        record['planned_positions_m'] = positions
        start_poses[arm] = list(target_pos)


def _check_pose_arrival(measured_pose, target_pose, pos_tol, ori_tol, *,
                        allow_vertical_compliance=False, max_upward_dz_m=None, max_downward_dz_m=None):
    """检查末端到位误差：支持标准 3D 容差，或在水平 (dx, dy) 严格对准下的 Z 向接触/负载顺应。"""
    from .feedback_motion import pose_error
    position, orientation = pose_error(measured_pose, target_pose)
    residual = compute_endpoint_residual(target_pose, measured_pose)
    if not math.isfinite(position + orientation):
        return False, position, orientation, residual, False
    if position <= pos_tol and orientation <= ori_tol:
        return True, position, orientation, residual, False
    if allow_vertical_compliance:
        xy_error = math.hypot(residual['dx_m'], residual['dy_m'])
        dz = residual['dz_m']
        z_down_tol = max_downward_dz_m if max_downward_dz_m is not None else pos_tol
        z_up_tol = max_upward_dz_m if max_upward_dz_m is not None else max(pos_tol, 0.040)
        if xy_error <= pos_tol and -z_down_tol <= dz <= z_up_tol and orientation <= ori_tol:
            return True, position, orientation, residual, True
    return False, position, orientation, residual, False


def preview_batch(adapter, action):
    try:
        from ...runtime.actions.plan_continuation import resolve_backend_continuation
        action, continuation_of = resolve_backend_continuation(adapter, action)
    except ValueError as error:
        return dict(status='failed', reason=str(error), physics_advanced=False, executed_native_steps=0)
    dependencies = validate_batch_action(action, adapter.capabilities())
    if getattr(adapter, 'motion_execution_mode', 'resampled') != 'resampled':
        raise ValueError('batch_requires_resampled_execution')
    if adapter.env.end_flag[0]:
        raise ValueError('batch_episode_ended')
    before = deepcopy(adapter.state())
    binding = adapter._preview_state_binding(before)
    held = adapter._held_row(before)
    initial_jaws = {}
    for arm in ARMS:
        try:
            initial_jaws[arm] = adapter._measured_gripper_goal(before, arm)
        except (ValueError, KeyError, AttributeError):
            # Missing jaw measurements matter only if a slow/synchronized jaw is
            # requested; a motion-only batch preserves existing jaw commands.
            initial_jaws[arm] = None
    cap = int(math.ceil(10*adapter.control_hz))
    deadline = time.monotonic()+action['timeout_s']
    stages = action['stages']
    pending = {s['id']: s for s in stages}
    completed, active, records = set(), {}, {}
    targets = {arm: deepcopy(before['arms'][arm]['ee_pose']) for arm in ARMS}
    base_thresholds = motion_thresholds(adapter)
    tolerances = dict.fromkeys(ARMS, float(base_thresholds.get('default_position_tolerance_m', .010)))
    orientation_tolerances = dict.fromkeys(ARMS, float(base_thresholds.get('orientation_tolerance_rad', .15)))
    rows, contexts, checkpoints, boundaries, diagnostics = [], [], [], [], []
    jaw_commanded = set()
    def verify():
        if adapter._preview_state_binding() != binding:
            raise ValueError('batch_state_changed_during_preview')
        if time.monotonic() > deadline:
            raise ValueError('batch_planning_timeout')
    try:
        while pending or active:
            verify()
            ready = [s for s in pending.values() if dependencies[s['id']] <= completed
                     and (s.get('arm') is None or s['arm'] not in active)]
            observe = [s for s in ready if s['kind'] == 'observe']
            if observe:
                checkpoints.append(dict(cursor=len(rows), stage_ids=[s['id'] for s in observe],
                    reasons={s['id']: s['reason'] for s in observe}))
                for stage in observe:
                    identifier = stage['id']
                    completed.add(identifier)
                    del pending[identifier]
                    records[identifier] = dict(kind='observe', arm=stage.get('arm'), start=len(rows), end=len(rows))
                continue
            for stage in ready:
                arm, identifier, own = stage['arm'], stage['id'], stage['action']
                offset = 0 if arm == 'left' else 7
                expected_q = list(held[offset:offset+6])
                if arm in jaw_commanded or (stage['kind'] == 'move' and own.get('gripper_speed_scale', 1.) >= 1.):
                    start_jaw = held[offset+6]
                else:
                    start_jaw = initial_jaws[arm]
                    if start_jaw is not None and len(rows) and held[offset+6] > start_jaw:
                        start_jaw = min(held[offset+6], start_jaw+len(rows)*adapter.eef_gripper_dwell_step)
                jaw_events = own.get('gripper_events') or []
                is_closing_jaw = bool(jaw_events) and float(jaw_events[0].get('goal_normalized', 1.0)) < 0.85
                grip_guard = resolve_grip_guard(stage, action if is_closing_jaw else None)
                joint_travel_options = extract_joint_travel_options(stage, action) if stage['kind'] == 'move' else {}
                record = dict(kind=stage['kind'], arm=arm, start=len(rows), expected_start_q=expected_q,
                    expected_start_jaw=start_jaw, action=own, target=deepcopy(targets[arm]),
                    position_tolerance_m=tolerances[arm], orientation_tolerance_rad=orientation_tolerances[arm],
                    source_kind='planned_command_state',
                    checkpoint_after_gripper=stage.get('checkpoint_after_gripper'),
                    grip_guard=grip_guard, joint_travel_options=joint_travel_options)
                if stage['kind'] == 'move':
                    path = own.get('path', {})
                    path_tol_fields = {
                        k: path[k] for k in TOLERANCE_FIELDS
                        if k in path and not (k == 'position_tolerance_m' and 'tolerance_preset' not in path and path[k] == 0.005)
                    }
                    stage_tol_spec = path_tol_fields | {k: stage[k] for k in TOLERANCE_FIELDS if k in stage}
                    resolved_tol = resolve_stage_tolerance(stage_tol_spec, action)
                    stage_guard_spec = {k: path[k] for k in STALL_GUARD_FIELDS if k in path} | {k: stage[k] for k in STALL_GUARD_FIELDS if k in stage}
                    resolved_guard = resolve_stall_guard(stage_guard_spec, action)
                    robot = adapter.robot.get_robot_by_arm_name(arm+'_arm')
                    target = own['path']['waypoints'][0][arm]
                    options = dict(start_q=np.asarray(expected_q))
                    if getattr(adapter, 'joint_replan_enabled', False):
                        options['joint_replan_context'] = dict(deadline=deadline, speed_scale=own.get('speed_scale', 1.),
                            position_tolerance_m=resolved_tol['position_tolerance_m'])
                    local_diagnostics = []
                    proxies = action.get('review_proxies') or {}
                    from .scene_obstacles import plan_move_with_obstacle_clearance
                    plan, resampled_arc = plan_move_with_obstacle_clearance(
                        adapter,
                        robot,
                        arm,
                        own,
                        target,
                        (targets.get(arm) or {}).get('position_m'),
                        expected_q,
                        proxies.get('obstacles') or [],
                        proxies.get('contact_entity_id'),
                        local_diagnostics,
                        before_state=before,
                        cap_remaining=cap - len(rows),
                        **options,
                    )
                    diagnostics.extend(dict(item, arm=arm, stage_id=identifier) for item in local_diagnostics)
                    verify()
                    if plan is None:
                        return dict(status='failed', reason='path_infeasible', failed_stage_id=identifier,
                            failed_arm=arm, physics_advanced=False, executed_native_steps=0,
                            failure_feedback=adapter._failure_feedback(own, arm, 0, local_diagnostics))
                    positions = resampled_arc if resampled_arc is not None else adapter._resample_action_plan(plan[0], arm, own, start_q=np.asarray(expected_q))
                    event = next(iter(own.get('gripper_events', [])), None)
                    jaw_values, schedule = None, None
                    if event:
                        if start_jaw is None:
                            raise ValueError('batch_jaw_requires_measured_initial_opening')
                        from .gripper_motion import synchronize
                        synchronized = synchronize(positions, expected_q, start_jaw, event,
                            adapter.eef_gripper_dwell_step*own.get('gripper_speed_scale', 1.), cap)
                        if synchronized is None:
                            raise ValueError('motion_duration_limit')
                        positions, jaw_values, schedule = synchronized
                        jaw_commanded.add(arm)
                    local = []
                    for i, q in enumerate(positions):
                        row = list(held)
                        row[offset:offset+6] = q.tolist()
                        if jaw_values is not None:
                            row[offset+6] = float(jaw_values[i])
                        local.append(row)
                    record.update(
                        target=deepcopy(target),
                        tolerance_preset=resolved_tol['tolerance_preset'],
                        position_tolerance_m=resolved_tol['position_tolerance_m'],
                        orientation_tolerance_rad=resolved_tol['orientation_tolerance_rad'],
                        max_upward_dz_m=resolved_tol.get('max_upward_dz_m'),
                        stall_guard_preset=resolved_guard['stall_guard_preset'],
                        max_tracking_divergence_m=resolved_guard['max_tracking_divergence_m'],
                        gripper_motion=schedule,
                    )
                else:
                    if start_jaw is None and own.get('gripper_speed_scale', 1.) < 1:
                        raise ValueError('batch_jaw_requires_measured_initial_opening')
                    local = _jaw_rows(adapter, stage, held, start_jaw)
                    jaw_commanded.add(arm)
                if not local:
                    raise ValueError('batch_empty_stage_commands')
                record.update(rows=local, end=len(rows)+len(local))
                records[identifier] = record
                active[arm] = dict(id=identifier, rows=local, cursor=0)
                del pending[identifier]
            if not active:
                raise ValueError('batch_dependency_deadlock')
            if len(rows) >= cap:
                raise ValueError('motion_duration_limit')
            context = {}
            for arm in ARMS:
                offset = 0 if arm == 'left' else 7
                current = active.get(arm)
                if current:
                    held[offset:offset+7] = current['rows'][current['cursor']][offset:offset+7]
                    context[arm] = dict(phase=records[current['id']]['kind'], stage_id=current['id'])
                else:
                    context[arm] = dict(phase='hold', stage_id=None)
            rows.append(list(held))
            contexts.append(context)
            finished = []
            for arm, current in list(active.items()):
                current['cursor'] += 1
                if current['cursor'] == len(current['rows']):
                    identifier = current['id']
                    completed.add(identifier)
                    finished.append(identifier)
                    if records[identifier]['kind'] == 'move':
                        targets[arm] = records[identifier]['target']
                        tolerances[arm] = records[identifier]['position_tolerance_m']
                        orientation_tolerances[arm] = records[identifier]['orientation_tolerance_rad']
                    del active[arm]
            if finished:
                boundaries.append(dict(cursor=len(rows), finished_stage_ids=finished))
        batch_jaw_checkpoint = bool(action.get('checkpoint_after_gripper', False))
        for identifier, record in records.items():
            events = record.get('action', {}).get('gripper_events', [])
            if not events:
                continue
            # 同一手臂在进入下一个 move 或显式 observe 前若还有后续纯夹爪阶段，由最后一个纯夹爪阶段代表该窗口
            later_same_arm_jaw = any(
                other_id != identifier and other['kind'] == 'gripper' and other['arm'] == record['arm']
                and other['start'] >= record['end']
                and other.get('action', {}).get('gripper_events')
                and not any(
                    mid['start'] >= record['end'] and mid['end'] <= other['start']
                    and (mid['kind'] == 'observe' or (mid['kind'] == 'move' and mid['arm'] == record['arm']))
                    for mid in records.values()
                )
                for other_id, other in records.items()
            )
            if later_same_arm_jaw:
                continue
            window_records = [
                other for other in records.values()
                if other['arm'] == record['arm'] and other['end'] <= record['end']
                and other.get('action', {}).get('gripper_events')
                and not any(
                    mid['start'] >= other['end'] and mid['end'] <= record['start']
                    and (mid['kind'] == 'observe' or (mid['kind'] == 'move' and mid['arm'] == record['arm']))
                    for mid in records.values()
                )
            ]
            if record.get('checkpoint_after_gripper') is False:
                continue
            stage_explicit = any(item.get('checkpoint_after_gripper') is True for item in window_records)
            if not (stage_explicit or batch_jaw_checkpoint):
                continue
            dependent_move_starts = [
                records[stage['id']]['start'] for stage in stages
                if stage['kind'] == 'move' and identifier in dependencies[stage['id']]
            ]
            if not dependent_move_starts:
                continue
            offset = 0 if record['arm'] == 'left' else 7
            previous = rows[record['start']-1] if record['start'] else adapter._held_row(before)
            goal = events[0]['goal_normalized']
            if goal == previous[offset+6] or (not stage_explicit and goal >= previous[offset+6] - 1e-4):
                continue
            first_move_start = min(dependent_move_starts)
            # 若窗口 [record['end'], first_move_start] 内已有显式 observe 阶段，不再重复插入自动检查点
            if any(
                record['end'] <= item['cursor'] <= first_move_start
                and any(s['id'] in item['stage_ids'] and s['kind'] == 'observe' for s in stages)
                for item in checkpoints
            ):
                continue
            target_cursor = first_move_start
            marker = next((item for item in checkpoints if item['cursor'] == target_cursor), None)
            if marker is None:
                marker = dict(cursor=target_cursor, stage_ids=[], reasons={})
                checkpoints.append(marker)
            if identifier not in marker['stage_ids']:
                marker['stage_ids'].append(identifier)
            marker['reasons'][identifier] = (
                'Jaw stage completed before dependent move (pre-lift); inspect finger contact alignment in RGB before continuing, '
                'or verify object co-motion after a lift stage.')
        checkpoints.sort(key=lambda item: item['cursor'])
        verify()
        from .motion_safety import adapter_assessment
        safety = []
        for identifier, record in records.items():
            if record['kind'] == 'observe':
                continue
            arm = record['arm']; offset = 0 if arm == 'left' else 7
            state = deepcopy(before)
            state['arms'][arm]['q_rad'] = record['expected_start_q']
            assessment = adapter_assessment(adapter, state, {arm: [r[offset:offset+6] for r in record['rows']]},
                native_steps=record['end'], request_remaining=cap, **record.get('joint_travel_options', {}))
            assessment['start_state_source'] = 'initial_measurement' if record['start'] == 0 else 'preceding_planned_command'
            if record['start']:
                assessment['information_sources'][0] = 'preceding planned named joint endpoint, not future measurement'
            safety.append(dict(stage_id=identifier, **assessment))
        rejected_checks = [s for s in safety if s['enabled'] and s['status'] == 'reject']
        if rejected_checks:
            first_rej = rejected_checks[0]
            first_arm = next((a for a, v in (first_rej.get('arms') or {}).items() if v.get('status') == 'reject'), None)
            return dict(status='failed', reason='motion_safety_rejected',
                failed_stage_id=first_rej.get('stage_id'), failed_arm=first_arm,
                planned_native_steps=len(rows),
                stage_costs={key: dict(arm=value['arm'], kind=value['kind'], start_native_step=value['start'],
                    end_native_step=value['end'], native_steps=value['end']-value['start']) for key, value in records.items()},
                motion_safety_checks=safety,
                physics_advanced=False, executed_native_steps=0)
        compact, full = _evidence(adapter, before, action, rows, contexts)
        _populate_planned_positions(adapter, before, records, full)
        from .batch_geometry import review_batch
        from .scene_obstacles import filter_review_obstacles
        review_proxies = deepcopy(action.get('review_proxies', {}))
        if review_proxies.get('obstacles'):
            kept = filter_review_obstacles(review_proxies['obstacles'], records)
            if kept:
                review_proxies['obstacles'] = kept
            else:
                review_proxies.pop('obstacles', None)
        geometry_review = review_batch(adapter, before, full, review_proxies)
        full['batch_geometry_review'] = geometry_review
        verify()
    except ValueError as error:
        return dict(status='failed', reason=str(error), physics_advanced=False, executed_native_steps=0)
    identifier = 'batch-'+uuid.uuid4().hex
    cache = getattr(adapter, '_batch_plans', None)
    if cache is None:
        cache = adapter._batch_plans = {}
    cache_limit = int(base_thresholds.get('batch_plan_cache_size', 16))
    while len(cache) >= cache_limit:
        cache.pop(next(iter(cache)))
    cache[identifier] = dict(action_digest=motion_digest(action), state_binding=binding, cursor=0,
        checkpoint_index=0, revision=0, status='planned', executable=True, rows=rows, contexts=contexts, stages=records,
        source_action=deepcopy({k: v for k, v in action.items() if k != 'stages'}),
        source_stages=deepcopy(action['stages']), completed_stage_ids=[],
        pending_stage_ids=[s['id'] for s in action['stages']], failed_stage_id=None,
        continuation_of=continuation_of,
        checkpoints=checkpoints, boundaries=boundaries, completed=set(), targets=deepcopy({a: before['arms'][a]['ee_pose'] for a in ARMS}),
        tolerances=dict.fromkeys(ARMS, float(base_thresholds.get('default_position_tolerance_m', .010))),
        orientation_tolerances=dict.fromkeys(ARMS, float(base_thresholds.get('orientation_tolerance_rad', .15))),
        initial_state=before)
    arm_costs = {}
    for arm, offset in (('left', 0), ('right', 7)):
        trajectory = np.asarray([before['arms'][arm]['q_rad'], *[row[offset:offset+6] for row in rows]])
        travel = np.abs(np.diff(trajectory, axis=0)).sum(axis=0)
        arm_costs[arm] = dict(motion_native_steps=sum(c[arm]['phase'] == 'move' for c in contexts),
            gripper_native_steps=sum(c[arm]['phase'] == 'gripper' for c in contexts),
            hold_native_steps=sum(c[arm]['phase'] == 'hold' for c in contexts),
            joint_travel_rad=travel.tolist(), sum_joint_travel_rad=float(travel.sum()))
    planned = dict(status='planned', plan_id=identifier, physics_advanced=False, executed_native_steps=0,
        planned_native_steps=len(rows), nominal_sim_time_s=len(rows)/adapter.control_hz,
        arm_costs=arm_costs,
        stage_costs={key: dict(arm=value['arm'], kind=value['kind'], start_native_step=value['start'],
            end_native_step=value['end'], native_steps=value['end']-value['start']) for key,value in records.items()},
        checkpoints=checkpoints, trajectory_evidence=compact, trajectory_data=full, motion_safety_checks=safety,
        geometry_review=geometry_review,
        execution_semantics='exact cached shared rows; stop on changed state or failed measured checkpoint; no replan')
    if continuation_of is not None:
        planned['continuation'] = continuation_of
        planned['continuation_of'] = continuation_of
    return planned


def _validate_paused_state(adapter, cache, action, binding_unchanged):
    """在恢复缓存批次前校验暂停期间是否插入了其他动作，并与阶段衔接容差统一对齐。"""
    from .feedback_motion import pose_error
    if adapter.env.end_flag[0] or not cache.get('paused_stop_confirmed', True):
        return False
    expected_obs = cache.get('expected_observation_id')
    if expected_obs is not None and action.get('based_on_observation_id') != expected_obs:
        return False
    binding = cache.get('state_binding')
    expected_steps = cache.get('paused_native_action_count',
        binding.get('native_action_count') if isinstance(binding, dict) else None)
    native_action_steps = int(adapter.env.take_action_cnt[0])
    if expected_steps is not None and native_action_steps != expected_steps:
        cache['resume_validation'] = dict(
            status='invalid',
            reason='native_action_steps_changed_during_pause',
            state_binding='unchanged' if binding_unchanged else 'revalidated',
            native_action_steps=native_action_steps,
            expected_native_action_steps=expected_steps)
        return False
    thresholds = motion_thresholds(adapter)
    joint_tol = float(action.get('resume_drift_tolerance_rad', getattr(adapter, 'resume_drift_tolerance_rad', thresholds['resume_max_joint_drift_rad'])))
    pos_tol = float(action.get('resume_ee_drift_tolerance_m', thresholds['resume_max_ee_drift_m']))
    angle_tol = float(action.get('resume_orientation_drift_tolerance_rad', max(thresholds['resume_max_orientation_drift_rad'], joint_tol)))
    jaw_tol = float(action.get('resume_jaw_drift_tolerance_normalized', thresholds['resume_max_jaw_drift_normalized']))
    current, paused = adapter.state(), cache['paused_state']
    valid_resume = adapter.last_dispatched_goals == cache['paused_jaw_goals']
    for arm in ARMS:
        q_error = float(np.max(np.abs(np.asarray(current['arms'][arm]['q_rad'])-paused['arms'][arm]['q_rad'])))
        position, angle = pose_error(current['arms'][arm]['ee_pose'], paused['arms'][arm]['ee_pose'])
        valid_resume = (valid_resume and math.isfinite(q_error+position+angle)
                        and q_error <= joint_tol and position <= pos_tol and angle <= angle_tol)
        old_jaw = cache['paused_measured_jaws'][arm]
        if old_jaw is not None:
            try:
                valid_resume = valid_resume and abs(adapter._measured_gripper_goal(current, arm)-old_jaw) <= jaw_tol
            except (ValueError, KeyError, AttributeError):
                valid_resume = False
    cache['resume_validation'] = dict(
        status='valid' if valid_resume else 'invalid',
        state_binding='unchanged' if binding_unchanged else 'revalidated',
        native_action_steps=native_action_steps,
        expected_native_action_steps=expected_steps,
        max_joint_drift_rad=joint_tol, max_ee_drift_m=pos_tol, max_orientation_drift_rad=angle_tol,
        max_jaw_opening_drift_normalized=jaw_tol,
        scope='public robot state and unchanged jaw goals; cached commands unchanged; no scene/contact proof')
    return valid_resume


def consume_batch(adapter, action):
    if 'source_plan_id' in action:
        from ...runtime.actions.plan_continuation import resolve_backend_continuation
        resolved, _ = resolve_backend_continuation(adapter, action)
        action.clear()
        action.update(resolved)
    validate_batch_action(action, adapter.capabilities())
    identifier = action.get('batch_plan_id')
    cache = getattr(adapter, '_batch_plans', {}).get(identifier)
    if cache is None:
        raise ValueError('batch_plan_unknown_or_expired')
    if cache['status'] not in ('planned', 'awaiting_observation'):
        raise ValueError('batch_plan_already_consumed_or_failed')
    if action.get('batch_expected_cursor',cache['cursor']) != cache['cursor']:
        raise ValueError('batch_plan_cursor_changed_since_review')
    if action.get('batch_expected_revision',cache['revision']) != cache['revision']:
        raise ValueError('batch_plan_revision_changed_since_review')
    if cache['action_digest'] != motion_digest(action):
        raise ValueError('batch_plan_commands_changed')
    cache['resuming'] = cache['status'] == 'awaiting_observation'
    binding_unchanged = cache['state_binding'] == adapter._preview_state_binding()
    if cache['resuming']:
        if not _validate_paused_state(adapter, cache, action, binding_unchanged):
            cache['status'] = 'invalidated'
            cache['executable'] = False
            raise ValueError('batch_plan_stale_state_requires_new_preview')
    elif not binding_unchanged:
        cache['status'] = 'invalidated'
        cache['executable'] = False
        raise ValueError('batch_plan_stale_state_requires_new_preview')
    cache['status'] = 'executing'
    cache['revision'] += 1
    return cache


def execute_batch(adapter, action, cache, receipt, segments, progress, interrupted, dispatch):
    from .feedback_motion import pose_error
    from .motion_safety import admit_motion
    thresholds = motion_thresholds(adapter)
    rows, records = cache['rows'], cache['stages']
    start = cache['cursor']
    trace = dict(action_space='native_joint', execution_mode='cached_batch', actions=deepcopy(rows[start:]),
        row_contexts=deepcopy(cache['contexts'][start:]), dt_s=1/adapter.control_hz,
        batch_start_native_step=start, complete_batch_native_steps=len(rows))
    segments.append(trace)
    stage_grip_facts = cache.setdefault('stage_grip_facts', {})
    stage_before_states = cache.setdefault('stage_before_states', {})
    active_arm_grip_guard = cache.setdefault('active_arm_grip_guard', {})
    results = {identifier: dict(kind=record['kind'], arm=record['arm'],
        status='completed' if identifier in cache['completed'] else 'pending',
        planned_steps=record['end']-record['start'],
        completed_steps=max(0, min(cache['cursor'], record['end'])-record['start']),
        **({'grip_facts': deepcopy(stage_grip_facts[identifier])} if identifier in stage_grip_facts else {}))
        for identifier,record in records.items()}
    receipt.update(batch_plan_id=action['batch_plan_id'], batch_status='executing', stage_results=results,
        preview_execution_semantics='exact cached native rows; no replanning, resampling or jaw rebase')
    if cache.get('resume_validation'):
        receipt['resume_validation'] = deepcopy(cache['resume_validation'])
    checked_starts = set()
    gripper_start_pos = {}
    stall_state = {}
    move_arms = {record['arm'] for record in records.values() if record['kind'] == 'move'}
    active_arms = {record['arm'] for record in records.values() if record['kind'] != 'observe' and record.get('arm') in ARMS}
    def fail(reason, identifier=None, **extra):
        receipt.update(status='failed', reason=reason, **extra)
        if identifier:
            receipt.update(failed_stage_id=identifier, failed_arm=records[identifier]['arm'])
            results[identifier]['status'] = 'failed'
        receipt['stop_confirmation'] = adapter._hold()
    def _is_open_jaw_pre_grasp(record):
        expected_jaw = record.get('expected_start_jaw')
        if expected_jaw is not None and float(expected_jaw) < 0.85:
            return False
        own_events = record.get('action', {}).get('gripper_events') or []
        if own_events and float(own_events[0].get('goal_normalized', 1.0)) < 0.5:
            return True
        arm = record['arm']
        for _, other in sorted(records.items(), key=lambda item: item[1]['start']):
            if other['arm'] != arm or other['start'] < record['end']:
                continue
            if other['kind'] == 'move':
                return False
            if other['kind'] == 'gripper':
                ev = other.get('action', {}).get('gripper_events') or []
                return bool(ev) and float(ev[0].get('goal_normalized', 1.0)) < 0.5
        return False
    def endpoint(identifier):
        record = records[identifier]; arm = record['arm']
        measured_pose = adapter.state()['arms'][arm]['ee_pose']
        ref_target = record.get('original_target', record['target'])
        pos_tol = float(record.get('position_tolerance_m', thresholds.get('default_position_tolerance_m', .010)))
        ori_tol = float(record.get('orientation_tolerance_rad', thresholds.get('orientation_tolerance_rad', .15)))
        is_contact_stop = record.get('stall_guard_preset') == 'contact_stop'
        expected_jaw = record.get('expected_start_jaw')
        is_open_jaw = expected_jaw is None or float(expected_jaw) >= 0.85
        is_pre_grasp = _is_open_jaw_pre_grasp(record)
        is_tight = record.get('tolerance_preset') in ('tight', 'fine')
        allow_vertical = is_contact_stop or (pos_tol >= 0.009 and not is_tight)
        if is_pre_grasp:
            max_upward_dz = max(pos_tol, 0.022 if is_contact_stop else 0.018)
        elif is_contact_stop:
            max_upward_dz = max(pos_tol, 0.040)
        elif is_open_jaw:
            max_upward_dz = max(pos_tol, 0.018)
        else:
            max_upward_dz = max(pos_tol * 1.2, 0.025)
        explicit_dz = record.get('max_upward_dz_m')
        if explicit_dz is not None:
            allow_vertical = float(explicit_dz) > 0
            max_upward_dz = float(explicit_dz)
        ok, position, orientation, residual, vertical_compliant = _check_pose_arrival(
            measured_pose, ref_target, pos_tol, ori_tol,
            allow_vertical_compliance=allow_vertical,
            max_upward_dz_m=max_upward_dz,
        )
        if not ok and pos_tol < 0.0055 and position <= 0.0055 and orientation <= ori_tol:
            planned_positions = record.get('planned_positions_m')
            if isinstance(planned_positions, list) and len(planned_positions) > 0:
                planned_end_err = float(np.linalg.norm(
                    np.asarray(measured_pose['position_m'], dtype=float)
                    - np.asarray(planned_positions[-1], dtype=float)
                ))
                if planned_end_err <= pos_tol:
                    ok = True
        results[identifier]['endpoint_error'] = dict(
            position_error_m=position,
            orientation_error_rad=orientation,
            effective_tolerances=dict(
                position_tolerance_m=round(pos_tol, 5),
                orientation_tolerance_rad=round(ori_tol, 5),
                max_upward_dz_m=round(max_upward_dz if allow_vertical else 0.0, 5),
            ),
        )
        if vertical_compliant:
            results[identifier]['endpoint_error']['xy_error_m'] = round(math.hypot(residual['dx_m'], residual['dy_m']), 5)
            results[identifier]['endpoint_error']['vertical_compliance_accepted'] = True
        results[identifier]['endpoint_residual_m'] = residual
        receipt['endpoint_residual_m'] = residual
        if record['action'].get('input_grasp_pose'):
            from ..model.grasp_contact import contact_target_feedback
            try:
                results[identifier]['grasp_contact_feedback'] = contact_target_feedback(record['action'], adapter.state())
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                results[identifier]['grasp_contact_feedback'] = dict(available=False, reason='diagnostic_unavailable', error_type=type(error).__name__)
        if not ok:
            if 'first_divergence' in results[identifier]:
                receipt['first_divergence'] = deepcopy(results[identifier]['first_divergence'])
            summary = build_contact_block_summary(
                identifier, arm, ref_target, measured_pose,
                results[identifier].get('grasp_contact_feedback'),
                completed_steps=results[identifier]['completed_steps'],
                planned_steps=results[identifier]['planned_steps'],
            )
            results[identifier]['contact_block_summary'] = summary
            receipt['contact_block_summary'] = summary
            expected_jaw = record.get('expected_start_jaw')
            if expected_jaw is not None and float(expected_jaw) < 0.85:
                cur_jaw = _extract_jaw(adapter.state(), adapter=adapter, arm=arm)
                cur_ap = _aperture_from_jaw(cur_jaw, adapter=adapter, arm=arm)
                if cur_ap is not None and cur_ap > 0.004:
                    results[identifier]['in_hand_tracking_error'] = True
                    receipt['in_hand_tracking_error'] = True
            fail('tracking_error', identifier)
            return False
        if vertical_compliant:
            record['vertical_compliance_accepted'] = True
            record['accepted_pose'] = deepcopy(measured_pose)
            results[identifier]['completion_quality'] = 'vertical_compliance_within_xy_tolerance'
        elif record.get('kind') == 'move' and 0.005 < position <= pos_tol:
            results[identifier]['endpoint_error']['within_1cm_soft_tolerance'] = True
            results[identifier]['completion_quality'] = 'within_1cm_tolerance'
        return True
    try:
        budget_limit = getattr(adapter, 'native_action_budget_limit', None)
        if budget_limit is not None:
            remaining = max(0, budget_limit-int(adapter.env.take_action_cnt[0]))
            receipt['batch_budget_admission'] = dict(remaining_native_steps=remaining,
                required_suffix_native_steps=len(rows)-cache['cursor'])
            if len(rows)-cache['cursor'] > remaining:
                fail('batch_remaining_native_budget_insufficient')
                return
        if cache.get('resuming') and start < len(rows):
            state = adapter.state()
            paused = cache.get('paused_state', state)
            paused_jaws = cache.get('paused_measured_jaws', {})
            for arm, offset in (('left', 0), ('right', 7)):
                identifier = cache['contexts'][start][arm]['stage_id']
                if identifier is None:
                    continue
                record = records[identifier]; own = record['action']
                reference_q = (np.asarray(state['arms'][arm]['q_rad']) if start == 0 else
                    np.asarray(rows[start-1][offset:offset+6])+np.asarray(state['arms'][arm]['q_rad'])-np.asarray(paused['arms'][arm]['q_rad']))
                jump = np.abs(np.asarray(rows[start][offset:offset+6])-reference_q)
                limits = adapter._speed_step_limits(arm, own.get('speed_scale', 1.))
                if start > 0:
                    planned_step = np.abs(np.asarray(rows[start][offset:offset+6]) - np.asarray(rows[start-1][offset:offset+6]))
                    limits = np.maximum(limits, planned_step)
                if (jump > limits+1e-9).any():
                    fail('batch_resume_first_joint_step_exceeds_limit', identifier,
                        first_joint_step_rad=jump.tolist(), max_joint_step_rad=limits.tolist())
                    return
                if own.get('gripper_events') and (own.get('gripper_speed_scale', 1.) < 1 or own['gripper_events'][0]['phase'] == 'during_motion'):
                    measured = adapter._measured_gripper_goal(state, arm)
                    planned_jaw = record['expected_start_jaw'] if record['start'] == start else rows[start-1][offset+6]
                    paused_jaw = paused_jaws.get(arm)
                    reference_jaw = (measured if planned_jaw is None or paused_jaw is None else
                        planned_jaw+(measured-paused_jaw))
                    bound = adapter.eef_gripper_dwell_step*own.get('gripper_speed_scale', 1.)
                    if abs(rows[start][offset+6]-reference_jaw) > bound+1e-12:
                        fail('batch_resume_first_jaw_step_exceeds_limit', identifier)
                        return
        while True:
            if interrupted('execution'):
                return
            cursor = cache['cursor']
            marker_index = cache['checkpoint_index']
            if marker_index < len(cache['checkpoints']) and cache['checkpoints'][marker_index]['cursor'] == cursor:
                marker = cache['checkpoints'][marker_index]
                cache['checkpoint_index'] += 1
                for identifier in marker['stage_ids']:
                    cache['completed'].add(identifier)
                    results[identifier]['status'] = 'completed'
                cache['status'] = 'awaiting_observation'
                stop_conf = adapter._hold()
                receipt.update(status='stopped', reason='observation_checkpoint', paused_at_checkpoint=True,
                    checkpoint_stage_ids=marker['stage_ids'], checkpoint_reasons=dict(marker['reasons']),
                    stop_confirmation=stop_conf)
                cache['paused_stop_confirmed'] = bool(
                    isinstance(stop_conf, dict)
                    and stop_conf.get('confirmed') is not False
                    and stop_conf.get('status') != 'unknown'
                )
                cache['state_binding'] = adapter._preview_state_binding()
                cache['paused_native_action_count'] = int(adapter.env.take_action_cnt[0])
                cache['paused_state'] = deepcopy(adapter.state())
                cache['paused_jaw_goals'] = dict(adapter.last_dispatched_goals)
                cache['paused_measured_jaws'] = {}
                for arm in ARMS:
                    try:
                        cache['paused_measured_jaws'][arm] = adapter._measured_gripper_goal(cache['paused_state'], arm)
                    except (ValueError, KeyError, AttributeError):
                        cache['paused_measured_jaws'][arm] = None
                return
            if cursor == len(rows):
                current_state = adapter.state()
                for arm in ARMS:
                    if arm not in active_arms:
                        initial_pose = cache['initial_state']['arms'][arm]['ee_pose']
                        drift_pos, _ = pose_error(current_state['arms'][arm]['ee_pose'], initial_pose)
                        if math.isfinite(drift_pos):
                            receipt['passive_arm_drift_m'] = round(float(drift_pos), 5)
                            receipt.setdefault('passive_arm_drifts_m', {})[arm] = round(float(drift_pos), 5)
                        continue
                    if arm not in move_arms:
                        continue
                    measured_pose = current_state['arms'][arm]['ee_pose']
                    pos_tol = cache['tolerances'][arm]
                    ori_tol = cache.get('orientation_tolerances', {}).get(arm, thresholds.get('orientation_tolerance_rad', .15))
                    ok, _, _, residual, _ = _check_pose_arrival(
                        measured_pose, cache['targets'][arm], pos_tol, ori_tol,
                        allow_vertical_compliance=(pos_tol >= 0.009),
                    )
                    if not ok:
                        receipt['endpoint_residual_m'] = residual
                        fail('tracking_error', failed_arm=arm, failed_phase='final_held')
                        return
                cache['status'] = 'completed'
                receipt.update(status='completed', reason=None)
                return
            state = adapter.state()
            starts = [identifier for identifier,record in records.items() if record['kind'] != 'observe'
                      and record['start'] == cursor and identifier not in checked_starts]
            base_drift_limit = float(thresholds.get('segment_start_joint_drift_rad', .10))
            for identifier in starts:
                record = records[identifier]; arm = record['arm']; own = record['action']
                offset = 0 if arm == 'left' else 7
                start_drift_limit = max(base_drift_limit, float(cache.get('orientation_tolerances', {}).get(arm, .10)))
                drift = float(np.max(np.abs(np.asarray(state['arms'][arm]['q_rad'])-record['expected_start_q'])))
                if drift > start_drift_limit:
                    fail('batch_segment_start_changed', identifier, max_joint_start_error_rad=drift)
                    return
                if record['kind'] == 'gripper':
                    gripper_start_pos[identifier] = np.asarray(state['arms'][arm]['ee_pose']['position_m'], dtype=float).copy()
                elif record['kind'] == 'move':
                    start_pos_arr = np.asarray(state['arms'][arm]['ee_pose']['position_m'], dtype=float).copy()
                    stall_state[identifier] = {
                        'consecutive': 0,
                        'last_div': None,
                        'last_pos': start_pos_arr,
                        'start_z': float(start_pos_arr[2]),
                    }
                events = own.get('gripper_events', [])
                if events and (own.get('gripper_speed_scale', 1.) < 1 or events[0]['phase'] == 'during_motion'):
                    measured = adapter._measured_gripper_goal(state, arm)
                    reference_jaw = record['expected_start_jaw'] if record['expected_start_jaw'] is not None else measured
                    first = record['rows'][0][offset+6]
                    if abs(first-reference_jaw) > adapter.eef_gripper_dwell_step*own.get('gripper_speed_scale', 1.)+1e-12:
                        fail('gripper_start_state_changed', identifier)
                        return
                if not admit_motion(adapter, state, {arm: [r[offset:offset+6] for r in record['rows']]}, receipt,
                        native_steps=len(rows)-cursor, request_remaining=int(math.ceil(10*adapter.control_hz))-cursor,
                        stage_id=identifier, **record.get('joint_travel_options', {})):
                    return
                stage_before_states[identifier] = _arm_grip_state(adapter, state, arm)
                checked_starts.add(identifier)
                results[identifier]['status'] = 'running'
            previous = receipt['executed_steps']
            attempted = [context['stage_id'] for context in cache['contexts'][cursor].values() if context['stage_id']]
            for identifier in attempted:
                results[identifier]['status'] = 'running'
                results[identifier]['dispatch_attempted_steps'] = results[identifier].get('dispatch_attempted_steps', 0)+1
            try:
                succeeded = dispatch(rows[cursor], command_phase='batch')
            except Exception:
                for identifier in attempted:
                    results[identifier]['execution_uncertain'] = True
                raise
            if receipt['executed_steps'] > previous:
                cache['cursor'] += 1
                for arm in ARMS:
                    identifier = cache['contexts'][cursor][arm]['stage_id']
                    if identifier:
                        results[identifier]['completed_steps'] += 1
            if not succeeded:
                return
            post_step_state = None
            for arm in ARMS:
                identifier = cache['contexts'][cursor][arm]['stage_id']
                if not identifier or records[identifier]['kind'] != 'move':
                    continue
                record = records[identifier]
                planned_positions = record.get('planned_positions_m')
                local_step = cursor - record['start']
                if not planned_positions or not (0 <= local_step < len(planned_positions)):
                    continue
                if post_step_state is None:
                    post_step_state = adapter.state()
                measured_pose = post_step_state['arms'][arm]['ee_pose']
                measured_pos = np.asarray(measured_pose['position_m'], dtype=float)
                planned_pos = np.asarray(planned_positions[local_step], dtype=float)
                step_div = float(np.linalg.norm(measured_pos - planned_pos))
                max_div = record.get('max_tracking_divergence_m')
                div_threshold = max_div if max_div is not None else record.get('position_tolerance_m', .010)
                if math.isfinite(step_div) and step_div > div_threshold and 'first_divergence' not in results[identifier]:
                    results[identifier]['first_divergence'] = {
                        'native_step': cursor + 1,
                        'stage_step': local_step + 1,
                        'divergence_m': round(step_div, 5),
                        'planned_position_m': [round(float(x), 5) for x in planned_pos],
                        'measured_position_m': [round(float(x), 5) for x in measured_pos],
                    }
                if max_div is not None and math.isfinite(step_div):
                    st = stall_state.setdefault(identifier, {'consecutive': 0, 'last_div': None, 'last_pos': None})
                    step_move = None if st['last_pos'] is None else float(np.linalg.norm(measured_pos - st['last_pos']))
                    st['last_pos'] = measured_pos.copy()
                    vel_thresh = float(thresholds.get('stall_guard_step_velocity_threshold_m', 0.0015))
                    consec_req = int(thresholds.get('stall_guard_consecutive_steps', 2))
                    prev_planned = np.asarray(planned_positions[local_step - 1], dtype=float) if local_step > 0 else None
                    planned_step_move = None if prev_planned is None else float(np.linalg.norm(planned_pos - prev_planned))
                    stalled = (step_move is None or step_move <= vel_thresh) and (planned_step_move is None or planned_step_move > vel_thresh)
                    if step_div > max_div and stalled and cursor + 1 < record['end'] - 1:
                        st['consecutive'] += 1
                        st['last_div'] = step_div
                    else:
                        st['consecutive'] = 0
                        st['last_div'] = step_div
                    if st['consecutive'] >= consec_req:
                        ref_target = record.get('original_target', record['target'])
                        pos_tol = float(record.get('position_tolerance_m', thresholds.get('default_position_tolerance_m', .010)))
                        ori_tol = float(record.get('orientation_tolerance_rad', thresholds.get('orientation_tolerance_rad', .15)))
                        is_pre_grasp = _is_open_jaw_pre_grasp(record)
                        ok_contact = False
                        if record.get('stall_guard_preset') == 'contact_stop':
                            explicit_dz = record.get('max_upward_dz_m')
                            contact_max_dz = (
                                float(explicit_dz) if explicit_dz is not None
                                else (max(pos_tol, 0.022) if is_pre_grasp else max(pos_tol, 0.040))
                            )
                            ok_contact, _, _, _, _ = _check_pose_arrival(
                                measured_pose, ref_target, pos_tol, ori_tol,
                                allow_vertical_compliance=contact_max_dz > 0,
                                max_upward_dz_m=contact_max_dz,
                            )
                        if ok_contact:
                            record.setdefault('original_target', deepcopy(record['target']))
                            held_q = [float(x) for x in post_step_state['arms'][arm]['q_rad']]
                            offset = 0 if arm == 'left' else 7
                            for r_idx in range(cache['cursor'], record['end']):
                                rows[r_idx][offset:offset+6] = list(held_q)
                            for _, other in sorted(records.items(), key=lambda item: item[1]['start']):
                                if other['arm'] != arm or other['start'] < record['end']:
                                    continue
                                if other['kind'] == 'move':
                                    break
                                if other['kind'] == 'gripper':
                                    other['expected_start_q'] = list(held_q)
                                    for r_idx in range(other['start'], other['end']):
                                        rows[r_idx][offset:offset+6] = list(held_q)
                            results[identifier]['contact_stopped_early'] = True
                            st['consecutive'] = 0
                        else:
                            position, orientation = pose_error(measured_pose, ref_target)
                            residual = compute_endpoint_residual(ref_target, measured_pose)
                            results[identifier]['endpoint_error'] = dict(position_error_m=position, orientation_error_rad=orientation)
                            results[identifier]['endpoint_residual_m'] = residual
                            results[identifier]['failed_phase'] = 'stall_guard_stopped'
                            receipt['endpoint_residual_m'] = residual
                            if 'first_divergence' in results[identifier]:
                                receipt['first_divergence'] = deepcopy(results[identifier]['first_divergence'])
                            summary = build_contact_block_summary(
                                identifier, arm, ref_target, measured_pose,
                                results[identifier].get('grasp_contact_feedback'),
                                completed_steps=results[identifier]['completed_steps'],
                                planned_steps=results[identifier]['planned_steps'],
                            )
                            results[identifier]['contact_block_summary'] = summary
                            receipt['contact_block_summary'] = summary
                            fail('tracking_error', identifier, failed_phase='stall_guard_stopped',
                                 tracking_divergence_m=round(step_div, 5), max_tracking_divergence_m=max_div)
                            return
            finished = [identifier for identifier, record in records.items() if record['kind'] != 'observe'
                        and record['end'] == cache['cursor']]
            # Check all move arms ending on the row before admitting any successor.
            finished_moves = [identifier for identifier in finished if records[identifier]['kind'] == 'move']
            valid = [endpoint(identifier) for identifier in finished_moves]
            if not all(valid):
                return
            after_state = adapter.state() if finished else None
            guard_triggered = []
            for identifier in finished:
                record = records[identifier]
                arm = record['arm']
                cache['completed'].add(identifier)
                results[identifier]['status'] = 'completed'
                if record['kind'] == 'move':
                    pos_tol = float(record.get('position_tolerance_m', thresholds.get('default_position_tolerance_m', .010)))
                    ori_tol = float(record.get('orientation_tolerance_rad', thresholds.get('orientation_tolerance_rad', .15)))
                    if record.get('vertical_compliance_accepted'):
                        cache['targets'][arm] = deepcopy(record.get('accepted_pose', record['target']))
                    else:
                        cache['targets'][arm] = record['target']
                    res_dz = (results[identifier].get('endpoint_residual_m') or {}).get('dz_m', 0.0)
                    if res_dz > 0.001 or record.get('vertical_compliance_accepted'):
                        held_q = [float(x) for x in after_state['arms'][arm]['q_rad']]
                        offset = 0 if arm == 'left' else 7
                        for _, other in sorted(records.items(), key=lambda item: item[1]['start']):
                            if other['arm'] != arm or other['start'] < record['end']:
                                continue
                            if other['kind'] == 'move':
                                break
                            if other['kind'] == 'gripper':
                                other['expected_start_q'] = list(held_q)
                                for r_idx in range(other['start'], other['end']):
                                    rows[r_idx][offset:offset+6] = list(held_q)
                    cache['tolerances'][arm] = pos_tol
                    cache.setdefault('orientation_tolerances', {})[arm] = ori_tol
                elif record['kind'] == 'gripper':
                    if post_step_state is None:
                        post_step_state = adapter.state()
                    end_pose = post_step_state['arms'][arm]['ee_pose']
                    end_pos = np.asarray(end_pose['position_m'], dtype=float)
                    start_pos = gripper_start_pos.get(identifier, np.asarray(record['target']['position_m'], dtype=float))
                    results[identifier]['compliance_shift_m'] = round(float(np.linalg.norm(end_pos - start_pos)), 5)
                    cache['targets'][arm] = deepcopy(end_pose)
                before_grip = stage_before_states.get(identifier)
                if before_grip is None:
                    before_grip = [*record['expected_start_q'], record['expected_start_jaw'] if record['expected_start_jaw'] is not None else 0.0]
                after_grip = _arm_grip_state(adapter, after_state, arm)
                grip_facts = compute_grip_facts(before_grip, after_grip, adapter=adapter, arm=arm)
                results[identifier]['grip_facts'] = grip_facts
                stage_grip_facts[identifier] = deepcopy(grip_facts)
                jaw_events = record.get('action', {}).get('gripper_events') or []
                is_closing_cmd = bool(jaw_events) and float(jaw_events[0].get('goal_normalized', 1.0)) < 0.5
                is_opening_cmd = bool(jaw_events) and float(jaw_events[0].get('goal_normalized', 1.0)) >= 0.5
                if is_opening_cmd:
                    active_arm_grip_guard.pop(arm, None)
                has_remaining_stages = any(other_id not in cache['completed'] and other_id not in finished for other_id in records)
                has_arm_remaining = any(
                    other_id not in cache['completed'] and other_id not in finished and records[other_id].get('arm') == arm
                    for other_id in records
                )
                if record.get('grip_guard'):
                    jaw_delta = grip_facts.get('jaw_delta')
                    guard_eval = evaluate_grip_guard(grip_facts, record['grip_guard'])
                    if not has_remaining_stages or (jaw_delta is not None and jaw_delta >= -1e-3):
                        guard_eval = dict(guard_eval, triggered=False, reason=None, message=None)
                    results[identifier]['grip_guard'] = guard_eval
                    if guard_eval['triggered']:
                        guard_triggered.append((identifier, guard_eval))
                        active_arm_grip_guard.pop(arm, None)
                    elif is_closing_cmd:
                        active_arm_grip_guard[arm] = deepcopy(record['grip_guard'])
                elif record['kind'] == 'move' and not jaw_events:
                    pending_guard = active_arm_grip_guard.pop(arm, None)
                    start_z = (stall_state.get(identifier) or {}).get('start_z')
                    target_z = float(record['target']['position_m'][2]) if record.get('target') else None
                    is_downward_move = (start_z is not None and target_z is not None and (target_z - start_z) < -0.005)
                    jaw_delta = grip_facts.get('jaw_delta') or 0.0
                    ap_after = grip_facts.get('measured_aperture_m')
                    if (
                        pending_guard
                        and has_arm_remaining
                        and not is_downward_move
                        and jaw_delta <= -0.08
                        and ap_after is not None
                        and ap_after <= 0.002
                    ):
                        guard_eval = evaluate_grip_guard(grip_facts, pending_guard)
                        if guard_eval['triggered'] and guard_eval['reason'] == 'aperture_below_min_m':
                            results[identifier]['grip_guard'] = guard_eval
                            guard_triggered.append((identifier, guard_eval))
            if guard_triggered:
                for identifier, guard_eval in guard_triggered:
                    results[identifier]['status'] = 'paused_at_checkpoint'
                    results[identifier]['checkpoint_reason'] = guard_eval['reason']
                if cache['checkpoint_index'] < len(cache['checkpoints']) and cache['checkpoints'][cache['checkpoint_index']]['cursor'] == cache['cursor']:
                    marker = cache['checkpoints'][cache['checkpoint_index']]
                    cache['checkpoint_index'] += 1
                    for obs_id in marker['stage_ids']:
                        cache['completed'].add(obs_id)
                        if obs_id not in dict(guard_triggered):
                            results[obs_id]['status'] = 'completed'
                cache['status'] = 'awaiting_observation'
                stop_conf = adapter._hold()
                receipt.update(
                    status='stopped',
                    reason='observation_checkpoint',
                    paused_at_checkpoint=True,
                    checkpoint_reason=guard_triggered[0][1]['reason'],
                    checkpoint_stage_ids=[i for i, _ in guard_triggered],
                    checkpoint_reasons={i: g['message'] for i, g in guard_triggered},
                    stop_confirmation=stop_conf,
                )
                cache['paused_stop_confirmed'] = bool(
                    isinstance(stop_conf, dict)
                    and stop_conf.get('confirmed') is not False
                    and stop_conf.get('status') != 'unknown'
                )
                cache['state_binding'] = adapter._preview_state_binding()
                cache['paused_native_action_count'] = int(adapter.env.take_action_cnt[0])
                cache['paused_state'] = deepcopy(adapter.state())
                cache['paused_jaw_goals'] = dict(adapter.last_dispatched_goals)
                cache['paused_measured_jaws'] = {}
                for arm in ARMS:
                    try:
                        cache['paused_measured_jaws'][arm] = adapter._measured_gripper_goal(cache['paused_state'], arm)
                    except (ValueError, KeyError, AttributeError):
                        cache['paused_measured_jaws'][arm] = None
                return
    finally:
        if cache['status'] == 'executing':
            cache['status'] = 'failed'
        completed_ids = [s['id'] for s in action['stages'] if s['id'] in cache['completed']]
        pending_ids = [s['id'] for s in action['stages'] if s['id'] not in cache['completed']]
        cache['executable'] = cache['status'] in ('planned', 'awaiting_observation')
        cache['completed_stage_ids'] = completed_ids
        cache['pending_stage_ids'] = pending_ids
        cache['failed_stage_id'] = receipt.get('failed_stage_id')
        for stage in results.values():
            if stage['status'] == 'running':
                stage['status'] = 'paused' if cache['status'] == 'awaiting_observation' else 'stopped'
        receipt.update(batch_status=cache['status'], batch_cursor=cache['cursor'], batch_revision=cache['revision'],
            complete_batch_native_steps=len(rows), completed_stage_ids=completed_ids,
            pending_stage_ids=pending_ids)
