"""显式容差预设、接触受阻早停与连坐 Bug 解绑回归测试。"""

from copy import deepcopy
import math
import threading
import unittest

import numpy as np

from auriga.robot.motion.motion_tolerance import (
    TOLERANCE_PRESETS,
    compute_endpoint_residual,
    resolve_stage_tolerance,
    resolve_stall_guard,
)
from auriga.runtime.actions.unified_actions import compile_request
import test_dual_arm_interface as interface_fixture
from test_batch_motion import batch, jaw, jaws, move
from test_joint_replan_runtime import periodic_pose
from test_motion_preview_backend import preview_adapter


class MotionToleranceAndStallTests(unittest.TestCase):
    setUp = interface_fixture.DualArmHarnessTests.setUp

    def execute_batch(self, adapter, action, plan):
        return adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda update: None, threading.Event())

    def test_tolerance_preset_and_priority_resolution(self):
        # 默认回退为 normal (0.010m, 0.15rad)
        default_tol = resolve_stage_tolerance({})
        self.assertEqual(default_tol, {
            'tolerance_preset': 'normal',
            'position_tolerance_m': 0.010,
            'orientation_tolerance_rad': 0.15,
        })
        # 各预设值核对
        for name, expected in TOLERANCE_PRESETS.items():
            resolved = resolve_stage_tolerance({'tolerance_preset': name})
            self.assertEqual(resolved['tolerance_preset'], name)
            self.assertAlmostEqual(resolved['position_tolerance_m'], expected['position_tolerance_m'], places=5)
            self.assertAlmostEqual(resolved['orientation_tolerance_rad'], expected['orientation_tolerance_rad'], places=5)

        # 优先级链：stage 显式数值 > stage preset > batch 显式数值 > batch preset > 默认
        batch_defaults = {
            'tolerance_preset': 'tight',
            'position_tolerance_m': 0.018,
        }
        # 继承 batch 预设 + batch 显式覆盖
        inh = resolve_stage_tolerance({}, batch_defaults)
        self.assertEqual(inh['tolerance_preset'], 'tight')
        self.assertAlmostEqual(inh['position_tolerance_m'], 0.018)
        self.assertAlmostEqual(inh['orientation_tolerance_rad'], 0.05)

        # stage preset 覆盖 batch
        stage_p = resolve_stage_tolerance({'tolerance_preset': 'coarse'}, batch_defaults)
        self.assertEqual(stage_p['tolerance_preset'], 'coarse')
        self.assertAlmostEqual(stage_p['position_tolerance_m'], 0.025)
        self.assertAlmostEqual(stage_p['orientation_tolerance_rad'], 0.35)

        # stage 显式数值只覆盖对应维度，另一维度保留 stage/batch preset
        stage_num = resolve_stage_tolerance(
            {'tolerance_preset': 'coarse', 'position_tolerance_m': 0.04},
            batch_defaults,
        )
        self.assertAlmostEqual(stage_num['position_tolerance_m'], 0.04)
        self.assertAlmostEqual(stage_num['orientation_tolerance_rad'], 0.35)

    def test_stall_guard_preset_and_priority_resolution(self):
        self.assertEqual(resolve_stall_guard({}), {
            'stall_guard_preset': 'off',
            'max_tracking_divergence_m': None,
        })
        self.assertEqual(resolve_stall_guard({'stall_guard_preset': 'contact_stop'}), {
            'stall_guard_preset': 'contact_stop',
            'max_tracking_divergence_m': 0.020,
        })
        self.assertEqual(resolve_stall_guard({'stall_guard_preset': 'strict'}), {
            'stall_guard_preset': 'strict',
            'max_tracking_divergence_m': 0.010,
        })
        # 优先级：stage 显式 > stage preset > batch 显式 > batch preset
        batch_def = {'stall_guard_preset': 'contact_stop', 'max_tracking_divergence_m': 0.03}
        self.assertEqual(resolve_stall_guard({}, batch_def)['max_tracking_divergence_m'], 0.03)
        self.assertEqual(resolve_stall_guard({'stall_guard_preset': 'strict'}, batch_def)['max_tracking_divergence_m'], 0.010)
        self.assertEqual(resolve_stall_guard({'stall_guard_preset': 'off'}, batch_def)['max_tracking_divergence_m'], None)
        self.assertEqual(
            resolve_stall_guard({'stall_guard_preset': 'strict', 'max_tracking_divergence_m': 0.04}, batch_def)['max_tracking_divergence_m'],
            0.04,
        )

    def test_compute_endpoint_residual_signed_components(self):
        target = {'position_m': [0.20, -0.10, 0.80], 'quaternion_xyzw': [0.0, 0.0, 0.0, 1.0]}
        measured = {'position_m': [0.212, -0.105, 0.785], 'quaternion_xyzw': [0.0, 0.0, math.sin(0.05), math.cos(0.05)]}
        res = compute_endpoint_residual(target, measured)
        self.assertAlmostEqual(res['dx_m'], 0.012, places=5)
        self.assertAlmostEqual(res['dy_m'], -0.005, places=5)
        self.assertAlmostEqual(res['dz_m'], -0.015, places=5)
        self.assertAlmostEqual(res['position_error_m'], round(math.sqrt(0.012**2 + 0.005**2 + 0.015**2), 5), places=5)
        self.assertAlmostEqual(res['orientation_error_rad'], 0.10, places=4)

    def test_coarse_tolerance_preset_allows_larger_arrival_error(self):
        adapter, _, _ = preview_adapter()
        action = batch([move('transit', arm='left', angle=-0.25, tolerance_preset='coarse')])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        original_state = adapter.state

        # 在终点注入 12mm 位置偏差与 0.20 rad 姿态偏差（超过 normal 5mm/0.1rad，但在 coarse 15mm/0.25rad 内）
        def perturbed_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= plan['planned_native_steps']:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [pose['position_m'][0] + 0.012, pose['position_m'][1], pose['position_m'][2]]
                angle = -0.25 + 0.20
                pose['quaternion_xyzw'] = [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
            return st

        adapter.state = perturbed_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        residual = result['stage_results']['transit']['endpoint_residual_m']
        self.assertAlmostEqual(residual['dx_m'], 0.012, places=4)
        self.assertAlmostEqual(residual['orientation_error_rad'], 0.20, places=4)
        self.assertEqual(result['endpoint_residual_m'], residual)

    def test_stall_guard_contact_stop_triggers_early_before_stage_completion(self):
        adapter, _, dispatched = preview_adapter()
        # angle=-0.5 规划 10 步
        action = batch([move('descent', arm='left', angle=-0.5, stall_guard_preset='contact_stop')])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['planned_native_steps'], 10)

        original_state = adapter.state
        frozen_pose = deepcopy(original_state()['arms']['left']['ee_pose'])

        # 模拟接触受阻：机械臂末端位置卡在初始位置不动，随规划推进与规划轨迹偏离超过 25mm
        def blocked_state():
            st = original_state()
            st['arms']['left']['ee_pose'] = deepcopy(frozen_pose)
            return st

        adapter.state = blocked_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'tracking_error')
        self.assertEqual(result['failed_phase'], 'stall_guard_stopped')
        self.assertEqual(result['failed_stage_id'], 'descent')
        # 连续 2 步超过 0.025m 即停机，远早于 10 步
        self.assertLess(len(dispatched), 5)
        self.assertEqual(result['executed_native_steps'], len(dispatched))
        self.assertIn('first_divergence', result)
        self.assertIn('first_divergence', result['stage_results']['descent'])
        self.assertIn('endpoint_residual_m', result)
        self.assertEqual(result['stage_results']['descent']['failed_phase'], 'stall_guard_stopped')

    def test_stall_guard_ignores_moving_pd_lag_and_contact_stop_accepts_compliant_endpoint(self):
        adapter, _, dispatched = preview_adapter()
        action = batch([move('sweep', arm='left', angle=-0.5, stall_guard_preset='contact_stop', position_tolerance_m=0.020)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        total_steps = plan['planned_native_steps']
        original_state = adapter.state

        # 模拟运动中的 PD 跟随滞后（滞后 22mm > 20mm，但每步仍在持续推进 > 1.5mm），以及终点 18mm 接触顺应偏差（<= 20mm）
        def lagging_but_moving_state():
            st = original_state()
            step = adapter.env.take_action_cnt[0]
            pos = list(st['arms']['left']['ee_pose']['position_m'])
            if 0 < step < total_steps:
                pos[0] -= 0.022
            elif step >= total_steps:
                pos[0] -= 0.018
            st['arms']['left']['ee_pose']['position_m'] = pos
            return st

        adapter.state = lagging_but_moving_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(dispatched), total_steps)
        self.assertAlmostEqual(result['stage_results']['sweep']['endpoint_error']['position_error_m'], 0.018, places=4)

    def test_pure_gripper_stage_ignores_ee_compliance_shift_and_records_metric(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        action = batch([jaw(identifier='close_jaw', arm='left', goal=0.0)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')

        original_state = adapter.state

        # 模拟闭合夹爪时因接触物体产生的 15mm 末端顺应偏移（超过默认 5mm/10mm 位置容差）
        def compliant_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= 1:
                pos = st['arms']['left']['ee_pose']['position_m']
                st['arms']['left']['ee_pose']['position_m'] = [pos[0] + 0.015, pos[1], pos[2]]
            return st

        adapter.state = compliant_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertIsNone(result['reason'])
        self.assertAlmostEqual(result['stage_results']['close_jaw']['compliance_shift_m'], 0.015, places=5)

    def test_passive_arm_drift_does_not_fail_final_held_on_single_arm_batch(self):
        adapter, _, _ = preview_adapter()
        # 仅左臂运动，右臂在本批次无任何阶段
        action = batch([move('left_only', arm='left', angle=-0.125)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')

        original_state = adapter.state

        # 模拟被动右臂在左臂运动期间发生 8mm 控制器漂移（超过 5mm）
        def passive_drift_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= 1:
                pos = st['arms']['right']['ee_pose']['position_m']
                st['arms']['right']['ee_pose']['position_m'] = [pos[0], pos[1] - 0.008, pos[2]]
            return st

        adapter.state = passive_drift_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertIsNone(result['reason'])
        self.assertAlmostEqual(result['passive_arm_drift_m'], 0.008, places=5)
        self.assertAlmostEqual(result['passive_arm_drifts_m']['right'], 0.008, places=5)

    def test_unified_action_compiles_batch_and_stage_tolerance_and_stall_guard(self):
        args = {
            'observation_id': self.h.latest['observation_id'],
            'tolerance_preset': 'coarse',
            'stall_guard_preset': 'contact_stop',
            'stages': [
                {
                    'id': 'transit',
                    'kind': 'move',
                    'arm': 'left',
                    'grasp_position_m': [0.2, -0.1, 0.8],
                    'angles_deg': [0, 0, 0],
                },
                {
                    'id': 'insert',
                    'kind': 'move',
                    'arm': 'left',
                    'grasp_position_m': [0.2, -0.1, 0.75],
                    'angles_deg': [0, 0, 0],
                    'tolerance_preset': 'tight',
                    'orientation_tolerance_rad': 0.12,
                    'stall_guard_preset': 'strict',
                },
            ],
        }
        for role in ('actor', 'program'):
            self.h.tool_registry.validate('robot_action', args, role=role)
        compiled = compile_request(self.h, args)
        p0 = compiled['stages'][0]['action']['path']
        p1 = compiled['stages'][1]['action']['path']
        self.assertEqual(p0['tolerance_preset'], 'coarse')
        self.assertAlmostEqual(p0['position_tolerance_m'], 0.025)
        self.assertAlmostEqual(p0['orientation_tolerance_rad'], 0.35)
        self.assertEqual(p0['stall_guard_preset'], 'contact_stop')
        self.assertAlmostEqual(p0['max_tracking_divergence_m'], 0.020)

        self.assertEqual(p1['tolerance_preset'], 'tight')
        self.assertAlmostEqual(p1['position_tolerance_m'], 0.003)
        self.assertAlmostEqual(p1['orientation_tolerance_rad'], 0.12)
        self.assertEqual(p1['stall_guard_preset'], 'strict')
        self.assertAlmostEqual(p1['max_tracking_divergence_m'], 0.010)

    def test_nonfinite_physx_state_raises_clean_runtime_error_without_poisoning_receipt_json(self):
        import io
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from auriga.robot.backends.robodojo.robot_worker import RoboDojoAdapter

        # 1. 直接验证 RoboDojoAdapter.state 在 PhysX 返回 NaN 位姿或零范数四元数时抛出 RuntimeError
        raw_adapter = object.__new__(RoboDojoAdapter)
        raw_adapter.np = np
        raw_adapter.control_step = 0
        raw_adapter.goals = {'left': 1.0}
        raw_adapter.robot_description = SimpleNamespace(frame_id='env_origin', tcp_config_id='arx-x5-link6-v1')
        target_robot = SimpleNamespace(
            type='target', arm_name='left_arm', gripper_name='left_gripper',
            gripper_scale=[0.0, 1.0], gripper_move={'sign': 1},
            arm_joints_name=[f'j{i}' for i in range(6)], arm_joint_indices=list(range(6)),
            gripper_joints_name=['jaw'],
        )
        class _DummyVel:
            def __getitem__(self, idx):
                return SimpleNamespace(cpu=lambda: np.zeros(6))

        key = SimpleNamespace(data=SimpleNamespace(joint_vel=_DummyVel()))
        raw_adapter.robot = SimpleNamespace(
            robot_list=[target_robot],
            robot_key=[key],
            get_real_endpose=lambda r, idx: np.array([[float('nan')] * 7]),
            get_joint=lambda r, idx: np.zeros((1, 6)),
            get_end_effector_real_val=lambda r, idx: np.zeros((1, 1)),
            process_name=lambda n: n,
            control_manager=SimpleNamespace(prev_control=[{'left_gripper': {'position': [0.5]}}]),
        )
        with self.assertRaisesRegex(RuntimeError, 'simulator_nonfinite_robot_state:left'):
            RoboDojoAdapter.state(raw_adapter)

        # 2. 验证在 execute 期间当 state() 抛出 simulator_nonfinite_robot_state 时，回执保持严格 JSON 合规（不含 NaN）
        adapter, _, _ = preview_adapter()
        action = batch([move('one', arm='left', angle=-0.125)])
        plan = adapter.preview_batch(action)
        original_state = adapter.state

        def nan_state():
            if adapter.env.take_action_cnt[0] >= 1:
                raise RuntimeError('simulator_nonfinite_robot_state:left')
            return original_state()

        adapter.state = nan_state
        with patch('sys.stderr', io.StringIO()):
            receipt = self.execute_batch(adapter, action, plan)
        serialized = json.dumps(receipt, allow_nan=False)
        self.assertIn('simulator_nonfinite_robot_state:left', serialized)

    def test_normal_soft_tolerance_and_persistent_obstacles_and_segment_start_drift(self):
        # 1. normal 预设下 5mm ~ 10mm 终点偏差按 1cm 软容差通过并标记 within_1cm_soft_tolerance
        adapter, _, _ = preview_adapter()
        action = batch([move('approach', arm='left', angle=-0.25)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        original_state = adapter.state

        def soft_perturbed_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= plan['planned_native_steps']:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [pose['position_m'][0] + 0.0082, pose['position_m'][1], pose['position_m'][2]]
            return st

        adapter.state = soft_perturbed_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['stage_results']['approach']['endpoint_error'].get('within_1cm_soft_tolerance'))
        self.assertEqual(result['stage_results']['approach']['completion_quality'], 'within_1cm_tolerance')

        # 2. 会话级持久化障碍物默认合并到同 episode 后续 compile_request，stateless 模式下不注入
        args1 = {
            'observation_id': self.h.latest['observation_id'],
            'obstacles': [{
                'id': 'box_wall',
                'min_m': [0.10, -0.20, 0.70],
                'max_m': [0.30, -0.05, 0.88],
                'evidence': 'measured front box wall from depth_query',
            }],
            'stages': [{
                'id': 'm1', 'kind': 'move', 'arm': 'left',
                'grasp_position_m': [0.20, -0.10, 0.92], 'angles_deg': [0, 0, 0],
            }],
        }
        c1 = compile_request(self.h, args1)
        self.assertEqual(len(c1['review_proxies']['obstacles']), 1)
        self.assertEqual(c1['review_proxies']['obstacles'][0]['id'], 'box_wall')
        args2 = {
            'observation_id': self.h.latest['observation_id'],
            'stages': [{
                'id': 'm2', 'kind': 'move', 'arm': 'left',
                'grasp_position_m': [0.20, 0.10, 0.92], 'angles_deg': [0, 0, 0],
            }],
        }
        c2 = compile_request(self.h, args2)
        self.assertEqual(len(c2['review_proxies']['obstacles']), 1)
        self.assertEqual(c2['review_proxies']['obstacles'][0]['id'], 'box_wall')
        c3 = compile_request(self.h, dict(args2, obstacle_policy='stateless'))
        self.assertNotIn('review_proxies', c3)

    def test_stage_tolerances_do_not_ratchet_and_preserve_normal_default(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        action = batch([
            move('transit', arm='left', angle=-0.125, tolerance_preset='coarse'),
            move('place', arm='left', angle=-0.25),
            jaw(identifier='open_jaw', arm='left', goal=1.0),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        cache = adapter._batch_plans[plan['plan_id']]
        # place 未显式传容差时保持 normal 默认 10mm，不做隐式 5mm 覆写，也不被阶段 1 的 coarse (25mm) 污染
        self.assertAlmostEqual(cache['stages']['place']['position_tolerance_m'], 0.010, places=5)
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertAlmostEqual(cache['tolerances']['left'], 0.010, places=5)

    def test_vertical_compliance_rejects_undeclared_downward_sag_and_xy_miss(self):
        # A 32 mm downward sag exceeds the declared 10 mm tolerance even with XY aligned.
        # Reject it and do not execute the subsequent release.
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        action = batch([
            move('carry_sag', arm='left', angle=-0.25),
            jaw(identifier='release_jaw', arm='left', goal=1.0),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        move_steps = plan['stage_costs']['carry_sag']['end_native_step']
        original_state = adapter.state

        def vertical_compliant_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= move_steps:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [
                    pose['position_m'][0] + 0.004,
                    pose['position_m'][1] - 0.003,
                    pose['position_m'][2] - 0.032,
                ]
            return st

        adapter.state = vertical_compliant_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'tracking_error')
        self.assertEqual(result['failed_stage_id'], 'carry_sag')
        self.assertEqual(result['stage_results']['release_jaw']['completed_steps'], 0)
        descent_res = result['stage_results']['carry_sag']
        self.assertFalse(descent_res['endpoint_error'].get('vertical_compliance_accepted', False))
        self.assertAlmostEqual(math.hypot(descent_res['endpoint_residual_m']['dx_m'],
                                         descent_res['endpoint_residual_m']['dy_m']), 0.005, places=4)
        self.assertAlmostEqual(descent_res['endpoint_residual_m']['dz_m'], -0.032, places=4)

        # 2. 未声明 contact_stop 的开爪抓取下探若被挡在目标上方 +32mm (> 12mm)，拒绝悬空闭爪
        adapter_air, _, _ = preview_adapter()
        jaws(adapter_air)
        adapter_air.last_dispatched_goals['left'] = 1.0
        action_air = batch([
            move('blocked_descent', arm='left', angle=-0.25),
            jaw(identifier='air_close', arm='left', goal=0.0),
        ])
        plan_air = adapter_air.preview_batch(action_air)
        orig_air = adapter_air.state

        def blocked_high_state():
            st = orig_air()
            if adapter_air.env.take_action_cnt[0] >= plan_air['stage_costs']['blocked_descent']['end_native_step']:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [pose['position_m'][0], pose['position_m'][1], pose['position_m'][2] + 0.032]
            return st

        adapter_air.state = blocked_high_state
        res_air = self.execute_batch(adapter_air, action_air, plan_air)
        self.assertEqual(res_air['status'], 'failed')
        self.assertEqual(res_air['reason'], 'tracking_error')
        self.assertEqual(res_air['failed_stage_id'], 'blocked_descent')
        self.assertEqual(res_air['stage_results']['air_close']['completed_steps'], 0)

        # 3. 水平 (dx, dy) 偏离 15mm (> 10mm normal 容差)，即使 Z 误差为 0 也严格拒绝
        adapter2, _, _ = preview_adapter()
        action2 = batch([move('miss_xy', arm='left', angle=-0.25)])
        plan2 = adapter2.preview_batch(action2)
        orig2 = adapter2.state

        def xy_miss_state():
            st = orig2()
            if adapter2.env.take_action_cnt[0] >= plan2['planned_native_steps']:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [pose['position_m'][0] + 0.015, pose['position_m'][1], pose['position_m'][2]]
            return st

        adapter2.state = xy_miss_state
        result2 = self.execute_batch(adapter2, action2, plan2)
        self.assertEqual(result2['status'], 'failed')
        self.assertEqual(result2['reason'], 'tracking_error')
        self.assertEqual(result2['failed_stage_id'], 'miss_xy')

    def test_contact_stop_mid_trajectory_vertical_touchdown_clamps_and_executes_attached_gripper(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        action = batch([
            move('touchdown', arm='left', angle=-0.5, stall_guard_preset='contact_stop'),
            jaw(identifier='release', arm='left', goal=1.0),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        target_pose = deepcopy(adapter._batch_plans[plan['plan_id']]['stages']['touchdown']['target'])
        original_state = adapter.state

        # 模拟下探在中途接触台面：水平 (x, y) 已对准目标 (dx=3mm, dy=2mm)，但高度停在目标上方 +30mm 不再下降
        def vertical_touchdown_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= 2:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_pose['position_m'][0] + 0.003,
                        target_pose['position_m'][1] + 0.002,
                        target_pose['position_m'][2] + 0.030,
                    ],
                    'quaternion_xyzw': list(target_pose['quaternion_xyzw']),
                }
            return st

        adapter.state = vertical_touchdown_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['completed_stage_ids'], ['touchdown', 'release'])
        self.assertTrue(result['stage_results']['touchdown'].get('contact_stopped_early'))
        self.assertAlmostEqual(result['stage_results']['touchdown']['endpoint_residual_m']['dz_m'], 0.030, places=4)

        # 开爪抓取下探（后续紧跟闭爪 goal=0.0）若即使声明 contact_stop 仍被物体顶部挡在 +28mm (> 15mm)，必须报 tracking_error 拒绝悬空闭爪
        adapter_grasp, _, _ = preview_adapter()
        jaws(adapter_grasp)
        adapter_grasp.last_dispatched_goals['left'] = 1.0
        action_grasp = batch([
            move('grasp_descent', arm='left', angle=-0.5, stall_guard_preset='contact_stop'),
            jaw(identifier='grasp_close', arm='left', goal=0.0),
        ])
        plan_grasp = adapter_grasp.preview_batch(action_grasp)
        target_grasp = deepcopy(adapter_grasp._batch_plans[plan_grasp['plan_id']]['stages']['grasp_descent']['target'])
        orig_grasp_state = adapter_grasp.state

        def blocked_grasp_state():
            st = orig_grasp_state()
            if adapter_grasp.env.take_action_cnt[0] >= 2:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_grasp['position_m'][0] + 0.002,
                        target_grasp['position_m'][1] + 0.001,
                        target_grasp['position_m'][2] + 0.028,
                    ],
                    'quaternion_xyzw': list(target_grasp['quaternion_xyzw']),
                }
            return st

        adapter_grasp.state = blocked_grasp_state
        res_grasp = self.execute_batch(adapter_grasp, action_grasp, plan_grasp)
        self.assertEqual(res_grasp['status'], 'failed')
        self.assertEqual(res_grasp['reason'], 'tracking_error')
        self.assertEqual(res_grasp['failed_stage_id'], 'grasp_descent')
        self.assertEqual(res_grasp['stage_results']['grasp_close']['completed_steps'], 0)

    def test_planning_failure_preserves_explicit_grasp_orientation(self):
        adapter, _, _ = preview_adapter()
        adapter.joint_replan_enabled = True
        stage = move('place', arm='left', angle=-0.125)
        stage['action']['input_grasp_pose'] = {'left': {
            'position_m': [0.25, -0.20, 0.88], 'angles_deg': [0., 0., 0.], 'offset_m': 0.15007}}
        original = deepcopy(stage)
        calls = []
        original_plan = adapter._plan_segment
        def planner(robot, target, diagnostics, waypoint_index=0, **kwargs):
            calls.append(deepcopy(target))
            if len(calls) == 1:
                diagnostics.append({'stage': 'ik', 'feasible_seed_count': 0})
                return None
            return original_plan(robot, target, diagnostics, waypoint_index, **kwargs)
        adapter._plan_segment = planner
        plan = adapter.preview_batch(batch([stage]))
        self.assertEqual(plan['status'], 'failed')
        self.assertEqual(plan['executed_native_steps'], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stage, original)

    def test_vertical_compliance_respects_rotation_and_downward_tolerances(self):
        from auriga.robot.motion.batch_motion import _check_pose_arrival
        target = {'position_m': [0., 0., 0.8], 'quaternion_xyzw': [0., 0., 0., 1.]}
        for dz, angle, expected in [(0.001, 0.20, False), (-0.035, 0., False),
                                     (0.015, 0.02, True), (-0.002, 0.02, True)]:
            measured = {'position_m': [0., 0., 0.8 + dz],
                        'quaternion_xyzw': [0., 0., math.sin(angle/2), math.cos(angle/2)]}
            result = _check_pose_arrival(measured, target, .003, .05,
                                        allow_vertical_compliance=True, max_upward_dz_m=.02)
            self.assertEqual(result[0], expected, (dz, angle, result))

    def test_explicit_max_upward_dz_m_and_effective_tolerances_reported(self):
        # 模型显式传入阶段级 max_upward_dz_m 时，preview 与 execute 均透明返回 effective_tolerances 且严格按模型设置执行
        adapter_rel, _, _ = preview_adapter()
        jaws(adapter_rel)
        adapter_rel.last_dispatched_goals['left'] = 0.0
        action_rel = batch([
            jaw(identifier='close_first', arm='left', goal=0.0),
            move('place_descent', arm='left', angle=-0.5, stall_guard_preset='contact_stop', max_upward_dz_m=0.075, position_tolerance_m=0.020),
            jaw(identifier='place_open', arm='left', goal=1.0),
        ])
        plan_rel = adapter_rel.preview_batch(action_rel)
        self.assertEqual(adapter_rel._batch_plans[plan_rel['plan_id']]['stages']['place_descent']['max_upward_dz_m'], 0.075)
        place_end = plan_rel['stage_costs']['place_descent']['end_native_step']
        target_rel = deepcopy(adapter_rel._batch_plans[plan_rel['plan_id']]['stages']['place_descent']['target'])
        orig_rel_state = adapter_rel.state

        def stack_contact_state():
            st = orig_rel_state()
            if adapter_rel.env.take_action_cnt[0] >= place_end:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_rel['position_m'][0] + 0.010,
                        target_rel['position_m'][1] - 0.010,
                        target_rel['position_m'][2] + 0.061,
                    ],
                    'quaternion_xyzw': list(target_rel['quaternion_xyzw']),
                }
            return st

        adapter_rel.state = stack_contact_state
        res_rel = self.execute_batch(adapter_rel, action_rel, plan_rel)
        self.assertEqual(res_rel['status'], 'completed', res_rel)
        self.assertEqual(res_rel['completed_stage_ids'], ['close_first', 'place_descent', 'place_open'])
        eff_tol = res_rel['stage_results']['place_descent']['endpoint_error']['effective_tolerances']
        self.assertEqual(eff_tol['max_upward_dz_m'], 0.075)

        # 3. 开爪预抓取下探在阶段终点 (endpoint) 即使声明 contact_stop 被挡在 +28mm (> 15mm)，也必须报 tracking_error 并返回 contact_block_summary
        adapter_ep, _, _ = preview_adapter()
        jaws(adapter_ep)
        adapter_ep.last_dispatched_goals['left'] = 1.0
        action_ep = batch([
            move('grasp_ep', arm='left', angle=-0.25, stall_guard_preset='contact_stop'),
            jaw(identifier='close_ep', arm='left', goal=0.0),
        ])
        plan_ep = adapter_ep.preview_batch(action_ep)
        ep_end = plan_ep['stage_costs']['grasp_ep']['end_native_step']
        target_ep = deepcopy(adapter_ep._batch_plans[plan_ep['plan_id']]['stages']['grasp_ep']['target'])
        orig_ep_state = adapter_ep.state

        def endpoint_blocked_grasp_state():
            st = orig_ep_state()
            if adapter_ep.env.take_action_cnt[0] >= ep_end:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_ep['position_m'][0] + 0.002,
                        target_ep['position_m'][1] + 0.001,
                        target_ep['position_m'][2] + 0.028,
                    ],
                    'quaternion_xyzw': list(target_ep['quaternion_xyzw']),
                }
            return st

        adapter_ep.state = endpoint_blocked_grasp_state
        res_ep = self.execute_batch(adapter_ep, action_ep, plan_ep)
        self.assertEqual(res_ep['status'], 'failed')
        self.assertEqual(res_ep['reason'], 'tracking_error')
        self.assertEqual(res_ep['failed_stage_id'], 'grasp_ep')
        self.assertEqual(res_ep['stage_results']['close_ep']['completed_steps'], 0)
        self.assertIn('contact_block_summary', res_ep)

        # 4. 闭爪接触下探后原地旋转的 tight 阶段显式指定 max_upward_dz_m=0.02 时允许 +16mm 顺应高度
        adapter_turn, _, _ = preview_adapter()
        jaws(adapter_turn)
        adapter_turn.last_dispatched_goals['left'] = 0.0
        action_turn = batch([
            jaw(identifier='hold_nut', arm='left', goal=0.0),
            move('shallow_insert', arm='left', angle=-0.25, stall_guard_preset='contact_stop'),
            move('turn_quarter', arm='left', angle=-0.25, tolerance_preset='tight', max_upward_dz_m=0.02),
        ])
        plan_turn = adapter_turn.preview_batch(action_turn)
        insert_end = plan_turn['stage_costs']['shallow_insert']['end_native_step']
        target_turn = deepcopy(adapter_turn._batch_plans[plan_turn['plan_id']]['stages']['shallow_insert']['target'])
        orig_turn_state = adapter_turn.state

        def compliant_support_turn_state():
            st = orig_turn_state()
            if adapter_turn.env.take_action_cnt[0] >= insert_end:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_turn['position_m'][0] + 0.002,
                        target_turn['position_m'][1] - 0.001,
                        target_turn['position_m'][2] + 0.016,
                    ],
                    'quaternion_xyzw': list(target_turn['quaternion_xyzw']),
                }
            return st

        adapter_turn.state = compliant_support_turn_state
        res_turn = self.execute_batch(adapter_turn, action_turn, plan_turn)
        self.assertEqual(res_turn['status'], 'completed', res_turn)
        self.assertEqual(res_turn['completed_stage_ids'], ['hold_nut', 'shallow_insert', 'turn_quarter'])

        # 5. 同时设置 tolerance_preset='tight' 与 stall_guard_preset='contact_stop' 的阶段，在 contact_stop 接触停住后 endpoint() 不得将 allow_vertical 置为 False
        adapter_tc, _, _ = preview_adapter()
        jaws(adapter_tc)
        adapter_tc.last_dispatched_goals['left'] = 0.0
        action_tc = batch([
            jaw(identifier='hold_slice', arm='left', goal=0.0),
            move('short_insert', arm='left', angle=-0.5, tolerance_preset='tight', stall_guard_preset='contact_stop'),
            jaw(identifier='release_slice', arm='left', goal=1.0),
        ])
        plan_tc = adapter_tc.preview_batch(action_tc)
        target_tc = deepcopy(adapter_tc._batch_plans[plan_tc['plan_id']]['stages']['short_insert']['target'])
        orig_tc_state = adapter_tc.state

        def tight_contact_stop_state():
            st = orig_tc_state()
            if adapter_tc.env.take_action_cnt[0] >= 2:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_tc['position_m'][0] + 0.0005,
                        target_tc['position_m'][1] - 0.0005,
                        target_tc['position_m'][2] + 0.025,
                    ],
                    'quaternion_xyzw': list(target_tc['quaternion_xyzw']),
                }
            return st

        adapter_tc.state = tight_contact_stop_state
        res_tc = self.execute_batch(adapter_tc, action_tc, plan_tc)
        self.assertEqual(res_tc['status'], 'completed', res_tc)
        self.assertEqual(res_tc['completed_stage_ids'], ['hold_slice', 'short_insert', 'release_slice'])
        self.assertEqual(
            res_tc['stage_results']['short_insert']['endpoint_error']['effective_tolerances']['max_upward_dz_m'],
            0.04,
        )

        # 6. 显式传入严格 position_tolerance_m < 0.009（且无 contact_stop 与显式 max_upward_dz_m）时，不应放行 +20mm 的垂直偏差
        adapter_strict, _, _ = preview_adapter()
        jaws(adapter_strict)
        action_strict = batch([
            move('precise_insert', arm='left', angle=-0.5, position_tolerance_m=0.005),
        ])
        plan_strict = adapter_strict.preview_batch(action_strict)
        target_strict = deepcopy(adapter_strict._batch_plans[plan_strict['plan_id']]['stages']['precise_insert']['target'])
        orig_strict_state = adapter_strict.state

        def strict_pos_tol_state():
            st = orig_strict_state()
            if adapter_strict.env.take_action_cnt[0] >= 1:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_strict['position_m'][0] + 0.001,
                        target_strict['position_m'][1] - 0.001,
                        target_strict['position_m'][2] + 0.016,
                    ],
                    'quaternion_xyzw': list(target_strict['quaternion_xyzw']),
                }
            return st

        adapter_strict.state = strict_pos_tol_state
        res_strict = self.execute_batch(adapter_strict, action_strict, plan_strict)
        self.assertEqual(res_strict['status'], 'failed', res_strict)

        # 7. [P1-1] 显式设置 position_tolerance_m=0.003 时，5mm 末端偏差严禁被隐藏 5.5mm 阈值放行
        adapter_3mm, _, _ = preview_adapter()
        jaws(adapter_3mm)
        action_3mm = batch([
            move('s1', arm='left', angle=-0.25, position_tolerance_m=0.003, max_upward_dz_m=0.0),
            move('s2', arm='left', angle=-0.50),
        ])
        plan_3mm = adapter_3mm.preview_batch(action_3mm)
        s1_rec = adapter_3mm._batch_plans[plan_3mm['plan_id']]['stages']['s1']
        target_3mm = deepcopy(s1_rec['target'])
        shifted_pos = [
            target_3mm['position_m'][0] + 0.005,
            target_3mm['position_m'][1],
            target_3mm['position_m'][2],
        ]
        s1_rec['planned_positions_m'] = [shifted_pos]
        orig_3mm_state = adapter_3mm.state

        def state_5mm_err():
            st = orig_3mm_state()
            if adapter_3mm.env.take_action_cnt[0] >= 1:
                st['arms']['left']['ee_pose'] = {
                    'position_m': list(shifted_pos),
                    'quaternion_xyzw': list(target_3mm['quaternion_xyzw']),
                }
            return st

        adapter_3mm.state = state_5mm_err
        res_3mm = self.execute_batch(adapter_3mm, action_3mm, plan_3mm)
        self.assertEqual(res_3mm['status'], 'failed', res_3mm)
        self.assertEqual(res_3mm['failed_stage_id'], 's1')


if __name__ == '__main__':
    unittest.main()
