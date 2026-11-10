"""测试夹爪本体事实/条件门禁、重采样统一与关节行程准入参数化。"""

import math
import threading
import unittest
import numpy as np

from auriga.robot.motion.grip_guard import (
    DEFAULT_MAX_APERTURE_M,
    GRIP_GUARD_PRESETS,
    compute_grip_facts,
    evaluate_grip_guard,
    resolve_grip_guard,
)
from auriga.robot.motion.motion_safety import JOINT_TRAVEL_PRESETS, assess_motion
from test_batch_motion import batch, jaw, jaws, move
from test_motion_preview_backend import preview_adapter


class GripGuardAndTrajectoryTests(unittest.TestCase):
    def execute(self, adapter, action, plan):
        return adapter.execute(
            dict(action, batch_plan_id=plan["plan_id"]),
            lambda update: None,
            threading.Event(),
        )

    def test_compute_grip_facts_returns_objective_measurements_without_empty_grasp_guess(self):
        before = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        after = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]
        facts = compute_grip_facts(before, after)
        self.assertEqual(facts["measured_jaw_before"], 1.0)
        self.assertEqual(facts["measured_jaw_after"], 0.25)
        self.assertEqual(facts["jaw_delta"], -0.75)
        self.assertAlmostEqual(facts["measured_aperture_m"], 0.25 * DEFAULT_MAX_APERTURE_M, places=6)
        self.assertAlmostEqual(facts["aperture_delta_m"], -0.75 * DEFAULT_MAX_APERTURE_M, places=6)

        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 1.0
        action = batch(
            [jaw(identifier="close", goal=0.0, checkpoint_after_gripper=True), move("lift")],
        )
        plan = adapter.preview_batch(action)
        result = self.execute(adapter, action, plan)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["reason"], "observation_checkpoint")
        self.assertNotIn("likely empty grasp", result["checkpoint_reasons"]["close"])
        stage_facts = result["stage_results"]["close"]["grip_facts"]
        self.assertEqual(stage_facts["measured_jaw_before"], 1.0)
        self.assertEqual(stage_facts["measured_jaw_after"], 0.0)
        self.assertEqual(stage_facts["measured_aperture_m"], 0.0)

    def test_grip_guard_triggers_below_min_above_max_and_passes_within_bounds(self):
        # 1) 低于下界触发 aperture_below_min_m
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 1.0
        action_below = batch([
            jaw(identifier="close", goal=0.0, grip_guard_preset="rigid_object"),
            move("lift"),
        ])
        plan_below = adapter.preview_batch(action_below)
        res_below = self.execute(adapter, action_below, plan_below)
        self.assertEqual(res_below["status"], "stopped")
        self.assertTrue(res_below["paused_at_checkpoint"])
        self.assertEqual(res_below["checkpoint_reason"], "aperture_below_min_m")
        self.assertEqual(res_below["stage_results"]["close"]["status"], "paused_at_checkpoint")
        self.assertEqual(res_below["stage_results"]["close"]["checkpoint_reason"], "aperture_below_min_m")
        self.assertEqual(res_below["stage_results"]["lift"]["completed_steps"], 0)

        # 2) 显式指定 max_aperture_m 时，高于上界触发 aperture_above_max_m
        adapter2, _, _ = preview_adapter()
        jaws(adapter2)
        adapter2.last_dispatched_goals["left"] = 1.0
        action_above = batch([
            jaw(identifier="close", goal=0.8, grip_guard_preset="thin_object", max_aperture_m=0.025),
            move("lift"),
        ])
        plan_above = adapter2.preview_batch(action_above)
        res_above = self.execute(adapter2, action_above, plan_above)
        self.assertEqual(res_above["status"], "stopped")
        self.assertTrue(res_above["paused_at_checkpoint"])
        self.assertEqual(res_above["checkpoint_reason"], "aperture_above_max_m")
        self.assertEqual(res_above["stage_results"]["close"]["status"], "paused_at_checkpoint")
        self.assertEqual(res_above["stage_results"]["close"]["checkpoint_reason"], "aperture_above_max_m")

        # 3) 处于 [min_aperture_m, max_aperture_m] 区间内连续执行完成
        adapter3, _, _ = preview_adapter()
        jaws(adapter3)
        adapter3.last_dispatched_goals["left"] = 1.0
        action_ok = batch([
            jaw(identifier="close", goal=0.25, min_aperture_m=0.010, max_aperture_m=0.035),
            move("lift"),
        ])
        plan_ok = adapter3.preview_batch(action_ok)
        res_ok = self.execute(adapter3, action_ok, plan_ok)
        self.assertEqual(res_ok["status"], "completed")
        self.assertAlmostEqual(
            res_ok["stage_results"]["close"]["grip_facts"]["measured_aperture_m"],
            0.25 * DEFAULT_MAX_APERTURE_M,
            places=6,
        )

    def test_explicit_checkpoint_after_gripper_does_not_skip_opening(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 0.0
        action = batch([
            jaw(identifier="open", goal=1.0, checkpoint_after_gripper=True),
            move("retract"),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(len(plan["checkpoints"]), 1)
        self.assertEqual(plan["checkpoints"][0]["stage_ids"], ["open"])
        res = self.execute(adapter, action, plan)
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["checkpoint_stage_ids"], ["open"])

    def test_resample_action_plan_scale_one_preserves_time_domain_scurve_deceleration(self):
        adapter, _, _ = preview_adapter()
        # 构造带 S-curve 末端减速的原规划轨迹：在直线关节路径上，末端步长小于中间巡航步长
        t = np.linspace(0.0, 1.0, 11)
        s = 3.0 * t**2 - 2.0 * t**3
        scurve = np.zeros((11, 6))
        scurve[:, 0] = 0.40 * s
        rows = adapter._resample_action_plan(scurve, "left", {"speed_scale": 1.0}, start_q=np.zeros(6))
        np.testing.assert_allclose(rows[-1], scurve[-1])
        deltas = np.abs(np.diff(np.vstack([np.zeros(6), rows])[:, 0]))
        # 保留原规划时间域 S-curve 减速特征：最后一步增量显著小于峰值巡航增量
        self.assertLess(deltas[-1], deltas.max() * 0.6)

    def test_joint_travel_presets_and_numeric_overrides(self):
        names = [f"joint{i}" for i in range(1, 7)]
        types = ["revolute"] * 6
        limits = [[-6.28, 6.28]] * 6
        start = np.zeros(6)
        # 构造总行程 4.0 rad (> pi, <= 1.5*pi)，回撤 2.0 rad (> pi/2, <= pi) 的轨迹
        positions = np.array([
            [2.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        # 默认 standard 下被拒绝
        std = assess_motion(start, positions, joint_names=names, joint_types=types, joint_limits=limits)
        self.assertEqual(std["status"], "reject")
        self.assertIn("abnormal_joint_travel", std["reasons"])

        # extended 预设下通过
        ext = assess_motion(
            start, positions, joint_names=names, joint_types=types, joint_limits=limits,
            joint_travel_preset="extended",
        )
        self.assertEqual(ext["status"], "pass")
        self.assertEqual(ext["thresholds"]["min_total_travel_rad"], JOINT_TRAVEL_PRESETS["extended"]["min_total_travel_rad"])

        # 显式数值覆盖通过
        custom = assess_motion(
            start, positions, joint_names=names, joint_types=types, joint_limits=limits,
            max_joint_travel_rad=4.5, max_backtracking_rad=2.5,
        )
        self.assertEqual(custom["status"], "pass")
        self.assertEqual(custom["thresholds"]["min_total_travel_rad"], 4.5)
        self.assertEqual(custom["thresholds"]["min_periodic_or_backtracking_excess_rad"], 2.5)

        # unconstrained 跳过行程检查但仍保留步数预算检查
        huge = np.array([
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        unc = assess_motion(
            start, huge, joint_names=names, joint_types=types, joint_limits=limits,
            joint_travel_preset="unconstrained", native_steps=10, request_remaining=5,
        )
        self.assertEqual(unc["status"], "reject")
        self.assertEqual(unc["reasons"], ["insufficient_remaining_native_budget"])
        self.assertEqual(unc["witnesses"], [])

    def test_unified_actions_compile_request_propagates_grip_guard_and_joint_travel(self):
        from types import SimpleNamespace
        from auriga.runtime.actions.unified_actions import compile_request
        harness = SimpleNamespace(
            identity={
                "run_id": "r1",
                "episode_id": "e1",
                "env_instance_id": "i1",
                "env_generation": 1,
            },
            capabilities={
                "frame_id": "env_origin",
                "tcp_config_id": "arx-x5-link6-v1",
                "limits_profile": "simulation-slow-v1",
                "motion_execution_mode": "resampled",
                "dual_arm": True,
            },
            config=SimpleNamespace(features=SimpleNamespace(default_motion_speed_scale=1.0)),
            lock=threading.Lock(),
            store=SimpleNamespace(record_sequence=lambda kind, prefix: 0),
        )
        compiled = compile_request(
            harness,
            {
                "observation_id": "obs-1",
                "grip_guard_preset": "rigid_object",
                "joint_travel_preset": "extended",
                "stages": [
                    {
                        "id": "grasp_it",
                        "kind": "move",
                        "arm": "left",
                        "grasp_position_m": [0.3, 0.1, 0.8],
                        "angles_deg": [0.0, 0.0, 0.0],
                        "gripper_goal_normalized": 0.0,
                        "min_aperture_m": 0.005,
                        "max_aperture_m": 0.060,
                        "max_joint_travel_rad": 5.0,
                        "max_backtracking_rad": 2.5,
                    }
                ],
            },
        )
        self.assertEqual(compiled["grip_guard_preset"], "rigid_object")
        self.assertEqual(compiled["joint_travel_preset"], "extended")
        self.assertEqual(len(compiled["stages"]), 2)
        motion_stage, jaw_stage = compiled["stages"]
        self.assertEqual(motion_stage["id"], "grasp_it/motion")
        self.assertEqual(motion_stage["max_joint_travel_rad"], 5.0)
        self.assertEqual(motion_stage["max_backtracking_rad"], 2.5)
        self.assertEqual(jaw_stage["id"], "grasp_it")
        self.assertEqual(jaw_stage["min_aperture_m"], 0.005)
        self.assertEqual(jaw_stage["max_aperture_m"], 0.060)

    def test_physical_jaw_ignores_negative_overdrive_gripper_scale(self):
        """回归测试：当 robot.gripper_scale=[-0.01, 0.044] 含负向过压区间时，空抓 position_m=[0.0, 0.0] 必须报告 0.0m 而非 16.3mm。"""
        adapter, _, _ = preview_adapter()
        robot = adapter.robot.robot_list[0]
        robot.gripper_joints_name = ["joint7", "joint8"]
        robot.gripper_move = dict(base="joint7", sign=1)
        robot.gripper_scale = [-0.010, 0.044]
        original = adapter.state

        def state():
            result = original()
            for arm in ("left", "right"):
                goal = adapter.last_dispatched_goals[arm]
                phys_pos = 0.044 * float(goal)
                result["arms"][arm]["gripper"] = dict(
                    position_source_kind="measurement",
                    position_m=[phys_pos, phys_pos],
                    joint_names=["joint7", "joint8"],
                )
            return result

        adapter.state = state
        adapter.last_dispatched_goals["left"] = 1.0
        action_empty = batch([
            jaw(identifier="close", goal=0.0, grip_guard_preset="rigid_object"),
            move("lift"),
        ])
        plan = adapter.preview_batch(action_empty)
        res = self.execute(adapter, action_empty, plan)
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["checkpoint_reason"], "aperture_below_min_m")
        facts = res["stage_results"]["close"]["grip_facts"]
        self.assertEqual(facts["measured_jaw_before"], 1.0)
        self.assertEqual(facts["measured_jaw_after"], 0.0)
        self.assertEqual(facts["measured_aperture_m"], 0.0)

    def test_mid_move_in_hand_slip_triggers_grip_guard_checkpoint_before_carry(self):
        """回归测试：close 阶段结束时夹住锥面 (25mm >= min_aperture_m)，但随后 lift 阶段中物体滑脱闭合至 0mm 时，必须在 lift 结束时暂停拦截后续 carry。"""
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 1.0
        action_slip = batch([
            jaw(identifier="close", goal=0.0, grip_guard_preset="rigid_object"),
            move("lift", angle=-0.125),
            move("carry", angle=-0.25),
        ])
        plan = adapter.preview_batch(action_slip)
        close_end = plan["stage_costs"]["close"]["end_native_step"]
        lift_end = plan["stage_costs"]["lift"]["end_native_step"]
        original = adapter.state

        def slip_during_lift_state():
            st = original()
            step = adapter.env.take_action_cnt[0]
            if 1 <= step <= close_end:
                st["arms"]["left"]["gripper"]["position_m"] = [0.30]
            elif step >= lift_end:
                st["arms"]["left"]["gripper"]["position_m"] = [0.0]
            return st

        adapter.state = slip_during_lift_state
        res = self.execute(adapter, action_slip, plan)
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["reason"], "observation_checkpoint")
        self.assertTrue(res["paused_at_checkpoint"])
        self.assertEqual(res["checkpoint_reason"], "aperture_below_min_m")
        self.assertEqual(res["checkpoint_stage_ids"], ["lift"])
        self.assertEqual(res["stage_results"]["close"]["status"], "completed")
        self.assertEqual(res["stage_results"]["lift"]["status"], "paused_at_checkpoint")
        self.assertEqual(res["stage_results"]["carry"]["completed_steps"], 0)

        # 2. 单次待定门禁 (pending_guard) 仅在闭爪后紧邻的第一段 move (lift) 生效，后续 carry/place 阶段不再常驻拦截薄物
        adapter2, _, _ = preview_adapter()
        jaws(adapter2)
        adapter2.last_dispatched_goals["left"] = 1.0
        action_carry_slip = batch([
            jaw(identifier="close", goal=0.0, grip_guard_preset="non_empty"),
            move("lift", angle=-0.125),
            move("carry", angle=-0.25),
            move("place", angle=-0.375),
        ])
        plan2 = adapter2.preview_batch(action_carry_slip)
        lift2_end = plan2["stage_costs"]["lift"]["end_native_step"]
        carry2_end = plan2["stage_costs"]["carry"]["end_native_step"]
        original2 = adapter2.state

        def slip_during_carry_state():
            st = original2()
            step = adapter2.env.take_action_cnt[0]
            if 1 <= step <= lift2_end:
                st["arms"]["left"]["gripper"]["position_m"] = [0.059]
            elif step >= carry2_end:
                st["arms"]["left"]["gripper"]["position_m"] = [0.0]
            return st

        adapter2.state = slip_during_carry_state
        res2 = self.execute(adapter2, action_carry_slip, plan2)
        self.assertEqual(res2["status"], "completed")
        self.assertEqual(res2["completed_stage_ids"], ["close", "lift", "carry", "place"])

        # 3. [P2-1] 单阶段/末阶段显式设置 grip_guard 时不得因无后续 stage 而被静默关闭
        adapter3, _, _ = preview_adapter()
        jaws(adapter3)
        adapter3.last_dispatched_goals["left"] = 1.0
        action_single_close = batch([
            jaw(identifier="close_only", goal=0.0, grip_guard_preset="rigid_object", min_aperture_m=0.005),
        ])
        plan3 = adapter3.preview_batch(action_single_close)
        res3 = self.execute(adapter3, action_single_close, plan3)
        self.assertEqual(res3["status"], "stopped")
        self.assertTrue(res3["stage_results"]["close_only"]["grip_guard"]["triggered"])
        self.assertEqual(res3["stage_results"]["close_only"]["grip_guard"]["reason"], "aperture_below_min_m")


if __name__ == "__main__":
    unittest.main()
