"""基于 SAM + VGGT 与机械臂物理量的虚拟空间数值模拟场 (VirtualSpatialField) 单元测试。"""
from types import SimpleNamespace
import unittest

import numpy as np

from auriga.perception.external import object_table, scene3d_helpers as s3
from auriga.perception.external.virtual_field import (
    OBB3D,
    VirtualSpatialField,
    gripper_frame,
    obb_signed_distance,
)
from auriga.runtime.actions.grasp_preview_3d import audit_batch_grasp_3d

COLUMNS = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"]


class VirtualSpatialFieldTest(unittest.TestCase):
    def setUp(self):
        self.base_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"],
            ],
        )

    def test_obb_signed_distance_and_gripper_frame_alignment(self):
        """验证 3D OBB 有符号距离（分离正距离、穿透负深度）与夹爪三维标架约定。"""
        axes = np.eye(3)
        box_a = OBB3D([0.0, 0.0, 0.0], axes, [0.02, 0.02, 0.02], label="a")
        box_b_sep = OBB3D([0.06, 0.0, 0.0], axes, [0.02, 0.02, 0.02], label="b")
        box_c_overlap = OBB3D([0.03, 0.0, 0.0], axes, [0.02, 0.02, 0.02], label="c")

        self.assertAlmostEqual(obb_signed_distance(box_a, box_b_sep), 0.02, places=3)
        self.assertAlmostEqual(obb_signed_distance(box_a, box_c_overlap), -0.01, places=3)

        # [0, 0, 0] 竖直向下时，接近轴为 [0, 0, -1]，张合轴为 [1, 0, 0]，指尖低于 pad_center 7.5mm
        gf = gripper_frame([0.10, -0.20, 0.780], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(gf["approach_dir"], [0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(gf["closing_dir"], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(gf["fingertip_center_m"], [0.10, -0.20, 0.7725], atol=1e-6)

    def test_evaluate_grasp_single_and_dense_multi_object_scenarios(self):
        """验证 evaluate_grasp 对正常抓取、高度过高咬合不足、偏心空抓/撞击目标、以及张开双指撞邻近物体的检测。"""
        vf = VirtualSpatialField(objects=self.base_table)

        # 1) 正常横跨长轴对准抓取 (yaw=-60 deg 垂直于 30 deg 主轴，z=0.780)
        ok_eval = vf.evaluate_grasp("right", [0.10, -0.20, 0.780], [0.0, 0.0, -60.0], target_id="o1")
        self.assertTrue(ok_eval["theoretically_graspable"], ok_eval)
        self.assertEqual(ok_eval["failure_reasons"], [])
        self.assertFalse(ok_eval["finger_hits_target"])
        self.assertAlmostEqual(ok_eval["projected_span_m"], 0.026, places=3)
        self.assertGreaterEqual(ok_eval["vertical_bite_m"], 0.014)
        self.assertLessEqual(ok_eval["lateral_eccentricity_m"], 0.002)

        # 2) 悬空过高导致 15mm 夹持面无有效竖直咬合 (z=0.815 > top_z=0.795 + 0.0075)
        high_eval = vf.evaluate_grasp("right", [0.10, -0.20, 0.815], [0.0, 0.0, -60.0], target_id="o1")
        self.assertFalse(high_eval["theoretically_graspable"])
        self.assertIn("insufficient_vertical_bite", high_eval["failure_reasons"])
        self.assertEqual(high_eval["vertical_bite_m"], 0.0)

        # 3) 横向偏心过大导致单侧手指下探直接撞在物体上或偏出包络
        # 沿闭合方向 (-60 deg) 偏移 38mm（超过半开度余量 (88.8 - 26)/2 = 31.4mm）
        rad = np.radians(-60.0)
        ecc_pos = [0.10 + 0.038 * float(np.cos(rad)), -0.20 + 0.038 * float(np.sin(rad)), 0.780]
        ecc_eval = vf.evaluate_grasp("right", ecc_pos, [0.0, 0.0, -60.0], target_id="o1")
        self.assertFalse(ecc_eval["theoretically_graspable"])
        self.assertTrue(
            "excessive_lateral_eccentricity" in ecc_eval["failure_reasons"]
            or "finger_hits_target_on_descent" in ecc_eval["failure_reasons"],
            ecc_eval,
        )

        # 4) 密集多物体场景：在 o1 中心右侧闭合轴方向 46mm 处放置紧邻障碍物 o2（恰好位于张开右指 44.4..52.4mm 下探包络内）
        obs_pos = [0.10 + 0.046 * float(np.cos(rad)), -0.20 + 0.046 * float(np.sin(rad)), 0.785]
        dense_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"],
                ["o2", obs_pos, 0.810, 0.765, 0.030, 0.030, 0.0, None, "table", "ok"],
            ],
        )
        vf_dense = VirtualSpatialField(objects=dense_table)
        col_eval = vf_dense.evaluate_grasp("right", [0.10, -0.20, 0.780], [0.0, 0.0, -60.0], target_id="o1")
        self.assertFalse(col_eval["theoretically_graspable"])
        self.assertIn("open_finger_collides_with_neighbor", col_eval["failure_reasons"])
        self.assertIn("o2", col_eval["colliding_neighbor_ids"])
        self.assertLess(col_eval["neighbor_clearance_m"], 0.0)

        # 5) solve_optimal_grasp 自动沿 o1 主轴移开或调整角度，给出不撞 o2 的零碰撞抓取位姿
        opt = col_eval["optimal_grasp_pose"]
        self.assertIsNotNone(opt)
        self.assertTrue(opt["theoretically_graspable"], opt)
        self.assertEqual(opt["colliding_neighbor_ids"], [])
        self.assertGreater(opt["neighbor_clearance_m"], 0.0)

    def test_solve_optimal_grasp_rotates_yaw_for_compact_object_with_side_neighbor(self):
        """当紧凑方块物体的 X 轴方向存在紧邻障碍物时，solve_optimal_grasp 自动选择 Y 轴方向无碰撞抓取偏航角。"""
        compact_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.00, -0.20, 0.780], 0.795, 0.765, 0.040, 0.038, 0.0, None, "table", "ok"],
                # o2 紧贴在 o1 的 +X 侧 (x=0.048)，若以 yaw=0 (沿 X 轴张合) 抓取，右指必撞 o2
                ["o2", [0.048, -0.20, 0.785], 0.810, 0.765, 0.035, 0.035, 0.0, None, "table", "ok"],
            ],
        )
        vf = VirtualSpatialField(objects=compact_table)
        bad = vf.evaluate_grasp("right", [0.00, -0.20, 0.780], [0.0, 0.0, 0.0], target_id="o1")
        self.assertIn("o2", bad["colliding_neighbor_ids"])

        opt = vf.solve_optimal_grasp("o1", arm="right", preferred_yaw_deg=0.0)
        self.assertTrue(opt["theoretically_graspable"], opt)
        self.assertEqual(opt["colliding_neighbor_ids"], [])
        self.assertGreater(opt["neighbor_clearance_m"], 0.01)

    def test_simulate_stages_tracks_held_object_detects_path_collision_and_updates_world(self):
        """验证 simulate_stages 多阶段动态推演：抓起绑定 -> 搬运途中碰撞高障碍物预警 -> 抬高后干净通过并更新释放坐标。"""
        scene_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                # 待抓取物 o1 位于 x=0.0, y=-0.20
                ["o1", [0.00, -0.20, 0.780], 0.795, 0.765, 0.050, 0.030, 0.0, 88.0, "table", "ok"],
                # 中途高耸障碍物 o2 位于 x=0.10, y=-0.20，顶面高达 0.880m
                ["o2", [0.10, -0.20, 0.822], 0.880, 0.765, 0.060, 0.060, 0.0, None, "table", "ok"],
                # 目标容器/垫块 o3 位于 x=0.20, y=-0.20，顶面 0.790m
                ["o3", [0.20, -0.20, 0.778], 0.790, 0.765, 0.080, 0.080, 0.0, None, "table", "ok"],
            ],
        )
        vf = VirtualSpatialField(objects=scene_table)

        # 低空搬运 (z=0.830m < o2.top_z=0.880m) 必在 transport 阶段撞击 o2
        low_stages = [
            {"id": "grasp", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, 0.780], "angles_deg": [0.0, 0.0, 90.0], "gripper": 0.0},
            {"id": "lift_low", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, 0.830], "angles_deg": [0.0, 0.0, 90.0]},
            {"id": "transport_low", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.20, 0.830], "angles_deg": [0.0, 0.0, 90.0]},
            {"id": "release", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.20, 0.810], "angles_deg": [0.0, 0.0, 90.0], "gripper": 1.0},
        ]
        sim_low = vf.simulate_stages(low_stages)
        self.assertEqual(sim_low["status"], "risk_detected")
        self.assertFalse(sim_low["theoretically_valid"])
        transport_rep = sim_low["stage_reports"][2]
        self.assertEqual(transport_rep["status"], "path_collision")
        self.assertIn("o2", transport_rep["colliding_objects"])
        self.assertGreater(transport_rep["recommended_safe_z_m"], 0.880)

        # 采用 plan_pick_and_place 自动生成的安全高度阶段序列：无碰撞通过，释放后 o1 落位到 [0.20, -0.20] 且支撑在 o3 上
        plan = vf.plan_pick_and_place("o1", [0.20, -0.20, 0.808], arm="right")
        self.assertEqual(plan["simulation"]["status"], "clean", plan["simulation"])
        self.assertTrue(plan["simulation"]["theoretically_valid"])
        self.assertGreater(plan["transit_z_m"], 0.880)

        world_after = {item["id"]: item for item in plan["simulation"]["virtual_world_after"]}
        self.assertAlmostEqual(world_after["o1"]["centre_m"][0], 0.20, places=3)
        self.assertAlmostEqual(world_after["o1"]["centre_m"][1], -0.20, places=3)
        self.assertEqual(world_after["o1"]["support"], "o3")
        self.assertTrue(world_after["o1"]["virtual_moved"])
        self.assertIsNone(world_after["o1"]["held_by_arm"])

    def test_update_from_observation_realtime_iteration_with_arm_motion_and_occlusion(self):
        """验证随真实机械臂移动产生新观测时，虚拟场实时更新持物位姿并保留单帧被遮挡物体。"""
        vf = VirtualSpatialField(objects=self.base_table)
        # 模拟右臂在 [0.10, -0.20, 0.780] 抓取了 o1
        gf0 = gripper_frame([0.10, -0.20, 0.780], [0.0, 0.0, -60.0])
        vf.objects["o1"].attach_to_gripper("right", gf0)
        vf.arm_apertures_m["right"] = 0.025

        # 下一帧新观测：右臂移动到 [0.18, -0.12, 0.900] 且旋转到 yaw=30 deg（夹爪仍闭合 aperture=0.025m），
        # 同时桌面上原本在 [-0.15, -0.22] 的物体 o2 在下一帧被遮挡消失
        vf.objects["o2"] = vf.find_target_object(
            dict(id="o2", centre_m=[-0.15, -0.22, 0.780], top_z=0.795, bottom_z=0.765, length=0.04, width=0.04, yaw_deg=0.0)
        )
        next_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o3", [0.22, -0.25, 0.780], 0.795, 0.765, 0.05, 0.05, 0.0, None, "table", "ok"],
            ],
        )
        next_robot_state = {
            "arms": {
                "right": {
                    "grasp_pose": {"position_m": [0.18, -0.12, 0.900], "angles_deg": [0.0, 0.0, 30.0]},
                    "gripper": {"aperture_m": 0.025},
                }
            }
        }
        vf.update_from_observation(next_table, robot_state=next_robot_state)

        # 1) 持物 o1 随右臂实测位姿实时更新到 [0.18, -0.12, 0.900]
        self.assertIn("o1", vf.objects)
        self.assertEqual(vf.objects["o1"].held_by_arm, "right")
        np.testing.assert_allclose(vf.objects["o1"].centre_m, [0.18, -0.12, 0.900], atol=1e-3)
        # 2) 被遮挡的桌面物体 o2 被保留一帧且标记 occluded=True
        self.assertIn("o2", vf.objects)
        self.assertTrue(vf.objects["o2"].occluded)
        # 3) 新观测中的 o3 正常加入
        self.assertIn("o3", vf.objects)

    def test_integration_with_grasp_hints_grasp_preview_3d_and_scene3d_helpers(self):
        """验证虚拟场与 object_table.grasp_hints_from、grasp_preview_3d、scene3d_helpers 的端到端集成。"""
        # 1) object_table.grasp_hints_from 包含 safe_grasp_pose
        raw_objs = [
            dict(centre_m=[0.10, -0.20, 0.780], top_z_m=0.795, bottom_z_m=0.765, length_m=0.140, width_m=0.026, yaw_deg=30.0, axis_tilt_deg=88.0, elongation=2.5),
        ]
        hints = object_table.grasp_hints_from(raw_objs, support_z_m=0.765)
        self.assertIn("safe_grasp_pose", hints["o1"])
        self.assertEqual(hints["o1"]["safe_grasp_pose"]["across_yaw_deg"], -60.0)
        self.assertIn("safe_transit_z_m", hints["o1"]["safe_grasp_pose"])

        # 2) grasp_preview_3d 在存在邻近障碍物碰撞与低空搬运碰撞时输出 virtual_field_simulation 诊断与告警
        rad = np.radians(-60.0)
        obs_pos = [0.10 + 0.046 * float(np.cos(rad)), -0.20 + 0.046 * float(np.sin(rad)), 0.785]
        crowded_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"],
                ["o2", obs_pos, 0.815, 0.765, 0.032, 0.032, 0.0, None, "table", "ok"],
            ],
        )
        harness = SimpleNamespace(_obs3d_memo=("obs-crowd", {"object_table": (crowded_table, None, {})}))
        action = {
            "based_on_observation_id": "obs-crowd",
            "stages": [
                {
                    "id": "grasp_crowded",
                    "kind": "move",
                    "arm": "right",
                    "action": {
                        "input_grasp_pose": {
                            "position_m": [0.10, -0.20, 0.780],
                            "angles_deg": [0.0, 0.0, -60.0],
                        },
                        "gripper_events": [{"goal_normalized": 0.0}],
                    },
                }
            ],
        }
        review, _ = audit_batch_grasp_3d(harness, action, "obs-crowd")
        self.assertEqual(review["status"], "warnings_detected")
        self.assertIn("warn_finger_collides_neighbor", review["risk_flags"])
        self.assertIn("virtual_field_simulation", review)
        stage_audit = review["stages"][0]
        self.assertEqual(stage_audit["finger_collision_check"], "warn_finger_collides_neighbor")
        self.assertIn("o2", stage_audit["colliding_neighbor_ids"])
        self.assertTrue(stage_audit["optimal_grasp_pose"]["theoretically_graspable"])

        # 3) scene3d_helpers 沙箱便捷函数直接调用
        vf_s3 = s3.virtual_field(crowded_table)
        self.assertIsInstance(vf_s3, VirtualSpatialField)
        opt_s3 = s3.solve_optimal_grasp("o1", arm="right")
        self.assertTrue(opt_s3["theoretically_graspable"])
        ev_s3 = s3.evaluate_grasp("right", opt_s3["position_m"], opt_s3["angles_deg"], target_id="o1")
        self.assertTrue(ev_s3["theoretically_graspable"])
        plan_s3 = s3.plan_pick_and_place("o1", [-0.12, -0.20, 0.790], arm="right")
        self.assertEqual(len(plan_s3["stages"]), 6)
        self.assertTrue(plan_s3["simulation"]["theoretically_valid"])

    def test_adversarial_counterexamples_from_codex_review(self):
        """针对对抗审查发现的全部 9 类几何、状态与契约反例的回归测试。"""
        from auriga.perception.external.virtual_field import (
            angles_deg_to_quat_wxyz,
            flange_pose_to_grasp_pose,
            quat_wxyz_to_matrix,
        )
        from auriga.runtime.actions.robot_action_schema import _STAGE_SCHEMAS

        # 反例 1: 86x86mm 不可抓物体不得在 grasp_hints_from 中输出 safe_grasp_pose，且 solve_optimal_grasp 保留 failure_reasons
        wide_objs = [
            dict(centre_m=[0.0, 0.0, 0.780], top_z_m=0.795, bottom_z_m=0.765, length_m=0.086, width_m=0.086, yaw_deg=0.0),
        ]
        wide_hints = object_table.grasp_hints_from(wide_objs, support_z_m=0.765)
        self.assertNotIn("safe_grasp_pose", wide_hints["o1"])
        vf_wide = VirtualSpatialField(objects=wide_objs, table_z=0.765)
        opt_wide = vf_wide.solve_optimal_grasp("o1")
        self.assertFalse(opt_wide["theoretically_graspable"])
        self.assertTrue(len(opt_wide["failure_reasons"]) > 0)

        # 反例 2: plan_pick_and_place 生成的 stages 必须使用 gripper_goal_normalized 并通过正式 schema 校验器，
        # 且直接将含 gripper_goal_normalized 的公开阶段传入 simulate_stages 能正确抓起并移动物体
        from auriga.runtime.actions.robot_action_schema import _validate_stage
        vf = VirtualSpatialField(objects=self.base_table)
        plan = vf.plan_pick_and_place("o1", [0.18, -0.20, 0.780], arm="right", approach_clearance_m=0.01)
        self.assertGreaterEqual(plan["stages"][3]["grasp_position_m"][2], plan["stages"][4]["grasp_position_m"][2] + 0.01 - 1e-6)
        self.assertAlmostEqual(plan["stages"][5]["grasp_position_m"][2], plan["stages"][3]["grasp_position_m"][2], places=5)
        allowed_move_keys = set(_STAGE_SCHEMAS["move"]["properties"].keys())
        for st in plan["stages"]:
            self.assertNotIn("gripper", st)
            self.assertTrue(set(st.keys()).issubset(allowed_move_keys), st.keys())
            _validate_stage(st)
        self.assertEqual(plan["stages"][0]["gripper_goal_normalized"], 1.0)
        self.assertEqual(plan["stages"][1]["gripper_goal_normalized"], 0.0)
        self.assertEqual(plan["stages"][4]["gripper_goal_normalized"], 1.0)
        world_after = {o["id"]: o for o in plan["simulation"]["virtual_world_after"]}
        self.assertTrue(world_after["o1"]["virtual_moved"])
        self.assertAlmostEqual(world_after["o1"]["centre_m"][0], 0.18, places=3)

        # 反例 3: 140x26x30mm 刚体在 [0, 45, 0] 倾斜姿态下连续 4 次 update_from_gripper 尺寸严格幂等，绝不膨胀；
        # 且旋转立起释放后保留 3D 姿态
        obj = vf.objects["o1"]
        gf_init = gripper_frame([0.10, -0.20, 0.780], [0.0, 0.0, -60.0])
        obj.attach_to_gripper("right", gf_init)
        gf_tilt = gripper_frame([0.10, -0.20, 0.880], [0.0, 45.0, -60.0])
        heights = []
        for _ in range(4):
            obj.update_from_gripper(gf_tilt)
            heights.append(round(obj.top_z - obj.bottom_z, 6))
        self.assertEqual(heights[0], heights[1])
        self.assertEqual(heights[1], heights[2])
        self.assertEqual(heights[2], heights[3])
        self.assertAlmostEqual(obj.intrinsic_height_m, 0.030, places=6)
        obj.detach_from_gripper(support_z=0.765, support_id="table")
        self.assertAlmostEqual(obj.top_z - obj.bottom_z, heights[0], places=5)
        self.assertAlmostEqual(obj.bottom_z, 0.765, places=6)

        # 反例 4: 30x30x200mm 竖直柱体以 [0, 90, 0] 侧向抓取（闭合轴沿竖直 200mm 方向）被判定为不可抓且手指穿模，
        # 手掌穿入目标本体也被检出
        tall_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.0, 0.0, 0.865], 0.965, 0.765, 0.030, 0.030, 0.0, 2.0, "table", "ok"],
            ],
        )
        vf_tall = VirtualSpatialField(objects=tall_table)
        side_eval = vf_tall.evaluate_grasp("right", [0.0, 0.0, 0.865], [0.0, 90.0, 0.0], target_id="o1")
        self.assertFalse(side_eval["theoretically_graspable"], side_eval)
        self.assertTrue(side_eval["finger_hits_target"])
        self.assertIn("projected_span_exceeds_opening", side_eval["failure_reasons"])
        self.assertIn("finger_hits_target_on_descent", side_eval["failure_reasons"])

        # 下探过深使手掌穿入高柱体顶部 (z=0.865 时手掌中心在 0.865 + 0.0975 = 0.9625，若压低到 z=0.830 则手掌穿入 0.965 以下)
        palm_eval = vf_tall.evaluate_grasp("right", [0.0, 0.0, 0.830], [0.0, 0.0, 0.0], target_id="o1")
        self.assertFalse(palm_eval["theoretically_graspable"])
        self.assertTrue(palm_eval["palm_hits_target"])
        self.assertIn("palm_collides_with_target", palm_eval["failure_reasons"])

        # 反例 5: 原地旋转 0 -> 90 deg 途中 22.5 deg 存在障碍物时被检出 path_collision；下探至桌面以下 z=0.700m 被检出 table 碰撞
        rad22 = math.radians(22.5) if "math" in globals() else np.radians(22.5)
        obs_rot_pos = [0.046 * float(np.cos(rad22)), 0.046 * float(np.sin(rad22)), 0.800]
        rot_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o_obs", obs_rot_pos, 0.820, 0.765, 0.020, 0.020, 0.0, None, "table", "ok"],
            ],
        )
        vf_rot = VirtualSpatialField(
            objects=rot_table,
            robot_state={"arms": {"right": {"grasp_pose": {"position_m": [0.0, 0.0, 0.790], "angles_deg": [0.0, 0.0, 0.0]}}}},
        )
        sim_rot = vf_rot.simulate_stages([
            {"id": "spin", "arm": "right", "kind": "move", "grasp_position_m": [0.0, 0.0, 0.790], "angles_deg": [0.0, 0.0, 90.0]},
        ])
        self.assertEqual(sim_rot["status"], "risk_detected")
        self.assertIn("o_obs", sim_rot["stage_reports"][0]["colliding_objects"])

        sim_table_hit = vf_rot.simulate_stages([
            {"id": "dive", "arm": "right", "kind": "move", "grasp_position_m": [0.0, -0.20, 0.700], "angles_deg": [0.0, 0.0, 0.0]},
        ])
        self.assertEqual(sim_table_hit["status"], "risk_detected")
        self.assertIn("table", sim_table_hit["stage_reports"][0]["colliding_objects"])

        # 反例 6: 法兰位姿 position_m + quaternion_xyzw 与自定义 contact_depth_m 精确解析
        q_wxyz = angles_deg_to_quat_wxyz(0.0, 0.0, -60.0)
        R_mat = quat_wxyz_to_matrix(q_wxyz)
        flange_pos = np.array([0.10, -0.20, 0.780]) - (0.1501 - 0.05) * R_mat[:, 0]
        quat_xyzw = [float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])]
        spec = VirtualSpatialField._extract_stage_spec(
            {
                "id": "flange_move",
                "arm": "right",
                "kind": "move",
                "position_m": flange_pos.tolist(),
                "quaternion_xyzw": quat_xyzw,
                "contact_depth_m": 0.05,
            },
            {},
            {},
        )
        np.testing.assert_allclose(spec["position_m"], [0.10, -0.20, 0.780], atol=1e-5)
        np.testing.assert_allclose(spec["angles_deg"], [0.0, 0.0, -60.0], atol=1e-4)
        self.assertAlmostEqual(spec["contact_depth_m"], 0.05, places=5)

        # 反例 7: 真实 RoboDojo gripper: {"position_m": [0.005, 0.005]} 被正确解析为 0.010m 开度
        vf_robo = VirtualSpatialField(
            objects=self.base_table,
            robot_state={
                "arms": {
                    "right": {
                        "grasp_pose": {"position_m": [0.10, -0.20, 0.780], "angles_deg": [0.0, 0.0, -60.0]},
                        "gripper": {"position_m": [0.005, 0.005], "goal_normalized": 0.0},
                    }
                }
            },
        )
        self.assertAlmostEqual(vf_robo.arm_apertures_m["right"], 0.010, places=5)

        # 反例 8: 斜交分离 OBB 的 obb_signed_distance 与网格采样精确欧氏距离一致
        r1 = quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(25.0, -35.0, 40.0))
        r2 = quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(-50.0, 20.0, -70.0))
        b1 = OBB3D([0.0, 0.0, 0.0], r1, [0.04, 0.015, 0.025])
        b2 = OBB3D([0.055, 0.032, 0.018], r2, [0.035, 0.012, 0.020])
        d_obb = obb_signed_distance(b1, b2)
        # 在 b1 表面密集采样并 clamp 到 b2 验证欧氏距离
        grid = np.linspace(-1.0, 1.0, 25)
        u, v, w = np.meshgrid(grid, grid, grid, indexing="ij")
        pts1 = b1.center[None, :] + (np.column_stack([u.ravel(), v.ravel(), w.ravel()]) * b1.half_extents[None, :]) @ b1.axes.T
        clamped2 = b2.clamp_points(pts1)
        clamped1 = b1.clamp_points(clamped2)
        d_ref = float(np.min(np.linalg.norm(clamped1 - clamped2, axis=1)))
        self.assertAlmostEqual(d_obb, d_ref, delta=5e-4)

        # 反例 9: 显式传入不存在的 target_id="o99" 时绝不静默回退到 XY 最近邻物体
        self.assertIsNone(vf.find_target_object("o99", position_m=[0.10, -0.20, 0.780]))
        ev_missing = vf.evaluate_grasp("right", [0.10, -0.20, 0.780], [0.0, 0.0, -60.0], target_id="o99")
        self.assertFalse(ev_missing["theoretically_graspable"])
        self.assertIn("no_target_object_in_reach", ev_missing["failure_reasons"])

    def test_adversarial_round2_counterexamples(self):
        """针对第二轮对抗审查发现的全部 P1-1~P1-3, P2-1~P2-11, P3-1 反例的回归测试。"""
        import base64
        import json
        from types import SimpleNamespace
        from auriga.perception.external.virtual_field import VirtualObject
        from auriga.robot.motion.batch_motion import _check_pose_arrival
        from auriga.runtime.actions.action_state_ledger import (
            attribute_contact_block_3d,
            compute_object_table_delta,
            record_execution_outcome,
        )
        from auriga.runtime.actions.unified_actions import _preview_downgrade_reasons

        # [P1-1] _check_pose_arrival 严格执行显式 max_upward_dz_m=0.002，不被 position_tolerance_m=0.010 绕过
        target_p = {"position_m": [0.0, 0.0, 0.800], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
        meas_p = {"position_m": [0.0, 0.0, 0.808], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
        ok_dz, _, _, _, _ = _check_pose_arrival(
            meas_p, target_p, 0.010, 0.15, allow_vertical_compliance=True, max_upward_dz_m=0.002
        )
        self.assertFalse(ok_dz)

        # [P1-2] 200mm 高物体侧向水平夹爪 (angles=[0,90,0]) 不可抓时，grasp_preview_3d 与 _preview_downgrade_reasons 必须拦截
        tall_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[["o1", [0.0, 0.0, 0.865], 0.965, 0.765, 0.030, 0.030, 0.0, 0.0, "table", "ok"]],
        )
        fake_harness = SimpleNamespace(
            _obs3d_memo=(
                "obs-tall",
                {"object_table": (tall_table, None, {"status": "conditional"})},
            ),
            latest={
                "observation_id": "obs-tall",
                "robot_state": {
                    "arms": {
                        "right": {
                            "grasp_pose": {"position_m": [0.0, 0.0, 0.865], "angles_deg": [0.0, 90.0, 0.0]},
                            "gripper": {"position_m": [0.0444, 0.0444]},
                        }
                    }
                },
            },
        )
        review_tall, _ = audit_batch_grasp_3d(
            fake_harness,
            {
                "based_on_observation_id": "obs-tall",
                "stages": [
                    {
                        "id": "grasp_tall",
                        "arm": "right",
                        "kind": "move",
                        "grasp_position_m": [0.0, 0.0, 0.865],
                        "angles_deg": [0.0, 90.0, 0.0],
                        "gripper_goal_normalized": 0.0,
                    }
                ],
            },
            "obs-tall",
        )
        self.assertEqual(review_tall["status"], "warnings_detected")
        self.assertTrue(len(review_tall["risk_flags"]) > 0)
        downgrade = _preview_downgrade_reasons(
            {"status": "planned", "grasp_3d_review": review_tall}
        )
        self.assertIn("grasp_3d_review_warnings", downgrade)

        # [P1-3 & P2-11] centre_m 含 None 时 attribute_contact_block_3d 与 compute_object_table_delta 不崩溃；
        # record_execution_outcome 对缺失 executed_native_steps 保留 None 与 execution_uncertain=True
        bad_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[["o_bad", [None, 0.0, 0.8], 0.82, 0.765, 0.04, 0.04, 0.0, None, "table", "ok"]],
        )
        fake_h2 = SimpleNamespace(
            identity={},
            _obs3d_memo=("obs-bad", {"object_table": (bad_table, None, {})}),
            latest={"observation_id": "obs-bad"},
        )
        self.assertIsNone(
            attribute_contact_block_3d(
                fake_h2,
                {"based_on_observation_id": "obs-bad", "stages": []},
                {"status": "failed", "contact_block_summary": {"measured_position_m": [0.0, 0.0, 0.8]}},
            )
        )
        record_execution_outcome(
            fake_h2,
            "plan-u",
            {"stages": []},
            {"status": "unknown", "execution_uncertain": True},
        )
        last_out = fake_h2._action_state_ledger["recent_outcomes"][-1]
        self.assertIsNone(last_out["executed_native_steps"])
        self.assertTrue(last_out["execution_uncertain"])

        # [P2-2 & P2-3] 可见点质心与 OBB 中心 Z 不重合时，旋转 90° 后 OBB 中心横向位移精确为 -20mm，且导出再导入保留 3D 姿态与固有尺寸
        vo = VirtualObject("x", [0.0, 0.0, 0.820], 0.835, 0.765, 0.10, 0.02, 0.0)
        f0 = gripper_frame([0.0, 0.0, 0.820], [0.0, 0.0, 0.0])
        f1 = gripper_frame([0.0, 0.0, 0.820], [0.0, 90.0, 0.0])
        vo.attach_to_gripper("right", f0)
        vo.update_from_gripper(f1)
        np.testing.assert_allclose(vo.obb().center, [-0.020, 0.0, 0.820], atol=1e-6)
        restored = VirtualSpatialField([vo.to_dict()]).objects["x"]
        np.testing.assert_allclose(restored.local_half_extents, [0.050, 0.010, 0.035], atol=1e-6)
        np.testing.assert_allclose(restored.world_axes(), vo.world_axes(), atol=1e-5)
        np.testing.assert_allclose(restored.obb().center, [-0.020, 0.0, 0.820], atol=1e-4)

        # [P2-4] contact_region="finger_middle" 的 move 后紧跟省略坐标的 gripper close，正确继承规范化 pad_center
        cube_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[["o1", [0.0, 0.0, 0.800], 0.815, 0.785, 0.030, 0.030, 0.0, None, "table", "ok"]],
        )
        vf_cube = VirtualSpatialField(objects=cube_table)
        sim_fm = vf_cube.simulate_stages([
            {
                "id": "m1",
                "arm": "right",
                "kind": "move",
                "grasp_position_m": [0.0, 0.0, 0.8275],
                "angles_deg": [0.0, 0.0, 0.0],
                "contact_region": "finger_middle",
            },
            {"id": "g1", "arm": "right", "kind": "gripper", "goal_normalized": 0.0},
        ])
        self.assertEqual(sim_fm["status"], "clean", sim_fm)
        self.assertEqual(sim_fm["stage_reports"][1].get("grasped_object_id"), "o1")

        # [P2-5] goal_normalized=0.49 (43.5mm > 30mm 方块) 不会虚构抓持；1mm 开度持薄物跨帧更新不会丢失持物
        sim_half = vf_cube.simulate_stages([
            {
                "id": "m_half",
                "arm": "right",
                "kind": "move",
                "grasp_position_m": [0.0, 0.0, 0.800],
                "angles_deg": [0.0, 0.0, 0.0],
                "gripper_goal_normalized": 0.49,
            },
            {
                "id": "lift_half",
                "arm": "right",
                "kind": "move",
                "grasp_position_m": [0.0, 0.0, 0.900],
                "angles_deg": [0.0, 0.0, 0.0],
            },
        ])
        self.assertEqual(sim_half["status"], "risk_detected")
        self.assertNotIn("grasped_object_id", sim_half["stage_reports"][0])

        vf_thin = VirtualSpatialField(
            objects=[
                dict(
                    id="cloth1",
                    centre_m=[0.0, 0.0, 0.800],
                    top_z=0.802,
                    bottom_z=0.798,
                    length=0.05,
                    width=0.002,
                    yaw_deg=0.0,
                    support="held_right",
                )
            ],
            robot_state={
                "arms": {
                    "right": {
                        "grasp_pose": {"position_m": [0.0, 0.0, 0.800], "angles_deg": [0.0, 0.0, 0.0]},
                        "gripper": {"measured_aperture_m": 0.001},
                    }
                }
            },
        )
        vf_thin.update_from_observation(
            {"columns": COLUMNS, "table_z": 0.765, "rows": []},
            robot_state={
                "arms": {
                    "right": {
                        "grasp_pose": {"position_m": [0.05, 0.0, 0.850], "angles_deg": [0.0, 0.0, 0.0]},
                        "gripper": {"measured_aperture_m": 0.001},
                    }
                }
            },
        )
        self.assertIn("cloth1", vf_thin.objects)
        self.assertAlmostEqual(vf_thin.objects["cloth1"].centre_m[0], 0.05, places=4)

        # [P2-7] 不支持或不完整的 move 姿态字段明确抛 ValueError
        with self.assertRaises(ValueError):
            vf_cube.simulate_stages([
                {"id": "bad1", "arm": "right", "kind": "move", "pose_world_m": [0, 0, 0.8], "quat_wxyz": [1, 0, 0, 0]}
            ])
        with self.assertRaises(ValueError):
            vf_cube.simulate_stages([
                {"id": "bad2", "arm": "right", "kind": "move", "grasp_position_m": [0, 0, 0.8], "orientation_rpy_deg": [0, 45, 0]}
            ])

        # [P2-8] scene3d execute_cloud_queries 对 xyz=True 返回可 JSON 序列化的列表
        from auriga.perception.external.scene3d import execute_cloud_queries
        cloud_1pt = s3.Cloud(
            {
                "xyz": [[0.0, 0.0, 1.0]],
                "conf": [230],
                "uv": [[10, 10]],
                "support_z": 0.765,
                "voxel_m": 0.005,
                "image_size": [100, 100],
            }
        )
        _, q_res = execute_cloud_queries(
            SimpleNamespace(identity={}),
            cloud_1pt,
            [{"op": "points_in_box", "lo": [-1, -1, 0], "hi": [1, 1, 2], "xyz": True}],
        )
        serialized = json.dumps(q_res)
        self.assertIn("[[0.0, 0.0, 1.0]]", serialized)

        # [P2-10] 跨帧物体匹配优先保留静止零位移物体 B，将消失的前序物体 A 记入 vacated
        before_t = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.0, 0.0, 0.780], 0.795, 0.765, 0.03, 0.03, 0.0, None, "table", "ok"],
                ["o2", [0.08, 0.0, 0.780], 0.795, 0.765, 0.03, 0.03, 0.0, None, "table", "ok"],
            ],
        )
        after_t = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.08, 0.0, 0.780], 0.795, 0.765, 0.03, 0.03, 0.0, None, "table", "ok"],
            ],
        )
        delta_ab = compute_object_table_delta(before_t, after_t)
        self.assertEqual(delta_ab["moved"], [])
        self.assertEqual(delta_ab["unchanged_count"], 1)
        self.assertEqual(len(delta_ab["vacated"]), 1)
        self.assertEqual(delta_ab["vacated"][0]["before_id"], "o1")


if __name__ == "__main__":
    unittest.main()
