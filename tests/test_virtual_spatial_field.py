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


if __name__ == "__main__":
    unittest.main()
