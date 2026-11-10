"""3D 感知驱动的 robot_action(mode='preview') 抓取几何与夹爪门禁预审模块。

当外部 3D 感知（object_table / object_table_observe）启用且当前观察存在缓存的 object_table 时，
在 preview 阶段自动比对每个闭合抓取阶段的：
1. 夹爪闭合轴与细长物体主轴（yaw_deg）的夹角（< 45° 时预警并给出 across_yaw_deg 建议）；
2. 抓取高度 z 与物体 [bottom_z, top_z] 及支撑面安全间隙的匹配度（悬空过高或指尖触底预警）；
3. 声明的 grip_guard 阈值（min_aperture_m / max_aperture_m）与物体 3D 短轴宽度 width 的物理一致性。
"""
from __future__ import annotations

import math

from ...robot.motion.grip_guard import resolve_grip_guard
from ...perception.external.object_table import is_horizontally_elongated_graspable
from ...perception.external.pose_skill import (
    FINGERTIP_BELOW_PAD_M,
    MAX_OPENING_M,
    SUPPORT_CLEARANCE_M,
    attached_table,
    axis_difference,
    closure_yaws,
    nearest,
)
from ...perception.external.virtual_field import VirtualSpatialField, flange_pose_to_grasp_pose


def _resolve_robot_state(harness, obs_id):
    """从 harness 当前观测或历史观测归档中提取实测 robot_state。"""
    obs = None
    cur_fn = getattr(harness, "current_observation", None)
    if callable(cur_fn):
        try:
            cand = cur_fn()
            if isinstance(cand, dict) and (not obs_id or cand.get("observation_id") == obs_id):
                obs = cand
        except Exception:
            obs = None
    if obs is None and obs_id and getattr(harness, "scene", None) is not None:
        try:
            obs = harness.scene.observation(getattr(harness, "identity", {}), obs_id)
        except Exception:
            obs = None
    if not isinstance(obs, dict):
        return None
    if isinstance(obs.get("robot_state"), dict):
        return obs["robot_state"]
    ref = obs.get("robot_state_ref")
    if ref:
        resolve_json = getattr(getattr(harness, "scene", None), "resolve_json", None)
        if callable(resolve_json):
            try:
                res = resolve_json(ref)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
        resolve_art = getattr(getattr(harness, "store", None), "resolve_artifact", None)
        if callable(resolve_art):
            try:
                import json
                _, raw = resolve_art(ref)
                res = json.loads(raw)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
    return None


def _stage_pose_for_arm(stage, arm):
    own = (stage or {}).get("action") or {}
    gp = own.get("input_grasp_pose")
    if isinstance(gp, dict) and arm in gp:
        gp = gp[arm]
    if isinstance(gp, dict) and "position_m" in gp and "angles_deg" in gp:
        return gp
    waypoints = ((own.get("path") or {}).get("waypoints")) if isinstance(own.get("path"), dict) else None
    if isinstance(waypoints, list) and waypoints:
        wp = waypoints[-1]
        if isinstance(wp, dict) and arm in wp and isinstance(wp[arm], dict):
            wp = wp[arm]
        if isinstance(wp, dict) and "position_m" in wp and "quaternion_xyzw" in wp:
            conv = flange_pose_to_grasp_pose(wp["position_m"], wp["quaternion_xyzw"])
            return dict(position_m=conv["grasp_position_m"], angles_deg=conv["angles_deg"])
    return None


def _extract_closing_grasp_stages(action):
    """提取批次中真正发生由开到闭抓握动作的阶段（排除持物保持闭合的搬运阶段与高空预闭爪推块阶段）。"""
    stages = (action or {}).get("stages") or []
    aperture = {"left": 1.0, "right": 1.0}
    closing = []
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        arm = stage.get("arm")
        kind = stage.get("kind")
        own = stage.get("action") or {}
        events = own.get("gripper_events") or []
        if kind == "gripper":
            if arm in aperture and events:
                aperture[arm] = float(events[-1].get("goal_normalized", aperture[arm]))
            continue
        if kind != "move" or arm not in aperture:
            continue
        gp = _stage_pose_for_arm(stage, arm)
        if gp is None:
            if events:
                aperture[arm] = float(events[-1].get("goal_normalized", aperture[arm]))
            continue
        was_open = aperture[arm] >= 0.5
        own_closing = bool(events) and float(events[0].get("goal_normalized", 1.0)) < 0.5
        if events:
            aperture[arm] = float(events[-1].get("goal_normalized", aperture[arm]))
        is_closing = was_open and own_closing
        guard_stage = stage
        next_same_arm_move_z = None
        for follow in stages[idx + 1 :]:
            if not isinstance(follow, dict) or follow.get("arm") != arm:
                continue
            if follow.get("kind") == "move":
                fgp = _stage_pose_for_arm(follow, arm)
                if fgp is not None:
                    next_same_arm_move_z = float(fgp["position_m"][2])
                break
            if follow.get("kind") == "gripper" and not is_closing and was_open:
                f_events = (follow.get("action") or {}).get("gripper_events") or []
                if f_events and float(f_events[0].get("goal_normalized", 1.0)) < 0.5:
                    is_closing = True
                    guard_stage = follow
        if is_closing:
            cur_z = float(gp["position_m"][2])
            if next_same_arm_move_z is not None and (cur_z - next_same_arm_move_z) > 0.015:
                continue
            guard = resolve_grip_guard(guard_stage, action)
            closing.append(
                dict(
                    stage_id=stage["id"],
                    arm=arm,
                    position_m=[float(v) for v in gp["position_m"]],
                    angles_deg=[float(v) for v in gp["angles_deg"]],
                    contact_region=str(gp.get("contact_region") or stage.get("contact_region") or "pad_center"),
                    contact_depth_m=gp.get("contact_depth_m") if gp.get("contact_depth_m") is not None else stage.get("contact_depth_m"),
                    grip_guard=guard,
                )
            )
    return closing


def _match_object_row(table, gx, gy):
    """在 object_table 中匹配水平距离最近且可单爪抓取（排除超宽容器/支架及手中已夹持物）的 3D 物体行。"""
    columns = list(table.get("columns") or [])
    best_row, best_dist = None, math.inf
    for raw in table.get("rows") or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < len(columns):
            continue
        row = dict(zip(columns, raw))
        if str(row.get("support") or "").startswith("held_") or row.get("quality") == "held_fk_depth":
            continue
        if (
            not isinstance(row.get("length"), (int, float))
            or not isinstance(row.get("width"), (int, float))
            or not isinstance(row.get("yaw_deg"), (int, float))
        ):
            continue
        centre = row.get("centre_m")
        if not isinstance(centre, list) or len(centre) != 3:
            continue
        width = float(row.get("width") or 0.0)
        if width > MAX_OPENING_M:
            continue
        cx, cy = float(centre[0]), float(centre[1])
        length = float(row.get("length") or 0.0)
        dist = math.hypot(gx - cx, gy - cy)
        reach = max(0.5 * length + 0.035, 0.5 * width + 0.035, 0.065)
        if dist <= reach and dist < best_dist:
            best_row, best_dist = row, dist
    return best_row, best_dist


def audit_batch_grasp_3d(harness, action, observation_id=None):
    """对批次中的闭合抓取阶段与搬运轨迹执行 3D 几何与门禁审计；无缓存 object_table 时返回 (None, [])。"""
    obs_id = observation_id or (action or {}).get("based_on_observation_id")
    if not obs_id:
        return None, []
    table = attached_table(harness, obs_id)
    if not isinstance(table, dict) or not table.get("rows"):
        return None, []
    closing_stages = _extract_closing_grasp_stages(action)
    support_z = table.get("table_z")
    robot_state = _resolve_robot_state(harness, obs_id)
    vf = VirtualSpatialField(objects=table, table_z=support_z, robot_state=robot_state)
    sim_report = vf.simulate_stages((action or {}).get("stages") or [])
    stage_audits, targets_3d, warnings = [], [], []
    for item in closing_stages:
        gx, gy, gz = item["position_m"]
        pitch, roll, yaw = item["angles_deg"]
        row, xy_dist = _match_object_row(table, gx, gy)
        if row is None:
            continue
        oid = row["id"]
        centre = [float(v) for v in row["centre_m"]]
        length, width = float(row["length"]), float(row["width"])
        obj_yaw = float(row["yaw_deg"])
        top_z, bottom_z = row.get("top_z"), row.get("bottom_z")
        tilt = row.get("tilt_deg")
        ratio = length / width if width > 0 else 1.0
        elongated = is_horizontally_elongated_graspable(length, width, top_z, bottom_z, tilt)
        closing_angle = round(axis_difference(yaw, obj_yaw), 1)
        across_candidates = closure_yaws(obj_yaw)
        rec_yaw = nearest(across_candidates, yaw) if elongated else None

        axis_status = "ok"
        stage_warnings = []
        if elongated and closing_angle < 45.0:
            axis_status = "warn_along_long_axis"
            stage_warnings.append(
                f"Stage '{item['stage_id']}' jaw closing axis is {closing_angle:.0f} deg from elongated {oid} "
                f"(length/width={ratio:.2f}, yaw={obj_yaw:.0f} deg); closing across the long axis recommends "
                f"angles_deg yaw={rec_yaw:.1f} deg."
            )

        support = float(bottom_z) if isinstance(bottom_z, (int, float)) else (
            float(support_z) if isinstance(support_z, (int, float)) else None
        )
        # The recommended pad height is the object centre (whole gripping face on the object). The support facts are
        # reported beside it rather than folded into the recommendation; see object_table.grasp_hints_from.
        pad_z = centre[2]
        height_status = "ok"
        if isinstance(top_z, (int, float)) and gz > float(top_z) + 0.012:
            height_status = "warn_above_object_top"
            stage_warnings.append(
                f"Stage '{item['stage_id']}' grasp z={gz:.3f}m is {(gz - float(top_z))*1000:.0f}mm above {oid} "
                f"visible top_z={float(top_z):.3f}m (object centre pad_z={pad_z:.3f}m)."
            )
        elif support is not None and (gz - FINGERTIP_BELOW_PAD_M) < support - 0.003:
            height_status = "warn_fingertip_below_support"
            stage_warnings.append(
                f"Stage '{item['stage_id']}' fingertip z={gz - FINGERTIP_BELOW_PAD_M:.3f}m reaches below "
                f"support={support:.3f}m (object centre pad_z={pad_z:.3f}m; a contact_stop descent stops at the "
                f"support, which is normal for thin objects)."
            )

        guard = item.get("grip_guard") or {}
        guard_status = "ok"
        if width > MAX_OPENING_M:
            guard_status = "warn_wider_than_max_opening"
            stage_warnings.append(
                f"Object {oid} visible width {width*1000:.0f}mm exceeds the 88.8mm max opening; grasp a narrower rim/handle."
            )
        elif guard.get("min_aperture_m") is not None and float(guard["min_aperture_m"]) > width + 0.002:
            guard_status = "warn_guard_exceeds_object_width"
            stage_warnings.append(
                f"Stage '{item['stage_id']}' min_aperture_m={float(guard['min_aperture_m'])*1000:.1f}mm exceeds "
                f"{oid} estimated width={width*1000:.1f}mm and may false-pause; consider non_empty (1.5mm) or rigid_object (2.0mm)."
            )

        # 调用虚拟空间模拟场执行双指三维实体干涉与咬合体积评估
        grasp_eval = vf.evaluate_grasp(
            arm=item["arm"],
            position_m=[gx, gy, gz],
            angles_deg=[pitch, roll, yaw],
            target_id=oid,
            contact_region=item.get("contact_region", "pad_center"),
            contact_depth_m=item.get("contact_depth_m"),
        )
        colliding_neighbors = grasp_eval.get("colliding_neighbor_ids") or []
        neighbor_clearance = grasp_eval.get("neighbor_clearance_m")
        opt_pose = grasp_eval.get("optimal_grasp_pose")
        finger_collision_status = "ok"
        if colliding_neighbors and isinstance(neighbor_clearance, (int, float)) and neighbor_clearance < -0.005:
            finger_collision_status = "warn_finger_collides_neighbor"
            stage_warnings.append(
                f"Stage '{item['stage_id']}' open jaws collide with neighboring object(s) {', '.join(colliding_neighbors)} "
                f"(clearance={neighbor_clearance*1000:.1f}mm); candidate grasp pose is "
                f"position_m={opt_pose['position_m']}, angles_deg={opt_pose['angles_deg']}."
            )

        bite_status = "ok"
        reasons = grasp_eval.get("failure_reasons") or []
        lat_ecc = grasp_eval.get("lateral_eccentricity_m") or 0.0
        if height_status == "ok" and axis_status == "ok" and (
            "grasp_misses_target_footprint" in reasons or "excessive_lateral_eccentricity" in reasons
        ):
            bite_status = "warn_lateral_eccentricity"
            stage_warnings.append(
                f"Stage '{item['stage_id']}' grasp lateral eccentricity={lat_ecc*1000:.1f}mm causes open finger to strike "
                f"or miss {oid}; align grasp XY closer to {opt_pose['position_m'] if opt_pose else centre}."
            )

        stage_risk_flags = [
            s for s in (axis_status, height_status, guard_status, finger_collision_status, bite_status) if s != "ok"
        ]
        warnings.extend(stage_warnings)
        stage_audits.append(
            dict(
                stage_id=item["stage_id"],
                arm=item["arm"],
                matched_object_id=oid,
                xy_distance_m=round(xy_dist, 4),
                object_length_m=round(length, 3),
                object_width_m=round(width, 3),
                object_yaw_deg=round(obj_yaw, 1),
                planned_yaw_deg=round(yaw, 1),
                closing_to_long_axis_deg=closing_angle,
                recommended_across_yaw_deg=rec_yaw,
                axis_alignment=axis_status,
                planned_z_m=round(gz, 4),
                object_top_z_m=round(float(top_z), 4) if isinstance(top_z, (int, float)) else None,
                recommended_pad_z_m=round(pad_z, 4),
                height_check=height_status,
                grip_guard_check=guard_status,
                finger_collision_check=finger_collision_status,
                bite_check=bite_status,
                theoretically_graspable=grasp_eval["theoretically_graspable"],
                vertical_bite_m=grasp_eval["vertical_bite_m"],
                lateral_eccentricity_m=grasp_eval["lateral_eccentricity_m"],
                neighbor_clearance_m=neighbor_clearance,
                colliding_neighbor_ids=colliding_neighbors,
                optimal_grasp_pose=opt_pose,
                risk_flags=stage_risk_flags,
                warnings=stage_warnings,
            )
        )
        targets_3d.append(
            dict(
                id=oid,
                centre_m=centre,
                length_m=length,
                width_m=width,
                yaw_deg=obj_yaw,
                top_z_m=float(top_z) if isinstance(top_z, (int, float)) else centre[2],
            )
        )
    all_risk_flags = [flag for stage in stage_audits for flag in stage.get("risk_flags", [])]
    for sr in sim_report.get("stage_reports") or []:
        if sr.get("status") == "path_collision" and (sr.get("path_clearance_m") or 0.0) < -0.005:
            col_objs = sr.get("colliding_objects") or []
            rec_z = sr.get("recommended_safe_z_m")
            if "warn_transport_collision" not in all_risk_flags:
                all_risk_flags.append("warn_transport_collision")
            warnings.append(
                f"Stage '{sr['stage_id']}' transport trajectory collides with 3D object(s) {', '.join(col_objs)} "
                f"(clearance={sr['path_clearance_m']*1000:.1f}mm); raise transit z to at least {rec_z:.3f}m."
            )
    if not stage_audits and not warnings:
        return None, []
    review = dict(
        observation_id=obs_id,
        status="warnings_detected" if warnings else "aligned",
        risk_flags=all_risk_flags,
        stages=stage_audits,
        warnings=warnings,
        virtual_field_simulation=dict(
            status=sim_report["status"],
            theoretically_valid=sim_report["theoretically_valid"],
            min_path_clearance_m=sim_report["min_path_clearance_m"],
            transport_clearance_m=sim_report["min_path_clearance_m"],
            stage_reports=sim_report["stage_reports"],
            virtual_world_after=sim_report["virtual_world_after"],
            suggested_fixes=sim_report["suggested_fixes"],
        ),
    )
    return review, targets_3d
