"""会话级动作状态账本 (Action-State Ledger) 与重复失败热点告警。"""
from copy import deepcopy
import math

import numpy as np

from ...core.config import motion_thresholds
from ...core.contracts import IDENTITY

_DEFAULT_THRESHOLDS = motion_thresholds()
HOTSPOT_RADIUS_M = _DEFAULT_THRESHOLDS["hotspot_xy_radius_m"]
HOTSPOT_XY_RADIUS_M = _DEFAULT_THRESHOLDS["hotspot_xy_radius_m"]
HOTSPOT_Z_RADIUS_M = _DEFAULT_THRESHOLDS["hotspot_z_radius_m"]
MAX_RECENT_OUTCOMES = _DEFAULT_THRESHOLDS["ledger_max_recent_outcomes"]
MAX_HOTSPOTS = _DEFAULT_THRESHOLDS["ledger_max_hotspots"]


def _ledger(harness):
    """获取当前 episode 绑定的动作状态账本；环境或局切换时自动重置。"""
    ident = getattr(harness, "_action_ledger_identity", None)
    current = {k: harness.identity.get(k) for k in IDENTITY}
    if ident != current or not hasattr(harness, "_action_state_ledger"):
        harness._action_ledger_identity = current
        harness._action_state_ledger = dict(
            actions_executed=0,
            recent_outcomes=[],
            failed_hotspots=[],
        )
    return harness._action_state_ledger


def _stage_target_pos(stage):
    own = stage.get("action") or {}
    gp = own.get("input_grasp_pose")
    if isinstance(gp, dict) and isinstance(gp.get("position_m"), list) and len(gp["position_m"]) == 3:
        return [round(float(v), 4) for v in gp["position_m"]]
    waypoints = (own.get("path") or {}).get("waypoints") or []
    arm = stage.get("arm")
    if waypoints and arm and isinstance(waypoints[-1].get(arm), dict):
        pos = waypoints[-1][arm].get("position_m")
        if isinstance(pos, list) and len(pos) == 3:
            return [round(float(v), 4) for v in pos]
    return None


def _extract_stage_failure_telemetry(receipt, failed_stage_id, arm):
    sr = ((receipt.get("stage_results") or {}).get(failed_stage_id or "")) or {}
    ee = sr.get("endpoint_error") or {}
    pos_err = ee.get("position_error_m")
    cf = (((sr.get("grasp_contact_feedback") or {}).get("arms") or {}).get(arm or "")) or {}
    actual_contact = cf.get("actual_contact_position_m") if cf.get("available") else None
    shortfall = cf.get("approach_shortfall_m") if cf.get("available") else None
    out = {}
    if isinstance(pos_err, (int, float)) and math.isfinite(pos_err):
        out["position_error_m"] = round(float(pos_err), 4)
    if isinstance(actual_contact, list) and len(actual_contact) == 3:
        out["actual_contact_m"] = [round(float(x), 4) for x in actual_contact]
    if isinstance(shortfall, (int, float)) and math.isfinite(shortfall):
        out["shortfall_m"] = round(float(shortfall), 4)
    return out


def attribute_contact_block_3d(harness, action, receipt, fallback_pos=None):
    """结合缓存的 3D object_table 对受阻/失败接触位置做三维物体表面归因（返回 contact_object_3d 并挂载到 contact_block_summary）。"""
    obs_id = (action or {}).get("based_on_observation_id")
    if not obs_id:
        return None
    from ...perception.external.pose_skill import attached_table
    table = attached_table(harness, obs_id)
    if not isinstance(table, dict) or not table.get("rows"):
        return None
    summary = (receipt or {}).get("contact_block_summary")
    pos = None
    if isinstance(summary, dict):
        pos = summary.get("actual_contact_position_m") or summary.get("measured_position_m") or summary.get("target_position_m")
    if not (isinstance(pos, list) and len(pos) == 3):
        pos = fallback_pos
    if not (isinstance(pos, list) and len(pos) == 3):
        return None
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    columns = list(table.get("columns") or [])
    best_row, best_dist = None, math.inf
    for raw in table.get("rows") or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < len(columns):
            continue
        row = dict(zip(columns, raw))
        if str(row.get("support") or "").startswith("held_") or row.get("quality") == "held_fk_depth":
            continue
        c = row.get("centre_m")
        if not isinstance(c, list) or len(c) != 3:
            continue
        dist = math.hypot(px - float(c[0]), py - float(c[1]))
        reach = max(0.5 * float(row.get("length") or 0.0) + 0.04, 0.5 * float(row.get("width") or 0.0) + 0.04, 0.075)
        if dist <= reach and dist < best_dist:
            best_row, best_dist = row, dist
    if best_row is None:
        return None
    cx, cy, cz = (float(v) for v in best_row["centre_m"])
    top_z = float(best_row["top_z"]) if isinstance(best_row.get("top_z"), (int, float)) else cz
    bottom_z = float(best_row["bottom_z"]) if isinstance(best_row.get("bottom_z"), (int, float)) else None
    dx, dy = round(px - cx, 4), round(py - cy, 4)
    dz_top = round(pz - top_z, 4)
    attr = dict(
        id=best_row["id"],
        centre_m=[round(cx, 4), round(cy, 4), round(cz, 4)],
        top_z=round(top_z, 4),
        bottom_z=round(bottom_z, 4) if bottom_z is not None else None,
        xy_offset_from_centre_m=[dx, dy],
        xy_distance_m=round(best_dist, 4),
        height_above_top_m=dz_top,
        diagnosis=(
            f"Contact/stop near 3D object {best_row['id']} (centre={[round(cx,3),round(cy,3),round(cz,3)]}, top_z={top_z:.3f}m, "
            f"xy_offset=[{dx:+.3f},{dy:+.3f}]m, dz_from_top={dz_top:+.3f}m)."
        ),
    )
    if isinstance(summary, dict):
        summary["contact_object_3d"] = deepcopy(attr)
    return attr


def _is_held_table_row(row):
    """判断 object_table 行是否为随夹爪运动的夹持物合成行。"""
    return str(row.get("support") or "").startswith("held_") or row.get("quality") == "held_fk_depth"


def compute_object_table_delta(before_table, after_table, *, move_threshold_m=0.015, match_radius_m=0.14):
    """对比动作前后两帧 3D object_table，识别发生明显位移/抬升、从原位消失或新出现的 3D 物体。"""
    if not isinstance(before_table, dict) or not isinstance(after_table, dict):
        return None
    b_cols, a_cols = list(before_table.get("columns") or []), list(after_table.get("columns") or [])
    b_rows = [
        row for r in (before_table.get("rows") or [])
        if isinstance(r, (list, tuple)) and len(r) >= len(b_cols)
        and not _is_held_table_row(row := dict(zip(b_cols, r)))
    ]
    a_rows = [
        row for r in (after_table.get("rows") or [])
        if isinstance(r, (list, tuple)) and len(r) >= len(a_cols)
        and not _is_held_table_row(row := dict(zip(a_cols, r)))
    ]
    if not b_rows and not a_rows:
        return None
    used_after = set()
    moved, unchanged, vacated = [], [], []
    for b in b_rows:
        bc = b.get("centre_m")
        if not isinstance(bc, list) or len(bc) != 3:
            continue
        bx, by, bz = float(bc[0]), float(bc[1]), float(bc[2])
        best_idx, best_d = None, math.inf
        for idx, a in enumerate(a_rows):
            if idx in used_after:
                continue
            ac = a.get("centre_m")
            if not isinstance(ac, list) or len(ac) != 3:
                continue
            d3 = math.sqrt((bx - float(ac[0])) ** 2 + (by - float(ac[1])) ** 2 + (bz - float(ac[2])) ** 2)
            if d3 <= match_radius_m and d3 < best_d:
                best_idx, best_d = idx, d3
        if best_idx is None:
            vacated.append(dict(before_id=b["id"], before_centre_m=[round(bx, 3), round(by, 3), round(bz, 3)]))
            continue
        used_after.add(best_idx)
        a = a_rows[best_idx]
        ax, ay, az = (float(v) for v in a["centre_m"])
        dx, dy, dz = round(ax - bx, 4), round(ay - by, 4), round(az - bz, 4)
        if best_d >= move_threshold_m or abs(dz) >= move_threshold_m:
            moved.append(
                dict(
                    before_id=b["id"],
                    after_id=a["id"],
                    from_centre_m=[round(bx, 3), round(by, 3), round(bz, 3)],
                    to_centre_m=[round(ax, 3), round(ay, 3), round(az, 3)],
                    delta_m=[dx, dy, dz],
                    displacement_m=round(best_d, 4),
                    lifted=bool(dz >= 0.020),
                )
            )
        else:
            unchanged.append(dict(before_id=b["id"], after_id=a["id"]))
    appeared = [
        dict(after_id=a["id"], centre_m=[round(float(v), 3) for v in a["centre_m"]])
        for idx, a in enumerate(a_rows)
        if idx not in used_after and isinstance(a.get("centre_m"), list) and len(a["centre_m"]) == 3
    ]
    return dict(moved=moved, vacated=vacated, appeared=appeared, unchanged_count=len(unchanged))


def record_object_table_delta(harness, before_obs_id, after_obs_id, delta):
    """将最近一次动作前后的 3D 物体位移差分写入会话动作账本。"""
    if not isinstance(delta, dict):
        return
    ledger = _ledger(harness)
    ledger["last_object_delta"] = dict(
        before_observation_id=before_obs_id,
        after_observation_id=after_obs_id,
        **deepcopy(delta),
    )


def _within_hotspot(p, hp):
    xy_dist = float(np.linalg.norm(p[:2] - hp[:2]))
    z_dist = abs(float(p[2] - hp[2]))
    return xy_dist <= HOTSPOT_XY_RADIUS_M and z_dist <= HOTSPOT_Z_RADIUS_M


def _clear_resolved_hotspots(hotspots, arm, pos):
    p = np.asarray(pos, dtype=float)
    hotspots[:] = [
        item
        for item in hotspots
        if item.get("arm") != arm
        or float(np.linalg.norm(p[:2] - np.asarray(item["position_m"], dtype=float)[:2])) > HOTSPOT_XY_RADIUS_M
    ]


def _upsert_hotspot(hotspots, arm, pos, reason, plan_id, telemetry=None):
    p = np.asarray(pos, dtype=float)
    telemetry = telemetry or {}
    for item in hotspots:
        if item.get("arm") == arm:
            hp = np.asarray(item["position_m"], dtype=float)
            if _within_hotspot(p, hp):
                item["failure_count"] += 1
                item["last_reason"] = reason
                item["last_plan_id"] = plan_id
                item["position_m"] = [round(float(v), 4) for v in (0.5 * (hp + p))]
                item.update(telemetry)
                return
    entry = dict(
        arm=arm,
        position_m=[round(float(v), 4) for v in pos],
        failure_count=1,
        last_reason=reason,
        last_plan_id=plan_id,
    )
    entry.update(telemetry)
    hotspots.append(entry)
    while len(hotspots) > MAX_HOTSPOTS:
        hotspots.pop(0)


def record_execution_outcome(harness, plan_id, action, receipt):
    """在批次执行完成后更新会话动作账本与失败空间热点。"""
    if not isinstance(receipt, dict):
        return
    ledger = _ledger(harness)
    ledger["actions_executed"] += 1
    status = receipt.get("batch_status", receipt.get("status"))
    stop_reason = receipt.get("stop_reason") or receipt.get("reason")
    failed_stage_id = receipt.get("failed_stage_id")
    completed_ids = list(receipt.get("completed_stage_ids") or [])
    completed_set = set(completed_ids)

    stages = action.get("stages") or []
    stage_by_id = {s.get("id"): s for s in stages if isinstance(s, dict)}
    last_move_pos = {}
    fail_telemetry = {}
    for s in stages:
        arm = s.get("arm")
        pos = _stage_target_pos(s)
        if arm and pos is not None:
            last_move_pos[arm] = pos
        sid = s.get("id")
        if sid in completed_set and s.get("kind") == "move" and arm and pos is not None:
            _clear_resolved_hotspots(ledger["failed_hotspots"], arm, pos)
        if sid == failed_stage_id and status in ("stopped", "failed"):
            fpos = pos or last_move_pos.get(arm)
            fail_telemetry = _extract_stage_failure_telemetry(receipt, failed_stage_id, arm)
            attr_3d = attribute_contact_block_3d(harness, action, receipt, fallback_pos=fpos)
            if attr_3d is not None:
                fail_telemetry["contact_object_3d"] = attr_3d
            if arm and fpos is not None:
                _upsert_hotspot(
                    ledger["failed_hotspots"],
                    arm,
                    fpos,
                    stop_reason or status,
                    plan_id,
                    fail_telemetry,
                )

    entry = dict(
        plan_id=plan_id,
        status=status,
        completed_stages=completed_ids[-4:],
        executed_native_steps=receipt.get("executed_native_steps", 0),
    )
    if stop_reason:
        entry["stop_reason"] = stop_reason
    if failed_stage_id:
        entry["failed_stage_id"] = failed_stage_id
        if failed_stage_id in stage_by_id:
            fpos = _stage_target_pos(stage_by_id[failed_stage_id])
            if fpos is not None:
                entry["failed_target_m"] = fpos
        entry.update(fail_telemetry)
    ledger["recent_outcomes"].append(entry)
    while len(ledger["recent_outcomes"]) > MAX_RECENT_OUTCOMES:
        ledger["recent_outcomes"].pop(0)


def attach_preview_ledger(harness, action, summary):
    """在再次瞄准失败热点柱面邻域时于 preview 摘要发出告警并附带紧凑会话账本。"""
    ledger = _ledger(harness)
    if ledger["actions_executed"] == 0 and not ledger["failed_hotspots"]:
        return summary
    warnings = []
    for stage in action.get("stages") or []:
        if stage.get("kind") != "move":
            continue
        arm = stage.get("arm")
        pos = _stage_target_pos(stage)
        if not arm or pos is None:
            continue
        p = np.asarray(pos, dtype=float)
        for hp in ledger["failed_hotspots"]:
            if hp.get("arm") != arm:
                continue
            hp_vec = np.asarray(hp["position_m"], dtype=float)
            if _within_hotspot(p, hp_vec):
                dist = float(np.linalg.norm(p - hp_vec))
                xy_dist = float(np.linalg.norm(p[:2] - hp_vec[:2]))
                w = dict(
                    stage_id=stage.get("id"),
                    arm=arm,
                    target_position_m=pos,
                    hotspot_position_m=deepcopy(hp["position_m"]),
                    distance_m=round(dist, 4),
                    xy_distance_m=round(xy_dist, 4),
                    prior_failures=hp["failure_count"],
                    last_reason=hp["last_reason"],
                    severity="repeated_hotspot" if hp["failure_count"] >= 2 else "prior_failure_nearby",
                    guidance=(
                        f"Target is within the failed hotspot zone (XY<={HOTSPOT_XY_RADIUS_M*100:.1f}cm, Z<={HOTSPOT_Z_RADIUS_M*100:.1f}cm); "
                        "adjust XY alignment, approach angles_deg, or contact_region instead of repeating Z nudges."
                    ),
                )
                for k in ("position_error_m", "actual_contact_m", "shortfall_m", "contact_object_3d"):
                    if k in hp:
                        w[k] = deepcopy(hp[k])
                warnings.append(w)
    if warnings:
        summary["episode_ledger"] = deepcopy(ledger)
        summary["repeat_failure_warning"] = warnings
    return summary


def attach_execution_ledger(harness, result, summary=None):
    """在 execute 返回结果上附加最新会话动作账本与热点重复告警。"""
    ledger = _ledger(harness)
    out = dict(result)
    out["episode_ledger"] = deepcopy(ledger)
    if isinstance(summary, dict) and "repeat_failure_warning" in summary:
        out["repeat_failure_warning"] = deepcopy(summary["repeat_failure_warning"])
    return out
