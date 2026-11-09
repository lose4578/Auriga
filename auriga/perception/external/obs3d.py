"""features.object_table_observe / features.scene3d_observe (both on by default with external_perception): 3D scene information as part of
every model observation reply instead of an on-demand geometry_query kind.

With a switch on, a full head-camera observation in a reply to the model (not to a program job) gets, after the
observation metadata and camera images, one extra JSON text part and at most one image. Replies covered:

* robot_observe without camera_id (a full observation, not an ROI view): text part + overlay image;
* the final observation of robot_action (mode execute / direct) and robot_wait replies: text part only; the overlay
  image is added there too only with features.obs3d_action_overlay (default false). Temporal wait samples get nothing.

Parts:

* object_table_observe: the object table of the head frame, computed exactly as geometry_query kind object_table
  (same service call POST /object_table, same rows: ids o1..oN, centre, top/bottom z, length, width, yaw, tilt,
  support, quality; no names, no truth), limited to features.obs3d_max_objects rows, and one overlay image of the head
  frame with the ids drawn (same size as the camera frame). Wrist cameras get nothing.
* scene3d_observe: the VGGT point cloud of the head frame saved as a scene3d_points artifact, exactly as
  geometry_query kind scene3d (POST /points); the reply carries only the artifact handle, the point count, table_z and
  a one-line pointer to scene3d_helpers (copied into program_run workspaces while the switch is on). No table, no image.

One service call per kind and observation: /object_table first, then /points (the service reuses its per-image
reconstruction). A successful result is kept for the most recent observation, so a robot_observe of an observation
that an action reply already covered reuses it (same rows, same artifact, no new call). The /height path is
unchanged. A service error or timeout (features.obs3d_timeout_s) never fails the observation or the action reply: the
part then says "unavailable: <reason>" in one field. Everything lives inside the tool result, so the request
projection (lean_projection block compression, old-result summaries, image placeholders) treats it like any other
tool output. The stored ObservationBundle is unchanged; per-observation diagnostics are written as event
geometry.obs3d_attached.

With analysis.scene_truth_log on (analysis only, off by default), every attached object table (also a reused one,
and an unavailable one) is additionally written as internal event analysis.obs3d_object_table: observation id, head
frame id, reply kind, reused flag and the rows exactly as attached. It holds the model-visible estimate only (no
truth), is never returned to the model, and is read offline by scripts/score_geometry_truth.py (kind
object_table_obs). With the switch off nothing is written.

With both switches off only flags() is consulted: tool definitions and replies are unchanged.
"""
from __future__ import annotations

import threading
import time

CAMERA = "cam_head"
OBJECT_TABLE_NOTE = ("head camera, VGGT monocular-depth estimates in env_origin m (not measurements); ids are per "
                     "observation and drawn on the next image; low:<flags> = implausible depth")
SCENE3D_USE = "program_run: import scene3d_helpers as s3; s3.load(artifact); print(s3.API) lists the queries"
CATALOGUE_PREFIX = " Head-camera observations here and in robot_action / robot_wait replies also carry the frame's "
CATALOGUE_TABLE = "VGGT-estimated object table (ids o1..oN: centre, top/bottom z, footprint, tilt, support, quality)"
CATALOGUE_CLOUD = "VGGT point cloud as a saved artifact (handle, point count, table_z) for scene3d_helpers in program_run."
CATALOGUE_BOTH = " plus a point-cloud artifact handle for scene3d_helpers"
CATALOGUE_OVERLAY = ("; robot_observe adds an id overlay image.", ", with an id overlay image.")
ANALYSIS_EVENT = "analysis.obs3d_object_table"


def flags(runtime):
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return (bool(getattr(features, "object_table_observe", False)),
            bool(getattr(features, "scene3d_observe", False)))


def action_overlay(runtime):
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "obs3d_action_overlay", False))


def catalogue_sentence(object_table=False, scene3d=False, action_overlay=False):
    """Text appended to the robot_observe description ('' when both switches are off)."""
    if not (object_table or scene3d):
        return ""
    if not object_table:
        return CATALOGUE_PREFIX + CATALOGUE_CLOUD
    return (CATALOGUE_PREFIX + CATALOGUE_TABLE + (CATALOGUE_BOTH if scene3d else "")
            + CATALOGUE_OVERLAY[bool(action_overlay)])


def developer_instructions(features) -> str:
    """Developer-instruction summary for active 3D perception switches; '' when all are off."""
    if features is None:
        return ""
    parts = []
    if getattr(features, "object_table_observe", False):
        parts.append(
            "Head-camera observations in robot_observe and the final observation of robot_action / robot_wait "
            "automatically attach a VGGT 3D object_table (ids o1..oN: centre_m, top_z, bottom_z, length, width, "
            "yaw_deg in [-90,90), tilt_deg, support, quality; robot_observe also attaches an id overlay image) "
            "plus inline `grasp_hints` per object (`pad_z_m`: the object's centre z, the pad height that puts the whole "
            "15 mm gripping face on the object; `fingertip_support_clearance_m`: how far the fingertips, 7.5 mm below "
            "the pad centre, stay above the support at that height; negative for thin objects, where a contact_stop "
            "descent touches the support and stops there, which is normal, with the pads as low as the object allows; "
            "`opening_check` "
            "against the 88.8 mm max opening, `elongated` flag; for horizontally elongated graspable objects, "
            "`across_yaw_deg` gives the two angles_deg yaws closing perpendicular to the long axis `yaw_deg +/- 90`, "
            "and for lying elongated objects `stand_upright_angles_deg` gives the 3D `[pitch,roll,yaw]` release "
            "orientation to stand the object upright). For compact (`elongated:false`) or `too_wide` objects/fixtures, "
            "choose the grasp point and orientation directly from the RGB image (such as a handle, rim, prongs, or slot) "
            "and execute complete pick-and-place sequences in a single continuous batch relying on `grip_guard` rather "
            "than splitting batches or inserting redundant mid-batch `:observe` stages. `robot_action(mode='preview')` "
            "automatically returns `grasp_3d_review`, and after execution the reply reports `object_state_delta` and "
            "any contact stop's `contact_object_3d` in `contact_block_summary` / `episode_ledger`."
        )
    if getattr(features, "object_table", False):
        parts.append(
            "For custom segmentation (including thin/flat objects and containers omitted from the default table) "
            "or wrist-camera projection, call geometry_query with "
            "request={'kind':'object_table', 'observation_id':..., 'labels':['noun phrase',...], "
            "'pixels':[[u,v],...], 'target_camera_ids':['cam_left_wrist','cam_right_wrist']}."
        )
    if getattr(features, "scene3d_observe", False) or getattr(features, "scene3d", False):
        parts.append(
            "When a scene3d point-cloud artifact handle is attached (or created via geometry_query kind='scene3d'), "
            "you can inspect local 3D surface geometry and run 3D virtual spatial simulations in program_run with "
            "`import scene3d_helpers as s3; s3.load(artifact)` "
            "(queries: s3.top_z, s3.table_z, s3.points_in_box, s3.points_in_cylinder, s3.nearest, s3.cluster_at_pixel, "
            "s3.surface_extrema(region) for unfiltered slot/hole/rim height extrema, "
            "s3.in_hand_axis(grasp_center_m) for 3D PCA yaw/tilt/posture and bottom_point_m / tip_offset_from_grasp_m "
            "of a held object near the fingertips using robot=True, "
            "s3.triangulate_rays(...) for two-camera stereo depth anchoring, "
            "and `vf = s3.virtual_field(object_table)` for fast 3D OBB grasp/stage simulation: "
            "`vf.evaluate_grasp(arm, position_m, angles_deg, target_id)` checks open-finger 3D collision, "
            "`vertical_bite_m`, and `lateral_eccentricity_m`; `vf.solve_optimal_grasp(target_id, arm)` finds a "
            "collision-free grasp pose (`position_m` in m, `angles_deg` in deg, `safe_lift_z_m` in m); "
            "`vf.simulate_stages(stages)` simulates multi-stage grasp/transport/release and reports `min_path_clearance_m` "
            "and `virtual_world_after`). Pass `obstacle_object_ids=['o1',...]` "
            "in `robot_action` when you want specific 3D objects from `object_table` converted into AABB obstacles."
        )
    if getattr(features, "vggt_height", False):
        mode = getattr(features, "vggt_height_mode", "optional")
        if mode == "default":
            parts.append(
                "geometry_query(kind='pixel_plane') automatically uses VGGT-estimated object centre height at the "
                "pixel unless you set plane_source='caller' (your plane_origin_m is retained as the alternate plane)."
            )
        else:
            parts.append(
                "geometry_query(kind='pixel_plane') accepts plane_source='vggt' to intersect the ray with the "
                "VGGT-estimated object centre height at that pixel."
            )
    return ("\n\n## 3D Perception (VGGT + SAM)\n" + "\n".join("- " + p for p in parts)) if parts else ""


def short_reason(reason):
    text = str(reason or "")
    text = text.split("_unavailable:", 1)[-1]
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "refused" in lowered or "connectionreset" in lowered or "remotedisconnected" in lowered:
        return "service_down"
    return text[:80] or "unknown"


def analysis_log_enabled(runtime):
    """analysis.scene_truth_log: the offline-scoring switch (the truth log itself is written by the worker only)."""
    analysis = getattr(getattr(runtime, "config", None), "analysis", None)
    return bool(getattr(analysis, "scene_truth_log", False))


def _analysis_log(runtime, correlation, observation_id, reply, table, result, reused):
    """Internal event with the attached table for offline scoring; never part of a reply, never raises."""
    if not analysis_log_enabled(runtime):
        return
    result = result if isinstance(result, dict) else {}
    payload = dict(observation_id=observation_id, observation_ids=[observation_id], camera_id=CAMERA, reply=reply,
                   reused=bool(reused), frame_ids=list(result.get("frame_ids") or []), available=isinstance(table, dict),
                   service_version=(result.get("diagnostics") or {}).get("service_version"))
    if isinstance(table, dict):
        payload.update(columns=list(table["columns"]), rows=table["rows"], table_z=table.get("table_z"),
                       omitted=table.get("omitted", 0))
    else:
        payload["unavailable"] = str(table)
    try:
        runtime.store.event(ANALYSIS_EVENT, payload, correlation, world_kind="internal")
    except Exception:
        pass


def _head_camera(bundle):
    for camera in (bundle or {}).get("cameras") or []:
        if isinstance(camera, dict) and camera.get("camera_id") == CAMERA:
            return camera
    return None


def _object_table(runtime, observation_id, timeout_s, max_objects):
    from .object_table import object_table_query
    result, overlays = object_table_query(runtime, {"kind": "object_table", "observation_id": observation_id,
                                                    "camera_id": CAMERA},
                                          timeout_s=timeout_s, max_objects=max_objects, slack_s=0.0)
    value = result.get("value") if result.get("status") == "conditional" else None
    if not isinstance(value, dict):
        return "unavailable: " + short_reason(result.get("reason")), None, result
    shown = dict(columns=value["columns"], rows=value["rows"], table_z=value.get("support_z_m"), note=OBJECT_TABLE_NOTE)
    if value.get("grasp_hints"):
        shown["grasp_hints"] = value["grasp_hints"]
    if value.get("omitted"):
        shown["omitted"] = value["omitted"]
    if value.get("held"):   # features.held_perception only
        shown["held"] = value["held"]
    overlay = next(((data, original) for camera_id, data, original in overlays if camera_id == CAMERA), None)
    return shown, overlay, result


def _scene3d_fetch(runtime, observation_id, timeout_s):
    from .scene3d import scene3d_query
    return scene3d_query(runtime, {"kind": "scene3d", "observation_id": observation_id, "camera_id": CAMERA},
                         timeout_s=timeout_s, slack_s=0.0)


def _scene3d_store(runtime, observation_id, correlation, result, data):
    from .scene3d import ARTIFACT_KIND, ARTIFACT_MIME
    if data is None or result.get("status") != "conditional":
        return "unavailable: " + short_reason(result.get("reason")), result
    reference = runtime.store.put_artifact(data, ARTIFACT_KIND, ARTIFACT_MIME)
    result["value"]["artifact"] = reference
    runtime.store.event("geometry.scene3d_stored", {"artifact": reference, "bytes": len(data),
        "points": result["value"]["points"], "observation_id": observation_id, "camera_id": CAMERA,
        "source_kind": "obs3d"}, correlation, world_kind="internal")
    value = result["value"]
    return dict(artifact=reference, points=value["points"], table_z=value["table_z"], use=SCENE3D_USE), result


def _scene3d(runtime, observation_id, timeout_s, correlation):
    result, data = _scene3d_fetch(runtime, observation_id, timeout_s)
    return _scene3d_store(runtime, observation_id, correlation, result, data)


def _memo(runtime, observation_id):
    """Successful results for observation_id (checks latest _obs3d_memo and bounded _obs3d_history): {'object_table': (...), 'scene3d': (...)}."""
    memo = getattr(runtime, "_obs3d_memo", None)
    if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == observation_id:
        return memo[1]
    history = getattr(runtime, "_obs3d_history", None)
    if isinstance(history, dict) and observation_id in history:
        return history[observation_id]
    return {}


def _remember(runtime, observation_id, key, value):
    entry = dict(_memo(runtime, observation_id))
    entry[key] = value
    try:
        runtime._obs3d_memo = (observation_id, entry)
        history = getattr(runtime, "_obs3d_history", None)
        if not isinstance(history, dict):
            history = {}
            runtime._obs3d_history = history
        history[observation_id] = entry
        while len(history) > 8:
            history.pop(next(iter(history)))
    except Exception:
        pass


def _sync_virtual_field(runtime, bundle, table):
    """将会话级虚拟空间模拟场 `runtime._virtual_spatial_field` 与最新观测的 object_table 及双臂位姿同步。"""
    if not isinstance(table, dict):
        return
    try:
        from .virtual_field import VirtualSpatialField
        robot_state = (bundle or {}).get("robot_state") if isinstance(bundle, dict) else None
        if not isinstance(robot_state, dict) and isinstance(bundle, dict) and bundle.get("robot_state_ref"):
            resolve_art = getattr(getattr(runtime, "store", None), "resolve_artifact", None)
            if callable(resolve_art):
                import json
                _, raw_bytes = resolve_art(bundle["robot_state_ref"])
                robot_state = json.loads(raw_bytes)
        vf = getattr(runtime, "_virtual_spatial_field", None)
        if not isinstance(vf, VirtualSpatialField):
            vf = VirtualSpatialField(objects=table, robot_state=robot_state)
            runtime._virtual_spatial_field = vf
        else:
            vf.update_from_observation(table, robot_state=robot_state)
    except Exception:
        pass


def attachments(runtime, bundle, correlation, *, reply="observe", before_observation_id=None):
    """Extra display parts for one model reply: [] or [StructuredText, (inputImage)].

    reply 'observe' (robot_observe, full observation): text + overlay; 'action' (final observation of robot_action /
    robot_wait): text, plus the overlay only with features.obs3d_action_overlay."""
    object_table_on, scene3d_on = flags(runtime)
    if not (object_table_on or scene3d_on) or correlation.get("program_id"):
        return []
    camera = _head_camera(bundle)
    if camera is None:
        return []
    from ...core.tool_output import StructuredText
    features = runtime.config.features
    timeout_s = float(getattr(features, "obs3d_timeout_s", 10.0))
    with_overlay = reply == "observe" or action_overlay(runtime)
    observation_id = bundle["observation_id"]
    shown = {"observation_id": observation_id}
    images, diagnostics = [], {"observation_id": observation_id, "camera_id": CAMERA, "reply": reply}
    started = time.monotonic()
    memo = _memo(runtime, observation_id)
    s3_prefetch = None
    if object_table_on and scene3d_on and "object_table" not in memo and "scene3d" not in memo:
        # 并发发起 /object_table 与 /points HTTP 请求，确保 /object_table 率先发出，所有 store 落盘保持主线程串行顺序
        s3_box = {}
        def _run_s3():
            time.sleep(min(0.005, max(0.0, timeout_s * 0.1)))
            t_s3 = time.monotonic()
            try:
                s3_box["res"] = _scene3d_fetch(runtime, observation_id, timeout_s)
            except Exception as exc:
                s3_box["err"] = exc
            s3_box["elapsed_s"] = time.monotonic() - t_s3
        s3_thread = threading.Thread(target=_run_s3, daemon=True)
        s3_thread.start()
        s3_prefetch = (s3_thread, s3_box)
    if object_table_on:
        t0 = time.monotonic()
        reused = "object_table" in memo
        if reused:
            table, overlay, result = memo["object_table"]
            diagnostics.setdefault("reused", []).append("object_table")
        else:
            try:
                table, overlay, result = _object_table(runtime, observation_id, timeout_s,
                                                       int(getattr(features, "obs3d_max_objects", 10)))
            except Exception as error:  # an optional attachment never hides the observation
                table, overlay, result = ("unavailable: " + short_reason(type(error).__name__ + ":" + str(error)),
                                          None, {})
            if isinstance(table, dict):
                _remember(runtime, observation_id, "object_table", (table, overlay, result))
                _sync_virtual_field(runtime, bundle, table)
        shown["object_table"] = table
        if isinstance(table, dict) and before_observation_id and before_observation_id != observation_id:
            before_obs = before_observation_id
            before_entry = _memo(runtime, before_obs).get("object_table")
            if not (isinstance(before_entry, tuple) and before_entry):
                history = getattr(runtime, "_obs3d_history", None)
                if isinstance(history, dict):
                    for hist_obs_id in reversed(list(history)):
                        if hist_obs_id != observation_id and "object_table" in history[hist_obs_id]:
                            before_obs = hist_obs_id
                            before_entry = history[hist_obs_id]["object_table"]
                            break
            before_table = before_entry[0] if isinstance(before_entry, tuple) and before_entry else None
            if isinstance(before_table, dict):
                from ...runtime.actions.action_state_ledger import compute_object_table_delta, record_object_table_delta
                delta = compute_object_table_delta(before_table, table)
                if delta:
                    shown["object_state_delta"] = delta
                    record_object_table_delta(runtime, before_obs, observation_id, delta)
        _analysis_log(runtime, correlation, observation_id, reply, table, result, reused)
        diagnostics["object_table"] = dict(status=result.get("status"), reason=result.get("reason"),
            rows=len(table["rows"]) if isinstance(table, dict) else None, latency_s=round(time.monotonic() - t0, 3),
            service=(result.get("diagnostics") or {}).get("service_version"),
            service_timing=(result.get("diagnostics") or {}).get("service_timing"))
        if overlay is not None and with_overlay:
            from ...core.storage.images import image_settings, image_url
            data, original = overlay
            settings = runtime.image_settings if runtime.image_settings is not None else image_settings(None)
            reference = runtime.store.put_artifact(data, "geometry_overlay", "image/jpeg")
            url, encoding = image_url(data, settings)
            images.append(dict(type="inputImage", imageUrl=url))
            runtime.store.event("geometry.overlay_delivered", {"image_ref": reference, "source_sha256": encoding["sha256"],
                "sha256": encoding["sha256"], "encoding": encoding, "original_rgb_ref": original, "camera_id": CAMERA,
                "source_kind": "obs3d_object_table_overlay", "delivered_to_model": True},
                correlation, world_kind="internal")
            diagnostics["object_table"]["overlay_ref"] = reference
    if scene3d_on:
        t0 = time.monotonic()
        if "scene3d" in memo:
            cloud, result = memo["scene3d"]
            diagnostics.setdefault("reused", []).append("scene3d")
            s3_latency_s = time.monotonic() - t0
        elif s3_prefetch is not None:
            s3_thread, s3_box = s3_prefetch
            s3_thread.join()
            try:
                if "err" in s3_box:
                    raise s3_box["err"]
                s3_result, s3_data = s3_box["res"]
                cloud, result = _scene3d_store(runtime, observation_id, correlation, s3_result, s3_data)
            except Exception as error:
                cloud, result = "unavailable: " + short_reason(type(error).__name__ + ":" + str(error)), {}
            if isinstance(cloud, dict):
                _remember(runtime, observation_id, "scene3d", (cloud, result))
            s3_latency_s = s3_box.get("elapsed_s", 0.0) + (time.monotonic() - t0)
        else:
            try:
                cloud, result = _scene3d(runtime, observation_id, timeout_s, correlation)
            except Exception as error:
                cloud, result = "unavailable: " + short_reason(type(error).__name__ + ":" + str(error)), {}
            if isinstance(cloud, dict):
                _remember(runtime, observation_id, "scene3d", (cloud, result))
            s3_latency_s = time.monotonic() - t0
        shown["scene3d"] = cloud
        diagnostics["scene3d"] = dict(status=result.get("status"), reason=result.get("reason"),
            artifact=cloud.get("artifact") if isinstance(cloud, dict) else None,
            points=cloud.get("points") if isinstance(cloud, dict) else None,
            latency_s=round(s3_latency_s, 3),
            npz_bytes=(result.get("diagnostics") or {}).get("npz_bytes"),
            cache_hit=(result.get("diagnostics") or {}).get("cache_hit"))
    diagnostics["latency_s"] = round(time.monotonic() - started, 3)
    try:
        runtime.store.event("geometry.obs3d_attached", diagnostics, correlation, world_kind="internal")
    except Exception:
        pass
    return [StructuredText(shown, projection="data"), *images]


def action_attachments(runtime, bundle, correlation, *, before_observation_id=None):
    """Parts for the final observation of a robot_action / robot_wait reply ([] when off); never raises."""
    if not any(flags(runtime)) or correlation.get("program_id") or not isinstance(bundle, dict):
        return []
    try:
        return attachments(runtime, bundle, correlation, reply="action", before_observation_id=before_observation_id)
    except Exception:  # the action already executed; an optional attachment must not hide its receipt
        return []
