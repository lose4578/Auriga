"""features.object_table (on by default with external_perception): geometry_query kind 'object_table'.

With the switch on, geometry_query accepts {kind:'object_table', observation_id, camera_id?, pixels?, labels?,
target_camera_ids?}. labels are noun phrases chosen by the model; they are forwarded verbatim and the service (object
table v5, SAM 3.1 text prompts) segments each phrase; the harness never names objects itself.
The harness sends the saved RGB image of camera_id (default cam_head), its calibrated K and T_world_camera, the
configured support (table) height and the robot link segments of the same observation (public FK of the measured
joint state; used only to drop robot masks) to the local VGGT service (features.vggt_endpoint, POST /object_table).
The service segments the image automatically (SAM 2.1, point-prompt grid restricted to pixels whose VGGT point lies
above the support plane), drops support / background / robot masks and returns, per object mask, VGGT-based geometry:
centre, top and bottom height, oriented footprint (PCA yaw, length, width), principal-axis tilt, support (table or
the object below) and quality flags. The tool reports this table with ids o1..oN (no names: the model matches ids to
what it sees) and one overlay image per camera: ids at the mask in camera_id, and at the projected centres in each
target_camera_ids view (same ids; a projection, not a detection in that view). Inputs are permitted RGB, calibration
and proprioception only; no simulator truth is read. All values are estimates from monocular depth, not measurements.

With the switch off only enabled() is consulted: tool definitions and handler behaviour are unchanged.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from copy import deepcopy

import cv2
import numpy as np

KIND = "object_table"
SOURCE = "vggt_object_table"
DEFAULT_CAMERA = "cam_head"
MAX_PIXELS = 16
MAX_LABELS = 16
MAX_TARGETS = 3
GRIPPER_REACH_M = 0.19          # link6 origin -> beyond the fingertips (ARX X5 gripping-face centre is at 0.1501 m)
LINK_RADIUS_M = 0.06
COLUMNS = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"]

KIND_SENTENCE = ("object_table (this run; camera_id optional, default cam_head): estimated table of the objects on the "
                 "support with ids o1..oN (centre, top/bottom height, footprint length/width/yaw, axis tilt, support, quality; "
                 "VGGT monocular depth + automatic SAM masks, estimates not measurements) and an id overlay image per camera.")
FIELD_DESCRIPTIONS = {
    "pixels": "Optional for object_table: up to 16 [u,v] pixels on objects you want included even if the automatic "
              "segmentation missed them (one mask per pixel).",
    "target_camera_ids": "Optional for object_table: up to 3 other cameras of the same observation (such as wrist views) "
                         "on which the same ids are drawn at the projected object centres.",
    "labels": "Optional for object_table: noun phrases for the objects you need; each is segmented by text and its rows "
              "get column label.",
}


def enabled(runtime) -> bool:
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "object_table", False))


def held_perception_enabled(runtime) -> bool:
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "held_perception", False))


def applies(runtime, request):
    return enabled(runtime) and isinstance(request, dict) and request.get("kind") == KIND


def schema_branch():
    v2 = dict(type="array", items=dict(type="number"), minItems=2, maxItems=2)
    return dict(type="object", properties=dict(
        kind=dict(type="string", const=KIND), observation_id=dict(type="string", minLength=1),
        camera_id=dict(type="string", minLength=1),
        pixels=dict(type="array", items=v2, minItems=1, maxItems=MAX_PIXELS),
        labels=dict(type="array", items=dict(type="string"), maxItems=MAX_LABELS),
        target_camera_ids=dict(type="array", items=dict(type="string", minLength=1), minItems=1, maxItems=MAX_TARGETS,
                               uniqueItems=True)),
        required=["kind", "observation_id"], additionalProperties=False)


def extend_definition(geometry_schema, descriptions):
    """Model-facing schema/description changes, applied only when the switch is on (after other extensions)."""
    geometry_schema["oneOf"].append(schema_branch())
    return descriptions


def describe_branch(geometry_schema):
    """Branch-specific field help (called after the shared descriptions were applied)."""
    for branch in geometry_schema["oneOf"]:
        if branch.get("properties", {}).get("kind", {}).get("const") == KIND:
            branch["properties"]["kind"]["description"] = KIND_SENTENCE
            for field, text in FIELD_DESCRIPTIONS.items():
                branch["properties"][field]["description"] = text


class ServiceError(RuntimeError):
    pass


def query_service(endpoint, body, timeout_s, slack_s=5.0):
    request = urllib.request.Request(endpoint.rstrip("/") + "/object_table", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s + slack_s) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            answer = json.loads(error.read())
        except Exception:
            raise ServiceError("http_" + str(error.code)) from None
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise ServiceError(type(error).__name__ + ":" + str(getattr(error, "reason", error))[:200]) from None
    if not isinstance(answer, dict) or answer.get("status") != "ok":
        raise ServiceError(str((answer or {}).get("error", "service_error"))[:200])
    return answer


def robot_segments(tools, full, observation):
    """Line segments of the robot links (public FK of the measured joint state) for dropping robot masks."""
    segments = []
    model = None
    for frame in observation.get("cameras", []):
        try:
            result = tools.execute({**full, "kind": "fk", "camera_id": frame["camera_id"]})
        except Exception:
            continue
        poses = (result.get("value") or {}).get("link_poses") if isinstance(result, dict) else None
        if not poses:
            continue
        if model is None:
            model = tools.store.resolve_json(observation["robot_model_ref"])
        poses = {name: np.asarray(item, dtype=np.float64) for name, item in poses.items()}
        for joint in model.get("joints", []):
            parent, child = poses.get(joint.get("parent")), poses.get(joint.get("child"))
            if parent is not None and child is not None and np.linalg.norm(parent[:3, 3] - child[:3, 3]) > 1e-4:
                segments.append([parent[:3, 3].tolist(), child[:3, 3].tolist()])
        if "link6" in poses:
            tip = poses["link6"][:3, 3] + poses["link6"][:3, :3] @ np.array([GRIPPER_REACH_M, 0.0, 0.0])
            segments.append([poses["link6"][:3, 3].tolist(), tip.tolist()])
    unique = {json.dumps(np.round(np.asarray(s), 4).tolist()): s for s in segments}
    return list(unique.values())


def _r(value, digits=3):
    return None if value is None else round(float(value), digits)


def rows_from(objects, with_label=False):
    rows = []
    for index, item in enumerate(objects):
        support = item.get("support")
        support = "o%d" % (support + 1) if isinstance(support, int) else support
        tilt = _r(item.get("axis_tilt_deg"), 0) if (item.get("elongation") or 0) >= 1.8 else None
        quality = item.get("quality", "ok")
        if item.get("flags"):
            quality = "low:" + ",".join(item["flags"])
        row = ["o%d" % (index + 1), [_r(v) for v in item["centre_m"]], _r(item["top_z_m"]), _r(item["bottom_z_m"]),
               _r(item["length_m"]), _r(item["width_m"]), _r(item["yaw_deg"], 0), tilt, support, quality]
        if with_label:
            row.append(item.get("label"))
        rows.append(row)
    return rows


def is_horizontally_elongated_graspable(length, width, top_z=None, bottom_z=None, tilt_deg=None):
    """判定物体是否为适合单爪横跨长轴抓取的水平细长单体（排除超宽容器/连体块、近似等轴物体及竖直高耸物体）。"""
    from .pose_skill import ELONGATED_RATIO, MAX_OPENING_M
    if width <= 0 or width > MAX_OPENING_M or (length / width) < ELONGATED_RATIO:
        return False
    if isinstance(top_z, (int, float)) and isinstance(bottom_z, (int, float)):
        if (float(top_z) - float(bottom_z)) >= 0.85 * float(length):
            return False
    if isinstance(tilt_deg, (int, float)) and float(tilt_deg) < 35.0:
        return False
    return True


def grasp_hints_from(objects, support_z_m=None):
    """为每个识别到的物体预计算零回合税抓取几何提示（仅对水平细长可抓单体输出 across_yaw_deg 及平躺立起 3D 姿态，并基于虚拟空间场附带零碰撞 safe_grasp_pose）。"""
    from .pose_skill import (
        FINGERTIP_BELOW_PAD_M,
        MAX_OPENING_M,
        _posture_transition_angles,
        closure_yaws,
        nearest,
    )
    from .virtual_field import VirtualSpatialField
    vf = VirtualSpatialField(objects=objects, table_z=support_z_m)
    hints = {}
    for index, item in enumerate(objects):
        oid = "o%d" % (index + 1)
        if str(item.get("support") or "").startswith("held_") or item.get("quality") == "held_fk_depth":
            continue
        if (
            not isinstance(item.get("yaw_deg"), (int, float))
            or not isinstance(item.get("width_m"), (int, float))
            or not isinstance(item.get("length_m"), (int, float))
        ):
            continue
        yaw = float(item["yaw_deg"])
        width = float(item["width_m"])
        length = float(item["length_m"])
        centre_z = float(item["centre_m"][2])
        top = item.get("top_z_m")
        bottom = item.get("bottom_z_m")
        tilt = _r(item.get("axis_tilt_deg"), 0) if (item.get("elongation") or 0) >= 1.8 else item.get("tilt_deg")
        support = bottom if isinstance(bottom, (int, float)) else support_z_m
        # pad_z_m is the object's centre height, the pad height that engages the whole gripping face. It is not
        # raised to keep the fingertips clear of the support: for a thin object (such as a 15 mm block, centre 7.5 mm
        # above the table) that would lift the pads above the object's half height and leave less of it under the
        # face, which slips. The two facts the model needs to decide are given beside it instead.
        pad_z = centre_z
        fit = "too_wide" if width > MAX_OPENING_M else ("tight" if width > MAX_OPENING_M - 0.010 else "ok")
        if fit == "too_wide":
            hints[oid] = dict(opening_check="too_wide", elongated=False)
            continue
        elongated = is_horizontally_elongated_graspable(length, width, top, bottom, tilt)
        hint = dict(
            pad_z_m=_r(pad_z, 3),
            opening_check=fit,
            elongated=bool(elongated),
        )
        if isinstance(support, (int, float)):
            # fingertips reach FINGERTIP_BELOW_PAD_M below pad_center; negative clearance means they reach the support
            # before the pads reach the centre, which is normal for thin objects: a contact_stop descent stops at the
            # support and the pads sit as low as the object allows. No "safe floor" is offered: a floor value is read as
            # the height to use and lifts the pads above a thin object's half height.
            hint["fingertip_support_clearance_m"] = _r(pad_z - FINGERTIP_BELOW_PAD_M - float(support), 3)
        if elongated:
            yaws = sorted(closure_yaws(yaw), key=lambda value: (abs(value), value))
            hint["across_yaw_deg"] = yaws
            if isinstance(tilt, (int, float)) and float(tilt) >= 55.0:
                up = _posture_transition_angles(nearest(yaws, 0.0), "lying", "upright")
                hint["stand_upright_angles_deg"] = up["release_angles_deg"]
        if oid in vf.objects:
            opt = vf.solve_optimal_grasp(oid)
            if opt.get("theoretically_graspable") and (elongated or len(vf.objects) > 1):
                hint["safe_grasp_pose"] = dict(
                    position_m=[_r(v, 3) for v in opt["position_m"]],
                    across_yaw_deg=opt["across_yaw_deg"],
                    neighbor_clearance_m=_r(opt["neighbor_clearance_m"], 3),
                    safe_transit_z_m=_r(opt["safe_transit_z_m"], 3),
                )
        hints[oid] = hint
    return hints


def _draw_label(image, text, u, v, scale=0.45):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(round(u - w / 2)), int(round(v + h / 2))
    cv2.rectangle(image, (x - 2, y - h - 2), (x + w + 1, y + 3), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_source(image, objects):
    for index, item in enumerate(objects):
        contour = np.asarray(item.get("contour_px") or [], dtype=np.int32).reshape(-1, 1, 2)
        if len(contour) >= 3:
            cv2.polylines(image, [contour], True, (0, 255, 255) if item.get("flags") else (0, 255, 0), 1, cv2.LINE_AA)
    for index, item in enumerate(objects):
        u, v = item["label_px"]
        _draw_label(image, "o%d" % (index + 1), u, v)
    return image


def render_target(image, camera, objects):
    from ...robot.model.geometry import project
    if not objects:
        return image, []
    pixels, depths, visible = project(camera, np.asarray([o["centre_m"] for o in objects], dtype=np.float64))
    shown = []
    for index, (pixel, ok) in enumerate(zip(pixels, visible)):
        if ok:
            cv2.circle(image, tuple(np.rint(pixel).astype(int)), 3, (0, 255, 255), -1)
            _draw_label(image, "o%d" % (index + 1), pixel[0], pixel[1] - 12)
            shown.append("o%d" % (index + 1))
    return image, shown


def triangulate_object_centres(camera_src, camera_dst, objects, dst_pixels, *, max_skew_gap_m=0.035):
    """利用副视角（如腕部相机）匹配像素对主视角 object_table 的三维中心与高度执行双目三角化深度锚定修正。"""
    from .scene3d_helpers import triangulate_rays
    K1, T1 = camera_src["K"], camera_src["T"]
    K2, T2 = camera_dst["K"], camera_dst["T"]
    anchored = []
    for index, obj in enumerate(objects):
        oid = "o%d" % (index + 1)
        dst_uv = (dst_pixels or {}).get(oid) if isinstance(dst_pixels, dict) else (
            dst_pixels[index] if isinstance(dst_pixels, (list, tuple)) and index < len(dst_pixels) else None
        )
        src_uv = obj.get("label_px")
        if dst_uv is None or src_uv is None or len(dst_uv) != 2 or len(src_uv) != 2:
            continue
        tri = triangulate_rays(src_uv[0], src_uv[1], K1, T1, dst_uv[0], dst_uv[1], K2, T2)
        if tri.get("status") != "ok" or tri.get("skew_gap_m") is None or tri["skew_gap_m"] > max_skew_gap_m:
            continue
        pt = tri["point_m"]
        old_c = [float(v) for v in obj["centre_m"]]
        dz = float(pt[2]) - old_c[2]
        anchored.append(
            dict(
                id=oid,
                vggt_centre_m=[_r(v) for v in old_c],
                stereo_centre_m=pt,
                dz_correction_m=_r(dz, 4),
                stereo_top_z_m=_r(float(obj["top_z_m"]) + dz) if obj.get("top_z_m") is not None else None,
                stereo_bottom_z_m=_r(float(obj["bottom_z_m"]) + dz) if obj.get("bottom_z_m") is not None else None,
                skew_gap_m=tri["skew_gap_m"],
            )
        )
    return anchored


def object_table_query(runtime, request, *, timeout_s=None, max_objects=None, slack_s=5.0):
    """Returns (result, overlays) where overlays is a list of (camera_id, encoded JPEG bytes, original rgb_ref).

    timeout_s / max_objects default to features.vggt_timeout_s / object_table_max_objects (obs3d passes its own)."""
    from ...robot.model.geometry import GeometryTools
    from ...core.storage.images import encode_image
    from PIL import Image
    features = runtime.config.features
    request = dict(request)
    camera_id = request.get("camera_id") or DEFAULT_CAMERA
    full = {**request, "camera_id": camera_id, **runtime.identity, "schema_version": "0.1"}
    tools = GeometryTools(runtime.scene, settings=runtime.image_settings)
    started = time.monotonic()
    limit = float(features.vggt_timeout_s if timeout_s is None else timeout_s)
    result = {"schema_version": "0.1", **runtime.identity, "world_kind": "internal", "query_type": KIND,
              "observation_ids": [request.get("observation_id")], "status": "supported", "frame_id": "env_origin",
              "unit": "m", "value": None}
    try:
        observation = runtime.scene.observation(runtime.identity, request["observation_id"])
        camera = tools.camera(full, full["observation_id"], camera_id)
        if np.any(np.abs(camera["D"]) > 1e-12):
            raise ServiceError("distorted_camera_not_supported")
        targets = []
        for target_id in request.get("target_camera_ids") or []:
            if target_id == camera_id:
                continue
            targets.append((target_id, tools.camera(full, full["observation_id"], target_id)))
        _, image_bytes = runtime.scene.resolve_artifact(camera["frame"]["rgb_ref"])
        prepared, held_error = None, None
        if held_perception_enabled(runtime):   # features.held_perception: FK robot pixel mask + held rows
            try:
                from .held_perception import Prepared
                prepared = Prepared(runtime, tools, full, observation, camera, camera_id)
                if not prepared.meshes or not prepared.poses:
                    prepared, held_error = None, "robot_meshes_or_fk_unavailable"
            except Exception as error:   # the capsule path below stays available
                prepared, held_error = None, type(error).__name__ + ":" + str(error)[:120]
        segments = robot_segments(tools, full, observation)
        body = dict(image_b64=base64.b64encode(image_bytes).decode("ascii"), K=np.asarray(camera["K"]).tolist(),
                    T_world_camera=np.asarray(camera["T"]).tolist(), support_z_m=float(features.vggt_support_z_m),
                    robot_segments_m=segments, robot_radius_m=LINK_RADIUS_M,
                    max_objects=int(features.object_table_max_objects if max_objects is None else max_objects),
                    max_object_height_m=float(features.object_table_max_height_m), timeout_s=limit)
        box = str(getattr(features, "object_table_head_exclude_px", "") or "")
        if camera_id == DEFAULT_CAMERA and box.strip():
            body["exclude_boxes_px"] = [[int(float(v)) for v in box.split(",")]]
        if prepared is not None:
            body.update(prepared.body_fields())
            if segments:
                body["robot_segments_m"] = segments
        if request.get("pixels"):
            body["pixels"] = [[float(u), float(v)] for u, v in request["pixels"]]
        labels = [str(text).strip()[:80] for text in request.get("labels") or [] if str(text).strip()]
        if labels:
            body["labels"] = labels
        answer = query_service(features.vggt_endpoint, body, limit, slack_s)
        if prepared is not None and answer.get("version") == "object-table-v5" and not answer.get("robot_mask"):
            # Backward-compatibility with an un-restarted object-table-v5 service: re-run with capsule segments
            # if available so robot links are still filtered while keeping client-side held rows.
            segments = robot_segments(tools, full, observation)
            if segments:
                fallback_body = {
                    k: v for k, v in body.items() if k not in ("robot_mask_b64", "robot_mask_shape", "held")
                }
                fallback_body["robot_segments_m"] = segments
                answer = query_service(features.vggt_endpoint, fallback_body, limit, slack_s)
    except (ServiceError, ValueError, KeyError, TypeError) as error:
        result.update(status="unknown", reason="object_table_unavailable:" + str(error)[:200])
        result["duration_s"] = time.monotonic() - started
        return result, []
    objects = answer.get("objects") or []
    held_meta = {}
    if prepared is not None:
        objects, held_meta = prepared.apply(objects, answer, features.vggt_endpoint, limit)
    overlays = []
    source = render_source(tools.image(camera), objects)
    encoded, _ = encode_image(Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)), tools.image_settings, storage=True)
    overlays.append((camera_id, encoded, camera["frame"]["rgb_ref"]))
    projected = {}
    for target_id, target in targets:
        drawn, shown = render_target(tools.image(target), target, objects)
        encoded, _ = encode_image(Image.fromarray(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)), tools.image_settings, storage=True)
        overlays.append((target_id, encoded, target["frame"]["rgb_ref"]))
        projected[target_id] = shown
    requested = []
    for item in answer.get("requested") or []:
        index = item.get("index")
        requested.append(dict(pixel=item.get("pixel"), status=item.get("status"),
                              id="o%d" % (index + 1) if isinstance(index, int) else None))
    with_label = bool(body.get("labels"))
    value = dict(source=SOURCE, measured=False, camera_id=camera_id, support_z_m=_r(answer.get("support_z_m")),
                 columns=COLUMNS + ["label"] if with_label else COLUMNS, rows=rows_from(objects, with_label),
                 grasp_hints=grasp_hints_from(objects, answer.get("support_z_m")),
                 omitted=answer.get("omitted", 0), overlay_cameras=[name for name, _, _ in overlays])
    if requested:
        value["requested"] = requested
    if with_label:
        value["labels"] = [dict(label=item.get("label"), status=item.get("status"),
                                ids=["o%d" % (index + 1) for index in item.get("indices") or [] if isinstance(index, int)],
                                **({"filtered": item["filtered"]} if item.get("filtered") else {}))
                           for item in answer.get("labels") or []]
    if projected:
        value["projected_ids"] = projected
    if held_meta:
        value["held"] = {"o%d" % (index + 1): meta for index, meta in sorted(held_meta.items())}
    result.update(status="conditional", value=value, frame_ids=[camera["frame"]["frame_id"]],
                  assumptions=["Values are VGGT-Omega monocular-depth estimates scaled to the configured table plane "
                               "(top = 98th-percentile height of the visible mask points; length/width/yaw from the "
                               "visible points); tilt is given for elongated objects only; low:<flags> marks implausible "
                               "depth. Ids are per call; target views show projected centres, not detections."])
    result["diagnostics"] = dict(service_version=answer.get("version"), backend=answer.get("backend"),
                                 latency_s=time.monotonic() - started,
                                 service_timing=answer.get("timing"), dropped=answer.get("dropped"),
                                 n_prompts=answer.get("n_prompts"), robot_segments=len(segments),
                                 support_fit=answer.get("support_fit"), scale=answer.get("scale"),
                                 objects=[{k: o.get(k) for k in ("mask_px", "n_points", "iou_pred", "conf_rank_p50",
                                           "contact_gap_m", "cluster_frac", "elongation", "axis_tilt_deg", "label_px")}
                                          for o in objects])
    if prepared is not None or held_error:
        result["diagnostics"]["held_perception"] = (dict(prepared.diagnostics, service_held_s=answer.get("held_s"))
                                                    if prepared is not None else dict(error=held_error))
    result["duration_s"] = time.monotonic() - started
    return result, overlays
