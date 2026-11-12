"""features.scene3d (on by default with external_perception): geometry_query kind 'scene3d' and the program helper module scene3d_helpers.

With the switch on, geometry_query accepts {kind:'scene3d', observation_id, camera_id?, voxel_m?}. The harness sends
the saved RGB image of camera_id (default cam_head), its calibrated K and T_world_camera, the configured support
(table) height and the robot link segments of the same observation (public FK of the measured joints; used only to
flag points near the robot) to the local VGGT service (features.vggt_endpoint, POST /points). The service returns the
VGGT-Omega point cloud of that image in env_origin, scaled to the support plane and voxel-downsampled, with a per-point
depth-confidence rank, the source pixel and a near-robot flag, as an npz file. The harness stores it as an artifact
(kind scene3d_points) and returns only a short summary: the artifact handle, point count, bounding boxes, table
height, a usage note and the helper API. Program jobs get scene3d_helpers.py in their workspace and query the cloud
there (nearest points, points in a box/cylinder, top surface height of a region, local table height, cluster at a
pixel); the cloud itself never enters the model context unless a program prints it. Inputs are permitted RGB,
calibration and proprioception only; no simulator truth is read. Values are monocular-depth estimates: no per-object
depth correction is applied (thin or white objects can appear above their true position; see conf and table_z).

With the switch off only enabled() is consulted: tool definitions, handler behaviour and program workspaces are
unchanged.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request

import numpy as np

KIND = "scene3d"
SOURCE = "vggt_scene3d"
DEFAULT_CAMERA = "cam_head"
ARTIFACT_KIND = "scene3d_points"
ARTIFACT_MIME = "application/x-npz"
WORKSPACE_MODULE = "scene3d_helpers.py"

KIND_SENTENCE = "scene3d (default cam_head): VGGT point cloud of one RGB image in env_origin m for scene3d_helpers in program_run (or in-process via queries)."
FIELD_DESCRIPTIONS = {"voxel_m": "scene3d voxel size in m (default 0.01)."}
USAGE = ("In program_run: import scene3d_helpers as s3; s3.load(artifact). Print only the numbers you need. "
         "Points are VGGT monocular-depth estimates, not measurements, scaled to the table plane; thin, white or transparent parts may sit "
         "above or in front of their true place (check conf and table_z near them). robot=True includes points "
         "within 6 cm of the arm (e.g. a held object).")
API = [
    "load(artifact) -> cloud; later calls use the last loaded cloud",
    "info() -> n, bbox, table_z, voxel_m",
    "table_z(xy=None, radius=0.08) -> global table plane z, or local median z of near-table points around xy",
    "points_in_box(lo, hi) / points_in_cylinder(center_xy, radius, z_min=None, z_max=None) -> n, centroid, min, max, top_z, conf_p50",
    "top_z(region, q=98) -> q-th percentile z of points above the table in region; region = (lo, hi) | "
    "{'center':[x,y],'radius':r} | {'polygon':[[x,y],...]} (optional 'z_min','z_max')",
    "nearest(p, k=5) -> k nearest points to p ([x,y] horizontal or [x,y,z]): xyz, dist, conf, uv",
    "cluster_at_pixel(u, v, radius=0.015) -> connected points above the table around the cloud point seen nearest "
    "to pixel (u,v): n, centroid, min, max, top_z, bottom_z, seed",
    "in_hand_axis(grasp_center_m, radius=0.065) -> 3D PCA principal axis (yaw_deg, tilt_deg, posture, elongation) "
    "and bottom_point_m / tip_offset_from_grasp_m of held-object points near the gripper (uses robot=True)",
    "surface_extrema(region, radius=0.04) -> unfiltered local 3D surface floor_z, rim_z, lowest_point_m, highest_point_m",
    "triangulate_rays(u1, v1, K1, T1, u2, v2, K2, T2) -> stereo 3D point_m and skew_gap_m from two calibrated views",
    "virtual_field(objects=None, cloud=None, table_z=None) -> VirtualSpatialField for 3D OBB grasp/stage simulation",
    "evaluate_grasp(arm, position_m, angles_deg, target_id=None) -> theoretically_graspable, vertical_bite_m, lateral_eccentricity_m, colliding_neighbor_ids, optimal_grasp_pose",
    "solve_optimal_grasp(target_id, arm='right') -> collision-free position_m, angles_deg, safe_lift_z_m, neighbor_clearance_m",
    "simulate_stages(stages) -> status, theoretically_valid, min_path_clearance_m, stage_reports, virtual_world_after",
    "plan_pick_and_place(target_id, place_position_m, arm='right') -> 6-stage collision-checked stages and simulation",
    "common keyword args: min_conf=0.1 (confidence rank 0-1), robot=False, xyz=False (True also returns the points)",
]


def enabled(runtime) -> bool:
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "scene3d", False))


def applies(runtime, request):
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    active = enabled(runtime) or bool(getattr(features, "scene3d_observe", False))
    return active and isinstance(request, dict) and request.get("kind") == KIND


def workspace_modules(runtime):
    """Files copied into each program job workspace when the switch (or features.scene3d_observe) is on."""
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return (WORKSPACE_MODULE,) if enabled(runtime) or bool(getattr(features, "scene3d_observe", False)) else ()


def schema_branch():
    return dict(type="object", properties=dict(
        kind=dict(type="string", const=KIND), observation_id=dict(type="string", minLength=1),
        camera_id=dict(type="string", minLength=1), voxel_m=dict(type="number", minimum=0.005, maximum=0.03),
        artifact=dict(type="string", minLength=1),
        queries=dict(type="array", minItems=1, maxItems=32, items=dict(type="object"))),
        required=["kind"], minProperties=2, additionalProperties=False)


def extend_definition(geometry_schema, descriptions):
    """Model-facing schema change, applied only when the switch is on."""
    geometry_schema["oneOf"].append(schema_branch())
    return descriptions


def describe_branch(geometry_schema):
    for branch in geometry_schema["oneOf"]:
        if branch.get("properties", {}).get("kind", {}).get("const") == KIND:
            branch["properties"]["kind"]["description"] = KIND_SENTENCE
            for field, text in FIELD_DESCRIPTIONS.items():
                branch["properties"][field]["description"] = text


class ServiceError(RuntimeError):
    pass


def query_service(endpoint, body, timeout_s, slack_s=5.0):
    request = urllib.request.Request(endpoint.rstrip("/") + "/points", data=json.dumps(body).encode(),
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
    if not isinstance(answer.get("npz_b64"), str):
        raise ServiceError("missing_points")
    return answer


def _r(values, digits=3):
    return None if values is None else [round(float(v), digits) for v in values]


def _resolve_cloud_artifact(runtime, ref):
    """解析点云 artifact 引用（支持完整 artifact:<sha256>-... 或短别名 artifact:ref_N / ref_N）。"""
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("invalid_scene3d_artifact_ref")
    store = getattr(runtime, "store", None)
    scene = getattr(runtime, "scene", None)
    resolver = getattr(store, "resolve_artifact", None) or getattr(scene, "resolve_artifact", None)
    if resolver is None:
        raise ValueError("scene3d_artifact_resolver_unavailable")
    candidate = ref.strip()
    try:
        info, data = resolver(candidate)
    except Exception:
        short = candidate.split(":", 1)[-1] if candidate.startswith("artifact:") else candidate
        read_state = getattr(store, "read_state", None)
        record = read_state("actor_reference", short) if callable(read_state) else None
        target = (record.get("reference") or record.get("value")) if isinstance(record, dict) else None
        if not isinstance(target, str) or not target:
            raise ValueError(f"unknown_scene3d_artifact:{ref}") from None
        info, data = resolver(target)
    return data


def _extract_region(q):
    if "region" in q and q["region"] is not None:
        return q["region"]
    if "lo" in q and "hi" in q:
        return (q["lo"], q["hi"])
    region = {k: q[k] for k in ("center", "radius", "polygon", "z_min", "z_max") if k in q}
    if "center" not in region and ("center_xy" in q or "xy" in q):
        region["center"] = q.get("center_xy") if "center_xy" in q else q.get("xy")
    return region


def _resolve_observation_id(runtime, observation_id=None, q=None):
    if isinstance(q, dict) and q.get("observation_id"):
        return q["observation_id"]
    if observation_id:
        return observation_id
    latest = getattr(runtime, "latest", None)
    if isinstance(latest, dict):
        if latest.get("observation_id"):
            return latest["observation_id"]
        obs = latest.get("observation")
        if isinstance(obs, dict) and obs.get("observation_id"):
            return obs["observation_id"]
        if isinstance(obs, str) and obs:
            return obs
    store = getattr(runtime, "store", None)
    records_fn = getattr(store, "records", None)
    if callable(records_fn):
        try:
            obs_list = records_fn("observation")
            if obs_list and isinstance(obs_list[-1], dict):
                return obs_list[-1].get("observation_id")
        except Exception:
            pass
    return None


def _resolve_arm_grasp_center(runtime, arm, observation_id=None, q=None):
    obs_id = _resolve_observation_id(runtime, observation_id=observation_id, q=q)
    obs = None
    latest = getattr(runtime, "latest", None)
    latest_obs = None
    if isinstance(latest, dict):
        latest_obs = latest if ("observation_id" in latest or "robot_state" in latest or "robot_state_ref" in latest) else latest.get("observation")
    if obs_id and getattr(runtime, "scene", None) is not None:
        try:
            obs = runtime.scene.observation(runtime.identity, obs_id)
        except Exception:
            obs = None
    if not isinstance(obs, dict):
        obs = latest_obs
    if not isinstance(obs, dict):
        return None
    robot_state = obs.get("robot_state")
    if not isinstance(robot_state, dict) and isinstance(latest_obs, dict) and isinstance(latest_obs.get("robot_state"), dict):
        robot_state = latest_obs["robot_state"]
    if not isinstance(robot_state, dict) and obs.get("robot_state_ref"):
        resolve_json = getattr(getattr(runtime, "scene", None), "resolve_json", None)
        if callable(resolve_json):
            try:
                robot_state = resolve_json(obs["robot_state_ref"])
            except Exception:
                robot_state = None
        if not isinstance(robot_state, dict):
            store = getattr(runtime, "store", None)
            resolve_art = getattr(store, "resolve_artifact", None)
            if callable(resolve_art):
                try:
                    _, state_bytes = resolve_art(obs["robot_state_ref"])
                    robot_state = json.loads(state_bytes)
                except Exception:
                    robot_state = None
    arm_state = ((robot_state or {}).get("arms") or {}).get(arm) or {}
    gp = (arm_state.get("grasp_pose") or {}).get("position_m")
    if gp is not None:
        return gp
    ee = arm_state.get("ee_pose") or {}
    pos = ee.get("position_m")
    quat = ee.get("quaternion_xyzw")
    if isinstance(pos, (list, tuple)) and len(pos) == 3 and isinstance(quat, (list, tuple)) and len(quat) == 4:
        try:
            from ...robot.model import roboprobe_pose
            x, y, z, w = quat
            vals = roboprobe_pose.pose_to_values([*pos, w, x, y, z])
            return [vals["x"], vals["y"], vals["z"]]
        except Exception:
            pass
    return pos


def _execute_one_cloud_query(runtime, cloud, q, observation_id=None):
    from . import scene3d_helpers as s3
    if not isinstance(q, dict):
        raise ValueError("scene3d_query_item_must_be_object")
    op = str(q.get("op") or q.get("fn") or "info").strip()
    common = {}
    if "min_conf" in q:
        common["min_conf"] = float(q["min_conf"])
    if "robot" in q:
        common["robot"] = bool(q["robot"])
    if "xyz" in q:
        common["xyz"] = bool(q["xyz"])
    if op == "info":
        return dict(op=op, **cloud.info())
    if op == "table_z":
        xy = q.get("xy") if "xy" in q else (q.get("center") if "center" in q else q.get("point"))
        radius = float(q.get("radius", 0.08))
        res = cloud.table_z(xy=xy, radius=radius)
        return dict(op=op, **(res if isinstance(res, dict) else {"table_z": res}))
    if op == "points_in_box":
        lo = q.get("lo") if "lo" in q else (q.get("min_m") if "min_m" in q else q.get("min"))
        hi = q.get("hi") if "hi" in q else (q.get("max_m") if "max_m" in q else q.get("max"))
        if lo is None or hi is None:
            raise ValueError("points_in_box_requires_lo_and_hi")
        return dict(op=op, **cloud.points_in_box(lo, hi, **common))
    if op == "points_in_cylinder":
        center = q.get("center_xy") if "center_xy" in q else (q.get("center") if "center" in q else q.get("xy"))
        if center is None or "radius" not in q:
            raise ValueError("points_in_cylinder_requires_center_and_radius")
        return dict(op=op, **cloud.points_in_cylinder(
            center, float(q["radius"]), z_min=q.get("z_min"), z_max=q.get("z_max"), **common))
    if op == "top_z":
        region = _extract_region(q)
        if not region:
            raise ValueError("top_z_requires_region")
        pct = float(q.get("q", 98))
        return dict(op=op, **cloud.top_z(
            region, q=pct, min_conf=float(q.get("min_conf", 0.1)), robot=bool(q.get("robot", False))))
    if op == "nearest":
        p = q.get("p") if "p" in q else (q.get("point") if "point" in q else (q.get("xyz") if "xyz" in q else (q.get("xy") if "xy" in q else q.get("center"))))
        if p is None:
            raise ValueError("nearest_requires_point_p")
        return dict(op=op, **cloud.nearest(
            p, k=int(q.get("k", 5)), min_conf=float(q.get("min_conf", 0.1)), robot=bool(q.get("robot", False))))
    if op == "cluster_at_pixel":
        pixel = q.get("pixel") if "pixel" in q else q.get("uv")
        if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
            u, v = pixel[0], pixel[1]
        elif "u" in q and "v" in q:
            u, v = q["u"], q["v"]
        else:
            raise ValueError("cluster_at_pixel_requires_pixel_or_u_v")
        return dict(op=op, **cloud.cluster_at_pixel(u, v, radius=float(q.get("radius", 0.015)), **common))
    if op in ("in_hand_axis", "in_hand_geometry"):
        center = (q.get("grasp_center_m") if "grasp_center_m" in q else
                  (q.get("grasp_position_m") if "grasp_position_m" in q else
                   (q.get("center") if "center" in q else q.get("p"))))
        if center is None and q.get("arm") in ("left", "right"):
            center = _resolve_arm_grasp_center(runtime, q["arm"], observation_id=observation_id, q=q)
        if center is None:
            raise ValueError("in_hand_axis_requires_grasp_center_m_or_arm")
        return dict(op=op, **cloud.in_hand_axis(
            center, radius=float(q.get("radius", 0.065)), min_conf=float(q.get("min_conf", 0.05)),
            above=float(q.get("above", q.get("z_clearance_above_table", 0.015))), xyz=bool(q.get("xyz", False))))
    if op == "surface_extrema":
        region = _extract_region(q)
        if not region:
            region = q.get("center") if "center" in q else (q.get("xy") if "xy" in q else q.get("p"))
        if region is None:
            raise ValueError("surface_extrema_requires_region_or_center")
        return dict(op=op, **cloud.surface_extrema(
            region, radius=float(q.get("radius", 0.04)), min_conf=float(q.get("min_conf", 0.0)),
            robot=bool(q.get("robot", False)), xyz=bool(q.get("xyz", False))))
    if op == "triangulate_rays":
        p1 = q.get("pixel_a") or q.get("uv1") or ([q["u1"], q["v1"]] if "u1" in q and "v1" in q else None)
        p2 = q.get("pixel_b") or q.get("uv2") or ([q["u2"], q["v2"]] if "u2" in q and "v2" in q else None)
        if not (isinstance(p1, (list, tuple)) and len(p1) == 2 and isinstance(p2, (list, tuple)) and len(p2) == 2):
            raise ValueError("triangulate_rays_requires_pixel_pairs")
        K1, T1, K2, T2 = q.get("K1"), q.get("T1"), q.get("K2"), q.get("T2")
        eff_obs_id = _resolve_observation_id(runtime, observation_id=observation_id, q=q)
        if (K1 is None or T1 is None or K2 is None or T2 is None) and eff_obs_id:
            from ...robot.model.geometry import GeometryTools
            tools = GeometryTools(runtime.scene, settings=runtime.image_settings)
            full = {"observation_id": eff_obs_id, **runtime.identity, "schema_version": "0.1"}
            c1 = tools.camera(full, eff_obs_id, q.get("camera_id") or DEFAULT_CAMERA)
            c2 = tools.camera(full, eff_obs_id, q.get("camera_id_b") or q.get("camera_id_2") or "cam_right_wrist")
            K1, T1, K2, T2 = c1["K"], c1["T"], c2["K"], c2["T"]
        if K1 is None or T1 is None or K2 is None or T2 is None:
            raise ValueError("triangulate_rays_requires_K_T_or_observation_cameras")
        return dict(op=op, **s3.triangulate_rays(p1[0], p1[1], K1, T1, p2[0], p2[1], K2, T2))
    if op in ("evaluate_grasp", "solve_optimal_grasp", "simulate_stages", "plan_pick_and_place"):
        eff_obs_id = _resolve_observation_id(runtime, observation_id=observation_id, q=q)
        objs = q.get("objects") or q.get("object_table")
        if objs is None and eff_obs_id:
            from .pose_skill import attached_table
            objs = attached_table(runtime, eff_obs_id)
        vf = getattr(runtime, "_virtual_spatial_field", None) if objs is None else None
        if vf is None:
            vf = s3.VirtualSpatialField(objects=objs, cloud=cloud)
        if op == "evaluate_grasp":
            return dict(op=op, **vf.evaluate_grasp(
                arm=str(q.get("arm") or "right"),
                position_m=q.get("position_m") or q.get("grasp_position_m"),
                angles_deg=q.get("angles_deg") or [0.0, 0.0, 0.0],
                target_id=q.get("target_id") or q.get("object"),
                open_aperture_m=float(q.get("open_aperture_m", 0.0888)),
                contact_region=str(q.get("contact_region") or "pad_center"),
                contact_depth_m=float(q["contact_depth_m"]) if q.get("contact_depth_m") is not None else None,
            ))
        if op == "solve_optimal_grasp":
            tid = q.get("target_id") or q.get("object")
            if not tid:
                raise ValueError("solve_optimal_grasp_requires_target_id")
            return dict(op=op, **vf.solve_optimal_grasp(
                target_id=tid,
                arm=str(q.get("arm") or "right"),
                preferred_yaw_deg=q.get("preferred_yaw_deg"),
                open_aperture_m=float(q.get("open_aperture_m", 0.0888)),
                contact_region=str(q.get("contact_region") or "pad_center"),
                contact_depth_m=float(q["contact_depth_m"]) if q.get("contact_depth_m") is not None else None,
            ))
        if op == "simulate_stages":
            stages = q.get("stages")
            if not isinstance(stages, list):
                raise ValueError("simulate_stages_requires_stages_list")
            return dict(op=op, **vf.simulate_stages(stages, step_m=float(q.get("step_m", 0.015))))
        if op == "plan_pick_and_place":
            tid = q.get("target_id") or q.get("object")
            place_pos = q.get("place_position_m") or q.get("position_m")
            if not tid or not place_pos:
                raise ValueError("plan_pick_and_place_requires_target_id_and_place_position_m")
            return dict(op=op, **vf.plan_pick_and_place(
                target_id=tid,
                place_position_m=place_pos,
                arm=str(q.get("arm") or "right"),
                place_angles_deg=q.get("place_angles_deg"),
                approach_clearance_m=float(q.get("approach_clearance_m", 0.06)),
            ))
    raise ValueError(f"unsupported_scene3d_op:{op}")


def execute_cloud_queries(runtime, data_bytes, raw_queries, observation_id=None):
    from . import scene3d_helpers as s3
    cloud = data_bytes if isinstance(data_bytes, s3.Cloud) else s3.from_bytes(data_bytes)
    results = []
    for item in raw_queries:
        try:
            results.append(_execute_one_cloud_query(runtime, cloud, item, observation_id=observation_id))
        except Exception as error:
            results.append(dict(op=(item.get("op") if isinstance(item, dict) else None),
                                status="error", error=str(error)[:200]))
    return cloud, results


def scene3d_query(runtime, request, *, timeout_s=None, slack_s=5.0):
    """Returns (result, npz bytes or None). The caller stores the bytes and fills value.artifact."""
    from ...robot.model.geometry import GeometryTools
    from .object_table import robot_segments, LINK_RADIUS_M
    features = runtime.config.features
    request = dict(request)
    camera_id = request.get("camera_id") or DEFAULT_CAMERA
    obs_id = request.get("observation_id")
    raw_queries = list(request.get("queries") or ([] if request.get("query") is None else [request["query"]]))
    started = time.monotonic()
    result = {"schema_version": "0.1", **runtime.identity, "world_kind": "internal", "query_type": KIND,
              "observation_ids": [obs_id] if obs_id else [], "status": "supported", "frame_id": "env_origin",
              "unit": "m", "value": None}
    # Fast path: in-process query against an existing scene3d artifact or memoized obs3d point cloud.
    artifact_ref = request.get("artifact")
    if artifact_ref is None and raw_queries and obs_id and camera_id == DEFAULT_CAMERA and not request.get("voxel_m"):
        from .obs3d import _memo
        cached = _memo(runtime, obs_id).get("scene3d")
        if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[0], dict):
            artifact_ref = cached[0].get("artifact")
    if artifact_ref is not None:
        try:
            data = _resolve_cloud_artifact(runtime, artifact_ref)
            cloud, query_results = execute_cloud_queries(runtime, data, raw_queries or [{"op": "info"}], observation_id=obs_id)
            info = cloud.info()
        except Exception as error:
            result.update(status="unknown", reason="scene3d_unavailable:" + str(error)[:200])
            result["duration_s"] = time.monotonic() - started
            return result, None
        value = dict(source=SOURCE, measured=False, camera_id=camera_id, artifact=artifact_ref,
                     points=info["n"], voxel_m=round(float(info["voxel_m"]), 4),
                     table_z=round(float(info["table_z"]), 4), bbox=info["bbox"],
                     query_results=query_results)
        if request.get("query") is not None and len(query_results) == 1:
            value["query_result"] = query_results[0]
        result.update(status="conditional", value=value,
                      assumptions=["VGGT-Omega monocular depth point cloud queried in-process via scene3d_helpers."])
        result["duration_s"] = time.monotonic() - started
        return result, None
    full = {**request, "camera_id": camera_id, **runtime.identity, "schema_version": "0.1"}
    tools = GeometryTools(runtime.scene, settings=runtime.image_settings)
    limit = float(features.vggt_timeout_s if timeout_s is None else timeout_s)
    try:
        observation = runtime.scene.observation(runtime.identity, request["observation_id"])
        camera = tools.camera(full, full["observation_id"], camera_id)
        if np.any(np.abs(camera["D"]) > 1e-12):
            raise ServiceError("distorted_camera_not_supported")
        _, image_bytes = runtime.scene.resolve_artifact(camera["frame"]["rgb_ref"])
        segments = robot_segments(tools, full, observation)
        voxel = float(request.get("voxel_m") or features.scene3d_voxel_m)
        body = dict(image_b64=base64.b64encode(image_bytes).decode("ascii"), K=np.asarray(camera["K"]).tolist(),
                    T_world_camera=np.asarray(camera["T"]).tolist(), support_z_m=float(features.vggt_support_z_m),
                    voxel_m=voxel, robot_segments_m=segments, robot_radius_m=LINK_RADIUS_M,
                    params=dict(z_above=float(features.scene3d_z_above_m), z_below=float(features.scene3d_z_below_m)),
                    timeout_s=limit)
        answer = query_service(features.vggt_endpoint, body, limit, slack_s)
        data = base64.b64decode(answer["npz_b64"])
    except (ServiceError, ValueError, KeyError, TypeError) as error:
        result.update(status="unknown", reason="scene3d_unavailable:" + str(error)[:200])
        result["duration_s"] = time.monotonic() - started
        return result, None
    bbox, above = answer.get("bbox_m"), answer.get("bbox_above_support_m")
    value = dict(source=SOURCE, measured=False, camera_id=camera_id, artifact=None, points=int(answer.get("n_points", 0)),
                 points_above_table=int(answer.get("n_above_support", 0)), near_robot=int(answer.get("n_robot", 0)),
                 voxel_m=round(float(answer.get("voxel_m", voxel)), 4),
                 table_z=round(float(answer.get("support_z_m", features.vggt_support_z_m)), 4),
                 bbox=[_r(bbox[0]), _r(bbox[1])] if bbox else None,
                 bbox_above_table=[_r(above[0]), _r(above[1])] if above else None,
                 usage=USAGE, api=list(API))
    if raw_queries:
        _, query_results = execute_cloud_queries(runtime, data, raw_queries, observation_id=obs_id)
        value["query_results"] = query_results
        if request.get("query") is not None and len(query_results) == 1:
            value["query_result"] = query_results[0]
    result.update(status="conditional", value=value, frame_ids=[camera["frame"]["frame_id"]],
                  assumptions=["VGGT-Omega monocular depth of one image, back-projected with the calibrated camera and "
                               "scaled so the fitted table plane lies at table_z; no per-object correction."])
    result["diagnostics"] = dict(service_version=answer.get("version"), latency_s=time.monotonic() - started,
                                 service_timing=answer.get("timing"), npz_bytes=len(data), cache_hit=answer.get("cache_hit"),
                                 n_pixels_kept=answer.get("n_pixels_kept"), support_fit=answer.get("support_fit"),
                                 scale=answer.get("scale"), robot_segments=len(segments))
    result["duration_s"] = time.monotonic() - started
    return result, data
