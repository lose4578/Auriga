"""Queries on a scene3d point cloud inside program_run (features.scene3d; copied into the job workspace).

    import scene3d_helpers as s3
    s3.load("artifact:ref_3")              # handle returned by geometry_query kind scene3d
    s3.top_z({"center": [0.1, -0.2], "radius": 0.03})

The cloud (npz: xyz float32 env_origin m, conf uint8 depth-confidence rank 0..255, uv uint16 source pixel (column,
row), count uint16, robot bool, support_z, voxel_m, image_size) is a VGGT-Omega monocular estimate scaled to the
table plane. Functions return small dicts of numbers; the points themselves only with xyz=True. No per-object depth
correction is applied here: thin, white or transparent parts can sit above or in front of their true place.
Only numpy is required; load() reads the artifact through AurigaClient.
"""
import io

import numpy as np

API = (  # same text as auriga/scene3d.py API (checked by tests/test_obs3d.py)
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
)
ABOVE_M = 0.004          # "above the table" threshold, same as the VGGT service
NEAR_TABLE_M = 0.015     # |z - table| for the local table height
_DEFAULT = None


class Cloud:
    def __init__(self, data):
        self.xyz = np.asarray(data["xyz"], dtype=np.float64).reshape(-1, 3)
        self.conf = np.asarray(data["conf"], dtype=np.float64).reshape(-1) / 255.0
        self.uv = np.asarray(data["uv"], dtype=np.int64).reshape(-1, 2)
        self.count = np.asarray(data["count"]).reshape(-1) if "count" in data else np.ones(len(self.xyz), np.int64)
        self.robot = np.asarray(data["robot"], dtype=bool).reshape(-1) if "robot" in data else np.zeros(len(self.xyz), bool)
        self.table = float(data["support_z"])
        self.voxel = float(data["voxel_m"]) if "voxel_m" in data else None
        self.image_size = [int(v) for v in data["image_size"]] if "image_size" in data else None

    def mask(self, min_conf=0.1, robot=False):
        keep = self.conf >= min_conf
        if not robot:
            keep &= ~self.robot
        return keep


def _r(value, digits=4):
    if value is None:
        return None
    if np.ndim(value):
        return [round(float(v), digits) for v in np.asarray(value).reshape(-1)]
    return round(float(value), digits)


def from_bytes(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        return Cloud({key: npz[key] for key in npz.files})


def load(artifact, client=None):
    """Load the cloud of a scene3d artifact handle (or raw bytes / Cloud / dict) and make it the default for later calls."""
    global _DEFAULT
    if isinstance(artifact, Cloud):
        _DEFAULT = artifact
        return _DEFAULT
    if isinstance(artifact, (bytes, bytearray, memoryview)):
        _DEFAULT = from_bytes(bytes(artifact))
        return _DEFAULT
    if isinstance(artifact, dict) and "artifact" in artifact:
        artifact = artifact["artifact"]
    if client is None:
        from auriga_client import AurigaClient
        client = AurigaClient()
    data = client.artifact(artifact)
    _DEFAULT = data if isinstance(data, Cloud) else from_bytes(data)
    return _DEFAULT


def use(cloud):
    global _DEFAULT
    _DEFAULT = cloud
    return cloud


def _cloud(cloud):
    if isinstance(cloud, (str, bytes, bytearray, memoryview, dict)):
        return load(cloud)
    cloud = cloud if cloud is not None else _DEFAULT
    if cloud is None:
        raise ValueError("scene3d_cloud_not_loaded: call load(artifact) first")
    return cloud


def info(cloud=None):
    c = _cloud(cloud)
    return dict(n=int(len(c.xyz)), bbox=[_r(c.xyz.min(0)), _r(c.xyz.max(0))] if len(c.xyz) else None,
                table_z=_r(c.table), voxel_m=c.voxel, near_robot=int(c.robot.sum()), image_size=c.image_size)


def _summary(c, keep, xyz=False):
    P = c.xyz[keep]
    out = dict(n=int(len(P)))
    if len(P):
        out.update(centroid=_r(P.mean(0)), min=_r(P.min(0)), max=_r(P.max(0)),
                   top_z=_r(np.percentile(P[:, 2], 98)), conf_p50=_r(np.median(c.conf[keep]), 2))
    else:
        out.update(centroid=None, min=None, max=None, top_z=None, conf_p50=None)
    if xyz:
        out["xyz"] = P
    return out


def _region_mask(c, region):
    """region = (lo, hi) | {'center': [x, y], 'radius': r} | {'polygon': [[x, y], ...]}, optional 'z_min'/'z_max'."""
    P = c.xyz
    if isinstance(region, (tuple, list)) and len(region) == 2 and np.size(region[0]) == 3:
        lo, hi = np.asarray(region[0], float), np.asarray(region[1], float)
        return np.all((P >= lo) & (P <= hi), axis=1)
    if not isinstance(region, dict):
        raise ValueError("region must be (lo, hi), {'center','radius'} or {'polygon'}")
    if "center" in region:
        centre = np.asarray(region["center"], float)[:2]
        keep = np.linalg.norm(P[:, :2] - centre, axis=1) <= float(region["radius"])
    elif "polygon" in region:
        keep = _in_polygon(P[:, :2], np.asarray(region["polygon"], float).reshape(-1, 2))
    elif "min_m" in region and "max_m" in region:
        lo, hi = np.asarray(region["min_m"], float), np.asarray(region["max_m"], float)
        keep = np.all((P[:, :len(lo)] >= lo) & (P[:, :len(hi)] <= hi), axis=1)
    elif "lo" in region and "hi" in region:
        lo, hi = np.asarray(region["lo"], float), np.asarray(region["hi"], float)
        keep = np.all((P[:, :len(lo)] >= lo) & (P[:, :len(hi)] <= hi), axis=1)
    elif "min" in region and "max" in region:
        lo, hi = np.asarray(region["min"], float), np.asarray(region["max"], float)
        keep = np.all((P[:, :len(lo)] >= lo) & (P[:, :len(hi)] <= hi), axis=1)
    else:
        raise ValueError("region dict needs 'center' + 'radius', 'polygon', or 'min_m' + 'max_m'")
    if region.get("z_min") is not None:
        keep &= P[:, 2] >= float(region["z_min"])
    if region.get("z_max") is not None:
        keep &= P[:, 2] <= float(region["z_max"])
    return keep


def _in_polygon(xy, poly):
    """Even-odd rule, vectorised over points."""
    inside = np.zeros(len(xy), bool)
    x, y = xy[:, 0], xy[:, 1]
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xc)
    return inside


def points_in_box(lo, hi=None, z_or_hi=None, cloud=None, min_conf=0.1, robot=False, xyz=False):
    if isinstance(lo, Cloud):
        cloud, lo, hi = lo, hi, z_or_hi
    c = _cloud(cloud)
    return _summary(c, _region_mask(c, (lo, hi)) & c.mask(min_conf, robot), xyz)


def points_in_cylinder(center_xy, radius=0.05, z_min=None, z_max=None, cloud=None, min_conf=0.1, robot=False, xyz=False):
    if isinstance(center_xy, Cloud):
        cloud, center_xy, radius, z_min = center_xy, radius, (0.05 if z_min is None else z_min), z_max
        z_max = None
    c = _cloud(cloud)
    region = dict(center=center_xy, radius=radius, z_min=z_min, z_max=z_max)
    return _summary(c, _region_mask(c, region) & c.mask(min_conf, robot), xyz)


def top_z(region, q=98, cloud=None, min_conf=0.1, robot=False, above=ABOVE_M, xyz=False, radius=None, z_min=None, z_max=None):
    """q-th percentile z of the points above the table (table_z + above) in region; top_z None when there are none."""
    if isinstance(region, Cloud):
        cloud, region, q = region, q, 98
    c = _cloud(cloud)
    if not isinstance(region, dict) and not (isinstance(region, (tuple, list)) and len(region) == 2 and np.size(region[0]) == 3):
        pt = np.asarray(region, dtype=float).reshape(-1)
        if len(pt) in (2, 3):
            rad = float(radius) if radius is not None else (float(q) if isinstance(q, float) and q < 1.0 else 0.05)
            if isinstance(q, float) and q < 1.0 and radius is None:
                q = 98
            region = dict(center=pt[:2].tolist(), radius=rad, z_min=z_min, z_max=z_max)
    elif isinstance(region, dict) and (z_min is not None or z_max is not None):
        region = {**region, **({"z_min": z_min} if z_min is not None else {}), **({"z_max": z_max} if z_max is not None else {})}
    keep = _region_mask(c, region) & c.mask(min_conf, robot) & (c.xyz[:, 2] > c.table + above)
    P = c.xyz[keep]
    out = dict(n=int(len(P)), top_z=_r(np.percentile(P[:, 2], q)) if len(P) else None,
               height_above_table=_r(np.percentile(P[:, 2], q) - c.table) if len(P) else None,
               conf_p50=_r(np.median(c.conf[keep]), 2) if len(P) else None)
    if xyz:
        out["xyz"] = P
    return out


def table_z(xy=None, radius=0.08, cloud=None, min_conf=0.1):
    """Without xy: the table plane z the cloud was scaled to. With xy: median z of the near-table points
    (|z - table| < 1.5 cm) within radius of xy, i.e. the local table height of this reconstruction."""
    if isinstance(xy, Cloud):
        cloud = xy
        xy = radius if isinstance(radius, (list, tuple, np.ndarray)) else None
        if xy is not None:
            radius = 0.08
    c = _cloud(cloud)
    if xy is None:
        return c.table
    keep = (c.mask(min_conf, False) & (np.abs(c.xyz[:, 2] - c.table) < NEAR_TABLE_M)
            & (np.linalg.norm(c.xyz[:, :2] - np.asarray(xy, float)[:2], axis=1) <= radius))
    z = c.xyz[keep, 2]
    return dict(n=int(len(z)), z=_r(np.median(z)) if len(z) else None,
                z_p10_p90=[_r(np.percentile(z, 10)), _r(np.percentile(z, 90))] if len(z) else None)


def nearest(p, k=5, cloud=None, min_conf=0.1, robot=False):
    if isinstance(p, Cloud):
        cloud, p, k = p, k, (cloud if isinstance(cloud, int) else 5)
    c = _cloud(cloud)
    keep = np.flatnonzero(c.mask(min_conf, robot))
    p = np.asarray(p, float).reshape(-1)
    dims = 2 if len(p) == 2 else 3
    d = np.linalg.norm(c.xyz[keep, :dims] - p[:dims], axis=1)
    order = keep[np.argsort(d)[:int(k)]]
    dist = np.linalg.norm(c.xyz[order, :dims] - p[:dims], axis=1)
    return dict(xyz=[_r(v) for v in c.xyz[order]], dist=_r(dist), conf=_r(c.conf[order], 2),
                uv=c.uv[order].tolist())


def cluster_at_pixel(u, v=None, radius=0.015, cloud=None, min_conf=0.1, robot=False, above=ABOVE_M, max_extent=0.4,
                     xyz=False):
    """Points above the table connected (link distance <= radius) to the cloud point whose source pixel is nearest to
    (u, v). Growth stops at max_extent from the seed. Returns n, centroid, min, max, top_z, bottom_z, seed."""
    if isinstance(u, Cloud):
        cloud = u
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2:
            u, v = v[0], v[1]
        else:
            u, v, radius = v, radius, 0.015
    elif isinstance(u, (list, tuple, np.ndarray)) and len(u) == 2 and v is None:
        u, v = u[0], u[1]
    c = _cloud(cloud)
    keep = np.flatnonzero(c.mask(min_conf, robot) & (c.xyz[:, 2] > c.table + above))
    if not len(keep):
        return dict(n=0, seed=None)
    pix = np.linalg.norm(c.uv[keep] - np.asarray([u, v], float), axis=1)
    seed = keep[int(np.argmin(pix))]
    P = c.xyz[keep]
    local = np.linalg.norm(P - c.xyz[seed], axis=1) <= max_extent
    ids, Q = keep[local], P[local]
    member = np.zeros(len(Q), bool)
    member[int(np.flatnonzero(ids == seed)[0])] = True
    frontier = member.copy()
    while frontier.any():
        grown = np.zeros(len(Q), bool)
        for point in Q[frontier]:
            grown |= np.linalg.norm(Q - point, axis=1) <= radius
        frontier = grown & ~member
        member |= grown
    sel = np.zeros(len(c.xyz), bool)
    sel[ids[member]] = True
    out = _summary(c, sel, xyz)
    out["bottom_z"] = _r(np.percentile(c.xyz[sel, 2], 2))
    out["seed"] = dict(xyz=_r(c.xyz[seed]), uv=c.uv[seed].tolist(), pixel_distance=_r(pix.min(), 1))
    return out


def in_hand_axis(grasp_center_m, radius=0.065, cloud=None, min_conf=0.1, above=0.015, xyz=False,
                 z_clearance_above_table=None):
    """基于 robot=True 的指尖邻域 3D 点云簇做主成分分析 (PCA)，估计夹爪中持握物体的实际 3D 主轴姿态。"""
    if z_clearance_above_table is not None:
        above = float(z_clearance_above_table)
    if isinstance(grasp_center_m, Cloud):
        cloud = grasp_center_m
        grasp_center_m = radius
        radius = float(min_conf) if isinstance(min_conf, (int, float)) and min_conf != 0.1 else 0.065
    c = _cloud(cloud)
    center = np.asarray(grasp_center_m, dtype=float).reshape(3)
    keep = (
        c.mask(min_conf, robot=True)
        & (c.xyz[:, 2] > c.table + above)
        & (np.linalg.norm(c.xyz - center[None, :], axis=1) <= float(radius))
    )
    P = c.xyz[keep]
    if len(P) < 5:
        return dict(n=int(len(P)), centroid=None, yaw_deg=None, tilt_deg=None, posture="unknown", elongation=None)
    mean = P.mean(0)
    centered = P - mean
    cov = (centered.T @ centered) / max(len(P) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-12)
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0].copy()
    if axis[2] < 0 or (abs(axis[2]) < 1e-6 and axis[0] < 0):
        axis = -axis
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
    yaw_deg = float((np.degrees(np.arctan2(axis[1], axis[0])) + 90.0) % 180.0 - 90.0)
    elongation = float(np.sqrt(eigvals[0] / eigvals[1]))
    posture = "upright" if tilt_deg < 35.0 else ("lying" if tilt_deg > 55.0 else "tilted")
    z_low_thr = float(np.percentile(P[:, 2], 5))
    low_pts = P[P[:, 2] <= z_low_thr + 1e-6]
    bottom_pt = low_pts.mean(0) if len(low_pts) else P[int(np.argmin(P[:, 2]))]
    tip_offset = bottom_pt - center
    out = _summary(c, keep, xyz)
    out.update(
        axis_world=_r(axis, 4),
        yaw_deg=_r(yaw_deg, 1),
        tilt_deg=_r(tilt_deg, 1),
        elongation=_r(elongation, 2),
        posture=posture,
        bottom_point_m=_r(bottom_pt, 4),
        tip_offset_from_grasp_m=_r(tip_offset, 4),
    )
    return out


in_hand_geometry = in_hand_axis


def surface_extrema(region, radius=0.04, cloud=None, min_conf=0.05, robot=False, xyz=False):
    """无固定高度截断的局部表面 3D 高程极值分析（直接返回槽底/孔位 lowest_point_m 与边缘/最高点 highest_point_m）。"""
    if isinstance(region, Cloud):
        cloud, region = region, radius
        radius = 0.04
    c = _cloud(cloud)
    if not isinstance(region, dict) and not (isinstance(region, (tuple, list)) and len(region) == 2 and np.size(region[0]) == 3):
        pt = np.asarray(region, dtype=float).reshape(-1)
        region = dict(center=pt[:2].tolist(), radius=float(radius))
    keep = _region_mask(c, region) & c.mask(min_conf, robot)
    P = c.xyz[keep]
    if not len(P):
        return dict(n=0, floor_z=None, rim_z=None, relief_m=None, lowest_point_m=None, highest_point_m=None)
    floor_z = float(np.percentile(P[:, 2], 5))
    rim_z = float(np.percentile(P[:, 2], 95))
    low_band = P[P[:, 2] <= floor_z + 0.003]
    high_band = P[P[:, 2] >= rim_z - 0.003]
    low_pt = low_band.mean(0) if len(low_band) else P[int(np.argmin(P[:, 2]))]
    high_pt = high_band.mean(0) if len(high_band) else P[int(np.argmax(P[:, 2]))]
    out = dict(
        n=int(len(P)),
        floor_z=_r(floor_z, 4),
        rim_z=_r(rim_z, 4),
        relief_m=_r(rim_z - floor_z, 4),
        lowest_point_m=_r(low_pt, 4),
        highest_point_m=_r(high_pt, 4),
        table_z=_r(c.table, 4),
    )
    if xyz:
        out["xyz"] = P
    return out


def triangulate_rays(u1, v1, K1, T1, u2, v2, K2, T2):
    """双目校准相机像素射线三角化：返回三维交点中点 point_m、双射线异面最小间距 skew_gap_m 及各相机深度。"""
    K1, T1 = np.asarray(K1, dtype=float).reshape(3, 3), np.asarray(T1, dtype=float).reshape(4, 4)
    K2, T2 = np.asarray(K2, dtype=float).reshape(3, 3), np.asarray(T2, dtype=float).reshape(4, 4)
    o1, o2 = T1[:3, 3], T2[:3, 3]
    d1 = T1[:3, :3] @ np.linalg.inv(K1) @ np.array([float(u1), float(v1), 1.0])
    d2 = T2[:3, :3] @ np.linalg.inv(K2) @ np.array([float(u2), float(v2), 1.0])
    d1 = d1 / max(float(np.linalg.norm(d1)), 1e-12)
    d2 = d2 / max(float(np.linalg.norm(d2)), 1e-12)
    w0 = o1 - o2
    b = float(np.dot(d1, d2))
    denom = 1.0 - b * b
    if denom < 1e-8:
        return dict(status="parallel_rays", point_m=None, skew_gap_m=None)
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    mid = 0.5 * (p1 + p2)
    gap = float(np.linalg.norm(p1 - p2))
    return dict(
        status="ok" if (s > 0 and t > 0) else "behind_camera",
        point_m=_r(mid, 4),
        ray1_point_m=_r(p1, 4),
        ray2_point_m=_r(p2, 4),
        skew_gap_m=_r(gap, 4),
        depth1_m=_r(s, 4),
        depth2_m=_r(t, 4),
    )


Cloud.info = lambda self: info(cloud=self)
Cloud.points_in_box = lambda self, *a, **kw: points_in_box(*a, cloud=self, **kw)
Cloud.points_in_cylinder = lambda self, *a, **kw: points_in_cylinder(*a, cloud=self, **kw)
Cloud.top_z = lambda self, *a, **kw: top_z(*a, cloud=self, **kw)
Cloud.table_z = lambda self, *a, **kw: table_z(*a, cloud=self, **kw)
Cloud.nearest = lambda self, *a, **kw: nearest(*a, cloud=self, **kw)
Cloud.cluster_at_pixel = lambda self, *a, **kw: cluster_at_pixel(*a, cloud=self, **kw)
Cloud.in_hand_axis = lambda self, *a, **kw: in_hand_axis(*a, cloud=self, **kw)
Cloud.in_hand_geometry = Cloud.in_hand_axis
Cloud.surface_extrema = lambda self, *a, **kw: surface_extrema(*a, cloud=self, **kw)

try:
    from .virtual_field import (
        OBB3D,
        VirtualObject,
        VirtualSpatialField,
        evaluate_grasp as _vf_evaluate_grasp,
        gripper_frame,
        gripper_obbs,
        obb_signed_distance,
        plan_pick_and_place as _vf_plan_pick_and_place,
        simulate_stages as _vf_simulate_stages,
        solve_optimal_grasp as _vf_solve_optimal_grasp,
        virtual_field as _vf_virtual_field,
    )
except ImportError:
    from virtual_field import (
        OBB3D,
        VirtualObject,
        VirtualSpatialField,
        evaluate_grasp as _vf_evaluate_grasp,
        gripper_frame,
        gripper_obbs,
        obb_signed_distance,
        plan_pick_and_place as _vf_plan_pick_and_place,
        simulate_stages as _vf_simulate_stages,
        solve_optimal_grasp as _vf_solve_optimal_grasp,
        virtual_field as _vf_virtual_field,
    )


def virtual_field(objects=None, cloud=None, table_z=None, robot_state=None):
    """构建或更新 3D 虚拟空间数值模拟场 (VirtualSpatialField)，默认关联当前加载的 scene3d 点云。"""
    eff_cloud = cloud if cloud is not None else _DEFAULT
    return _vf_virtual_field(objects=objects, cloud=eff_cloud, table_z=table_z, robot_state=robot_state)


def evaluate_grasp(arm="right", position_m=None, angles_deg=None, target_id=None, open_aperture_m=0.0888,
                   contact_region="pad_center", objects=None, field=None):
    """评估单次 3D 抓取位姿的理论可夹持性、竖直咬合高度、横向偏心距与周围邻近物体干涉。"""
    return _vf_evaluate_grasp(
        arm=arm, position_m=position_m, angles_deg=angles_deg, target_id=target_id,
        open_aperture_m=open_aperture_m, contact_region=contact_region, objects=objects, field=field,
    )


def solve_optimal_grasp(target_id, arm="right", preferred_yaw_deg=None, open_aperture_m=0.0888, objects=None, field=None):
    """求解目标物体无碰撞、最大咬合、最小偏心的最优 3D 抓取位姿。"""
    return _vf_solve_optimal_grasp(
        target_id=target_id, arm=arm, preferred_yaw_deg=preferred_yaw_deg,
        open_aperture_m=open_aperture_m, objects=objects, field=field,
    )


def simulate_stages(stages, step_m=0.015, objects=None, field=None):
    """在虚拟空间场内按顺序动态推演多阶段动作序列，检测路径碰撞并预测执行后物体新位置。"""
    return _vf_simulate_stages(stages=stages, step_m=step_m, objects=objects, field=field)


def plan_pick_and_place(target_id, place_position_m, arm="right", place_angles_deg=None,
                        approach_clearance_m=0.06, objects=None, field=None):
    """在虚拟空间场内自动规划并验证完整的 6 阶段抓取-抬升-搬运-放置-撤离动作列表。"""
    return _vf_plan_pick_and_place(
        target_id=target_id, place_position_m=place_position_m, arm=arm,
        place_angles_deg=place_angles_deg, approach_clearance_m=approach_clearance_m,
        objects=objects, field=field,
    )


Cloud.virtual_field = lambda self, objects=None, table_z=None, robot_state=None: virtual_field(
    objects=objects, cloud=self, table_z=table_z, robot_state=robot_state
)
