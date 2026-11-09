"""基于 SAM + VGGT 与机械臂物理量的虚拟空间数值模拟场 (Virtual Spatial Simulation Field)。

本模块构建一个纯数值（基于 NumPy，零物理渲染开销、毫秒级运算）、随机械臂移动实时演化的三维空间模拟场：
1. 精确建模 ARX X5 双臂末端执行器与平行夹爪三维几何（法兰偏移 0.1501m、指垫深 15mm、指宽 18mm、指厚 8mm、指长 70mm、最大开度 88.8mm）与场景物体 3D 定向包围盒 (OBB)；
2. 支持抓取几何可夹持性判定 (`evaluate_grasp`)：检测张开双指与目标/邻近物体的三维实体干涉、闭合扫掠体积、15mm 夹持面竖直咬合高度 `vertical_bite_m`、横向偏心距 `lateral_eccentricity_m` 与支撑台面净空 `support_clearance_m`；
3. 支持零碰撞最优抓取位姿求解 (`solve_optimal_grasp`)：围绕目标主轴正交方向与主轴偏移向量化搜索不撞邻物、不撞桌面、咬合最大且偏心最小的 3D 抓取位姿；
4. 支持多阶段动作序列动态推演 (`simulate_stages`)：在虚拟场副本中模拟抓起绑定、持物随臂移动、路径三维碰撞检测（自动给出最小安全搬运高度 `recommended_safe_z_m`）以及开爪释放后的虚拟世界状态更新；
5. 支持随真实观测与机械臂位姿实时迭代更新 (`update_from_observation`)。

所有输入均来自受控观测 (`object_table`, `scene3d`, `robot_state`)，绝不读取仿真器真值，也绝不绕过 `Harness.execute` 发送物理指令。
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np

# ------------------------------------------------------------------ 机械臂末端与夹爪物理常量 (单位: m, deg)
# 法兰 (link6) 到指尖 15mm 夹持面中心 (pad_center) 的轴向距离 (m)
GRASP_POINT_OFFSET_M: float = 0.1501
# 远端扁平夹持面沿工具接近轴的深度 (m)；指尖位于 pad_center 前方 +0.0075 m
PAD_DEPTH_M: float = 0.015
FINGERTIP_BELOW_PAD_M: float = 0.0075
# 手指沿切向（垂直于接近轴与开合轴）的宽度 (m)
FINGER_WIDTH_M: float = 0.018
# 单侧手指沿开合轴的厚度 (m)
FINGER_THICKNESS_M: float = 0.008
# 手指从指尖向手掌方向的暴露长度 (m)
FINGER_LENGTH_M: float = 0.070
# 手掌壳体几何尺寸 (m)
PALM_LENGTH_M: float = 0.065
PALM_WIDTH_M: float = 0.096
PALM_THICKNESS_M: float = 0.056
# 平行夹爪最大指间内侧开度 (m)
MAX_OPENING_M: float = 0.0888
# 细长物体长宽比阈值
ELONGATED_RATIO: float = 1.5
# 默认工作台高度 (m)
DEFAULT_TABLE_Z_M: float = 0.765
# 最小有效竖直咬合高度 (m)
MIN_VERTICAL_BITE_M: float = 0.0025
# 接触截面预设偏移 (m)，正方向为从 pad_center 沿接近轴反方向朝向手掌
CONTACT_DEPTH_PRESETS_M: dict[str, float] = {
    "pad_center": 0.0,
    "finger_middle": 0.0275,
    "fingertip": -0.0075,
}
# 竖直向下基准姿态四元数 (w, x, y, z)，此时接近轴为世界 -Z，夹爪开合轴为世界 X
TOP_DOWN_QUAT_WXYZ: tuple[float, float, float, float] = (0.5, -0.5, 0.5, 0.5)
_REFERENCE_QUAT = np.asarray(TOP_DOWN_QUAT_WXYZ, dtype=np.float64)

DEFAULT_ARM_POSES: dict[str, dict[str, list[float]]] = {
    "left": {"position_m": [-0.25, -0.25, 0.92], "angles_deg": [0.0, 0.0, 0.0]},
    "right": {"position_m": [0.25, -0.25, 0.92], "angles_deg": [0.0, 0.0, 0.0]},
}


def _r(value, digits: int = 4):
    """数值与数组四舍五入辅助函数。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [round(float(v), digits) for v in np.asarray(value, dtype=float).reshape(-1)]
    return round(float(value), digits)


def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def wrap180(angle_deg: float) -> float:
    """将角度归一化到 [-180, 180) 度。"""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def wrap90(angle_deg: float) -> float:
    """将无向主轴角度归一化到 [-90, 90) 度。"""
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def axis_difference_deg(a_deg: float, b_deg: float) -> float:
    """计算两个水平无向轴之间的夹角，范围 [0, 90] 度。"""
    return abs(wrap90(float(a_deg) - float(b_deg)))


def closure_yaws_deg(object_yaw_deg: float) -> list[float]:
    """计算垂直横跨物体主轴 (yaw_deg +/- 90°) 的两个等价夹爪偏航角 (deg)。"""
    return [round(wrap180(float(object_yaw_deg) + 90.0), 1), round(wrap180(float(object_yaw_deg) - 90.0), 1)]


# ------------------------------------------------------------------ 三维姿态与旋转矩阵运算
def _axis_quat(axis: int, radians: float) -> np.ndarray:
    half = 0.5 * float(radians)
    q = np.zeros(4, dtype=np.float64)
    q[0] = math.cos(half)
    q[axis + 1] = math.sin(half)
    return q


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """四元数乘法 (w, x, y, z)。"""
    w0, x0, y0, z0 = (float(v) for v in left)
    w1, x1, y1, z1 = (float(v) for v in right)
    return np.asarray(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=np.float64,
    )


def quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def quat_normalize(quat_wxyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(arr))
    return arr / max(norm, 1e-12)


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """将单位四元数 (w, x, y, z) 转换为 3x3 旋转矩阵（列向量为刚体坐标轴）。"""
    w, x, y, z = quat_normalize(quat_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def angles_deg_to_quat_wxyz(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """将 angles_deg=[pitch_deg, roll_deg, yaw_deg] 转换为法兰四元数 (w, x, y, z)，严格对齐 roboprobe_pose。"""
    rot = _axis_quat(0, math.radians(float(pitch_deg)))
    rot = quat_multiply(_axis_quat(1, math.radians(float(roll_deg))), rot)
    rot = quat_multiply(_axis_quat(2, math.radians(float(yaw_deg))), rot)
    return quat_multiply(rot, _REFERENCE_QUAT)


def quat_wxyz_to_angles_deg(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """从法兰四元数 (w, x, y, z) 恢复 (pitch_deg, roll_deg, yaw_deg)。"""
    rel = quat_multiply(quat_normalize(quat_wxyz), quat_conjugate(_REFERENCE_QUAT))
    mat = quat_wxyz_to_matrix(rel)
    roll = math.asin(float(np.clip(-mat[2, 0], -1.0, 1.0)))
    if abs(math.cos(roll)) > 1e-7:
        pitch = math.atan2(float(mat[2, 1]), float(mat[2, 2]))
        yaw = math.atan2(float(mat[1, 0]), float(mat[0, 0]))
    else:
        yaw = 0.0
        sign = 1.0 if mat[2, 0] < 0 else -1.0
        pitch = math.atan2(sign * float(mat[0, 1]), float(mat[1, 1]))
    return (math.degrees(pitch) + 0.0, math.degrees(roll) + 0.0, math.degrees(yaw) + 0.0)


def gripper_frame(
    position_m,
    angles_deg,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
) -> dict:
    """根据请求抓取位姿计算夹爪在世界系下的正交标架与关键几何参考点。

    参数:
        position_m: 长度为 3 的目标位置 [x, y, z]，单位 m (env_origin)。
        angles_deg: 长度为 3 的姿态角 [pitch_deg, roll_deg, yaw_deg]，单位 deg。
        contact_region: 接触截面名称 ('pad_center' | 'fingertip' | 'finger_middle')，默认 'pad_center'。
        contact_depth_m: 可选自定义接触深度偏移，单位 m；若提供则覆盖 contact_region。

    返回:
        包含 pad_center_m, fingertip_center_m, flange_position_m, approach_dir, closing_dir, tangent_dir, axes 的字典。
    """
    pos = np.asarray(position_m, dtype=np.float64).reshape(3)
    ang = np.asarray(angles_deg, dtype=np.float64).reshape(3)
    quat = angles_deg_to_quat_wxyz(ang[0], ang[1], ang[2])
    R = quat_wxyz_to_matrix(quat)
    # R[:, 0] 为法兰 +X（即接近轴，[0,0,0] 时指向世界 -Z）
    # -R[:, 1] 为两指张合轴（[0,0,yaw] 时指向 [cos(yaw), sin(yaw), 0]）
    # R[:, 2] 为手指宽度切向轴（[0,0,yaw] 时指向 [-sin(yaw), cos(yaw), 0]）
    approach_dir = R[:, 0].copy()
    closing_dir = -R[:, 1].copy()
    tangent_dir = R[:, 2].copy()
    depth = float(contact_depth_m) if contact_depth_m is not None else float(CONTACT_DEPTH_PRESETS_M.get(contact_region, 0.0))
    pad_center = pos + depth * approach_dir
    fingertip_center = pad_center + FINGERTIP_BELOW_PAD_M * approach_dir
    flange_pos = pad_center - GRASP_POINT_OFFSET_M * approach_dir
    axes = np.column_stack([approach_dir, closing_dir, tangent_dir])
    return dict(
        pad_center_m=pad_center,
        fingertip_center_m=fingertip_center,
        flange_position_m=flange_pos,
        approach_dir=approach_dir,
        closing_dir=closing_dir,
        tangent_dir=tangent_dir,
        axes=axes,
        quat_wxyz=quat,
    )


# ------------------------------------------------------------------ 三维定向包围盒 (3D OBB) 与碰撞/距离计算
class OBB3D:
    """三维定向包围盒 (Oriented Bounding Box)。"""

    def __init__(self, center_m, axes, half_extents_m, label: str = ""):
        self.center = np.asarray(center_m, dtype=np.float64).reshape(3)
        self.axes = np.asarray(axes, dtype=np.float64).reshape(3, 3)
        self.half_extents = np.maximum(np.asarray(half_extents_m, dtype=np.float64).reshape(3), 1e-5)
        self.label = str(label)

    def vertices(self) -> np.ndarray:
        """返回 OBB 的 8 个三维顶点坐标 (8, 3)。"""
        signs = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        scaled = signs * self.half_extents[None, :]
        return self.center[None, :] + scaled @ self.axes.T

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """返回世界系下的轴对齐外包盒 (min_xyz, max_xyz)。"""
        ext = np.abs(self.axes) @ self.half_extents
        return self.center - ext, self.center + ext

    def clamp_point(self, point_m: np.ndarray) -> np.ndarray:
        """将任意世界系三维点投影（钳位）到本 OBB 表面或内部最近点。"""
        local = (point_m - self.center) @ self.axes
        clamped = np.clip(local, -self.half_extents, self.half_extents)
        return self.center + clamped @ self.axes.T


def obb_signed_distance(obb_a: OBB3D, obb_b: OBB3D) -> float:
    """计算两个 3D OBB 之间的有符号距离 (单位: m)。

    返回值:
        > 0: 两包围盒分离，返回最小欧氏净空距离 (m)；
        <= 0: 两包围盒发生实体穿透/碰撞，返回负的最大穿透深度 (m)。
    """
    diff = obb_b.center - obb_a.center
    A = obb_a.axes
    B = obb_b.axes
    ha = obb_a.half_extents
    hb = obb_b.half_extents

    # 15 个分离轴 (SAT): A 的 3 个面法向、B 的 3 个面法向、以及 9 个边叉积
    cross_axes = []
    for i in range(3):
        for j in range(3):
            c = np.cross(A[:, i], B[:, j])
            norm = float(np.linalg.norm(c))
            if norm > 1e-6:
                cross_axes.append(c / norm)
    if cross_axes:
        L_all = np.column_stack([A, B, np.column_stack(cross_axes)])  # (3, K)
    else:
        L_all = np.column_stack([A, B])

    dist_proj = np.abs(diff @ L_all)
    ra = np.abs(A.T @ L_all).T @ ha
    rb = np.abs(B.T @ L_all).T @ hb
    gaps = dist_proj - (ra + rb)
    max_gap = float(np.max(gaps))
    if max_gap <= 0.0:
        return max_gap

    # 分离状态下使用交替凸投影求解精确三维欧氏最近点距离
    pb = obb_b.center.copy()
    pa = obb_a.center.copy()
    for _ in range(6):
        pa = obb_a.clamp_point(pb)
        pb = obb_b.clamp_point(pa)
    euclid = float(np.linalg.norm(pa - pb))
    return max(euclid, max_gap)


def gripper_obbs(
    position_m,
    angles_deg,
    aperture_m: float = MAX_OPENING_M,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
) -> dict[str, OBB3D]:
    """构建指定位姿与开度下夹爪各部件（左指、右指、手掌、指间闭合扫掠区）的 3D OBB。"""
    frame = gripper_frame(position_m, angles_deg, contact_region=contact_region, contact_depth_m=contact_depth_m)
    pad_center = frame["pad_center_m"]
    u_app = frame["approach_dir"]
    u_close = frame["closing_dir"]
    axes = frame["axes"]
    ap = float(np.clip(aperture_m, 0.0, MAX_OPENING_M))

    # 手指从指尖 (pad_center + 0.0075*u_app) 延伸至指根 (pad_center - 0.0625*u_app)
    finger_axial_center = pad_center + (FINGERTIP_BELOW_PAD_M - 0.5 * FINGER_LENGTH_M) * u_app
    finger_half = np.array([0.5 * FINGER_LENGTH_M, 0.5 * FINGER_THICKNESS_M, 0.5 * FINGER_WIDTH_M])
    lat_offset = 0.5 * ap + 0.5 * FINGER_THICKNESS_M

    left_finger = OBB3D(finger_axial_center - lat_offset * u_close, axes, finger_half, label="left_finger")
    right_finger = OBB3D(finger_axial_center + lat_offset * u_close, axes, finger_half, label="right_finger")

    palm_center = pad_center - (FINGER_LENGTH_M - FINGERTIP_BELOW_PAD_M + 0.5 * PALM_LENGTH_M) * u_app
    palm_half = np.array([0.5 * PALM_LENGTH_M, 0.5 * PALM_WIDTH_M, 0.5 * PALM_THICKNESS_M])
    palm = OBB3D(palm_center, axes, palm_half, label="palm")

    swept_half = np.array([0.5 * PAD_DEPTH_M, max(0.5 * ap, 5e-4), 0.5 * FINGER_WIDTH_M])
    swept_pad = OBB3D(pad_center, axes, swept_half, label="swept_pad")

    return dict(
        left_finger=left_finger,
        right_finger=right_finger,
        palm=palm,
        swept_pad=swept_pad,
        frame=frame,
    )


def _clip_polygon_1d(poly: np.ndarray, axis: int, bound: float, keep_greater: bool) -> np.ndarray:
    """用半平面 poly[:, axis] >= bound (或 <= bound) 裁剪二维凸多边形 (Sutherland-Hodgman)。"""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prev = poly[(i - 1) % n]
        cur_in = (cur[axis] >= bound) if keep_greater else (cur[axis] <= bound)
        prev_in = (prev[axis] >= bound) if keep_greater else (prev[axis] <= bound)
        if cur_in != prev_in:
            denom = cur[axis] - prev[axis]
            t = (bound - prev[axis]) / denom if abs(denom) > 1e-12 else 0.0
            out.append(prev + np.clip(t, 0.0, 1.0) * (cur - prev))
        if cur_in:
            out.append(cur)
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 2), dtype=np.float64)


# ------------------------------------------------------------------ 场景虚拟物体 (VirtualObject)
class VirtualObject:
    """场景物体的三维体积与动态运动学状态表征。"""

    def __init__(
        self,
        oid: str,
        centre_m,
        top_z: float,
        bottom_z: float,
        length: float,
        width: float,
        yaw_deg: float,
        tilt_deg: float | None = None,
        support: str | None = "table",
        quality: str = "ok",
        held_by_arm: str | None = None,
        occluded: bool = False,
    ):
        self.id = str(oid)
        self.centre_m = [float(v) for v in np.asarray(centre_m, dtype=float).reshape(3)]
        self.top_z = float(top_z)
        self.bottom_z = float(bottom_z)
        self.length = max(float(length), 0.004)
        self.width = max(float(width), 0.004)
        self.yaw_deg = float(yaw_deg)
        self.tilt_deg = float(tilt_deg) if _num(tilt_deg) else None
        self.support = support
        self.quality = str(quality or "ok")
        self.held_by_arm = held_by_arm
        self.occluded = bool(occluded)
        self.virtual_moved = False
        # 被夹爪抓起后，物体中心与主轴相对于夹爪标架 (pad_center, axes) 的刚体相对变换
        self.rel_offset_in_gripper: np.ndarray | None = None
        self.rel_axes_in_gripper: np.ndarray | None = None
        self._world_axes_override: np.ndarray | None = None

    @property
    def height_m(self) -> float:
        return max(self.top_z - self.bottom_z, 0.004)

    @property
    def is_elongated(self) -> bool:
        if self.width <= 0 or self.width > MAX_OPENING_M or (self.length / self.width) < ELONGATED_RATIO:
            return False
        if (self.top_z - self.bottom_z) >= 0.85 * self.length:
            return False
        if _num(self.tilt_deg) and float(self.tilt_deg) < 35.0:
            return False
        return True

    def world_axes(self) -> np.ndarray:
        if self._world_axes_override is not None:
            return self._world_axes_override
        yaw_rad = math.radians(self.yaw_deg)
        u_long = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float64)
        u_short = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)
        u_vert = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return np.column_stack([u_long, u_short, u_vert])

    def obb(self) -> OBB3D:
        """返回物体当前在世界坐标系下的 3D OBB。"""
        axes = self.world_axes()
        z_mid = 0.5 * (self.top_z + self.bottom_z)
        center = np.array([self.centre_m[0], self.centre_m[1], z_mid], dtype=np.float64)
        half_ext = np.array([0.5 * self.length, 0.5 * self.width, max(0.5 * (self.top_z - self.bottom_z), 0.002)], dtype=np.float64)
        return OBB3D(center, axes, half_ext, label=self.id)

    def footprint_corners_xy(self) -> np.ndarray:
        """返回物体水平截面四个角点的世界坐标 (4, 3)。"""
        axes = self.world_axes()
        u_long, u_short = axes[:, 0], axes[:, 1]
        c = np.asarray(self.centre_m, dtype=np.float64)
        hl, hw = 0.5 * self.length, 0.5 * self.width
        return np.array(
            [
                c - hl * u_long - hw * u_short,
                c + hl * u_long - hw * u_short,
                c + hl * u_long + hw * u_short,
                c - hl * u_long + hw * u_short,
            ],
            dtype=np.float64,
        )

    def attach_to_gripper(self, arm: str, g_frame: dict):
        """将本物体刚性绑定到指定机械臂的夹爪标架上。"""
        self.held_by_arm = arm
        pad_c = g_frame["pad_center_m"]
        g_axes = g_frame["axes"]
        obb_c = self.obb().center
        # 记录物体几何中心及点位中心相对于夹爪标架的局部坐标
        self.rel_offset_in_gripper = (obb_c - pad_c) @ g_axes
        self._rel_centre_in_gripper = (np.asarray(self.centre_m, dtype=np.float64) - pad_c) @ g_axes
        self.rel_axes_in_gripper = g_axes.T @ self.world_axes()

    def update_from_gripper(self, g_frame: dict):
        """当持物机械臂移动时，根据夹爪最新标架实时更新物体的 3D 位姿与包围盒。"""
        if self.rel_offset_in_gripper is None or self.rel_axes_in_gripper is None:
            return
        pad_c = g_frame["pad_center_m"]
        g_axes = g_frame["axes"]
        new_axes = g_axes @ self.rel_axes_in_gripper
        self._world_axes_override = new_axes
        new_centre = pad_c + self._rel_centre_in_gripper @ g_axes.T
        new_obb_c = pad_c + self.rel_offset_in_gripper @ g_axes.T
        half_ext = np.array([0.5 * self.length, 0.5 * self.width, 0.5 * self.height_m], dtype=np.float64)
        z_extent = float(np.abs(new_axes[2, :]) @ half_ext)
        self.centre_m = [float(v) for v in new_centre]
        self.top_z = float(new_obb_c[2] + z_extent)
        self.bottom_z = float(new_obb_c[2] - z_extent)
        # 更新水平投影主轴 yaw_deg
        u_long = new_axes[:, 0]
        if math.hypot(u_long[0], u_long[1]) > 1e-4:
            self.yaw_deg = float(wrap90(math.degrees(math.atan2(u_long[1], u_long[0]))))
        self.virtual_moved = True

    def detach_from_gripper(self, support_z: float, support_id: str = "table"):
        """开爪释放物体时，解除刚体绑定并将物体沉降到下方支撑面上。"""
        drop = max(0.0, self.bottom_z - float(support_z))
        self.centre_m[2] = float(self.centre_m[2] - drop)
        self.bottom_z = float(self.bottom_z - drop)
        self.top_z = float(self.top_z - drop)
        self.support = support_id
        self.held_by_arm = None
        self.rel_offset_in_gripper = None
        self.rel_axes_in_gripper = None
        self._world_axes_override = None
        self.virtual_moved = True

    def to_dict(self) -> dict:
        return dict(
            id=self.id,
            centre_m=_r(self.centre_m, 4),
            top_z=_r(self.top_z, 4),
            bottom_z=_r(self.bottom_z, 4),
            length=_r(self.length, 4),
            width=_r(self.width, 4),
            yaw_deg=_r(self.yaw_deg, 1),
            tilt_deg=_r(self.tilt_deg, 1) if self.tilt_deg is not None else None,
            support=self.support,
            quality=self.quality,
            held_by_arm=self.held_by_arm,
            virtual_moved=self.virtual_moved,
            occluded=self.occluded,
        )


def _parse_virtual_object(item, index: int = 0, table_z: float = DEFAULT_TABLE_Z_M, columns=None) -> VirtualObject | None:
    """将 object_table 的行列表或字典解析为 VirtualObject。"""
    if isinstance(item, VirtualObject):
        return deepcopy(item)
    if isinstance(item, (list, tuple)):
        cols = list(columns or ("id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"))
        if len(item) < len(cols):
            return None
        row = dict(zip(cols, item))
    elif isinstance(item, dict):
        row = item
    else:
        return None

    oid = str(row.get("id") or f"o{index + 1}")
    support = row.get("support", "table")
    if isinstance(support, int):
        support = f"o{support + 1}"
    quality = str(row.get("quality") or "ok")
    held_by = None
    if isinstance(support, str) and support.startswith("held_"):
        held_by = support.split("held_", 1)[-1]

    centre = row.get("centre_m")
    if not isinstance(centre, (list, tuple, np.ndarray)) or len(centre) != 3 or not all(_num(v) for v in centre):
        return None
    length = row.get("length") if _num(row.get("length")) else row.get("length_m")
    width = row.get("width") if _num(row.get("width")) else row.get("width_m")
    yaw = row.get("yaw_deg")
    if not (_num(length) and _num(width) and _num(yaw)):
        return None
    cz = float(centre[2])
    bottom = row.get("bottom_z") if _num(row.get("bottom_z")) else row.get("bottom_z_m")
    top = row.get("top_z") if _num(row.get("top_z")) else row.get("top_z_m")
    bottom_z = float(bottom) if _num(bottom) else float(table_z)
    top_z = float(top) if _num(top) else max(cz + (cz - bottom_z), bottom_z + 0.012)
    if top_z < bottom_z:
        top_z, bottom_z = bottom_z, top_z
    tilt = row.get("tilt_deg") if _num(row.get("tilt_deg")) else row.get("axis_tilt_deg")
    return VirtualObject(
        oid=oid,
        centre_m=centre,
        top_z=top_z,
        bottom_z=bottom_z,
        length=float(length),
        width=float(width),
        yaw_deg=float(yaw),
        tilt_deg=float(tilt) if _num(tilt) else None,
        support=support,
        quality=quality,
        held_by_arm=held_by,
        occluded=bool(row.get("occluded", False)),
    )


# ------------------------------------------------------------------ 虚拟空间数值模拟场 (VirtualSpatialField)
class VirtualSpatialField:
    """基于 SAM + VGGT 与机械臂物理量的实时三维虚拟空间模拟场。"""

    def __init__(self, objects=None, cloud=None, table_z: float | None = None, robot_state: dict | None = None):
        self.table_z: float = float(table_z) if _num(table_z) else DEFAULT_TABLE_Z_M
        self.cloud = cloud
        if cloud is not None and hasattr(cloud, "table") and table_z is None:
            self.table_z = float(cloud.table)
        self.objects: dict[str, VirtualObject] = {}
        self.arm_poses: dict[str, dict[str, list[float]]] = deepcopy(DEFAULT_ARM_POSES)
        self.arm_apertures_m: dict[str, float] = {"left": MAX_OPENING_M, "right": MAX_OPENING_M}
        self.step_count: int = 0
        if objects is not None or robot_state is not None:
            self.update_from_observation(objects, robot_state=robot_state, cloud=cloud)

    def clone(self) -> "VirtualSpatialField":
        """创建当前虚拟场的深拷贝副本，用于无副作用的推演与规划。"""
        copied = VirtualSpatialField(table_z=self.table_z)
        copied.cloud = self.cloud
        copied.objects = deepcopy(self.objects)
        copied.arm_poses = deepcopy(self.arm_poses)
        copied.arm_apertures_m = deepcopy(self.arm_apertures_m)
        copied.step_count = self.step_count
        return copied

    def _update_robot_state(self, robot_state: dict | None):
        if not isinstance(robot_state, dict):
            return
        arms = robot_state.get("arms") or {}
        for arm in ("left", "right"):
            arm_data = arms.get(arm)
            if not isinstance(arm_data, dict):
                continue
            gp = arm_data.get("grasp_pose")
            if isinstance(gp, dict) and isinstance(gp.get("position_m"), (list, tuple)) and len(gp["position_m"]) == 3:
                ang = gp.get("angles_deg") or [0.0, 0.0, 0.0]
                self.arm_poses[arm] = {
                    "position_m": [float(v) for v in gp["position_m"]],
                    "angles_deg": [float(v) for v in ang[:3]],
                }
            else:
                ee = arm_data.get("ee_pose") or {}
                pos = ee.get("position_m")
                quat_xyzw = ee.get("quaternion_xyzw")
                if isinstance(pos, (list, tuple)) and len(pos) == 3 and isinstance(quat_xyzw, (list, tuple)) and len(quat_xyzw) == 4:
                    x, y, z, w = (float(v) for v in quat_xyzw)
                    quat_wxyz = np.array([w, x, y, z], dtype=np.float64)
                    p_deg, r_deg, y_deg = quat_wxyz_to_angles_deg(quat_wxyz)
                    t_axis = quat_wxyz_to_matrix(quat_wxyz)[:, 0]
                    grasp_pos = np.asarray(pos, dtype=np.float64) + GRASP_POINT_OFFSET_M * t_axis
                    self.arm_poses[arm] = {
                        "position_m": [float(v) for v in grasp_pos],
                        "angles_deg": [float(p_deg), float(r_deg), float(y_deg)],
                    }
            # 解析夹爪开度
            grip = arm_data.get("gripper") or arm_data.get("ee_joint_state") or {}
            if isinstance(grip, dict):
                if _num(grip.get("aperture_m")):
                    self.arm_apertures_m[arm] = float(np.clip(grip["aperture_m"], 0.0, MAX_OPENING_M))
                elif _num(grip.get("position_normalized")):
                    self.arm_apertures_m[arm] = float(np.clip(grip["position_normalized"], 0.0, 1.0)) * MAX_OPENING_M
            elif _num(arm_data.get("aperture_m")):
                self.arm_apertures_m[arm] = float(np.clip(arm_data["aperture_m"], 0.0, MAX_OPENING_M))

    def update_from_observation(self, object_table=None, robot_state: dict | None = None, cloud=None) -> "VirtualSpatialField":
        """随真实机械臂移动与新观测实时融合更新虚拟空间场。

        参数:
            object_table: VGGT object_table 字典（含 rows, columns, table_z）或物体列表。
            robot_state: 可选实测双臂状态字典（含 arms.left / arms.right 的位姿与夹爪开度）。
            cloud: 可选 scene3d Cloud 实例。
        """
        if cloud is not None:
            self.cloud = cloud
            if hasattr(cloud, "table"):
                self.table_z = float(cloud.table)
        self._update_robot_state(robot_state)

        prev_objects = self.objects
        prev_held = {
            arm: obj
            for obj in prev_objects.values()
            if obj.held_by_arm in ("left", "right")
            for arm in (obj.held_by_arm,)
        }

        columns = None
        raw_items = []
        if isinstance(object_table, dict):
            if _num(object_table.get("table_z")):
                self.table_z = float(object_table["table_z"])
            elif _num(object_table.get("support_z_m")):
                self.table_z = float(object_table["support_z_m"])
            columns = object_table.get("columns")
            raw_items = list(object_table.get("rows") or object_table.get("objects") or [])
        elif isinstance(object_table, (list, tuple)):
            raw_items = list(object_table)

        new_objects: dict[str, VirtualObject] = {}
        for idx, item in enumerate(raw_items):
            parsed = _parse_virtual_object(item, index=idx, table_z=self.table_z, columns=columns)
            if parsed is not None:
                new_objects[parsed.id] = parsed

        # 实时同步持物状态：若某臂上一轮持物且当前夹爪未张开释放 (aperture > 1.5mm 且 < 75mm)，用实测末端位姿更新持物 OBB
        for arm, held_obj in prev_held.items():
            ap = self.arm_apertures_m.get(arm, MAX_OPENING_M)
            if 0.0015 < ap < 0.075:
                pose = self.arm_poses.get(arm)
                if pose is not None:
                    g_frame = gripper_frame(pose["position_m"], pose["angles_deg"])
                    held_copy = deepcopy(held_obj)
                    held_copy.update_from_gripper(g_frame)
                    key = held_copy.id if held_copy.id not in new_objects else f"held_{arm}_{held_copy.id}"
                    held_copy.id = key
                    new_objects[key] = held_copy

        # 跨帧遮挡记忆：若上一帧未持握的桌面物体在本帧未出现对应邻近物体（如被机械臂遮挡），保留一帧已知包络并标记 occluded=True
        if prev_objects and new_objects:
            for old_obj in prev_objects.values():
                if old_obj.held_by_arm is not None or old_obj.occluded:
                    continue
                ox, oy = old_obj.centre_m[0], old_obj.centre_m[1]
                matched = any(
                    math.hypot(ox - cur.centre_m[0], oy - cur.centre_m[1]) <= max(0.5 * old_obj.length, 0.045)
                    for cur in new_objects.values()
                )
                if not matched:
                    occ = deepcopy(old_obj)
                    occ.occluded = True
                    occ_id = occ.id if occ.id not in new_objects else f"occluded_{occ.id}"
                    occ.id = occ_id
                    new_objects[occ_id] = occ

        if object_table is not None:
            self.objects = new_objects
        elif prev_held:
            for key, obj in new_objects.items():
                self.objects[key] = obj

        self.step_count += 1
        return self

    def find_target_object(self, target_id: str | dict | VirtualObject | None, position_m=None) -> VirtualObject | None:
        """根据 target_id 或空间坐标查找目标 VirtualObject。"""
        if isinstance(target_id, VirtualObject):
            return target_id
        if isinstance(target_id, dict):
            return _parse_virtual_object(target_id, table_z=self.table_z)
        if isinstance(target_id, str) and target_id in self.objects:
            return self.objects[target_id]
        if position_m is not None:
            gx, gy = float(position_m[0]), float(position_m[1])
            best_obj, best_dist = None, math.inf
            for obj in self.objects.values():
                if obj.held_by_arm is not None:
                    continue
                dist = math.hypot(gx - obj.centre_m[0], gy - obj.centre_m[1])
                reach = max(0.5 * obj.length + 0.04, 0.5 * obj.width + 0.04, 0.075)
                if dist <= reach and dist < best_dist:
                    best_obj, best_dist = obj, dist
            return best_obj
        return None

    def _evaluate_grasp_core(
        self,
        arm: str,
        position_m,
        angles_deg,
        target: VirtualObject,
        open_aperture_m: float = MAX_OPENING_M,
        contact_region: str = "pad_center",
    ) -> dict:
        """内部高速抓取几何与体积干涉评估核心（不递归触发 solve_optimal_grasp）。"""
        pos = np.asarray(position_m, dtype=np.float64).reshape(3)
        ang = np.asarray(angles_deg, dtype=np.float64).reshape(3)
        ap = float(np.clip(open_aperture_m, 0.0, MAX_OPENING_M))

        g_parts = gripper_obbs(pos, ang, aperture_m=ap, contact_region=contact_region)
        g_frame = g_parts["frame"]
        pad_c = g_frame["pad_center_m"]
        u_close = g_frame["closing_dir"]
        u_tan = g_frame["tangent_dir"]
        left_obb = g_parts["left_finger"]
        right_obb = g_parts["right_finger"]
        palm_obb = g_parts["palm"]
        swept_obb = g_parts["swept_pad"]

        failure_reasons: list[str] = []

        # 1. 计算目标物体在夹爪闭合-切向平面 (u_close, u_tan) 内的截面裁剪多边形
        corners_3d = target.footprint_corners_xy()
        rel_3d = corners_3d - pad_c[None, :]
        poly_2d = np.column_stack([rel_3d @ u_close, rel_3d @ u_tan])
        half_fw = 0.5 * FINGER_WIDTH_M
        clipped = _clip_polygon_1d(poly_2d, axis=1, bound=-half_fw, keep_greater=True)
        clipped = _clip_polygon_1d(clipped, axis=1, bound=half_fw, keep_greater=False)

        obj_axes = target.world_axes()
        delta_c = np.asarray(target.centre_m, dtype=np.float64) - pad_c
        axial_offset_m = float(abs(delta_c @ obj_axes[:, 0]))

        if len(clipped) > 0:
            c_min = float(np.min(clipped[:, 0]))
            c_max = float(np.max(clipped[:, 0]))
            projected_span_m = max(0.0, c_max - c_min)
            slice_center = 0.5 * (c_min + c_max)
            lateral_eccentricity_m = abs(slice_center)
            footprint_hit = True
        else:
            c_min = float(np.min(poly_2d[:, 0]))
            c_max = float(np.max(poly_2d[:, 0]))
            projected_span_m = max(0.0, c_max - c_min)
            lateral_eccentricity_m = abs(0.5 * (c_min + c_max))
            footprint_hit = False
            failure_reasons.append("grasp_misses_target_footprint")

        # 2. 竖直咬合高度 (vertical_bite_m) 与指尖支撑台面净空 (support_clearance_m)
        swept_min, swept_max = swept_obb.aabb()
        pad_z_low, pad_z_high = float(swept_min[2]), float(swept_max[2])
        vertical_bite_m = max(0.0, min(pad_z_high, target.top_z) - max(pad_z_low, target.bottom_z))
        if vertical_bite_m < MIN_VERTICAL_BITE_M:
            failure_reasons.append("insufficient_vertical_bite")

        support_z = target.bottom_z if _num(target.bottom_z) else self.table_z
        finger_min_z = min(float(left_obb.aabb()[0][2]), float(right_obb.aabb()[0][2]))
        finger_max_z = max(float(left_obb.aabb()[1][2]), float(right_obb.aabb()[1][2]))
        support_clearance_m = finger_min_z - float(support_z)
        if support_clearance_m < -0.006:
            failure_reasons.append("fingertip_penetrates_support")

        # 3. 张开手指下探时是否直接撞击目标物体本体 (finger_hits_target) 及闭合跨度/偏心校验
        z_overlaps_target = (min(finger_max_z, target.top_z) - max(finger_min_z, target.bottom_z)) > 0.001
        half_ap = 0.5 * ap
        hits_left = footprint_hit and z_overlaps_target and (c_min < -half_ap + 0.0015) and (c_max > -half_ap - FINGER_THICKNESS_M - 0.0015)
        hits_right = footprint_hit and z_overlaps_target and (c_max > half_ap - 0.0015) and (c_min < half_ap + FINGER_THICKNESS_M + 0.0015)
        finger_hits_target = bool(hits_left or hits_right)

        if projected_span_m > ap - 0.004:
            failure_reasons.append("projected_span_exceeds_opening")
        max_allowed_ecc = max(0.0, 0.5 * (ap - projected_span_m))
        if footprint_hit and lateral_eccentricity_m > max_allowed_ecc + 0.0015:
            failure_reasons.append("excessive_lateral_eccentricity")
        if finger_hits_target:
            failure_reasons.append("finger_hits_target_on_descent")

        # 4. 细长物体闭合方向夹角检验
        closing_to_long_axis = axis_difference_deg(ang[2], target.yaw_deg)
        if target.is_elongated and closing_to_long_axis < 45.0:
            failure_reasons.append("closing_along_elongated_axis")

        # 5. 与周围其余邻近物体的三维体积干涉检测
        colliding_neighbors: list[str] = []
        min_neighbor_clearance = 1.0
        for other in self.objects.values():
            if other.id == target.id or other.held_by_arm == arm:
                continue
            other_obb = other.obb()
            d_left = obb_signed_distance(left_obb, other_obb)
            d_right = obb_signed_distance(right_obb, other_obb)
            d_palm = obb_signed_distance(palm_obb, other_obb)
            d_min = min(d_left, d_right, d_palm)
            if d_min < min_neighbor_clearance:
                min_neighbor_clearance = d_min
            if d_min < -0.001:
                colliding_neighbors.append(other.id)

        if colliding_neighbors:
            failure_reasons.append("open_finger_collides_with_neighbor")

        return dict(
            arm=arm,
            target_id=target.id,
            theoretically_graspable=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
            finger_hits_target=finger_hits_target,
            projected_span_m=_r(projected_span_m, 4),
            vertical_bite_m=_r(vertical_bite_m, 4),
            lateral_eccentricity_m=_r(lateral_eccentricity_m, 4),
            max_allowed_eccentricity_m=_r(max_allowed_ecc, 4),
            axial_offset_m=_r(axial_offset_m, 4),
            closing_to_long_axis_deg=_r(closing_to_long_axis, 1),
            support_clearance_m=_r(support_clearance_m, 4),
            neighbor_clearance_m=_r(min_neighbor_clearance, 4),
            colliding_neighbor_ids=colliding_neighbors,
        )

    def evaluate_grasp(
        self,
        arm: str = "right",
        position_m=None,
        angles_deg=None,
        target_id: str | dict | None = None,
        open_aperture_m: float = MAX_OPENING_M,
        contact_region: str = "pad_center",
    ) -> dict:
        """评估指定三维抓取位姿的理论可夹持性、咬合体积与周围障碍物干涉。

        参数:
            arm: 执行抓取的机械臂名称 ('left' | 'right')，默认 'right'。
            position_m: 抓取目标位置 [x, y, z]，单位 m (env_origin)。
            angles_deg: 抓取姿态角 [pitch_deg, roll_deg, yaw_deg]，单位 deg；默认 [0, 0, 0]。
            target_id: 目标物体 ID (如 'o1') 或物体属性字典；省略时自动匹配水平距离最近的可抓物体。
            open_aperture_m: 下探前夹爪张开宽度，单位 m，默认 0.0888 (88.8mm)。
            contact_region: 接触截面 ('pad_center' | 'fingertip' | 'finger_middle')，默认 'pad_center'。

        返回:
            包含 theoretically_graspable (bool), failure_reasons (list[str]), projected_span_m (m),
            vertical_bite_m (m), lateral_eccentricity_m (m), support_clearance_m (m),
            neighbor_clearance_m (m), colliding_neighbor_ids (list[str]), optimal_grasp_pose (dict) 的诊断字典。
        """
        if angles_deg is None:
            angles_deg = [0.0, 0.0, 0.0]
        target = self.find_target_object(target_id, position_m=position_m)
        if target is None:
            return dict(
                arm=arm,
                target_id=target_id if isinstance(target_id, str) else None,
                theoretically_graspable=False,
                failure_reasons=["no_target_object_in_reach"],
                finger_hits_target=False,
                projected_span_m=None,
                vertical_bite_m=0.0,
                lateral_eccentricity_m=None,
                support_clearance_m=None,
                neighbor_clearance_m=None,
                colliding_neighbor_ids=[],
                optimal_grasp_pose=None,
            )
        if position_m is None:
            position_m = list(target.centre_m)
        report = self._evaluate_grasp_core(
            arm=arm,
            position_m=position_m,
            angles_deg=angles_deg,
            target=target,
            open_aperture_m=open_aperture_m,
            contact_region=contact_region,
        )
        opt = self.solve_optimal_grasp(
            target_id=target,
            arm=arm,
            preferred_yaw_deg=float(angles_deg[2]),
            open_aperture_m=open_aperture_m,
        )
        report["optimal_grasp_pose"] = opt
        return report

    def solve_optimal_grasp(
        self,
        target_id: str | dict | VirtualObject,
        arm: str = "right",
        preferred_yaw_deg: float | None = None,
        open_aperture_m: float = MAX_OPENING_M,
    ) -> dict:
        """在虚拟场内围绕目标物体主轴与偏移网格搜索零碰撞、最大咬合、最小偏心的最优 3D 抓取位姿。

        参数:
            target_id: 目标物体 ID (如 'o1') 或物体几何字典。
            arm: 规划使用的机械臂 ('left' | 'right')，默认 'right'。
            preferred_yaw_deg: 可选偏好偏航角 (deg)；未指定时优先选择最接近 0° 的无碰撞横跨主轴角。
            open_aperture_m: 下探前夹爪张开宽度，单位 m，默认 0.0888。

        返回:
            包含 position_m (m), angles_deg (deg), across_yaw_deg (deg), pre_grasp_position_m (m),
            safe_lift_z_m (m), safe_transit_z_m (m), neighbor_clearance_m (m), vertical_bite_m (m) 的最优抓取字典。
        """
        target = self.find_target_object(target_id)
        if target is None:
            raise ValueError(f"target_object_not_found:{target_id}")

        base_yaws = closure_yaws_deg(target.yaw_deg)
        ref_yaw = float(preferred_yaw_deg) if _num(preferred_yaw_deg) else 0.0
        base_yaws.sort(key=lambda y: (abs(wrap180(y - ref_yaw)), abs(y)))

        if target.is_elongated:
            yaw_deltas = (0.0, -12.0, 12.0, -25.0, 25.0, -38.0, 38.0)
            candidate_yaws = []
            for by in base_yaws:
                for dy in yaw_deltas:
                    y = round(wrap180(by + dy), 1)
                    if y not in candidate_yaws:
                        candidate_yaws.append(y)
        else:
            candidate_yaws = list(base_yaws)
            for extra in (0.0, -45.0, 45.0, -90.0, 90.0, -135.0, 135.0, 180.0):
                y = round(wrap180(extra), 1)
                if y not in candidate_yaws:
                    candidate_yaws.append(y)

        # 沿物体长轴与短轴的候选偏移 (单位: m)
        max_along = max(0.0, 0.35 * target.length - 0.006)
        if max_along >= 0.012:
            along_offsets = (0.0, -0.5 * max_along, 0.5 * max_along, -max_along, max_along)
        else:
            along_offsets = (0.0,)
        across_offsets = (0.0, -0.005, 0.005)

        support_z = target.bottom_z if _num(target.bottom_z) else self.table_z
        pad_z = float(target.centre_m[2])
        z_candidates = [pad_z]
        safe_floor_z = float(support_z) + FINGERTIP_BELOW_PAD_M + 0.001
        if pad_z < safe_floor_z and (target.top_z - safe_floor_z) >= -0.003:
            z_candidates.append(round(safe_floor_z, 4))

        axes = target.world_axes()
        u_long, u_short = axes[:, 0], axes[:, 1]
        cx, cy = float(target.centre_m[0]), float(target.centre_m[1])

        best_score = -math.inf
        best_candidate = None

        for gz in z_candidates:
            for along in along_offsets:
                for across in across_offsets:
                    px = cx + along * float(u_long[0]) + across * float(u_short[0])
                    py = cy + along * float(u_long[1]) + across * float(u_short[1])
                    pos = [px, py, gz]
                    for yaw in candidate_yaws:
                        ang = [0.0, 0.0, yaw]
                        ev = self._evaluate_grasp_core(
                            arm=arm,
                            position_m=pos,
                            angles_deg=ang,
                            target=target,
                            open_aperture_m=open_aperture_m,
                        )
                        valid_bonus = 100.0 if ev["theoretically_graspable"] else -100.0 * len(ev["failure_reasons"])
                        clr = float(ev["neighbor_clearance_m"])
                        clr_score = min(clr, 0.05) * 400.0 if clr >= 0.0 else clr * 2000.0
                        bite_score = float(ev["vertical_bite_m"]) * 300.0
                        ecc_penalty = float(ev["lateral_eccentricity_m"]) * 250.0 + abs(along) * 60.0
                        yaw_diff = abs(wrap180(yaw - ref_yaw))
                        axis_bonus = float(ev["closing_to_long_axis_deg"]) * 0.25 if target.is_elongated else 0.0
                        score = valid_bonus + clr_score + bite_score + axis_bonus - ecc_penalty - yaw_diff * 0.05
                        if score > best_score:
                            best_score = score
                            best_candidate = (pos, ang, ev)

        best_pos, best_ang, best_ev = best_candidate
        max_scene_top_z = max([obj.top_z for obj in self.objects.values()] + [target.top_z, self.table_z])
        obj_bottom_below_pad = max(0.0, best_pos[2] - target.bottom_z)
        safe_lift_z = max(best_pos[2] + 0.06, max_scene_top_z + obj_bottom_below_pad + 0.05)
        pre_grasp_z = max(best_pos[2] + 0.06, target.top_z + 0.04)

        return dict(
            target_id=target.id,
            arm=arm,
            theoretically_graspable=best_ev["theoretically_graspable"],
            position_m=_r(best_pos, 4),
            angles_deg=_r(best_ang, 1),
            across_yaw_deg=round(float(best_ang[2]), 1),
            pre_grasp_position_m=_r([best_pos[0], best_pos[1], pre_grasp_z], 4),
            safe_lift_z_m=_r(safe_lift_z, 4),
            safe_transit_z_m=_r(safe_lift_z, 4),
            neighbor_clearance_m=best_ev["neighbor_clearance_m"],
            vertical_bite_m=best_ev["vertical_bite_m"],
            projected_span_m=best_ev["projected_span_m"],
            lateral_eccentricity_m=best_ev["lateral_eccentricity_m"],
            colliding_neighbor_ids=best_ev["colliding_neighbor_ids"],
        )

    @staticmethod
    def _extract_stage_spec(stage: dict, default_poses: dict, default_apertures: dict) -> dict:
        """兼容公开 robot_action stage 格式与内部 normalized action stage 格式。"""
        stage_id = str(stage.get("id") or stage.get("stage_id") or "stage")
        arm = str(stage.get("arm") or "right")
        kind = str(stage.get("kind") or "move")
        own = stage.get("action") if isinstance(stage.get("action"), dict) else {}

        cur_pose = default_poses.get(arm) or DEFAULT_ARM_POSES.get(arm, DEFAULT_ARM_POSES["right"])
        pos = None
        ang = None

        # 1) 检查顶层 grasp_position_m / position_m / waypoints
        if isinstance(stage.get("grasp_position_m"), (list, tuple)) and len(stage["grasp_position_m"]) == 3:
            pos = [float(v) for v in stage["grasp_position_m"]]
        elif isinstance(stage.get("position_m"), (list, tuple)) and len(stage["position_m"]) == 3:
            pos = [float(v) for v in stage["position_m"]]
        if isinstance(stage.get("angles_deg"), (list, tuple)) and len(stage["angles_deg"]) == 3:
            ang = [float(v) for v in stage["angles_deg"]]
        if pos is None and isinstance(stage.get("waypoints"), list) and stage["waypoints"]:
            wp = stage["waypoints"][-1]
            if isinstance(wp, dict):
                wpos = wp.get("grasp_position_m") or wp.get("position_m")
                if isinstance(wpos, (list, tuple)) and len(wpos) == 3:
                    pos = [float(v) for v in wpos]
                if isinstance(wp.get("angles_deg"), (list, tuple)) and len(wp["angles_deg"]) == 3:
                    ang = [float(v) for v in wp["angles_deg"]]

        # 2) 检查内部 action.input_grasp_pose
        if pos is None:
            gp = own.get("input_grasp_pose")
            if isinstance(gp, dict) and arm in gp:
                gp = gp[arm]
            if isinstance(gp, dict):
                if isinstance(gp.get("position_m"), (list, tuple)) and len(gp["position_m"]) == 3:
                    pos = [float(v) for v in gp["position_m"]]
                if isinstance(gp.get("angles_deg"), (list, tuple)) and len(gp["angles_deg"]) == 3:
                    ang = [float(v) for v in gp["angles_deg"]]

        if pos is None:
            pos = list(cur_pose["position_m"])
        if ang is None:
            ang = list(cur_pose["angles_deg"])

        # 3) 提取夹爪开合指令 goal_normalized (0.0=闭合, 1.0=全开)
        goal_norm = None
        if _num(stage.get("gripper")):
            goal_norm = float(stage["gripper"])
        elif _num(stage.get("goal_normalized")):
            goal_norm = float(stage["goal_normalized"])
        else:
            events = own.get("gripper_events") or stage.get("gripper_events") or []
            if isinstance(events, list) and events and isinstance(events[-1], dict):
                if _num(events[-1].get("goal_normalized")):
                    goal_norm = float(events[-1]["goal_normalized"])

        contact_region = str(stage.get("contact_region") or own.get("contact_region") or "pad_center")
        return dict(
            stage_id=stage_id,
            arm=arm,
            kind=kind,
            position_m=pos,
            angles_deg=ang,
            goal_normalized=goal_norm,
            contact_region=contact_region,
        )

    def simulate_stages(self, stages, step_m: float = 0.015) -> dict:
        """在虚拟场克隆副本上按顺序动态推演多阶段动作序列 [stage_1, stage_2, ...]。

        支持闭爪抓起判定、持物随机械臂同步移动、搬运轨迹 3D OBB 碰撞检测与安全高度计算、以及开爪释放后物体位置更新。

        参数:
            stages: 阶段列表 `[stage_1, ...]` 或包含 `'stages'` 的动作字典。
            step_m: 路径线性插值采样步长，单位 m，默认 0.015 (1.5cm)。

        返回:
            包含 status ('clean' | 'risk_detected'), theoretically_valid (bool),
            min_path_clearance_m (m), stage_reports (list[dict]),
            virtual_world_after (list[dict]), suggested_fixes (list[dict]) 的推演报告。
        """
        raw_stages = stages.get("stages") if isinstance(stages, dict) else stages
        raw_stages = list(raw_stages or [])
        sim = self.clone()

        stage_reports: list[dict] = []
        suggested_fixes: list[dict] = []
        overall_min_clearance = 1.0
        has_risk = False

        for idx, raw in enumerate(raw_stages):
            if not isinstance(raw, dict):
                continue
            spec = self._extract_stage_spec(raw, sim.arm_poses, sim.arm_apertures_m)
            arm = spec["arm"]
            stage_id = spec["stage_id"]
            kind = spec["kind"]
            start_pose = deepcopy(sim.arm_poses.get(arm, DEFAULT_ARM_POSES["right"]))
            start_pos = np.asarray(start_pose["position_m"], dtype=np.float64)
            start_ang = np.asarray(start_pose["angles_deg"], dtype=np.float64)
            target_pos = np.asarray(spec["position_m"], dtype=np.float64)
            target_ang = np.asarray(spec["angles_deg"], dtype=np.float64)

            prev_ap = sim.arm_apertures_m.get(arm, MAX_OPENING_M)
            was_open = prev_ap >= 0.5 * MAX_OPENING_M
            goal_norm = spec["goal_normalized"]
            # 判断本阶段或紧邻的纯 gripper 阶段是否在该位置闭爪抓取
            next_gripper_closing = False
            if goal_norm is None and kind == "move" and idx + 1 < len(raw_stages):
                nxt = raw_stages[idx + 1]
                if isinstance(nxt, dict) and str(nxt.get("kind")) == "gripper" and str(nxt.get("arm") or arm) == arm:
                    nxt_spec = self._extract_stage_spec(nxt, sim.arm_poses, sim.arm_apertures_m)
                    if nxt_spec["goal_normalized"] is not None and nxt_spec["goal_normalized"] < 0.5:
                        next_gripper_closing = True

            is_closing_grasp = was_open and ((goal_norm is not None and goal_norm < 0.5) or next_gripper_closing)
            is_releasing = (not was_open) and (goal_norm is not None and goal_norm >= 0.5)

            held_obj = next((o for o in sim.objects.values() if o.held_by_arm == arm), None)
            intended_target = sim.find_target_object(None, position_m=target_pos) if (is_closing_grasp and held_obj is None) else None

            # 1. 若为移动阶段，沿轨迹插值检测夹爪 + 持物与场内其他物体的 3D OBB 净空与碰撞
            stage_min_clearance = 1.0
            colliding_ids: list[str] = []
            if kind == "move":
                dist = float(np.linalg.norm(target_pos - start_pos))
                n_steps = max(2, int(math.ceil(dist / max(float(step_m), 0.005))) + 1)
                for s_i in range(1, n_steps + 1):
                    alpha = float(s_i) / float(n_steps)
                    interp_pos = start_pos + alpha * (target_pos - start_pos)
                    interp_ang = start_ang + alpha * np.array([wrap180(target_ang[k] - start_ang[k]) for k in range(3)])
                    g_obbs = gripper_obbs(interp_pos, interp_ang, aperture_m=prev_ap, contact_region=spec["contact_region"])
                    moving_obbs = [g_obbs["left_finger"], g_obbs["right_finger"], g_obbs["palm"]]
                    if held_obj is not None:
                        held_obj.update_from_gripper(g_obbs["frame"])
                        moving_obbs.append(held_obj.obb())

                    for other in sim.objects.values():
                        if other.held_by_arm == arm:
                            continue
                        # 下探抓取目标阶段，目标物体本体由 evaluate_grasp 专项审计，不计入路径障碍物
                        if intended_target is not None and other.id == intended_target.id:
                            continue
                        # 放置阶段在终点释放时允许接触目标支撑容器顶面
                        if is_releasing and alpha > 0.85 and other.top_z <= (held_obj.bottom_z if held_obj else interp_pos[2]) + 0.008:
                            continue
                        other_obb = other.obb()
                        d = min(obb_signed_distance(m_obb, other_obb) for m_obb in moving_obbs)
                        if d < stage_min_clearance:
                            stage_min_clearance = d
                        if d < -0.002 and other.id not in colliding_ids:
                            colliding_ids.append(other.id)

                sim.arm_poses[arm] = {
                    "position_m": [float(v) for v in target_pos],
                    "angles_deg": [float(v) for v in target_ang],
                }
                if held_obj is not None:
                    end_frame = gripper_frame(target_pos, target_ang, contact_region=spec["contact_region"])
                    held_obj.update_from_gripper(end_frame)

            if stage_min_clearance < overall_min_clearance:
                overall_min_clearance = stage_min_clearance

            stage_entry = dict(
                stage_id=stage_id,
                arm=arm,
                kind=kind,
                target_position_m=_r(target_pos, 4),
                target_angles_deg=_r(target_ang, 1),
                held_object_id=held_obj.id if held_obj is not None else None,
                path_clearance_m=_r(stage_min_clearance, 4),
                colliding_objects=colliding_ids,
                status="ok",
            )

            if colliding_ids:
                has_risk = True
                stage_entry["status"] = "path_collision"
                max_col_top = max(sim.objects[cid].top_z for cid in colliding_ids if cid in sim.objects)
                bottom_offset = max(0.0075, float(target_pos[2]) - held_obj.bottom_z) if held_obj is not None else FINGERTIP_BELOW_PAD_M
                rec_z = round(max_col_top + bottom_offset + 0.04, 4)
                stage_entry["recommended_safe_z_m"] = rec_z
                suggested_fixes.append(
                    dict(
                        stage_id=stage_id,
                        issue="path_collision",
                        colliding_objects=colliding_ids,
                        recommended_safe_z_m=rec_z,
                        suggestion=f"Raise stage '{stage_id}' transit height z to at least {rec_z:.3f}m to clear {', '.join(colliding_ids)} (top_z={max_col_top:.3f}m).",
                    )
                )

            # 2. 处理闭爪抓取事件
            if was_open and goal_norm is not None and goal_norm < 0.5:
                grasp_eval = sim.evaluate_grasp(
                    arm=arm,
                    position_m=target_pos,
                    angles_deg=target_ang,
                    open_aperture_m=prev_ap,
                    contact_region=spec["contact_region"],
                )
                stage_entry["grasp_evaluation"] = grasp_eval
                if grasp_eval["theoretically_graspable"] and grasp_eval["target_id"] in sim.objects:
                    grasped = sim.objects[grasp_eval["target_id"]]
                    g_frame = gripper_frame(target_pos, target_ang, contact_region=spec["contact_region"])
                    grasped.attach_to_gripper(arm, g_frame)
                    sim.arm_apertures_m[arm] = max(float(grasp_eval["projected_span_m"] or 0.01), 0.002)
                    stage_entry["grasped_object_id"] = grasped.id
                else:
                    has_risk = True
                    stage_entry["status"] = "grasp_risk"
                    sim.arm_apertures_m[arm] = float(goal_norm) * MAX_OPENING_M
                    opt = grasp_eval.get("optimal_grasp_pose")
                    if opt is not None:
                        suggested_fixes.append(
                            dict(
                                stage_id=stage_id,
                                issue="grasp_infeasible",
                                failure_reasons=grasp_eval["failure_reasons"],
                                optimal_grasp_pose=opt,
                                suggestion=(
                                    f"Adjust stage '{stage_id}' grasp pose to position_m={opt['position_m']}, "
                                    f"angles_deg={opt['angles_deg']} (reasons: {', '.join(grasp_eval['failure_reasons'])})."
                                ),
                            )
                        )
            elif goal_norm is not None and goal_norm >= 0.5:
                # 3. 处理开爪释放事件
                if held_obj is not None:
                    hx, hy = held_obj.centre_m[0], held_obj.centre_m[1]
                    support_id, support_top = "table", sim.table_z
                    for other in sim.objects.values():
                        if other.id == held_obj.id or other.held_by_arm is not None:
                            continue
                        if math.hypot(hx - other.centre_m[0], hy - other.centre_m[1]) <= 0.5 * max(other.length, other.width) + 0.02:
                            if other.top_z >= support_top and other.top_z <= held_obj.bottom_z + 0.03:
                                support_top = other.top_z
                                support_id = other.id
                    held_obj.detach_from_gripper(support_z=support_top, support_id=support_id)
                    stage_entry["released_object_id"] = held_obj.id
                    stage_entry["new_object_centre_m"] = _r(held_obj.centre_m, 4)
                    stage_entry["new_support"] = support_id
                sim.arm_apertures_m[arm] = float(np.clip(goal_norm, 0.0, 1.0)) * MAX_OPENING_M

            stage_reports.append(stage_entry)

        return dict(
            status="risk_detected" if has_risk else "clean",
            theoretically_valid=not has_risk,
            min_path_clearance_m=_r(overall_min_clearance, 4),
            stage_reports=stage_reports,
            virtual_world_after=[obj.to_dict() for obj in sim.objects.values()],
            suggested_fixes=suggested_fixes,
        )

    def plan_pick_and_place(
        self,
        target_id: str | dict | VirtualObject,
        place_position_m,
        arm: str = "right",
        place_angles_deg=None,
        approach_clearance_m: float = 0.06,
    ) -> dict:
        """在虚拟场内自动规划并验证完整的 6 阶段抓取-安全抬升-避障搬运-放置-撤离动作序列。

        参数:
            target_id: 待抓取的目标物体 ID (如 'o1') 或物体字典。
            place_position_m: 放置目标位置 [x, y, z]，单位 m (env_origin)。
            arm: 规划使用的机械臂 ('left' | 'right')，默认 'right'。
            place_angles_deg: 放置释放时的 3D 姿态角 [pitch_deg, roll_deg, yaw_deg]，单位 deg；省略时保持抓取姿态。
            approach_clearance_m: 抓取与放置上方的安全悬停净空高度，单位 m，默认 0.06 (6cm)。

        返回:
            包含 stages (可直接传入 robot_action 的 6 阶段列表), grasp_plan, transit_z_m (m), simulation (推演验证结果) 的字典。
        """
        grasp_plan = self.solve_optimal_grasp(target_id=target_id, arm=arm)
        g_pos = grasp_plan["position_m"]
        g_ang = grasp_plan["angles_deg"]
        p_pos = [round(float(v), 4) for v in np.asarray(place_position_m, dtype=float).reshape(3)]
        p_ang = [round(float(v), 1) for v in np.asarray(place_angles_deg if place_angles_deg is not None else g_ang, dtype=float).reshape(3)]

        clearance = max(float(approach_clearance_m), 0.03)
        transit_z = round(max(float(grasp_plan["safe_lift_z_m"]), g_pos[2] + clearance, p_pos[2] + clearance), 4)
        pre_grasp_z = round(max(float(grasp_plan["pre_grasp_position_m"][2]), transit_z), 4)
        retreat_z = round(max(p_pos[2] + clearance, transit_z), 4)

        stages = [
            dict(
                id="pre_grasp",
                arm=arm,
                kind="move",
                grasp_position_m=[g_pos[0], g_pos[1], pre_grasp_z],
                angles_deg=list(g_ang),
                gripper=1.0,
            ),
            dict(
                id="grasp",
                arm=arm,
                kind="move",
                grasp_position_m=list(g_pos),
                angles_deg=list(g_ang),
                gripper=0.0,
                grip_guard_preset="non_empty",
            ),
            dict(
                id="lift",
                arm=arm,
                kind="move",
                grasp_position_m=[g_pos[0], g_pos[1], transit_z],
                angles_deg=list(g_ang),
            ),
            dict(
                id="pre_place",
                arm=arm,
                kind="move",
                grasp_position_m=[p_pos[0], p_pos[1], transit_z],
                angles_deg=list(p_ang),
            ),
            dict(
                id="place",
                arm=arm,
                kind="move",
                grasp_position_m=list(p_pos),
                angles_deg=list(p_ang),
                gripper=1.0,
            ),
            dict(
                id="retreat",
                arm=arm,
                kind="move",
                grasp_position_m=[p_pos[0], p_pos[1], retreat_z],
                angles_deg=list(p_ang),
            ),
        ]
        sim_report = self.simulate_stages(stages)
        return dict(
            target_id=grasp_plan["target_id"],
            arm=arm,
            stages=stages,
            grasp_plan=grasp_plan,
            transit_z_m=transit_z,
            simulation=sim_report,
        )


# ------------------------------------------------------------------ 模块级默认实例与便捷函数
_DEFAULT_FIELD: VirtualSpatialField | None = None


def virtual_field(objects=None, cloud=None, table_z: float | None = None, robot_state: dict | None = None) -> VirtualSpatialField:
    """构建或更新默认虚拟空间模拟场 `VirtualSpatialField` 实例。

    参数:
        objects: object_table 字典（含 rows, columns, table_z）或物体列表；省略且已存在默认场时复用当前场。
        cloud: 可选 scene3d Cloud 点云对象。
        table_z: 支撑桌面高度，单位 m (默认 0.765)。
        robot_state: 可选实测双臂状态字典。

    用法示例:
        vf = s3.virtual_field(object_table)
        res = vf.evaluate_grasp('right', [0.10, -0.20, 0.78], [0, 0, -60], target_id='o1')
    """
    global _DEFAULT_FIELD
    if objects is None and cloud is None and table_z is None and robot_state is None and _DEFAULT_FIELD is not None:
        return _DEFAULT_FIELD
    if _DEFAULT_FIELD is None or objects is not None:
        _DEFAULT_FIELD = VirtualSpatialField(objects=objects, cloud=cloud, table_z=table_z, robot_state=robot_state)
    else:
        _DEFAULT_FIELD.update_from_observation(object_table=objects, robot_state=robot_state, cloud=cloud)
    return _DEFAULT_FIELD


def evaluate_grasp(
    arm: str = "right",
    position_m=None,
    angles_deg=None,
    target_id: str | dict | None = None,
    open_aperture_m: float = MAX_OPENING_M,
    contact_region: str = "pad_center",
    objects=None,
    field: VirtualSpatialField | None = None,
) -> dict:
    """在虚拟空间场中评估单次抓取位姿的理论可夹持性、咬合高度与周围物体干涉（详见 VirtualSpatialField.evaluate_grasp）。"""
    vf = field if field is not None else virtual_field(objects=objects)
    return vf.evaluate_grasp(
        arm=arm,
        position_m=position_m,
        angles_deg=angles_deg,
        target_id=target_id,
        open_aperture_m=open_aperture_m,
        contact_region=contact_region,
    )


def solve_optimal_grasp(
    target_id: str | dict | VirtualObject,
    arm: str = "right",
    preferred_yaw_deg: float | None = None,
    open_aperture_m: float = MAX_OPENING_M,
    objects=None,
    field: VirtualSpatialField | None = None,
) -> dict:
    """在虚拟空间场中求解目标物体的零碰撞、最大咬合最优 3D 抓取位姿（详见 VirtualSpatialField.solve_optimal_grasp）。"""
    vf = field if field is not None else virtual_field(objects=objects)
    return vf.solve_optimal_grasp(
        target_id=target_id,
        arm=arm,
        preferred_yaw_deg=preferred_yaw_deg,
        open_aperture_m=open_aperture_m,
    )


def simulate_stages(stages, step_m: float = 0.015, objects=None, field: VirtualSpatialField | None = None) -> dict:
    """在虚拟空间场中动态推演多阶段动作序列并输出碰撞风险与执行后各物体预期位置（详见 VirtualSpatialField.simulate_stages）。"""
    vf = field if field is not None else virtual_field(objects=objects)
    return vf.simulate_stages(stages, step_m=step_m)


def plan_pick_and_place(
    target_id: str | dict | VirtualObject,
    place_position_m,
    arm: str = "right",
    place_angles_deg=None,
    approach_clearance_m: float = 0.06,
    objects=None,
    field: VirtualSpatialField | None = None,
) -> dict:
    """在虚拟空间场中一键生成经过碰撞验证的 6 阶段 pick-and-place 动作序列（详见 VirtualSpatialField.plan_pick_and_place）。"""
    vf = field if field is not None else virtual_field(objects=objects)
    return vf.plan_pick_and_place(
        target_id=target_id,
        place_position_m=place_position_m,
        arm=arm,
        place_angles_deg=place_angles_deg,
        approach_clearance_m=approach_clearance_m,
    )
