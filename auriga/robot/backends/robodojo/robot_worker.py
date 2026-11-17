"""Isaac-thread-owned RoboDojo execution. Invoke with the installed simulator Python."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

from auriga.robot.model.arx_contract import (FRAME_ID, TCP_CONFIG_ID, validate_action, validate_ee_chunk,
    validate_joint_prefix, validate_preview_action)
from auriga.robot.backends.robodojo.worker_protocol import WIRE_PREFIX, serve_worker
from auriga.core.storage.images import camera_path, encode_image, image_settings
from auriga.core.contracts import MAX_HOLD_NATIVE_STEPS
from auriga.core.observation_policy import ACTOR_PROFILE, ACTOR_PROFILES, allows_depth, has_calibration

from auriga.robot.model.embodiments import get_robodojo_robot
from auriga.robot.backends.robodojo.environment import (
    configure_depth_capture, host_articulation_properties, native_eval_config,
    launch_application, create_environment, episode_terminal, native_validity,
)

protocol_output = sys.stdout
protocol_lock = threading.Lock()
# Keep extracted setup and robot bindings in the native adapter's provenance.
ADAPTER_SOURCE_SHA256 = hashlib.sha256(b"\0".join(
    (Path(__file__).parent / name).read_bytes()
    for name in ("robot_worker.py", "environment.py", "../../model/embodiments.py")
)).hexdigest()

def emit(message):
    with protocol_lock:
        print(WIRE_PREFIX + json.dumps(message, allow_nan=False), file=protocol_output, flush=True)


def save_depth_capture(path, depth, image_shape, clip_m):
    """Save metric z-depth and validity without sending arrays through the IPC protocol."""
    import numpy as np
    from auriga.core.contracts import validate_depth_metadata

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    height, width = image_shape[:2]
    fields = dict(depth_path=str(path), depth_kind="z", depth_unit="m",
                  depth_clip_m=[float(item) for item in clip_m], depth_shape=[int(height), int(width)])
    validate_depth_metadata(dict(fields, width=int(width), height=int(height)), "depth_path")
    if depth.shape != (height, width):
        raise ValueError("depth_rgb_shape_mismatch")
    near, far = fields["depth_clip_m"]
    valid = np.isfinite(depth) & (depth > 0) & (depth >= near) & (depth <= far)
    np.savez(path, depth=depth, valid=valid)
    return fields


class RoboDojoAdapter:
    robot_binding = get_robodojo_robot()
    robot_description = robot_binding.robot

    def __init__(self, arguments):
        self.tracking_guard_enabled = bool(getattr(arguments, "tracking_guard", False))
        self.joint_replan_enabled = bool(getattr(arguments, "joint_replan", False))
        self.motion_safety_guard_enabled = bool(getattr(arguments, "motion_safety_guard", True))
        self.trajectory_preview_enabled = bool(getattr(arguments, "trajectory_preview", True))
        self.motion_execution_mode = getattr(arguments, "motion_execution_mode", "resampled")
        if self.motion_execution_mode not in {"resampled", "feedback"}:
            raise ValueError("invalid_motion_execution_mode")
        if self.motion_execution_mode == "feedback" and self.tracking_guard_enabled:
            raise ValueError("feedback_requires_tracking_guard_disabled")
        self.robot_binding = get_robodojo_robot(getattr(arguments, "robot_model_id", "arx-x5"))
        self.robot_description = self.robot_binding.robot
        self.app = launch_application(arguments.gpu)
        import numpy as np

        self.np, self.directory = np, Path(arguments.run_dir)
        self.root, self.task = Path(arguments.robodojo_root), arguments.task
        self.observation_profile = arguments.observation_profile
        # The parent serialized its resolved run settings; do not overlay host
        # configuration again inside the worker, especially during resume.
        self.image_settings = image_settings(json.loads(os.environ["AURIGA_IMAGE_SETTINGS"])) if "AURIGA_IMAGE_SETTINGS" in os.environ else image_settings()
        from auriga.core.config import motion_thresholds
        self.motion_thresholds = (
            motion_thresholds(json.loads(os.environ["AURIGA_MOTION_THRESHOLDS"]))
            if "AURIGA_MOTION_THRESHOLDS" in os.environ
            else motion_thresholds()
        )
        # A parent imported before the image upgrade still expects PNG-only IPC.
        # New RobotBackend always supplies the frozen settings environment field.
        self.legacy_png_ipc = "AURIGA_IMAGE_SETTINGS" not in os.environ
        self.recording, self.recorded_frames = False, []
        self.episode_index, self.evaluator_actions = 0, 0
        self.directory.mkdir(parents=True, exist_ok=True)
        root = self.root
        # Keep this alias call compatible with hd_replay's camera override.
        config = native_eval_config(root, arguments.task, arguments.seed, arguments.layout_id,
                                    self.observation_profile, robot_binding=self.robot_binding)
        self.env, self.layout_acceptance = create_environment(
            arguments, self.app, self.directory, config, self.robot_binding)
        self.robot, self.obs = self.env.robot_manager, self.env.obs_manager
        self.goals = {"left": 1.0, "right": 1.0}
        self.last_dispatched_goals = dict(self.goals)
        self.control_step, self.dispatch_log = 0, None
        # Controller budget counters are also available on a real deployment.
        # Snapshot the configured public limit; never use task score/success.
        self.native_action_budget_limit = int(self.env.step_lim)
        self.native_actions_dispatched = 0
        self.control_hz = float(self.obs.collect_freq)
        self.native_step_ticks = int(self.obs.collect_interval)
        self.arm_step_limits, self.arm_velocity_step_limits = {}, {}
        self.arm_joint_limits, self.arm_joint_names = {}, {}
        for robot, asset in zip(self.robot.robot_list, self.robot.robot_key):
            if robot.type == "target":
                velocity = asset.data.joint_vel_limits[0, robot.arm_joint_indices].cpu().numpy()
                side = robot.arm_name.split("_")[0]
                self.arm_step_limits[side] = np.minimum(velocity / self.control_hz, 0.05)
                self.arm_velocity_step_limits[side] = velocity / self.control_hz
                self.arm_joint_limits[side] = asset.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().tolist()
                self.arm_joint_names[side] = list(robot.arm_joints_name)
        self.gripper_step_limit = min(1.0, 0.2 * self.native_step_ticks, 0.25)
        self.eef_gripper_dwell_step = min(self.gripper_step_limit, 0.18)
        native_step = self.env.step
        def tracked_step(env_idx_list, decimation=1):
            result = native_step(env_idx_list, decimation=decimation)
            self.control_step += decimation
            if self.dispatch_log is not None:
                self.dispatch_log.append({"sim_step": self.control_step, "commands": {
                    name: {kind: np.asarray(value).tolist() for kind, value in command.items()}
                    for name, command in self.robot.control_manager.prev_control[0].items()}})
            if self.recording and self.control_step % 50 == 0:
                self.observe(video_only=True)
            return result
        self.env.step = tracked_step
        self.versions = self._versions(root)
        from auriga.robot.model.robot_model import urdf_manifest
        self.robot_model = urdf_manifest(root / self.robot_binding.urdf_path) if has_calibration(self.observation_profile) else None
        self.seed, self.layout_id = arguments.seed, arguments.layout_id
        self._initialize_evaluator()
        self.camera_calibration = self._camera_calibration() if has_calibration(self.observation_profile) else None
        self._segment_plan_cache = {}
        from auriga.robot.backends.robodojo.scene_truth import make_recorder as make_scene_truth_recorder
        self.scene_truth = make_scene_truth_recorder(getattr(arguments, "scene_truth_log", None), self.env)

    def _versions(self, root):
        def git(*arguments):
            return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
        isaaclab_diff = subprocess.check_output(["git", "-C", str(root / "third_party/IsaacLab"), "diff"], text=True)
        return {"adapter_source_sha256": ADAPTER_SOURCE_SHA256,
                "robodojo_commit": git("rev-parse", "HEAD"), "robodojo_local_diff_sha256": hashlib.sha256(git("diff").encode()).hexdigest(),
                "isaaclab_local_diff_sha256": hashlib.sha256(isaaclab_diff.encode()).hexdigest(),
                "submodules": git("submodule", "status"), "python": sys.version,
                "packages": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab", "torch")},
                "robot_config_sha256": hashlib.sha256((root / self.robot_binding.robot_config_path).read_bytes()).hexdigest(),
                "curobo_config_sha256": hashlib.sha256((root / self.robot_binding.curobo_config_path).read_bytes()).hexdigest()}

    def capabilities(self):
        from auriga.robot.model.roboprobe_pose import GRASP_POINT_OFFSET_M
        from auriga.robot.model.grasp_contact import contact_capabilities
        feedback = getattr(self, "motion_execution_mode", "resampled") == "feedback"
        return {"backend": "robodojo_curobo", "data_origin": "robodojo_runtime", "world_kind": "observed",
                "motion_execution_mode": "feedback" if feedback else "resampled",
                "action_speed": {"available": not feedback,
                    "speed_scale": {"min": .25, "max": 2., "default": 1.},
                    "gripper_speed_scale": {"min": .25, "max": 1., "default": 1.},
                    "waypoint_speed_scales": {"available": not feedback, "min": .25, "max": 2.,
                        "semantics": "one resolved scale per waypoint, overriding top-level speed_scale; omitted list inherits top-level for all segments"},
                    "semantics": "resampled joint-command spacing; top-level speed_scale supplies the default, with optional per-waypoint overrides; no physical speed guarantee",
                    "resampling": "scale 1 preserves legacy sampling; other scales preserve significant turns within 0.001 rad per joint and bound adjacent command deltas by scaled nominal and native velocity limits",
                    "gripper": "scale below 1 ramps from measured opening then retains legacy final-goal dwell; during_motion scales the synchronized ramp limit; direct after_motion jaws compile from fresh measured opening after successful arrival, within the shared request cap; exact cached previews keep their rows and reject gripper_start_state_changed if the old first jaw step exceeds the bound",
                    "accounting": "25 Hz native actions and physics ticks unchanged; slower commands consume more native steps within the same 10 s request cap"},
                "frame_id": self.robot_description.frame_id, "tcp_config_id": self.robot_description.tcp_config_id, "robot_model_id": self.robot_description.robot_model_id,
                "observation_profile": self.observation_profile, "observation_profiles": [self.observation_profile],
                "modes": ["absolute"], "path_kinds": ["backend_free", "hold"],
                "max_waypoints": 8, "max_simultaneous_arms": 1 if feedback else 2,
                "dual_arm": not feedback, "standalone_gripper": True,
                **({"arm_queue": {"available": True, "max_stages_per_arm": 8,
                                  "checkpoint": "stops_whole_request"}} if not feedback else {}),
                "motion_cost_preview": {"available": not feedback and has_calibration(self.observation_profile),
                                        "max_candidates": 3, "max_waypoints": 1},
                "motion_safety_guard": {"enabled": bool(getattr(self, "motion_safety_guard_enabled", False)),
                    "policy": "joint-travel-admission-v1", "scope": "resampled segments before dispatch; no wrapping or automatic replanning"},
                "trajectory_preview": {"available": bool(getattr(self, "trajectory_preview_enabled", False)),
                    "scope": "single-segment candidate joint/FK timeline; robot-model near-approach only; no task-object truth"},
                "action_budget": {"native_step_limit": int(self.env.step_lim) if getattr(self.env, "step_lim", None) is not None else None,
                                  "source": "native_task_configuration"},
                **({"dual_arm_progression": {"available": True, "modes": ["paired", "independent"],
                                            "default": "paired"}} if not feedback else {}),
                "dual_arm_semantics": "unsupported in feedback mode" if feedback else
                    "paired default checks both endpoints at each waypoint barrier; independent advances each arm after its own measured arrival and starts its own after_motion jaw, with final held-endpoint recheck; both initial plans precede dispatch; one merged row per native step, one shared 10 s cap; no joint collision or held-object constraint guarantee",
                "waypoint_semantics": "sequential fixed joint endpoints with measured arrival and stall detection; no planner-path or linear Cartesian guarantee" if feedback else "sequential native plans with endpoint verification; no blending or linear Cartesian guarantee",
                "waypoint_failure_semantics": "prior waypoints may already have executed when a later plan fails; receipt retains partial motion",
                "reset": True, "native_evaluator": "privileged IPC only; never actor observation", "observer_video": True,
                "gripper_phases": ["after_motion"] if feedback else ["after_motion", "during_motion"], "gripper_after_motion": True, "orientation_modes": ["backend_free"],
                "gripper_semantics": {"goal_normalized": "0=closed, 1=open; latched through subsequent moves", "contact_feedback": "unavailable",
                    "during_motion": "unsupported in feedback mode: arrival duration is determined by measurements" if feedback else "linear ramp from latched goal (measured opening for gripper_speed_scale below 1) over start_fraction..end_fraction of final segment native steps; uniformly subdivide motion if needed for command rate limits; command completion does not prove measured opening or grasp"},
                "tool_geometry": {"source": "pinned ARX X5 URDF gripping-face geometry, not task object state",
                                  "link6_to_grasp_center_local_m": [GRASP_POINT_OFFSET_M, 0, 0],
                                  "grasp_center_reference": "distal_15_mm_flat_pad_center",
                                  "contact_selection": contact_capabilities(),
                                  "approach_axis_local": [1, 0, 0], "jaw_opening_axis_local": [0, 1, 0],
                                  "conversion": "p_grasp = p_link6 + R(quaternion_xyzw) @ offset. robot_action stages use kind='move' with a unique id and arm. position_m plus quaternion_xyzw targets link6; alternatively grasp_position_m plus angles_deg targets the selected grasp section. Both pose forms use the same public move stage; no separate EE tool or manual grasp offset is needed."},
                "limits_profile": "robodojo-native-v1", "native_control_hz": self.control_hz,
                "bounded_wait": {"max_native_steps": MAX_HOLD_NATIVE_STEPS, "native_control_hz": self.control_hz,
                                 "semantics": "hold both measured arm joint positions and last gripper goals; each step consumes one native action, not wall-clock sleep"},
                "physics_dt_s": float(self.env.dt), "native_physics_steps_per_action": self.native_step_ticks,
                "arm_max_step_rad": {name: values.tolist() for name, values in self.arm_step_limits.items()},
                "arm_joint_limits_rad": self.arm_joint_limits, "arm_joint_names": self.arm_joint_names,
                "gripper_max_step": self.gripper_step_limit, "motion_duration_limit_s": 10,
                "eef_gripper_dwell_step": self.eef_gripper_dwell_step,
                "joint_prefix": {"max_actions": 50, "dimension": 14, "timing": "one official native action per row"},
                "ee_chunk": {"max_actions": 50, "pose_convention": "xyz_qwxyz", "frame_id": FRAME_ID,
                             "timing": "one official native action per row",
                             "native_interface": "eval_env.take_action *_ee_pose keys; per-arm native IK"},
                "clock_policy": "physics advances only in the worker's bounded execution; paused during agent reasoning",
                "cancellation": "checked before dispatch and between native actions; native planning is not interruptible",
                "collision_scope": "native planned endpoint only; feedback target execution does not follow the collision-checked planner trajectory" if feedback else "native robot self-collision and configured static planner table; no task-object or other-arm collision proof",
                "motion_preview": {"available": has_calibration(self.observation_profile) and not feedback,
                                   "max_waypoints": 1, "execution_reference": "preview_plan_id",
                                   "semantics": "read-only native planning; exact cached resampled plan, single use, rejected after state changes; cached after_motion jaw rows are never rebased and can stop with gripper_start_state_changed after arm motion"},
                "unified_action": {"available": not feedback, "modes": ["preview", "execute", "inspect", "direct"],
                    "max_stages": 32, "max_stages_per_arm": 16, "max_native_steps": int(self.np.ceil(10*self.control_hz)),
                    "checkpoint_resume": "validate current measured state then retain exact cached suffix; never replan",
                    "planning": "complete stage graph, future starts from prior commanded endpoints without physics"},
                "versions": self.versions}

    def _initialize_evaluator(self):
        # EvalEnv.reset already reset and initialized RewardManager. Register the
        # task predicates exactly where native run_eval would register them.
        self.env.run_reward()
        if hasattr(self.env, "get_score"):
            self.env.get_score()
        self.evaluator_actions = 0
        self.evaluator_failure = None

    def reset(self, seed=0, layout_id=0):
        raise ValueError("native_reset_requires_fresh_worker_use_RobotBackend_reset")

    def episode_terminal(self):
        return episode_terminal(self.env)

    def fast_forward_slice(self, checkpoint_spec):
        from auriga.robot.backends.robodojo.native_replay import fast_forward_adapter
        return fast_forward_adapter(self, checkpoint_spec)

    def _matrix(self, pose):
        from transforms3d.quaternions import quat2mat
        transform = self.np.eye(4)
        transform[:3, :3] = quat2mat(self.np.asarray(pose[3:]) / self.np.linalg.norm(pose[3:]))
        transform[:3, 3] = pose[:3]
        return transform

    def _camera_pose(self, camera_id):
        from auriga.robot.backends.robodojo.native_evaluator import host_array
        manager = self.env.camera_manager
        index = manager.camera_names[0].index(camera_id)
        camera = manager.cameras[0][index]
        position, quaternion = camera.get_world_pose(camera_axes="ros")
        transform = self._matrix(list(host_array(position)) + list(host_array(quaternion)))
        transform[:3, 3] -= self.env.env_origins[0].cpu().numpy()
        sample = {"T_world_camera": transform.tolist(), "source_kind": "simulator_camera_transform"}
        mount = manager.camera_config[camera_id].camera.get("mount_link")
        if mount:
            arm = "left" if "left" in camera_id else "right"
            robot = self.robot.get_robot_by_arm_name(arm + "_arm")
            link = str(mount).split("/")[-1]
            key = self.robot.robot_key[self.robot.robot_list.index(robot)]
            parents = {joint["child"]: joint["parent"] for joint in self.robot_model["joints"] if joint["type"] == "fixed"}
            while link not in key.body_names and link in parents:
                link = parents[link]
            sample.update(arm=arm, link_name=link,
                          T_world_link=self._matrix(self.robot.get_link_pose(robot, link, [0], is_relative=True)[0]).tolist(),
                          T_world_base=self._matrix(self.robot.get_link_pose(robot, self.robot_model["base_link"], [0], is_relative=True)[0]).tolist())
            calibration = getattr(self, "camera_calibration", None)
            if calibration is not None:
                # With Fabric enabled, native Camera.get_world_pose reads the unchanged
                # USD stage. The rendered mount follows measured robot link kinematics.
                mounted = self.np.asarray(sample["T_world_link"]) @ self.np.asarray(calibration["cameras"][camera_id]["T_link_camera"])
                sample.update(T_world_camera=mounted.tolist(), source_kind="robot_kinematics",
                              native_usd_pose_diagnostic={"T_world_camera": transform.tolist(),
                                  "translation_residual_m": float(self.np.linalg.norm(mounted[:3, 3] - transform[:3, 3])),
                                  "reason": "USD stage camera pose may be stale under Fabric; never used as dynamic optical pose"})
        return sample

    def _camera_calibration(self):
        from auriga.robot.backends.robodojo.native_evaluator import host_array
        manager, cameras = self.env.camera_manager, {}
        for index, camera_id in enumerate(manager.camera_names[0]):
            camera = manager.cameras[0][index]
            sample = self._camera_pose(camera_id)
            # Isaac 仅按方形像素渲染（与 RoboDojo get_camera_intrinsics 一致），竖直焦距须等于水平焦距。
            intrinsics = self.np.asarray(host_array(camera.get_intrinsics_matrix())).copy()
            intrinsics[1, 1] = intrinsics[0, 0]
            item = {"K": intrinsics.tolist(), "distortion": [0.0] * 5,
                    "distortion_model": "pinhole_no_distortion", "image_size": list(camera.get_resolution()),
                    "optical_axes": "+x right,+y down,+z forward", "calibration_source": "native rendered Camera pinhole parameters and optical child pose"}
            if "T_world_link" in sample:
                item.update(arm=sample["arm"], link_name=sample["link_name"], T_world_base=sample["T_world_base"],
                            T_link_camera=(self.np.linalg.inv(self.np.asarray(sample["T_world_link"])) @ self.np.asarray(sample["T_world_camera"])).tolist())
            else:
                item["T_world_camera"] = sample["T_world_camera"]
            cameras[camera_id] = item
        return {"profile_id": "robodojo-arx-x5-pinhole-v1", "source_kind": "simulator_configuration",
                "reference_frame_id": self.robot_description.frame_id, "cameras": cameras}

    def evaluate(self, final_check=True):
        """Trusted native task predicates; result is only exposed to the observer."""
        if self.evaluator_failure:
            raise RuntimeError(f"native evaluator unavailable: {self.evaluator_failure}")
        terminal = self.episode_terminal()
        reward = float(self.env.reward_manager.get_reward(final_check=final_check and not terminal["ended"])[0])
        task_success = terminal["success"] if terminal["ended"] else reward > 1 - 1e-3
        process_score = float(self.env.reward_manager.get_score([reward])[0]) if hasattr(self.env, "get_score") else None
        # Native run_eval also excludes samples marked unstable after reset
        # (for example, when a support-arm demonstration finishes). Read the
        # live flags without rerunning stability checks or task predicates.
        native_validity = self._native_validity()
        result = {"schema_version": "0.1", "evaluation_id": str(uuid.uuid4()), "task": self.task,
                  "seed": self.seed, "layout_id": self.layout_id, "episode_index": self.episode_index,
                  "task_success": task_success, "native_reward": reward, "native_process_score": process_score,
                  "normalized_score": 1.0 if task_success else (process_score / 100.0 if process_score is not None else 0.0),
                  "valid_episode": (self.layout_acceptance["valid"] and self.layout_acceptance["stable"]
                                    and native_validity["valid"]),
                  "layout_acceptance": self.layout_acceptance,
                  "native_validity": native_validity,
                  "authority": "installed RoboDojo task.run_reward / RewardManager", "audience": "observer_only",
                  "final_check": bool(final_check and not terminal["ended"]), "evaluator_actions": self.evaluator_actions,
                  "success_authority": "locked_native_terminal" if terminal["ended"] else "native_final_check",
                  "episode_terminal": self.episode_terminal(),
                  "sim_step": self.control_step, "step_domain": "adapter_control_step", "versions": self.versions}
        directory = self.directory / "evaluations"
        directory.mkdir(exist_ok=True)
        path = directory / f"{result['evaluation_id']}.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        return {**result, "evaluation_path": str(path)}

    def _native_validity(self):
        return native_validity(self.env)

    def record(self, enabled=True):
        self.recording = enabled
        if enabled:
            self.recorded_frames = []
            self.video_export = None
            self.observe(video_only=True)
        return {"enabled": enabled, "sampling": "actual RGB every 50 adapter steps (5 Hz), plus observation boundaries"}

    def observer_snapshot(self):
        from auriga.robot.backends.robodojo.native_evaluator import host_array
        manager = self.env.scene_manager.layout_manager
        objects = []
        for kind in ("Rigid", "Articulation", "Garment"):
            for record in manager.get_layout_records(0, kind):
                position, quaternion = manager.get_instance_pose(inst_name=record["inst_name"], env_idx=0)
                item = {"instance_name": record["inst_name"], "kind": kind,
                        "position_m": self.np.asarray(host_array(position)).tolist(), "quaternion_wxyz": self.np.asarray(host_array(quaternion)).tolist()}
                obj = manager.get_scene_object(0, record["inst_name"])
                if hasattr(obj, "_rigid_prim_view"):
                    item["velocity_linear_m_s_angular_rad_s"] = self.np.asarray(host_array(obj._rigid_prim_view.get_velocities()))[0].tolist()
                objects.append(item)
        return {"audience": "observer_only", "source_kind": "simulator_hidden_scene_state", "frame_id": self.robot_description.frame_id,
                "sim_step": self.control_step, "step_domain": "adapter_control_step", "objects": objects}

    def save_video(self):
        from concurrent.futures import ThreadPoolExecutor
        import cv2
        if getattr(self, "video_export", None) is not None:
            return self.video_export
        if self.recording:
            self.observe(video_only=True)
        self.recording = False
        if not self.recorded_frames:
            raise ValueError("no recorded frames; call record before execution")
        directory = self.directory / "videos" / str(uuid.uuid4())
        directory.mkdir(parents=True)
        cameras = list(self.recorded_frames[0]["cameras"])

        def _encode_camera(camera):
            name, width, height = camera["camera_id"], camera["width"], camera["height"]
            path = directory / f"{name}.mp4"
            log_path = directory / f"{name}.ffmpeg.log"
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                       "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}", "-framerate", "5",
                       "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
            mapping = []
            try:
                with log_path.open("xb") as log:
                    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log) as encoder:
                        for frame in self.recorded_frames:
                            source = next(item for item in frame["cameras"] if item["camera_id"] == name)
                            image = cv2.imread(str(camera_path(source)))
                            if image is None or image.shape != (height, width, 3):
                                raise ValueError(f"missing or mismatched RGB frame: {camera_path(source)}")
                            encoder.stdin.write(image.tobytes())
                            mapping.append({"video_frame": len(mapping), "pts_s": len(mapping) / 5.0,
                                            "sim_step": frame["sim_step"], "step_domain": frame["step_domain"],
                                            "sim_time_s": frame["sim_time_s"], "source_rgb": str(camera_path(source)),
                                            "source_retained": True,
                                            "source_sha256": source.get("image_encoding", {}).get("sha256"),
                                            "raw_observation_id": frame["raw_observation_id"], "frame_id": source["frame_id"]})
                    if encoder.returncode:
                        raise RuntimeError(f"ffmpeg exited with status {encoder.returncode}")
                    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error",
                        "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=codec_name,pix_fmt,nb_read_frames",
                        "-of", "json", str(path)], stderr=log))["streams"][0]
                    if probe["codec_name"] != "h264" or probe["pix_fmt"] != "yuv420p" or int(probe["nb_read_frames"]) != len(mapping):
                        raise ValueError("encoded video codec, pixel format or frame count mismatch")
            except Exception as error:
                raise RuntimeError(f"video encoding failed: {path}; inspect {log_path}: {error}") from error
            return {"camera_id": name, "video_path": str(path), "frames": len(mapping), "fps": 5.0,
                    "codec": "h264", "pixel_format": "yuv420p", "faststart": True, "encoder_log_path": str(log_path),
                    "timing": "5 fps observation sequence; frame mapping carries actual nonuniform simulation times", "frame_mapping": mapping}

        with ThreadPoolExecutor(max_workers=max(1, min(3, len(cameras)))) as pool:
            futures = [pool.submit(_encode_camera, camera) for camera in cameras]
            videos = [future.result() for future in futures]
        manifest = {"audience": "observer_only", "data_origin": "robodojo_runtime", "videos": videos}
        # Seal the verified video/index before removing its temporary inputs.
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        # Only private video-only frames are temporary. Published observation
        # paths and immutable Store artifacts remain available to history/tools.
        cleanup_errors = []
        for frame in self.recorded_frames:
            if frame.get("video_only"):
                for camera in frame["cameras"]:
                    try:
                        camera_path(camera).unlink(missing_ok=True)
                    except OSError as error:
                        cleanup_errors.append(dict(path=str(camera_path(camera)), error=type(error).__name__))
                try:
                    camera_path(frame["cameras"][0]).parent.rmdir()
                except OSError:
                    pass  # A failed unlink or extra file must not invalidate the video.
        for video in videos:
            for frame in video["frame_mapping"]:
                frame["source_retained"] = Path(frame["source_rgb"]).exists()
        manifest["temporary_frame_cleanup_errors"] = cleanup_errors
        self.video_export = manifest
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    def state(self):
        arms = {}
        for robot in self.robot.robot_list:
            if robot.type != "target":
                continue
            name = robot.arm_name.split("_")[0]
            key = self.robot.robot_key[self.robot.robot_list.index(robot)]
            pose = self.robot.get_real_endpose(robot, [0])[0].tolist()
            q_rad = self.robot.get_joint(robot, [0])[0].tolist()
            qdot_rad_s = key.data.joint_vel[0, robot.arm_joint_indices].cpu().tolist()
            ee_pos = self.robot.get_end_effector_real_val(robot, [0])[0].tolist()
            qxyzw = self.np.asarray(pose[4:7] + pose[3:4], dtype=float)
            norm = float(self.np.linalg.norm(qxyzw))
            if (not self.np.isfinite(pose).all() or not self.np.isfinite(q_rad).all()
                    or not self.np.isfinite(qdot_rad_s).all() or not self.np.isfinite(ee_pos).all()
                    or not self.np.isfinite(norm) or norm <= 1e-12):
                raise RuntimeError(f"simulator_nonfinite_robot_state:{name}")
            qxyzw = qxyzw / norm
            gripper_key = self.robot.process_name(robot.gripper_name)
            command = self.robot.control_manager.prev_control[0][gripper_key]["position"][0]
            lower, upper = robot.gripper_scale
            normalized = (command - lower) / (upper - lower) if robot.gripper_move["sign"] == 1 else (upper - command) / (upper - lower)
            arms[name] = {"joint_names": list(robot.arm_joints_name), "q_rad": q_rad,
                "qdot_rad_s": qdot_rad_s, "source_kind": "measurement",
                "ee_pose": {"position_m": pose[:3], "quaternion_xyzw": qxyzw.tolist(), "frame_id": self.robot_description.frame_id,
                            "tcp_config_id": self.robot_description.tcp_config_id, "source_kind": "measurement"},
                "gripper": {"goal_normalized": self.goals[name], "goal_source_kind": "command",
                    "command_normalized": float(normalized), "command_source_kind": "command",
                    "joint_names": list(robot.gripper_joints_name),
                    "position_m": ee_pos, "position_source_kind": "measurement",
                    "contact_feedback": None, "contact_feedback_reason": "no contact sensor exposed"}}
        return {"arms": arms, "frame_id": self.robot_description.frame_id, "tcp_config_id": self.robot_description.tcp_config_id,
                "sim_step": self.control_step, "step_domain": "adapter_control_step"}

    def observe(self, *, video_only=False):
        from PIL import Image
        from auriga.core.observation_policy import actor_robot_state
        if video_only and self.recorded_frames and self.recorded_frames[-1]["sim_step"] == self.control_step:
            return None
        self.obs.render_for_capture()
        raw = self.obs.get_obs([0])[0]
        observation_id = "raw-" + str(uuid.uuid4())
        directory = self.directory / "observations" / observation_id
        directory.mkdir(parents=True)
        cameras = []
        for camera_id, values in raw["vision"].items():
            image = self.np.asarray(values["color"], dtype=self.np.uint8)
            if getattr(self, "legacy_png_ipc", False):
                path = directory / f"{camera_id}.png"
                Image.fromarray(image).save(path)
                image_fields = {"png_path": str(path)}
            else:
                path = directory / f"{camera_id}.jpg"
                data, encoding = encode_image(Image.fromarray(image), self.image_settings, storage=True)
                path.write_bytes(data)
                image_fields = {"rgb_path": str(path), "image_encoding": encoding}
            cameras.append({"camera_id": camera_id, "frame_id": f"{observation_id}-{camera_id}", **image_fields,
                "width": int(image.shape[1]), "height": int(image.shape[0]), "pixel_transform": "identity",
                "capture_alignment": "physics paused; native zero-delay camera delivery fence", "delivered_to_actor": False})
            if allows_depth(self.observation_profile) and not video_only:
                if "depth" not in values:
                    raise ValueError("rgbd_profile_requires_native_depth:" + camera_id + ":fields=" + ",".join(sorted(values)))
                manager = self.env.camera_manager
                camera = manager.cameras[0][manager.camera_names[0].index(camera_id)]
                cameras[-1].update(save_depth_capture(directory / f"{camera_id}.depth.npz", values["depth"],
                                                      image.shape, camera.get_clipping_range()))
        observation = {"raw_observation_id": observation_id, "data_origin": "robodojo_runtime", "world_kind": "observed",
                "observation_profile": self.observation_profile, "robot_model_id": self.robot_description.robot_model_id, "cameras": cameras,
                "robot_state": actor_robot_state(self.state(), "rgb_proprio_v1") if self.observation_profile == "rgb_proprio_v1" else self.state(),
                "sim_step": self.control_step, "step_domain": "adapter_control_step",
                "control_dt_s": float(self.env.dt), "sim_time_s": float(self.env.sim.sim.current_time),
                "captured_at": datetime.now(timezone.utc).isoformat(), "instruction": raw.get("instruction"),
                "versions": self.versions}
        if self.camera_calibration is not None:
            observation.update(camera_calibration=self.camera_calibration, robot_model=self.robot_model,
                camera_pose_sample={"reference_frame_id": self.robot_description.frame_id, "sim_step": self.control_step,
                    "step_domain": "adapter_control_step", "cameras": {item["camera_id"]: {
                        **{key: value for key, value in self._camera_pose(item["camera_id"]).items()
                           if key != "native_usd_pose_diagnostic"}, "frame_id": item["frame_id"]} for item in cameras}})
        if not video_only and getattr(self, "scene_truth", None) is not None:
            self.scene_truth.record(observation_id, self.control_step, observation["robot_state"], observation["sim_time_s"])
        if self.recording and (not self.recorded_frames or self.recorded_frames[-1]["sim_step"] != self.control_step):
            self.recorded_frames.append({"raw_observation_id": observation_id, "sim_step": self.control_step,
                                         "step_domain": observation["step_domain"], "sim_time_s": observation["sim_time_s"],
                                         "video_only": video_only, "cameras": cameras})
        return observation

    def _native_action(self, row):
        if self.env.end_flag[0]:
            return False
        action = {"left_arm_joint_state": row[:6], "left_ee_joint_state": row[6:7],
                  "right_arm_joint_state": row[7:13], "right_ee_joint_state": row[13:14]}
        # Count dispatched control commands locally; this does not inspect task
        # objects, evaluator success or the simulator's collision state.
        if hasattr(self, "native_actions_dispatched"):
            self.native_actions_dispatched += 1
        self.env.take_action(action)
        self.goals = {"left": float(self.np.clip(row[6], 0, 1)), "right": float(self.np.clip(row[13], 0, 1))}
        self.last_dispatched_goals = dict(self.goals)
        self.evaluator_actions = int(self.env.take_action_cnt[0])
        return True

    def _native_ee_action(self, row):
        """Dispatch one policy-authored EE row through RoboDojo's native EE interface.

        eval_env.take_action infers the "ee" action type from the *_ee_pose keys
        and solves IK per arm; an arm whose IK fails keeps its previous joint
        target. The per-arm IK outcome is recorded from that same native call,
        never re-solved, so the receipt can report it.
        """
        if self.env.end_flag[0]:
            return False
        statuses, manager = {}, self.robot
        native_solve, own_attribute = manager.solve_ik, "solve_ik" in vars(manager)

        def recorded_solve(*args, **kwargs):
            result = native_solve(*args, **kwargs)
            robot = kwargs.get("robot", args[2] if len(args) > 2 else None)
            side = robot.arm_name.split("_")[0] if robot is not None else "unknown"
            statuses[side] = isinstance(result, dict) and result.get("status") == "Success"
            return result

        manager.solve_ik = recorded_solve
        try:
            if hasattr(self, "native_actions_dispatched"):
                self.native_actions_dispatched += 1
            self.env.take_action({key: [float(x) for x in value] for key, value in row.items()})
        finally:
            if own_attribute:
                manager.solve_ik = native_solve
            else:
                del manager.solve_ik
            self.last_ee_ik = statuses
        self.goals = {side: float(self.np.clip(row[side + "_ee_joint_state"][0], 0, 1)) for side in ("left", "right")}
        self.last_dispatched_goals = dict(self.goals)
        self.evaluator_actions = int(self.env.take_action_cnt[0])
        return True

    def native_policy_observation(self):
        """Native observation dictionary for an external policy; physics does not advance."""
        self.obs.render_for_capture()
        return self.obs.get_obs([0])[0]

    def policy_observation(self):
        """Read-only capture of an external policy's inputs at the current step."""
        from auriga.robot.motion.ee_chunk_motion import PolicyCaptureWriter
        writer = PolicyCaptureWriter(self.directory / "policy_captures")
        writer.add(self.control_step, self.native_policy_observation())
        return dict(writer.write(), step_domain="adapter_control_step", physics_advanced=False)

    def _hold(self):
        # Native execution is synchronous: once it returns there are no queued
        # controls. Pausing the clock stops without spending hidden action steps.
        self.goals = dict(self.last_dispatched_goals)
        return {"clock": "paused", "actuator_target": "last_native_command_retained", "settling_steps": 0}

    def _held_row(self, state):
        return [value for side in ("left", "right")
                for value in [*state["arms"][side]["q_rad"], self.last_dispatched_goals[side]]]

    def _resample_plan(self, positions, arm_name):
        # Pinned official EEF policy: per-joint total variation / declared step,
        # then sample the original polyline starting one interval after its start.
        travel = self.np.abs(self.np.diff(positions, axis=0)).sum(axis=0)
        steps = max(1, int(self.np.ceil(travel / self.arm_step_limits[arm_name]).max()))
        source = self.np.linspace(0, 1, len(positions))
        target = self.np.linspace(0, 1, steps + 1)[1:]
        return self.np.stack([self.np.interp(target, source, positions[:, joint])
                             for joint in range(positions.shape[1])], axis=1)

    def _speed_step_limits(self, arm_name, speed_scale):
        nominal = self.arm_step_limits[arm_name]
        # Older CPU fixtures expose only nominal limits. Never infer extra
        # velocity headroom when the native limit is unavailable.
        native = getattr(self, "arm_velocity_step_limits", {}).get(arm_name, nominal)
        return self.np.minimum(native, nominal * speed_scale)

    @staticmethod
    def _waypoint_speed_scale(action, waypoint_index=0):
        scales = action.get("waypoint_speed_scales")
        return float(scales[waypoint_index] if scales is not None else action.get("speed_scale", 1.))

    def _resample_action_plan(self, positions, arm_name, action, waypoint_index=0, *, start_q=None):
        scale = self._waypoint_speed_scale(action, waypoint_index)
        if scale == 1:
            return self._resample_plan(positions, arm_name)
        from auriga.robot.motion.motion_speed import bounded_resample
        # Include the measured-to-first-target interval in the same command
        # bound; native planners need not return a duplicate start knot.
        current = self.np.asarray(self.state()["arms"][arm_name]["q_rad"] if start_q is None else start_q)
        if not self.np.array_equal(current, positions[0]):
            positions = self.np.vstack([current, positions])
        return bounded_resample(positions, self._speed_step_limits(arm_name, scale))

    def _joint_replan_geometry(self, robot, planner, target):
        from auriga.robot.model.geometry import fk
        from auriga.robot.model.robot_model import urdf_manifest
        from auriga.robot.model.roboprobe_pose import quat_to_matrix
        np = self.np
        names = list(robot.arm_joints_name)
        config = planner.robot_cfg["robot_cfg"]["kinematics"]
        path = Path(config["urdf_path"])
        if not path.is_absolute():
            path = Path(config["asset_root_path"]) / path
        if (len(names) != 6 or list(planner.cspace_joint_names) != names
                or list(planner.active_joints_name) != names or list(planner.tool_frames) != ["link6"]
                or np.asarray(planner.frame_bias).shape != (3,) or np.any(np.asarray(planner.frame_bias) != 0)
                or path.resolve() != (self.root / self.robot_binding.urdf_path).resolve()
                or config.get("extra_links")):
            raise ValueError("unsupported_joint_replan_geometry")
        model = getattr(self, "_joint_replan_model", None)
        if model is None:
            model = urdf_manifest(path)
            self._joint_replan_model = model
            self._joint_replan_urdf_path = str(path.resolve())
        if config.get("base_link") != model["base_link"]:
            raise ValueError("joint_replan_base_link_mismatch")
        joints = {joint["name"]: joint for joint in model["joints"]}
        joint_types = [joints[name]["type"] for name in names]
        # Remaining X5 joints are prismatic jaws. Their fixed common value cannot
        # affect either link6 or the equality of all links under revolute turns.
        other = [joint for joint in model["joints"] if joint["name"] not in names and joint["type"] != "fixed"]
        if any(joint["type"] != "prismatic" for joint in other):
            raise ValueError("unsupported_unplanned_joint")
        pose = np.asarray(robot.entity_origin_pose, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("invalid_robot_base_pose")
        base = np.eye(4)
        base[:3, :3], base[:3, 3] = quat_to_matrix(pose[3:]), pose[:3]
        quaternion = target["quaternion_xyzw"]
        rotation = quat_to_matrix(np.asarray(quaternion[3:] + quaternion[:3]))
        def transforms(q):
            return fk(model, {"joint_names": names + [joint["name"] for joint in other],
                              "q_rad": list(q) + [0.] * len(other), "source_kind": "estimate"})
        def pose_error(q):
            world = base @ transforms(q)["link6"]
            cosine = (np.trace(rotation.T @ world[:3, :3]) - 1) / 2
            return (float(np.linalg.norm(world[:3, 3] - target["position_m"])),
                    float(np.arccos(np.clip(cosine, -1, 1))))
        return joint_types, transforms, pose_error

    def _feedback_endpoint(self, robot, target, current, goal, position_tolerance_m, diagnostic):
        """Select a legal equivalent endpoint, independently checked against URDF FK.

        Feedback already commands an endpoint rather than following the native
        trajectory. This changes only that endpoint's revolute turn count; it
        neither wraps/clips arbitrary joints nor claims collision-free transit.
        Validation failures must reject the move, not dispatch a winding fallback.
        """
        from auriga.robot.motion.joint_replan import nearest_equivalent
        np = self.np
        planner = self.robot.planner[robot.robot_name]
        joint_types, transforms, pose_error = self._joint_replan_geometry(robot, planner, target)
        key = self.robot.robot_key[self.robot.robot_list.index(robot)]
        limits = key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
        goal, current = np.asarray(goal, dtype=float), np.asarray(current, dtype=float)
        equivalent, turns = nearest_equivalent(current, goal, limits, joint_types)
        diagnostic.update(original_goal_q_rad=goal.tolist(), selected_goal_q_rad=equivalent.tolist(),
                          turn_shifts=turns, original_endpoint_distance_rad=float(np.linalg.norm(goal - current)),
                          selected_endpoint_distance_rad=float(np.linalg.norm(equivalent - current)))
        old, new = transforms(goal), transforms(equivalent)
        if not old or old.keys() != new.keys():
            raise ValueError("invalid_fk_links")
        differences = []
        for name in old:
            first, second = np.asarray(old[name]), np.asarray(new[name])
            if first.shape != (4, 4) or second.shape != (4, 4) or not np.isfinite([first, second]).all():
                raise ValueError("invalid_fk_transform")
            differences.append(float(np.max(np.abs(first - second))))
        diagnostic["all_link_fk_max_abs_error"] = max(differences)
        if max(differences) > 1e-9:
            raise ValueError("endpoint_not_fk_equivalent")
        for name, q in (("original_pose", goal), ("selected_pose", equivalent)):
            error = np.asarray(pose_error(q), dtype=float)
            if error.shape != (2,) or not np.isfinite(error).all() or np.any(error < 0):
                raise ValueError("invalid_pose_error")
            diagnostic[name] = {"position_error_m": float(error[0]), "orientation_error_rad": float(error[1])}
            if error[0] > position_tolerance_m or error[1] > 0.1:
                raise ValueError("endpoint_pose_target_mismatch")
        model = getattr(self, "_joint_replan_model", None)
        if model is not None:
            diagnostic.update(robot_model_sha256=model["source_sha256"],
                              robot_model_urdf_path=self._joint_replan_urdf_path)
        diagnostic.update(validated=True, changed=bool(any(turns)), reason="validated_nearest_legal_endpoint")
        return equivalent

    def _segment_cache_key(self, robot, current, native_target, joint_replan_context):
        """生成单段运动规划的缓存键，确保相同起点、目标、动态障碍物与重规划参数直接复用已校验轨迹。"""
        replan_enabled = bool(getattr(self, "joint_replan_enabled", False) and joint_replan_context is not None)
        speed = round(float(joint_replan_context.get("speed_scale", 1.0)), 6) if replan_enabled else None
        pos_tol = round(float(min(0.005, joint_replan_context.get("position_tolerance_m", 0.005))), 7) if replan_enabled else None
        origin = getattr(robot, "entity_origin_pose", None)
        try:
            origin_key = tuple(round(float(x), 7) for x in self.np.asarray(origin, dtype=float).ravel())
        except (TypeError, ValueError):
            origin_key = id(origin)
        obstacles = getattr(self, "_dynamic_scene_obstacles", None) or []
        obs_key = tuple(sorted(
            (str(b.get("id", "")), tuple(b.get("min_m", ())), tuple(b.get("max_m", ())), float(b.get("margin_m", 0.0)))
            for b in obstacles if isinstance(b, dict)
        ))
        return (
            robot.robot_name,
            tuple(round(float(x), 7) for x in self.np.asarray(current, dtype=float).ravel()),
            tuple(round(float(x), 7) for x in native_target),
            origin_key,
            obs_key,
            replan_enabled,
            speed,
            pos_tol,
        )

    def _plan_segment(self, robot, target, planning_diagnostics=None, waypoint_index=None, *, joint_replan_context=None, start_q=None):
        quaternion = target["quaternion_xyzw"]
        native_target = target["position_m"] + quaternion[3:] + quaternion[:3]
        current = self.robot.get_joint(robot, [0])[0] if start_q is None else self.np.asarray(start_q, dtype=float)
        cache = getattr(self, "_segment_plan_cache", None)
        interrupted_fn = joint_replan_context.get("interrupted", lambda: False) if joint_replan_context else (lambda: False)
        cache_key = self._segment_cache_key(robot, current, native_target, joint_replan_context) if isinstance(cache, dict) else None
        if cache_key is not None and cache_key in cache and not interrupted_fn():
            cached_positions, cached_attempts, cached_diagnostics = cache[cache_key]
            if planning_diagnostics is not None:
                planning_diagnostics.extend(dict(deepcopy(item), waypoint_index=waypoint_index, cached=True) for item in cached_diagnostics)
            return cached_positions.copy(), cached_attempts
        diag_start = len(planning_diagnostics) if planning_diagnostics is not None else 0
        planned = self._plan_segment_impl(
            robot, target, native_target, current, planning_diagnostics,
            waypoint_index=waypoint_index, joint_replan_context=joint_replan_context)
        if cache_key is not None and planned is not None and not interrupted_fn():
            if len(cache) >= 128:
                cache.pop(next(iter(cache)))
            added_diag = deepcopy(planning_diagnostics[diag_start:]) if planning_diagnostics is not None else []
            cache[cache_key] = (planned[0].copy(), planned[1], added_diag)
        return planned

    def _plan_segment_impl(self, robot, target, native_target, current, planning_diagnostics=None, *, waypoint_index=None, joint_replan_context=None):
        planner = self.robot.planner[robot.robot_name]
        with ExitStack() as restore:
            def capture_stage(stage, path):
                owner = planner.motion_planner
                for name in path[:-1]: owner = getattr(owner, name, None)
                name = path[-1]
                native = getattr(owner, name, None)
                if not callable(native): return
                namespace = vars(owner)
                def capture(*args, **kwargs):
                    started, result, failure = time.monotonic(), None, None
                    try:
                        result = native(*args, **kwargs)
                        return result
                    except BaseException as error:
                        failure = type(error).__name__
                        raise
                    finally:
                        diagnostic = {"waypoint_index": waypoint_index, "stage": stage,
                                      "elapsed_wall_s": time.monotonic() - started,
                                      "returned_none": result is None if failure is None else None,
                                      "exception_type": failure, "status": None, "success": None}
                        try:
                            status, success = getattr(result, "status", None), getattr(result, "success", None)
                            if hasattr(success, "detach"): success = success.detach().cpu()
                            if hasattr(success, "tolist"): success = success.tolist()
                            values = {"status": None if status is None else str(status), "success": success}
                            json.dumps(values, allow_nan=False)
                            diagnostic.update(values)
                            from auriga.robot.motion.motion_failure import solver_snapshot
                            diagnostic.update(solver_snapshot(result, stage,
                                active_joint_names=getattr(planner.motion_planner, "joint_names", robot.arm_joints_name)))
                        except Exception as error:
                            diagnostic["diagnostic_error"] = type(error).__name__
                        planning_diagnostics.append(diagnostic)

                # This worker owns the planner on one Isaac thread. Preserve the
                # instance dictionary exactly, including an existing override.
                if name in namespace: restore.callback(namespace.__setitem__, name, namespace[name])
                else: restore.callback(namespace.pop, name, None)
                namespace[name] = capture
            if planning_diagnostics is not None:
                for stage, path in (("plan_pose", ("plan_pose",)), ("ik", ("ik_solver", "solve_pose")),
                                    ("trajopt", ("trajopt_solver", "solve_pose")),
                                    ("graph_seed", ("_get_graph_seed_trajectories",))):
                    try:
                        capture_stage(stage, path)
                    except Exception as error:
                        planning_diagnostics.append({"waypoint_index": waypoint_index, "stage": stage,
                                                     "diagnostic_error": type(error).__name__})
            result = planner.plan_path(current, native_target, robot.entity_origin_pose)
        key = self.robot.robot_key[self.robot.robot_list.index(robot)]
        limits = key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
        if result["status"] != "Success":
            if getattr(self, "joint_replan_enabled", False) and joint_replan_context is not None and planning_diagnostics:
                ik_candidates = [
                    d.get("candidate") for d in planning_diagnostics
                    if isinstance(d, dict) and d.get("stage") == "ik" and isinstance(d.get("candidate"), dict)
                    and d["candidate"].get("q_rad")
                ]
                if ik_candidates and not joint_replan_context.get("interrupted", lambda: False)():
                    cand = ik_candidates[-1]
                    ik_q = self.np.asarray(cand["q_rad"][-1], dtype=float)
                    if cand.get("solver_feasible") is True or (
                        cand.get("position_error_m", 1.0) <= 0.005 and cand.get("orientation_error_rad", 1.0) <= 0.05
                    ):
                        from auriga.robot.motion.joint_replan import replan_from_ik_candidate
                        diagnostic = {"waypoint_index": waypoint_index, "stage": "joint_replan_from_ik",
                                      "selected": False, "attempted": False}
                        try:
                            joint_types, transforms, pose_error = self._joint_replan_geometry(robot, planner, target)
                            segment_speed = joint_replan_context.get("speed_scale", 1.)
                            pos_tol = min(0.005, joint_replan_context.get("position_tolerance_m", 0.005))
                            resample_fn = lambda value: self._resample_action_plan(
                                value, robot.arm_name.split("_")[0], {"speed_scale": segment_speed}, start_q=current)
                            rescued = replan_from_ik_candidate(
                                ik_q=ik_q, current=current, limits=limits,
                                joint_types=joint_types, transforms=transforms, pose_error=pose_error,
                                resample=resample_fn, plan_joint=planner.plan_joint,
                                deadline=joint_replan_context["deadline"],
                                diagnostic=diagnostic, clock=time.monotonic, position_tolerance_m=pos_tol)
                            if rescued is None and cand.get("feasible_q_rads"):
                                seeds = sorted(
                                    (self.np.asarray(q, dtype=float) for q in cand["feasible_q_rads"]),
                                    key=lambda q: (current[1] >= 0.05 and q[1] < -0.05, float(self.np.linalg.norm(q - current))),
                                )
                                for alt_q in seeds[:2]:
                                    if float(self.np.max(self.np.abs(alt_q - ik_q))) <= 1e-4:
                                        continue
                                    rescued = replan_from_ik_candidate(
                                        ik_q=alt_q, current=current, limits=limits,
                                        joint_types=joint_types, transforms=transforms, pose_error=pose_error,
                                        resample=resample_fn, plan_joint=planner.plan_joint,
                                        deadline=joint_replan_context["deadline"],
                                        diagnostic=diagnostic, clock=time.monotonic,
                                        position_tolerance_m=pos_tol, allow_zero_turns=True)
                                    if rescued is not None:
                                        break
                        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError) as error:
                            diagnostic.update(reason="geometry_validation_failed", validation_error=str(error))
                            rescued = None
                        planning_diagnostics.append(diagnostic)
                        if rescued is not None:
                            return rescued, 1
            return None
        positions = self.np.asarray(result["position"], dtype=float)
        if positions.ndim != 2 or not len(positions) or positions.shape[1] != len(robot.arm_joints_name) or not self.np.isfinite(positions).all():
            raise ValueError("invalid native trajectory")
        if ((positions < limits[:, 0] - 1e-5) | (positions > limits[:, 1] + 1e-5)).any():
            raise ValueError("native trajectory exceeds joint limits")
        if getattr(self, "joint_replan_enabled", False) and joint_replan_context is not None:
            from auriga.robot.motion.joint_replan import select_joint_replan
            diagnostic = {"waypoint_index": waypoint_index, "stage": "joint_replan", "selected": False, "attempted": False}
            if planning_diagnostics is not None:
                planning_diagnostics.append(diagnostic)
            if joint_replan_context.get("interrupted", lambda: False)():
                diagnostic["reason"] = "action_interrupted_before_replan"
                return positions, 1
            try:
                joint_types, transforms, pose_error = self._joint_replan_geometry(robot, planner, target)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError) as error:
                diagnostic.update(reason="geometry_validation_failed", validation_error=str(error))
                return positions, 1
            model = getattr(self, "_joint_replan_model", None)
            if model is not None:
                diagnostic["robot_model_sha256"] = model["source_sha256"]
                diagnostic["robot_model_urdf_path"] = self._joint_replan_urdf_path
            # Compare candidate costs with the same rate and resampler that will
            # dispatch this segment, including per-waypoint speed overrides.
            segment_speed = joint_replan_context.get("speed_scale", 1.)
            diagnostic["speed_scale"] = segment_speed
            pos_tol = min(0.005, joint_replan_context.get("position_tolerance_m", 0.005))
            resample_fn = lambda value: self._resample_action_plan(
                value, robot.arm_name.split("_")[0], {"speed_scale": segment_speed}, start_q=current)
            positions = select_joint_replan(original=positions, current=current, limits=limits,
                joint_types=joint_types, transforms=transforms, pose_error=pose_error,
                resample=resample_fn, plan_joint=planner.plan_joint, deadline=joint_replan_context["deadline"],
                diagnostic=diagnostic, clock=time.monotonic, position_tolerance_m=pos_tol)
            if (not diagnostic.get("selected") and planning_diagnostics
                    and current[1] >= 0.05 and positions[-1, 1] < -0.05
                    and abs(float(positions[-1, 0] - current[0])) >= 1.8
                    and float(self.np.abs(self.np.diff(positions, axis=0)).sum()) >= 7.0):
                from auriga.robot.motion.joint_replan import replan_from_ik_candidate
                ik_cands = [
                    d.get("candidate") for d in planning_diagnostics
                    if isinstance(d, dict) and d.get("stage") == "ik" and isinstance(d.get("candidate"), dict)
                    and d["candidate"].get("feasible_q_rads")
                ]
                if ik_cands:
                    normal_seeds = sorted(
                        (
                            self.np.asarray(q, dtype=float) for q in ik_cands[-1]["feasible_q_rads"]
                            if q[1] >= 0.0 and abs(float(q[0] - current[0])) < 1.8
                        ),
                        key=lambda q: float(self.np.linalg.norm(q - current)),
                    )
                    if normal_seeds:
                        branch_diag = {"waypoint_index": waypoint_index, "stage": "joint_replan_from_ik",
                                       "selected": False, "attempted": False}
                        alt = replan_from_ik_candidate(
                            ik_q=normal_seeds[0], current=current, limits=limits,
                            joint_types=joint_types, transforms=transforms, pose_error=pose_error,
                            resample=resample_fn, plan_joint=planner.plan_joint,
                            deadline=joint_replan_context["deadline"],
                            diagnostic=branch_diag, clock=time.monotonic,
                            position_tolerance_m=pos_tol, allow_zero_turns=True)
                        if alt is not None:
                            old_rows = len(resample_fn(positions))
                            old_travel = float(self.np.abs(self.np.diff(positions, axis=0)).sum())
                            new_metrics = branch_diag.get("candidate") or {}
                            if (new_metrics.get("resampled_rows", old_rows) < old_rows
                                    and new_metrics.get("total_joint_travel_rad", old_travel) < old_travel - 1.0):
                                positions = alt
                            else:
                                branch_diag.update(selected=False, reason="no_strict_path_improvement")
                        planning_diagnostics.append(branch_diag)
        return positions, 1

    def _failure_feedback(self, action, arm, waypoint_index, diagnostics):
        from auriga.robot.motion.motion_failure import failure_feedback
        from auriga.robot.motion.planner_witness import capture_candidate_witness

        def witness(q_rad, joint_names, joint_derivatives=None):
            robot = self.robot.get_robot_by_arm_name(arm + "_arm")
            planner = self.robot.planner[robot.robot_name]
            options = {"joint_derivatives": joint_derivatives} if joint_derivatives is not None else {}
            return capture_candidate_witness(planner, q_rad, joint_names, **options)

        from auriga.robot.motion.planner_witness import SCOPE
        return failure_feedback(action, arm, waypoint_index, diagnostics, SCOPE, witness)

    @staticmethod
    def _preview_action_digest(action):
        # Identity, observation references and timeout may be refreshed by the
        # Harness. Every field affecting the commanded motion remains bound.
        motion = {key: action.get(key) for key in (
            "type", "mode", "frame_id", "tcp_config_id", "limits_profile", "path", "gripper_events")}
        motion.update({key: float(action.get(key, 1.)) for key in ("speed_scale", "gripper_speed_scale")})
        if "waypoint_speed_scales" in action:
            motion["waypoint_speed_scales"] = [float(scale) for scale in action["waypoint_speed_scales"]]
        return hashlib.sha256(json.dumps(motion, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _preview_state_binding(self, state=None):
        state = self.state() if state is None else state
        value = {"state": state, "last_dispatched_goals": self.last_dispatched_goals,
                 "sim_step": self.control_step, "native_action_count": int(self.env.take_action_cnt[0]),
                 "episode_ended": bool(self.env.end_flag[0])}
        return {"sim_step": self.control_step, "native_action_count": value["native_action_count"],
                "sha256": hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()}

    def _measured_gripper_goal(self, state, arm):
        gripper = state["arms"][arm].get("gripper", {})
        if gripper.get("position_source_kind") != "measurement":
            raise ValueError("slow_gripper_requires_measured_opening")
        robot = self.robot.get_robot_by_arm_name(arm + "_arm")
        names = list(robot.gripper_joints_name)
        base = names.index(robot.gripper_move["base"])
        value = float(gripper["position_m"][base])
        lower, upper = robot.gripper_scale
        normalized = ((value - lower) if robot.gripper_move["sign"] == 1 else (upper - value)) / (upper - lower)
        if not self.np.isfinite(normalized):
            raise ValueError("slow_gripper_requires_finite_measured_opening")
        return float(self.np.clip(normalized, 0., 1.))

    def _gripper_rows(self, action, state):
        row = self._held_row(state)
        steps = 0
        events = [event for event in action.get("gripper_events", []) if event["phase"] == "after_motion"]
        scale = action.get("gripper_speed_scale", 1.)
        initial = {}
        for event in events:
            index = 6 if event["arm"] == "left" else 13
            steps = max(steps, int(self.np.ceil(abs(event["goal_normalized"] - row[index]) / self.eef_gripper_dwell_step)))
            if scale < 1:
                initial[index] = self._measured_gripper_goal(state, event["arm"])
            row[index] = event["goal_normalized"]
        dwell = [list(row) for _ in range(max(1, steps))] if events else []
        if not initial:
            return dwell
        ramp_steps = max(int(self.np.ceil(abs(row[index] - value) / (self.eef_gripper_dwell_step * scale)))
                         for index, value in initial.items())
        ramp = []
        for step in range(1, ramp_steps + 1):
            command = list(row)
            for index, value in initial.items():
                command[index] = value + (row[index] - value) * step / ramp_steps
            ramp.append(command)
        return ramp + dwell

    def _gripper_start_change(self, action, rows):
        """Guard exact cached jaw rows; direct execution compiles at arrival."""
        scale = action.get("gripper_speed_scale", 1.)
        if scale == 1 or not rows:
            return None
        state = self.state()
        bound = self.eef_gripper_dwell_step * scale
        for event in action.get("gripper_events", []):
            if event["phase"] != "after_motion":
                continue
            arm = event["arm"]
            measured = self._measured_gripper_goal(state, arm)
            first = rows[0][6 if arm == "left" else 13]
            if abs(first - measured) > bound + 1e-12:
                return dict(arm=arm, measured_opening_normalized=measured,
                    first_goal_normalized=first, max_goal_step=bound,
                    recovery="submit a separate gripper request to ramp from the new measured opening")
        return None

    def _arrival_gripper_rows(self, action, held_row, *, state=None):
        """Compile jaws at admission, preserving every other actuator target.

        Call only after the relevant measured endpoint/dependency checks pass.
        This does not advance physics. The caller must admit the new row count
        against its remaining shared budget before dispatch. Exact preview
        execution deliberately does not use this helper.
        """
        state = self.state() if state is None else state
        rows = self._gripper_rows(action, state)
        events = [event for event in action.get("gripper_events", []) if event["phase"] == "after_motion"]
        indices = [6 if event["arm"] == "left" else 13 for event in events]
        result = []
        for row in rows:
            command = list(held_row)
            for index in indices:
                command[index] = row[index]
            result.append(command)
        scale = action.get("gripper_speed_scale", 1.)
        compilation = dict(mode="arrival_state", compiled_sim_step=self.control_step,
            gripper_speed_scale=scale, planned_steps=len(result),
            semantics="fresh jaw admission state; non-jaw actuator targets retained; no physics advanced")
        if scale < 1:
            compilation.update(max_goal_step=self.eef_gripper_dwell_step * scale,
                measured_opening_normalized={event["arm"]: self._measured_gripper_goal(state, event["arm"])
                                             for event in events})
        return result, compilation

    def _motion_segment(self, action, state, positions, arm, max_native_steps, *, last_waypoint=True):
        """Compile arm rows and a start-state estimate of after-motion jaws.

        Direct execution recompiles those jaws at arrival; exact previews keep
        these cached rows and guard their first step instead.
        """
        held = self._held_row(state)
        offset = 0 if arm == "left" else 7
        extra_rows = self._gripper_rows(action, state) if last_waypoint else []
        schedule, targets = None, None
        event = next((event for event in action.get("gripper_events", [])
                      if event["phase"] == "during_motion"), None) if last_waypoint else None
        if event is not None:
            from auriga.robot.motion.gripper_motion import synchronize
            scale = action.get("gripper_speed_scale", 1.)
            initial = self._measured_gripper_goal(state, arm) if scale < 1 else held[offset + 6]
            result = synchronize(positions, held[offset:offset + 6], initial, event,
                                 self.eef_gripper_dwell_step * scale, max_native_steps)
            if result is None:
                return None
            positions, targets, schedule = result
        if len(positions) + len(extra_rows) > max_native_steps:
            return None
        motion_rows = []
        for index, q in enumerate(positions):
            row = list(held)
            row[offset:offset + 6] = q.tolist()
            if targets is not None:
                row[offset + 6] = float(targets[index])
            motion_rows.append(row)
        for row in extra_rows:
            row[offset:offset + 6] = positions[-1].tolist()
        return positions, motion_rows, extra_rows, schedule

    def preview_batch(self, action):
        from auriga.robot.motion.batch_motion import preview_batch
        return preview_batch(self, action)

    def preview_motion(self, action):
        """Plan a single segment without observing/rendering or advancing physics.

        Cached rows are owned by this worker. A preview is consumed on the first
        execution attempt and may never silently fall back to another plan.
        """
        from auriga.robot.motion.dual_arm_motion import is_dual_action
        if is_dual_action(action):
            raise ValueError("dual_arm_motion_preview_unsupported")
        if getattr(self, "motion_execution_mode", "resampled") == "feedback":
            raise ValueError("feedback_motion_preview_unsupported")
        if not has_calibration(self.observation_profile):
            raise ValueError("motion_preview_requires_calibrated_profile")
        if "preview_plan_id" in action:
            raise ValueError("motion_preview_cannot_reference_existing_plan")
        arm, target = validate_preview_action(action)
        segment_speed = self._waypoint_speed_scale(action)
        if self.env.end_flag[0]:
            raise ValueError("motion_preview_episode_ended")
        robot = self.robot.get_robot_by_arm_name(arm + "_arm")
        before = self.state()
        binding = self._preview_state_binding(before)
        started = time.monotonic()
        diagnostics = []
        options = {}
        if getattr(self, "joint_replan_enabled", False):
            options["joint_replan_context"] = {"deadline": started + action.get("timeout_s", 20),
                "speed_scale": segment_speed,
                "position_tolerance_m": action["path"].get("position_tolerance_m", 0.005)}
        plan = self._plan_segment(robot, target, diagnostics, 0, **options)
        if self._preview_state_binding() != binding:
            raise ValueError("motion_preview_state_changed_during_planning")
        result = {"status": "failed", "reason": "path_infeasible", "plan_id": None,
                  "jaw_execution_semantics": "exact cached after_motion rows; no arrival rebase; stale first jaw step stops before jaw dispatch",
                  "state_binding": binding, "before_state": before,
                  "speed_scale": action.get("speed_scale", 1.), "waypoint_speed_scales": [segment_speed],
                  "gripper_speed_scale": action.get("gripper_speed_scale", 1.),
                  "collision_scope": self.capabilities()["collision_scope"],
                  "physics_advanced": False, "executed_native_steps": 0,
                  "planning_elapsed_s": time.monotonic() - started}
        if plan is None:
            result["failure_feedback"] = self._failure_feedback(action, arm, 0, diagnostics)
            result["planner_failure_stage"] = result["failure_feedback"]["failure_stage"]
            result["failed_waypoint_index"] = 0
            return result
        positions = self._resample_action_plan(plan[0], arm, action)
        compiled = self._motion_segment(action, before, positions, arm, int(self.np.ceil(10 * self.control_hz)))
        if compiled is None:
            result["reason"] = "motion_duration_limit"
            return result
        positions, motion_rows, extra_rows, schedule = compiled
        if time.monotonic() - started > action.get("timeout_s", 20):
            result["reason"] = "planning_timeout"
            return result
        native_rows = motion_rows + extra_rows
        from auriga.robot.model.robot_model import urdf_manifest
        from auriga.robot.model.roboprobe_pose import quat_to_matrix
        from auriga.robot.model.geometry import array, fk, transform
        planner = self.robot.planner[robot.robot_name]
        config = planner.robot_cfg["robot_cfg"]["kinematics"]
        model = self.robot_model
        bases = {}
        geometry = {"available": False, "source_kind": "derived_from_public_proprioception_and_robot_model"}
        try:
            if model is None:
                model = urdf_manifest(self.root / self.robot_binding.urdf_path)
            for side, state in before["arms"].items():
                pose = state["ee_pose"]
                if (state.get("source_kind") != "measurement" or pose.get("source_kind") != "measurement"
                        or pose.get("frame_id") != action["frame_id"]
                        or pose.get("tcp_config_id") != action["tcp_config_id"]):
                    raise ValueError("preview_requires_public_measured_joint_and_ee_pose")
                names = list(state["joint_names"])
                values = array(state["q_rad"], (len(names),)).tolist()
                if not names or any(not isinstance(name, str) for name in names):
                    raise ValueError("preview_requires_named_arm_joints")
                gripper = state.get("gripper", {})
                finger_names = list(gripper.get("joint_names", []))
                if finger_names:
                    if gripper.get("position_source_kind") != "measurement":
                        raise ValueError("preview_requires_measured_finger_joints")
                    values += array(gripper["position_m"], (len(finger_names),)).tolist()
                    names += finger_names
                base_ee = fk({**model, "T_world_base": self.np.eye(4)},
                    {"joint_names": names, "q_rad": values, "source_kind": "measurement"})[robot.ee_link_name]
                quaternion = array(pose["quaternion_xyzw"], (4,))
                if abs(float(self.np.linalg.norm(quaternion)) - 1) > 1e-4:
                    raise ValueError("preview_requires_normalized_ee_quaternion")
                world_ee = self.np.eye(4)
                world_ee[:3, :3] = quat_to_matrix(quaternion[[3, 0, 1, 2]])
                world_ee[:3, 3] = array(pose["position_m"], (3,))
                bases[side] = transform(world_ee @ self.np.linalg.inv(base_ee)).tolist()
            if not bases or arm not in bases:
                raise ValueError("preview_requires_public_arm_geometry")
            geometry["available"] = True
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError) as error:
            # A plan remains reusable even when public geometry is insufficient
            # for its advisory review. Never substitute simulator base/scene data.
            bases = {}
            geometry.update(reason="public_proprioception_or_robot_model_unavailable", error_type=type(error).__name__)
        plan_id = "plan-" + uuid.uuid4().hex
        cache = getattr(self, "_motion_previews", None)
        if cache is None:
            cache = self._motion_previews = {}
        # Bounded worker-local storage. Eviction rejects old references instead
        # of regenerating a plan. Reset starts a fresh worker and empty cache.
        while len(cache) >= 16:
            cache.pop(next(iter(cache)))
        cache[plan_id] = {"state_binding": binding, "action_digest": self._preview_action_digest(action),
                          "positions": positions.copy(), "motion_rows": deepcopy(motion_rows),
                          "gripper_rows": deepcopy(extra_rows), "gripper_motion": deepcopy(schedule)}
        result.update(status="planned", reason=None, plan_id=plan_id, robot_model=model, preview_geometry=geometry,
            planned_trace={"arm": arm, "waypoint_index": 0, "joint_names": list(robot.arm_joints_name),
                           "q_rad": positions.tolist(), "dt_s": 1 / self.control_hz,
                           "resampling": "legacy total variation" if segment_speed == 1 else "bounded significant-turn polyline; 0.001 rad simplification tolerance",
                           "speed_scale": segment_speed, "gripper_speed_scale": action.get("gripper_speed_scale", 1.),
                           "command_step_limits_rad": self._speed_step_limits(arm, segment_speed).tolist(),
                           "planned_motion_steps": len(motion_rows), "planned_gripper_steps": len(extra_rows)},
            native_rows=native_rows, planned_native_steps=len(native_rows),
            robot_collision_model={"spheres": config.get("collision_spheres", {}),
                "collision_sphere_buffer": config.get("collision_sphere_buffer", 0),
                "self_collision_buffer": config.get("self_collision_buffer", {}),
                "self_collision_ignore": config.get("self_collision_ignore", {}),
                "scope": "native configured sphere approximation; no attached task object"})
        if geometry["available"]:
            result["T_world_bases"] = bases
            result["T_world_bases_source_kind"] = geometry["source_kind"]
        if schedule is not None:
            result["planned_trace"]["gripper_motion"] = schedule
        from auriga.robot.model.robot_model import collision_mesh_bounds
        try:
            result["robot_collision_model"]["finger_bounds"] = collision_mesh_bounds(self.root / self.robot_binding.urdf_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["robot_collision_model"]["finger_bounds"] = {}
            result["robot_collision_model"]["finger_bounds_unavailable"] = type(error).__name__
        return deepcopy(result)

    def _consume_motion_preview(self, action):
        plan_id = action.get("preview_plan_id")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("invalid_preview_plan_id")
        validate_preview_action(action)
        cached = getattr(self, "_motion_previews", {}).pop(plan_id, None)
        if cached is None:
            raise ValueError("preview_plan_unknown_expired_or_consumed")
        if cached["action_digest"] != self._preview_action_digest(action):
            raise ValueError("preview_plan_action_mismatch")
        if cached["state_binding"] != self._preview_state_binding():
            raise ValueError("preview_plan_stale_state")
        return cached

    def execute(self, action, progress, cancellation):
        from auriga.robot.motion.dual_arm_motion import is_dual_action

        feedback_mode = getattr(self, "motion_execution_mode", "resampled") == "feedback"
        batch_action = action.get("type") == "BatchAction"
        arm_queue = action.get("type") == "ArmQueueAction"
        dual_arm = is_dual_action(action)
        unsupported = ("feedback_batch_unsupported" if feedback_mode and batch_action else
                       "feedback_arm_queue_unsupported" if feedback_mode and arm_queue else
                       "arm_queue_preview_unsupported" if arm_queue and "preview_plan_id" in action else
                       "feedback_dual_arm_unsupported" if feedback_mode and dual_arm else
                       "dual_arm_motion_preview_unsupported" if dual_arm and "preview_plan_id" in action else
                       "feedback_action_speed_unsupported" if feedback_mode and (any(
                           action.get(key, 1.) != 1 for key in ("speed_scale", "gripper_speed_scale")) or
                           "waypoint_speed_scales" in action) else
                       "feedback_motion_preview_unsupported" if feedback_mode and "preview_plan_id" in action else
                       "feedback_during_motion_unsupported" if feedback_mode and any(
                           event.get("phase") == "during_motion" for event in action.get("gripper_events", [])) else None)
        if unsupported:
            return {"status": "rejected", "reason": unsupported,
                    "motion_execution_mode": "feedback" if feedback_mode else "resampled", "executed_steps": 0,
                    "executed_native_steps": 0, "completed_native_steps": 0, "executed_control_steps": 0,
                    "physics_advanced": False, "planned_trace": [], "dispatched_trace": [], "actual_trace": [],
                    "terminal_observation": None, "observation_unavailable_reason":
                        "feedback_rejected_before_dispatch" if feedback_mode else "preview_rejected_before_dispatch",
                    "execution_interval": {"start_sim_step": self.control_step, "end_sim_step": self.control_step,
                                           "step_domain": "adapter_control_step"}}
        cached_preview = None
        cached_batch = None
        if batch_action:
            from auriga.robot.motion.batch_motion import consume_batch
            try:
                cached_batch = consume_batch(self, action)
            except ValueError as error:
                return {"status": "rejected", "reason": str(error), "batch_plan_id": action.get("batch_plan_id"),
                    "executed_steps": 0, "executed_native_steps": 0, "completed_native_steps": 0,
                    "executed_control_steps": 0, "physics_advanced": False,
                    "planned_trace": [], "dispatched_trace": [], "actual_trace": [],
                    "terminal_observation": None, "observation_unavailable_reason": "batch_rejected_before_dispatch",
                    "execution_interval": {"start_sim_step": self.control_step, "end_sim_step": self.control_step,
                                           "step_domain": "adapter_control_step"}}
        if "preview_plan_id" in action:
            try:
                cached_preview = self._consume_motion_preview(action)
            except ValueError as error:
                # All preview validation precedes observation and dispatch.
                # A known zero-motion rejection must not be turned into a lost
                # acknowledgement/unknown-physics state by the RPC boundary.
                return {"status": "rejected", "reason": str(error), "preview_plan_id": action["preview_plan_id"],
                        "executed_steps": 0, "executed_native_steps": 0, "completed_native_steps": 0,
                        "executed_control_steps": 0, "physics_advanced": False,
                        "planned_trace": [], "dispatched_trace": [], "actual_trace": [],
                        "terminal_observation": None, "observation_unavailable_reason": "preview_rejected_before_dispatch",
                        "execution_interval": {"start_sim_step": self.control_step, "end_sim_step": self.control_step,
                                               "step_domain": "adapter_control_step"}}
        joint_prefix = action.get("type") == "JointPrefixAction"
        ee_chunk = action.get("type") == "EEChunkAction"
        native_policy = joint_prefix or ee_chunk
        if batch_action:
            dual_arm, arm_name, robot = True, "both", None
        elif arm_queue:
            from auriga.core.contracts import validate_arm_queue_action
            validate_arm_queue_action(action, self.capabilities())
            dual_arm, arm_name, robot = True, "both", None
        elif joint_prefix:
            dual_arm = False
            rows = validate_joint_prefix(action)
            for robot, asset in zip(self.robot.robot_list, self.robot.robot_key):
                if robot.type != "target":
                    continue
                offset = 0 if robot.arm_name.startswith("left") else 7
                limits = asset.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
                values = self.np.asarray(rows)[:, offset:offset + 6]
                if ((values < limits[:, 0] - 1e-5) | (values > limits[:, 1] + 1e-5)).any():
                    raise ValueError("joint_prefix_exceeds_live_joint_limits")
            arm_name, robot = None, None
        elif ee_chunk:
            dual_arm, arm_name, robot = False, None, None
            rows = validate_ee_chunk(action)
        else:
            arm_name, _ = validate_action(action)
            from auriga.core.contracts import validate_dual_arm_progression
            validate_dual_arm_progression(action, self.capabilities())
            dual_arm = arm_name == "both"
            robot = self.robot.get_robot_by_arm_name(arm_name + "_arm") if arm_name and not dual_arm else None
        before = self.observe()
        tracking_guards = {}
        if (robot is not None or dual_arm) and getattr(self, "tracking_guard_enabled", False):
            from auriga.robot.motion.tracking_guard import TrackingGuard
            tracking_guards = {side: TrackingGuard() for side in (("left", "right") if dual_arm else (arm_name,))}
        start_step = self.control_step
        start_native_count = int(self.env.take_action_cnt[0])
        self.dispatch_log = []
        receipt = {"status": "failed", "reason": None, "before_state": before["robot_state"], "before_observation": before,
            "planned_trace": [], "dispatched_trace": [], "actual_trace": [], "executed_steps": 0, "waypoint_results": [],
            "trace_sampling": {"actual_every_native_actions": 1, "actual_boundary_samples": True},
            "collision_scope": ("native policy joint targets; no planner collision proof" if joint_prefix else
                                "native policy EE targets solved by native IK; no planner collision proof" if ee_chunk
                                else self.capabilities()["collision_scope"]),
            "versions": self.versions}
        if not native_policy:
            receipt.update(speed_scale=action.get("speed_scale", 1.), gripper_speed_scale=action.get("gripper_speed_scale", 1.))
            if robot is not None:
                receipt["waypoint_speed_scales"] = [self._waypoint_speed_scale(action, index)
                    for index, _ in enumerate(action["path"]["waypoints"])]
        if not dual_arm and not native_policy:
            for event in action.get("gripper_events", []):
                if event["phase"] == "after_motion":
                    receipt.setdefault("gripper_execution", {})[event["arm"]] = {
                        "phase": "after_motion", "planned_steps": None, "completed_steps": 0,
                        "started": False, "command_complete": False, "dispatch_attempted_steps": 0,
                        "execution_uncertain": False,
                        "semantics": "completed command rows; does not prove grasp or retention"}
        if feedback_mode:
            receipt["motion_execution_mode"] = "feedback"
        deadline = time.monotonic() + action.get("timeout_s", 20)
        segments, planning_diagnostics = [], []
        from auriga.robot.motion.execution_evidence import ExecutionEvidence
        evidence = ExecutionEvidence(receipt, segments)

        def sample():
            evidence.sample(self.control_step, self.state())

        def interrupted(phase):
            if self.env.end_flag[0]:
                receipt.update(status="stopped", reason="episode_ended")
                receipt["stop_confirmation"] = self._hold()
                return True
            if cancellation.is_set() or time.monotonic() > deadline:
                receipt.update(status="stopped" if cancellation.is_set() else "failed",
                               reason="cancelled" if cancellation.is_set() else phase + "_timeout")
                receipt["stop_confirmation"] = self._hold()
                return True
            return False

        def dispatch(row, waypoint_index=None, *, command_phase="motion", arm_context=None, native=None):
            if interrupted("execution"):
                return False
            attempted = evidence.attempted(command_phase, arm_context)
            try:
                (native or self._native_action)(row)
            except Exception:
                evidence.failed(attempted)
                raise
            evidence.completed(command_phase, arm_context)
            sample()
            if tracking_guards:
                if interrupted("execution"):
                    return False
                for side, guard in tracking_guards.items():
                    offset = 0 if side == "left" else 7
                    measured = receipt["actual_trace"][-1]["state"]["arms"][side]["q_rad"]
                    divergence = guard.observe(row[offset:offset + 6], measured)
                    if divergence is not None:
                        receipt.update(status="failed", reason="tracking_divergence", failed_waypoint_index=waypoint_index,
                                       tracking_guard={"arm": side, **divergence})
                        if arm_context is not None:
                            receipt["failed_waypoint_index"] = arm_context[side]["waypoint_index"]
                            if arm_context[side].get("stage_id") is not None:
                                receipt["failed_stage_id"] = arm_context[side]["stage_id"]
                        if dual_arm:
                            receipt["failed_arm"] = side
                        receipt["stop_confirmation"] = self._hold()
                        return False
            if receipt["executed_steps"] == 1 or receipt["executed_steps"] % 5 == 0:
                progress({"phase": "executing", "waypoint_index": waypoint_index,
                          **({"arm_context": arm_context} if arm_context is not None else {}),
                          "executed_steps": receipt["executed_steps"], "sim_step": self.control_step,
                          "observation": self.observe(), "episode_terminal": self.episode_terminal()})
            return not interrupted("execution")

        def gripper_rows(state):
            return self._gripper_rows(action, state)

        sample()
        try:
            if interrupted("planning"):
                pass
            elif joint_prefix:
                segments.append({"action_space": "native_joint", "actions": rows, "dt_s": 1 / self.control_hz,
                                 "policy_provenance": action["policy_provenance"]})
                progress({"phase": "started", "planned_steps": len(rows), "sim_step": self.control_step})
                for row in rows:
                    if not dispatch(row, command_phase="joint_prefix"):
                        break
                else:
                    receipt.update(status="completed", reason=None)
            elif ee_chunk:
                from auriga.robot.motion.ee_chunk_motion import execute_ee_chunk
                execute_ee_chunk(self, action, rows, receipt, segments, progress, dispatch)
            elif batch_action:
                from auriga.robot.motion.batch_motion import execute_batch
                execute_batch(self, action, cached_batch, receipt, segments, progress, interrupted, dispatch)
            elif arm_queue:
                from auriga.robot.motion.arm_queue_motion import execute_arm_queue
                execute_arm_queue(self, action, receipt, segments, planning_diagnostics,
                                  progress, interrupted, dispatch, deadline)
            elif dual_arm:
                from auriga.robot.motion.dual_arm_motion import execute_dual
                execute_dual(self, action, receipt, segments, planning_diagnostics,
                             progress, interrupted, dispatch, deadline)
            elif robot is None:
                if "hold_native_steps" in action:
                    held = self._held_row(before["robot_state"])
                    rows = [list(held) for _ in range(action["hold_native_steps"])]
                else:
                    rows = gripper_rows(before["robot_state"])
                if len(rows) > int(self.np.ceil(10 * self.control_hz)):
                    raise ValueError("motion_duration_limit")
                segments.append({"action_space": "native_joint", "actions": rows, "dt_s": 1 / self.control_hz,
                                 "gripper_speed_scale": action.get("gripper_speed_scale", 1.), "planned_gripper_steps": len(rows)})
                for jaw in receipt.get("gripper_execution", {}).values():
                    jaw["planned_steps"] = len(rows)
                progress({"phase": "started", "planned_steps": len(rows), "sim_step": self.control_step})
                from auriga.robot.motion.motion_safety import admit_jaw_budget
                if admit_jaw_budget(self, before["robot_state"], ("left", "right"), receipt,
                        native_steps=len(rows), request_remaining=int(self.np.ceil(10*self.control_hz))):
                    for row in rows:
                        if not dispatch(row, command_phase="hold" if "hold_native_steps" in action else "gripper"):
                            break
                    else:
                        receipt.update(status="completed", reason=None)
            else:
                for waypoint_index, waypoint in enumerate(action["path"]["waypoints"]):
                    target = waypoint[arm_name]
                    segment_speed = self._waypoint_speed_scale(action, waypoint_index)
                    if interrupted("planning"):
                        break
                    progress({"phase": "planning", "waypoint_index": waypoint_index, "sim_step": self.control_step})
                    plan_options = {}
                    path = action["path"]
                    if (not feedback_mode and getattr(self, "joint_replan_enabled", False) and path.get("kind") == "backend_free"
                            and path.get("blend_radius_m", 0) == 0 and path.get("orientation", "backend_free") == "backend_free"
                            and not any(path.get(key) for key in ("constraints", "path_constraints", "constraint_pose"))):
                        plan_options["joint_replan_context"] = {"deadline": deadline,
                            "speed_scale": segment_speed,
                            "position_tolerance_m": path.get("position_tolerance_m", 0.005),
                            "interrupted": lambda: interrupted("planning")}
                    plan = ((cached_preview["positions"], 1) if cached_preview is not None else
                            self._plan_segment(robot, target, planning_diagnostics, waypoint_index, **plan_options))
                    if interrupted("planning"):
                        break
                    if plan is None:
                        receipt.update(reason="path_infeasible", failed_waypoint_index=waypoint_index)
                        receipt["failure_feedback"] = self._failure_feedback(action, arm_name, waypoint_index,
                                                                              planning_diagnostics)
                        receipt["planner_failure_stage"] = receipt["failure_feedback"]["failure_stage"]
                        break
                    if feedback_mode:
                        from auriga.robot.motion.feedback_motion import ArrivalMonitor, pose_error
                        current = self.state()
                        # Select the nearest legal revolute equivalent once, then
                        # keep it fixed. Preserve the native plan as provenance.
                        endpoint_diagnostic = {"waypoint_index": waypoint_index,
                            "stage": "feedback_endpoint", "validated": False}
                        planning_diagnostics.append(endpoint_diagnostic)
                        try:
                            endpoint = self._feedback_endpoint(robot, target, current["arms"][arm_name]["q_rad"],
                                plan[0][-1], action["path"].get("position_tolerance_m", 0.005), endpoint_diagnostic)
                        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, OverflowError) as error:
                            endpoint_diagnostic.update(reason="endpoint_validation_failed", validation_error=str(error))
                            receipt.update(reason="feedback_endpoint_invalid", failed_waypoint_index=waypoint_index)
                            break
                        row = self._held_row(current)
                        offset = 0 if arm_name == "left" else 7
                        row[offset:offset + 6] = endpoint.tolist()
                        monitor = ArrivalMonitor(target, endpoint, current["arms"][arm_name],
                                                 action["path"].get("position_tolerance_m", 0.005),
                                                 action["path"].get("orientation_tolerance_rad", 0.1))
                        segment = {"arm": arm_name, "waypoint_index": waypoint_index,
                            "joint_names": list(robot.arm_joints_name), "q_rad": [endpoint.tolist()],
                            "speed_scale": segment_speed,
                            "dt_s": 1 / self.control_hz, "execution_mode": "feedback",
                            "resampling": "none: repeat a fixed endpoint until measured arrival; not the original planner path",
                            "reference_resampled_steps": max(1, int(self.np.ceil(
                                self.np.abs(self.np.diff(plan[0], axis=0)).sum(axis=0) / self.arm_step_limits[arm_name]).max())),
                            "reference_steps_semantics": "diagnostic only; no dense trajectory is dispatched",
                            "feedback_endpoint": endpoint_diagnostic,
                            "feedback_controller": monitor.settings(), "feedback_samples": [],
                            "executed_motion_steps": 0}
                        segments.append(segment)
                        progress({"phase": "started", "waypoint_index": waypoint_index,
                                  "execution_mode": "feedback", "sim_step": self.control_step,
                                  "max_native_steps": int(self.np.ceil(10 * self.control_hz)) - receipt["executed_steps"]})
                        arrived = False
                        while receipt["executed_steps"] < int(self.np.ceil(10 * self.control_hz)):
                            completed_before = receipt["executed_steps"]
                            keep_running = dispatch(row, waypoint_index)
                            completed = receipt["executed_steps"] - completed_before
                            segment["executed_motion_steps"] += completed
                            if completed:
                                outcome, measured = monitor.observe(receipt["actual_trace"][-1]["state"]["arms"][arm_name])
                                segment["feedback_samples"].append(measured)
                            # A completed native step can simultaneously end the
                            # episode. Keep its sample, but terminal/cancel wins.
                            if not keep_running:
                                break
                            receipt.update(position_error_m=measured["position_error_m"],
                                           orientation_error_rad=measured["orientation_error_rad"])
                            if outcome == "arrived":
                                arrived = True
                                segment["feedback_outcome"] = "arrived"
                                break
                            if outcome == "stalled":
                                receipt.update(reason="feedback_stalled", failed_waypoint_index=waypoint_index)
                                segment["feedback_outcome"] = "stalled"
                                break
                        else:
                            receipt.update(reason="motion_duration_limit", failed_waypoint_index=waypoint_index)
                        if not arrived:
                            segment.setdefault("feedback_outcome", receipt.get("reason") or "interrupted")
                            break
                        last_waypoint = waypoint_index == len(action["path"]["waypoints"]) - 1
                        extra_rows = self._gripper_rows(action, self.state()) if last_waypoint else []
                        if last_waypoint:
                            for jaw in receipt.get("gripper_execution", {}).values():
                                jaw["planned_steps"] = len(extra_rows)
                        if receipt["executed_steps"] + len(extra_rows) > int(self.np.ceil(10 * self.control_hz)):
                            receipt.update(reason="motion_duration_limit", failed_waypoint_index=waypoint_index)
                            break
                        # Preserve the chosen arm endpoint during the unchanged
                        # after_motion jaw dwell. Never release on failed arrival.
                        for grip_row in extra_rows:
                            grip_row[offset:offset + 6] = endpoint.tolist()
                            if not dispatch(grip_row, waypoint_index, command_phase="gripper"):
                                break
                        else:
                            actual = self.state()["arms"][arm_name]["ee_pose"]
                            position_error, quaternion_error = pose_error(actual, target)
                            receipt["waypoint_results"].append({"waypoint_index": waypoint_index,
                                "speed_scale": segment_speed,
                                "position_error_m": position_error, "orientation_error_rad": quaternion_error,
                                "sim_step": self.control_step})
                            receipt.update(position_error_m=position_error, orientation_error_rad=quaternion_error)
                            if position_error > action["path"].get("position_tolerance_m", 0.005) or quaternion_error > action["path"].get("orientation_tolerance_rad", 0.1):
                                receipt.update(reason="tracking_error", failed_waypoint_index=waypoint_index)
                                break
                            continue
                        break
                    positions = cached_preview["positions"] if cached_preview is not None else self._resample_action_plan(plan[0], arm_name, action, waypoint_index)
                    last_waypoint = waypoint_index == len(action["path"]["waypoints"]) - 1
                    remaining = int(self.np.ceil(10 * self.control_hz)) - receipt["executed_steps"]
                    compiled = ((positions, deepcopy(cached_preview["motion_rows"]),
                                 deepcopy(cached_preview["gripper_rows"]), deepcopy(cached_preview["gripper_motion"]))
                                if cached_preview is not None else
                                self._motion_segment(action, self.state(), positions, arm_name, remaining,
                                                     last_waypoint=last_waypoint))
                    if compiled is None or len(compiled[1]) + len(compiled[2]) > remaining:
                        receipt.update(reason="motion_duration_limit", failed_waypoint_index=waypoint_index)
                        break
                    positions, motion_rows, extra_rows, schedule = compiled
                    from auriga.robot.motion.motion_safety import admit_motion
                    if not admit_motion(self, self.state(), {arm_name: positions}, receipt,
                            native_steps=len(motion_rows)+len(extra_rows), request_remaining=remaining,
                            waypoint_index=waypoint_index):
                        break
                    if last_waypoint:
                        for jaw in receipt.get("gripper_execution", {}).values():
                            jaw["planned_steps"] = len(extra_rows)
                    segments.append({"arm": arm_name, "waypoint_index": waypoint_index,
                        "joint_names": list(robot.arm_joints_name), "q_rad": positions.tolist(), "dt_s": 1 / self.control_hz,
                        "resampling": "legacy total variation; 10 s cap" if segment_speed == 1 else "bounded significant-turn polyline; 0.001 rad simplification tolerance; 10 s cap",
                        "speed_scale": segment_speed, "gripper_speed_scale": action.get("gripper_speed_scale", 1.),
                        "command_step_limits_rad": self._speed_step_limits(arm_name, segment_speed).tolist(),
                        "planned_motion_steps": len(motion_rows), "planned_gripper_steps": len(extra_rows)})
                    if schedule is not None:
                        segments[-1]["gripper_motion"] = schedule
                    progress({"phase": "started", "waypoint_index": waypoint_index, "speed_scale": segment_speed,
                              "planned_steps": len(positions) + len(extra_rows), "sim_step": self.control_step})
                    for row in motion_rows:
                        if not dispatch(row, waypoint_index):
                            break
                    else:
                        def checked_endpoint(phase):
                            from auriga.robot.motion.feedback_motion import pose_error
                            position_error, quaternion_error = pose_error(self.state()["arms"][arm_name]["ee_pose"], target)
                            result = {"waypoint_index": waypoint_index, "position_error_m": position_error,
                                      "orientation_error_rad": quaternion_error, "sim_step": self.control_step,
                                      "speed_scale": segment_speed, "phase": phase}
                            receipt.update(position_error_m=position_error, orientation_error_rad=quaternion_error,
                                endpoint_result_waypoint_index=waypoint_index, endpoint_result_sim_step=self.control_step,
                                endpoint_result_phase=phase)
                            valid = position_error <= path.get("position_tolerance_m", .005) and quaternion_error <= path.get("orientation_tolerance_rad", .1)
                            if not valid:
                                receipt.update(reason="tracking_error", failed_waypoint_index=waypoint_index,
                                               failed_arm=arm_name, failed_phase=phase)
                                receipt["stop_confirmation"] = self._hold()
                            return valid, result

                        # The jaw must not close or release at an unverified
                        # endpoint. A failed approach returns feedback first.
                        arrived, arrival_result = checked_endpoint("before_gripper" if extra_rows else "motion_complete")
                        receipt.setdefault("motion_arrival_results", []).append(arrival_result)
                        if not arrived:
                            receipt["waypoint_results"].append(arrival_result)
                            break
                        # Direct jaws use the actual arrival opening. Explicit
                        # previews retain exact rows and their stale-start guard.
                        if extra_rows and cached_preview is None:
                            initial_steps = len(extra_rows)
                            extra_rows, compilation = self._arrival_gripper_rows(action, motion_rows[-1])
                            segments[-1].update(initial_gripper_steps_estimate=initial_steps,
                                planned_gripper_steps=len(extra_rows), gripper_actions=extra_rows,
                                gripper_compilation=compilation)
                            for jaw in receipt.get("gripper_execution", {}).values():
                                jaw["planned_steps"] = len(extra_rows)
                            if receipt["executed_steps"] + len(extra_rows) > int(self.np.ceil(10 * self.control_hz)):
                                receipt.update(reason="motion_duration_limit", failed_waypoint_index=waypoint_index,
                                    failed_arm=arm_name, failed_phase="before_gripper")
                                receipt["stop_confirmation"] = self._hold()
                                break
                        elif cached_preview is not None:
                            changed = self._gripper_start_change(action, extra_rows)
                            if changed is not None:
                                receipt.update(reason="gripper_start_state_changed", failed_waypoint_index=waypoint_index,
                                               gripper_start_state=changed, failed_phase="before_gripper")
                                receipt["stop_confirmation"] = self._hold()
                                break
                        from auriga.robot.motion.motion_safety import admit_jaw_budget
                        if not admit_jaw_budget(self,self.state(),[arm_name],receipt,native_steps=len(extra_rows),
                                request_remaining=int(self.np.ceil(10*self.control_hz))-receipt['executed_steps'],
                                waypoint_index=waypoint_index):
                            break
                        for grip_row in extra_rows:
                            if not dispatch(grip_row, waypoint_index, command_phase="gripper"):
                                break
                        else:
                            valid, final_result = checked_endpoint("after_gripper") if extra_rows else (True, arrival_result)
                            receipt["waypoint_results"].append(final_result)
                            if not valid:
                                break
                            continue
                    break
                else:
                    receipt.update(status="completed", reason=None)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            receipt.update(status="failed", reason=f"backend_error: {type(error).__name__}: {error}")
            receipt["stop_confirmation"] = self._hold()
            if self.env.take_action_cnt[0] != self.evaluator_actions:
                receipt["status"] = "unknown"
                self.evaluator_failure = f"native_action_failed: {type(error).__name__}: {error}"
        if cached_preview is not None:
            receipt["preview_plan_id"] = action["preview_plan_id"]
            receipt["preview_state_binding"] = cached_preview["state_binding"]
            receipt["preview_execution_semantics"] = "cached native rows; no replanning, resampling or arrival jaw rebase"
        evidence.finish(start_control_step=start_step, control_step=self.control_step,
            start_native_count=start_native_count, native_count=int(self.env.take_action_cnt[0]),
            episode_terminal=self.episode_terminal(), dispatched_trace=self.dispatch_log)
        self.dispatch_log = None
        terminal_sample_captured = False
        try:
            sample()
            terminal_sample_captured = True
            receipt["terminal_observation"] = self.observe()
        except Exception as error:
            receipt.update(terminal_observation=None, terminal_observation_error=str(error))
            if not terminal_sample_captured:
                receipt["trace_sampling"].update(actual_boundary_samples=False,
                    terminal_state_sample_error=str(error))
        try:
            episode_directory = getattr(self, "directory", None)
            if episode_directory is not None:
                directory = episode_directory / "development-diagnostics"
                directory.mkdir(exist_ok=True)
                private = {"audience": "observer_only", "action_id": action.get("action_id"),
                           "planning_diagnostics": planning_diagnostics}
                if self.observation_profile == "rgb_proprio_v1":
                    private["receipt"] = receipt
                else:
                    private.update(observation_profile=self.observation_profile,
                                   joint_replan_enabled=bool(getattr(self, "joint_replan_enabled", False)),
                                   status=receipt["status"], reason=receipt["reason"],
                                   executed_native_steps=receipt["executed_native_steps"])
                (directory / f"action-{uuid.uuid4().hex}.json").write_text(json.dumps(private, allow_nan=False))
        except Exception as error:
            try:
                print(f"Private development diagnostic write failed: {error}", file=sys.stderr)
            except Exception:
                pass
        if self.observation_profile == "rgb_proprio_v1":
            from auriga.core.observation_policy import actor_robot_state
            receipt["before_state"] = actor_robot_state(receipt["before_state"], "rgb_proprio_v1")
            for item in receipt["actual_trace"]:
                item["state"] = actor_robot_state(item["state"], "rgb_proprio_v1")
        return receipt

    def close(self, progress=None):
        progress = progress or (lambda value: None)
        try:
            self.env.close()
            progress({"phase": "environment_closed"})
        finally:
            try:
                progress({"phase": "application_close_started",
                          "fast_shutdown": getattr(self.app, "config", {}).get("fast_shutdown")})
            finally:
                self.app.close()
            progress({"phase": "application_closed"})


def main():
    global protocol_output
    # Native Kit writes directly to fd 1; give it the log fd and retain a clean IPC fd.
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    parser = argparse.ArgumentParser()
    parser.add_argument("--robodojo-root", required=True)
    parser.add_argument("--robot-model-id", default="arx-x5")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task", default="stack_bowls")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--observation-profile", choices=ACTOR_PROFILES, default=ACTOR_PROFILE)
    parser.add_argument("--tracking-guard", action="store_true")
    parser.add_argument("--joint-replan", action="store_true")
    parser.add_argument("--motion-safety-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion-execution-mode", choices=("resampled", "feedback"), default="resampled")
    parser.add_argument("--scene-truth-log", default=None, help="analysis only: append simulator truth to this JSONL path")
    arguments = parser.parse_args()
    serve_worker(RoboDojoAdapter, arguments, emit)


if __name__ == "__main__":
    main()
