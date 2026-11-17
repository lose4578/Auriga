"""Configure RoboDojo operations above the independently owned worker transport."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import uuid

from ...model.arx_contract import FRAME_ID, TCP_CONFIG_ID, validate_joint_prefix, validate_preview_action
from ...model.arx_contract import validate_action as validate_arx_action
from ....core.observation_policy import ACTOR_PROFILES
from .robot_process import RobotProcess, WIRE_PREFIX
from ...model.embodiments import get_robodojo_robot
from ...motion.arx_actions import compile_action

_UNSET = object()


def validate_action(action):
    """Compatibility entry for native command validation, without starting a worker."""
    if action.get("type") == "BatchAction":
        from ...motion.batch_contract import validate_batch_action
        validate_batch_action(action)
        return "both", None
    if action.get("type") == "ArmQueueAction":
        from ...motion.arm_queue_contract import validate_arm_queue_action
        validate_arm_queue_action(action)
        if (action.get("mode") != "absolute" or action.get("frame_id") != FRAME_ID
                or action.get("tcp_config_id") != TCP_CONFIG_ID
                or action.get("limits_profile") not in {"robodojo-native-v1", "simulation-slow-v1"}):
            raise ValueError("unsupported_arm_queue_frame_or_limits")
        timeout = action.get("timeout_s")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not .1 <= timeout <= 300:
            raise ValueError("invalid_arm_queue_timeout")
        return "both", None
    return validate_arx_action(action)


class RobotBackend(RobotProcess):
    """One long-lived JSONL worker, with independently deliverable cancellation."""

    validate_action = staticmethod(validate_action)
    compile_action = staticmethod(compile_action)
    robot_binding = get_robodojo_robot()

    def __init__(self, robodojo_root=None, python=None,
                 gpu_id=None, task=None, run_dir="runs/robot", seed=None, layout_id=None,
                 observation_profile=None, tracking_guard=_UNSET, joint_replan=_UNSET, *,
                 motion_execution_mode=_UNSET, config=None):
        from ....core.config import Config, motion_thresholds
        config = config if config is not None else Config.load()
        if config.backend.environment != "robodojo":
            raise ValueError(f"unsupported_robodojo_environment: {config.backend.environment}")
        self.robot_binding = get_robodojo_robot(config.backend.robot_model_id)
        gpu_id = config.run.gpu if gpu_id is None else gpu_id
        task = config.run.task if task is None else task
        seed = config.run.seed if seed is None else seed
        layout_id = config.run.layout_id if layout_id is None else layout_id
        observation_profile = config.features.observation_profile if observation_profile is None else observation_profile
        tracking_guard = config.features.tracking_guard if tracking_guard is _UNSET else tracking_guard
        joint_replan = config.features.joint_replan if joint_replan is _UNSET else joint_replan
        if type(tracking_guard) is not bool:
            raise ValueError("invalid_tracking_guard")
        self.tracking_guard = tracking_guard
        if type(joint_replan) is not bool:
            raise ValueError("invalid_joint_replan")
        self.joint_replan = joint_replan
        self.motion_safety_guard = config.features.motion_safety_guard
        self.trajectory_preview = config.features.trajectory_preview
        motion_execution_mode = (config.features.motion_execution_mode
                                 if motion_execution_mode is _UNSET else motion_execution_mode)
        if type(motion_execution_mode) is not str or motion_execution_mode not in ('resampled', 'feedback'):
            raise ValueError("invalid_motion_execution_mode")
        self.motion_execution_mode = motion_execution_mode
        self.root = str(Path(robodojo_root if robodojo_root is not None else config.backend.root).expanduser())
        self.python = str(Path(python if python is not None else config.backend.python).expanduser())
        self.gpu_id = gpu_id
        self.task, self.seed, self.layout_id = task, seed, layout_id
        if observation_profile not in ACTOR_PROFILES:
            raise ValueError("unsupported observation profile")
        self.observation_profile = observation_profile
        from ....core.storage.images import image_settings
        self.image_settings = image_settings(config.images.to_dict())
        self.motion_thresholds = motion_thresholds(config.features)
        self.scene_truth_log = bool(getattr(getattr(config, "analysis", None), "scene_truth_log", False))
        super().__init__(run_dir)

    def start(self):
        if self.process is not None:
            return self.capabilities
        self.run_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        project = str(Path(__file__).resolve().parents[4])
        environment.update(CUDA_VISIBLE_DEVICES=str(self.gpu_id), ROBODOJO_RENDER_GPU=str(self.gpu_id),
                           AURIGA_IMAGE_SETTINGS=json.dumps(self.image_settings),
                           AURIGA_MOTION_THRESHOLDS=json.dumps(self.motion_thresholds),
                           ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES",
                           PYTHONDONTWRITEBYTECODE="1",
                           PYTHONPATH=os.pathsep.join((project, self.root, str(Path(self.root) / "XPolicyLab"), environment.get("PYTHONPATH", ""))))
        command = [self.python, "-u", "-m", "auriga.robot.backends.robodojo.robot_worker", "--robodojo-root", self.root,
                                        "--robot-model-id", self.robot_binding.robot.robot_model_id,
                                        "--gpu", str(self.gpu_id), "--task", self.task, "--run-dir", str(self.run_dir),
                                        "--seed", str(self.seed), "--layout-id", str(self.layout_id),
                                        "--observation-profile", self.observation_profile,
                                        "--motion-execution-mode", self.motion_execution_mode,
                                        "--motion-safety-guard" if self.motion_safety_guard else "--no-motion-safety-guard",
                                        "--trajectory-preview" if self.trajectory_preview else "--no-trajectory-preview",
                                        *(["--tracking-guard"] if getattr(self, "tracking_guard", False) else []),
                                        *(["--joint-replan"] if getattr(self, "joint_replan", False) else [])]
        if getattr(self, "scene_truth_log", False):
            from .scene_truth import truth_path
            command += ["--scene-truth-log", str(truth_path(self.run_dir))]
        try:
            return self.start_process(command, cwd=self.root, environment=environment)
        except RuntimeError as error:
            message = str(error)
            # A simulator CUDA fault during bootstrap precedes any action; restart once on a fresh worker.
            # Allocation failures are left to the queue's delayed OOM retry.
            if (not message.startswith("bootstrap_failed") or "CUDA error" not in message
                    or "out of memory" in message.lower() or getattr(self, "bootstrap_retries", 0) >= 1):
                raise
            cleanup = self.close()
            if cleanup.get("process_exit_confirmed") is not True:
                raise
            self.process, self.failure, self.capabilities = None, None, None
            self.bootstrap_retries = getattr(self, "bootstrap_retries", 0) + 1
            (self.run_dir / "bootstrap-retry.json").write_text(json.dumps(
                dict(attempt=self.bootstrap_retries + 1, previous_error=message[:500])) + "\n")
            return self.start_process(command, cwd=self.root, environment=environment)

    def observe(self):
        with self.operation_lock:
            return self._request("observe")

    def policy_observation(self):
        """External policy inputs at the current step (npz in the worker run dir); no physics."""
        with self.operation_lock:
            return self._request("policy_observation")

    def validate_reset(self, seed=0, layout_id=0):
        """Read-only preflight; a refusal leaves the current worker and world intact."""
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (seed, layout_id)):
            raise ValueError("seed and layout_id must be nonnegative integers")
        layout = Path(self.root) / "Assets" / self.robot_binding.layout_directory / str(seed) / f"{self.task}_{layout_id}.json"
        try:
            layout.read_bytes()
        except OSError as error:
            raise ValueError(f"reset_layout_unavailable: seed={seed}, layout_id={layout_id}") from error
        return layout

    def reset(self, seed=0, layout_id=0):
        with self.operation_lock:
            self.validate_reset(seed, layout_id)
            before = self._request("observe")["robot_state"]
            cleanup = self.close()
            if (cleanup.get("process_exit_confirmed") is not True
                    or cleanup.get("reader_exit_confirmed", True) is not True):
                raise RuntimeError("reset_requires_previous_worker_exit")
            self.process, self.failure, self.capabilities = None, None, None
            self.seed, self.layout_id = seed, layout_id
            self.start()
            observation = self._request("observe")
            return {"reset_id": str(uuid.uuid4()), "episode_index": 0, "seed": seed, "layout_id": layout_id,
                    "before_state": before, "robot_state": observation["robot_state"], "observation": observation,
                    "clock": "paused", "reset_strategy": "fresh_worker_official_eval_reset"}

    def evaluate(self, final_check=True):
        """Privileged native judgment. Harness must never publish it to actor tools."""
        with self.operation_lock:
            return self._request("evaluate", {"final_check": bool(final_check)})

    def observer_snapshot(self):
        """Hidden native scene state for independent observer audits only."""
        with self.operation_lock:
            return self._request("observer_snapshot")

    def record(self, enabled=True):
        """Observer recording at 5 Hz simulation time; physics remains worker-owned."""
        with self.operation_lock:
            return self._request("record", {"enabled": bool(enabled)})

    def save_video(self):
        with self.operation_lock:
            return self._request("save_video")

    def execute(self, action, progress_callback=None, cancel_flag=None):
        if action.get("type") == "JointPrefixAction":
            validate_joint_prefix(action)
        else:
            validate_action(action)
        with self.operation_lock:
            return self._request("execute", action, timeout=action.get("timeout_s", 20) + 60,
                                 progress=progress_callback, cancel=cancel_flag)

    def preview_motion(self, action):
        """Plan without advancing physics; the returned plan can be executed once."""
        validate_preview_action(action)
        if "preview_plan_id" in action:
            raise ValueError("motion_preview_cannot_reference_existing_plan")
        with self.operation_lock:
            return self._request("preview_motion", action, timeout=action.get("timeout_s", 20) + 60)

    def preview_batch(self, action):
        from ...motion.batch_contract import validate_batch_action
        validate_batch_action(action)
        with self.operation_lock:
            return self._request("preview_batch", action, timeout=action.get("timeout_s", 120) + 60)

    def execute_joint_prefix(self, action, progress_callback=None, cancel_flag=None):
        validate_joint_prefix(action)
        return self.execute(action, progress_callback, cancel_flag)

    def fast_forward_slice(self, checkpoint_spec, timeout=180.0):
        with self.operation_lock:
            return self._request("fast_forward_slice", checkpoint_spec, timeout=timeout)
