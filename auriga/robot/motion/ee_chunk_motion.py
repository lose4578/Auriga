"""Execute an external policy's native end-effector chunk inside the RoboDojo worker.

The rows are dispatched exactly as the policy emitted them through RoboDojo's
own end-effector interface (``eval_env.take_action`` with ``*_ee_pose`` keys,
which solves IK per arm with ``robot_manager.solve_ik`` and interpolates the
joint target over one native step). Nothing is converted to joints here, so the
recorded generator of the motion stays the policy. This module also captures the
policy's own inputs (head-camera frame per executed step, and the final
three-camera RGB plus end-effector state) from the native observation, keeping
them out of the Harness store and away from the actor.

Only ``numpy`` is needed; it runs in the simulator interpreter.
"""
from pathlib import Path
import uuid

from ..model.arx_contract import EE_CHUNK_GRIPPER_KEYS, EE_CHUNK_POSE_CONVENTION, FRAME_ID

POLICY_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
POLICY_STATE_KEYS = ("left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state")
NATIVE_INTERFACE = ("RoboDojo eval_env.take_action with left/right *_ee_pose and *_ee_joint_state keys; "
                    "native per-arm robot_manager.solve_ik, an arm whose IK fails keeps its previous joint target")


def native_policy_frame(raw):
    """Select the permitted policy inputs from one native ``get_obs`` dictionary.

    Only RGB colour of the three public cameras, the end-effector pose and
    normalized gripper state of each arm, and the public task instruction are
    read. Depth, intrinsics and any other native field are ignored.
    """
    import numpy as np

    vision = raw.get("vision") if isinstance(raw, dict) else None
    if not isinstance(vision, dict):
        raise ValueError("native_observation_missing_vision")
    images = {}
    for camera in POLICY_CAMERAS:
        entry = vision.get(camera)
        color = entry.get("color") if isinstance(entry, dict) else None
        if color is None:
            raise ValueError("native_observation_missing_policy_camera:" + camera)
        image = np.asarray(color)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("native_policy_camera_requires_uint8_hwc:" + camera)
        images[camera] = np.ascontiguousarray(image[..., :3])
    state = raw.get("state")
    if not isinstance(state, dict):
        raise ValueError("native_observation_missing_state")
    values = {}
    for key in POLICY_STATE_KEYS:
        if key not in state:
            raise ValueError("native_observation_missing_policy_state:" + key)
        value = np.asarray(state[key], dtype=np.float64).reshape(-1)
        if value.shape != ((7,) if key.endswith("_ee_pose") else (1,)) or not np.isfinite(value).all():
            raise ValueError("native_policy_state_invalid:" + key)
        values[key] = value
    instruction = raw.get("instruction")
    return images, values, instruction if isinstance(instruction, str) else None


class PolicyCaptureWriter:
    """Head frames of consecutive executed steps plus the inputs of the last one."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.heads, self.sim_steps, self.final = [], [], None

    def add(self, sim_step, raw):
        images, state, instruction = native_policy_frame(raw)
        self.heads.append(images["cam_head"])
        self.sim_steps.append(int(sim_step))
        self.final = (images, state, instruction, int(sim_step))

    def write(self):
        """Persist the capture as one uncompressed npz; None when nothing was captured."""
        if self.final is None:
            return None
        import numpy as np

        images, state, instruction, sim_step = self.final
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"capture-{uuid.uuid4().hex}.npz"
        np.savez(path, head=np.stack(self.heads), sim_steps=np.asarray(self.sim_steps, dtype=np.int64),
                 **{"final_" + camera: image for camera, image in images.items()},
                 **{"state_" + key: value for key, value in state.items()})
        return dict(path=str(path), frames=len(self.heads), sim_steps=list(self.sim_steps),
                    final_sim_step=sim_step, instruction=instruction,
                    semantics="head-camera RGB after each executed native step; final three-camera RGB and EE state")


def execute_ee_chunk(adapter, action, rows, receipt, segments, progress, dispatch):
    """Dispatch native EE rows one native step each, with transparent IK accounting.

    ``ik_failure_stop_steps`` (optional, from the request) stops the chunk once
    either arm's native IK has failed that many consecutive steps; otherwise IK
    failures are only counted, exactly as the native interface tolerates them.
    """
    segments.append({"action_space": "native_ee_pose", "pose_convention": EE_CHUNK_POSE_CONVENTION,
                     "frame_id": FRAME_ID, "actions": rows, "dt_s": 1 / adapter.control_hz,
                     "native_interface": NATIVE_INTERFACE, "policy_provenance": action["policy_provenance"]})
    limit = action.get("ik_failure_stop_steps")
    capture = (PolicyCaptureWriter(Path(adapter.directory) / "policy_captures")
               if action.get("policy_capture") else None)
    ik = {side: dict(failed_steps=0, unknown_steps=0, max_consecutive_failed_steps=0) for side in ("left", "right")}
    streak = dict(left=0, right=0)
    receipt["ee_chunk_ik"] = ik
    receipt["ik_failure_stop_steps"] = limit
    receipt["gripper_commands_outside_unit_interval_rows"] = 0
    progress({"phase": "started", "planned_steps": len(rows), "sim_step": adapter.control_step})
    for row in rows:
        if any(not 0 <= row[key][0] <= 1 for key in EE_CHUNK_GRIPPER_KEYS):
            # The native interface clips the opening command to [0, 1]; count it.
            receipt["gripper_commands_outside_unit_interval_rows"] += 1
        adapter.last_ee_ik = None
        dispatched = dispatch(row, command_phase="ee_chunk", native=adapter._native_ee_action)
        statuses = adapter.last_ee_ik
        if statuses is not None:
            for side in ("left", "right"):
                status = statuses.get(side)
                if status is None:
                    ik[side]["unknown_steps"] += 1
                elif status:
                    streak[side] = 0
                else:
                    streak[side] += 1
                    ik[side]["failed_steps"] += 1
                    ik[side]["max_consecutive_failed_steps"] = max(ik[side]["max_consecutive_failed_steps"], streak[side])
        if not dispatched:
            break
        if capture is not None:
            capture.add(adapter.control_step, adapter.native_policy_observation())
        failed = [side for side in ("left", "right") if limit is not None and streak[side] >= limit]
        if failed:
            receipt.update(status="stopped", reason="ik_failure_stop", failed_arm=failed[0] if len(failed) == 1 else "both")
            receipt["stop_confirmation"] = adapter._hold()
            break
    else:
        receipt.update(status="completed", reason=None)
    if capture is not None:
        receipt["policy_capture"] = capture.write()
