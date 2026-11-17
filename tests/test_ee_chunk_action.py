"""Native EE-pose chunks from an external policy: contract and worker execution (CPU fixture)."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

import numpy as np

from auriga.robot.backends.robodojo.robot_worker import RoboDojoAdapter
from auriga.robot.model.arx_contract import validate_action, validate_ee_chunk, validate_preview_action
from auriga.robot.motion.motion_speed import resolve_action_speed


def row(x=0.3, gripper=1.0, w=1.0):
    return dict(left_ee_pose=[x, 0.2, 0.9, w, 0.0, 0.0, 0.0], right_ee_pose=[x, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
                left_ee_joint_state=[gripper], right_ee_joint_state=[0.5])


def chunk(count=4, **updates):
    action = dict(type="EEChunkAction", frame_id="env_origin", pose_convention="xyz_qwxyz",
                  actions=[row(0.3 + 0.001 * index) for index in range(count)], timeout_s=20,
                  policy_provenance={"fixture": "ee_chunk"})
    action.update(updates)
    return action


class EEChunkContractCheck(unittest.TestCase):
    def test_valid_chunk_and_routing(self):
        self.assertEqual(len(validate_ee_chunk(chunk())), 4)
        self.assertEqual(validate_action(chunk()), ("both", None))
        self.assertEqual(resolve_action_speed(chunk(), 1.25), chunk())
        with self.assertRaisesRegex(ValueError, "dual_arm_motion_preview_unsupported"):
            validate_preview_action(chunk())

    def test_rejections_name_the_violated_bound(self):
        bad_quaternion = chunk(actions=[row(w=0.5)])
        outside = chunk(actions=[row(x=2.5)])
        nonfinite = chunk(actions=[dict(row(), left_ee_joint_state=[float("nan")])])
        missing = chunk(actions=[{key: value for key, value in row().items() if key != "right_ee_pose"}])
        for action, reason in ((chunk(frame_id="world"), "env_origin_xyz_qwxyz"),
                               (chunk(pose_convention="xyz_qxyzw"), "env_origin_xyz_qwxyz"),
                               (chunk(actions=[]), "1_to_50"), (chunk(count=51), "1_to_50"),
                               (missing, "both_arm_poses_and_grippers"), (nonfinite, "finite"),
                               (bad_quaternion, "quaternion must be normalized"), (outside, "2 m envelope"),
                               (chunk(timeout_s=0), "timeout"), (chunk(ik_failure_stop_steps=0), "ik_failure_stop_steps"),
                               (chunk(policy_capture="yes"), "policy_capture"), (chunk(policy_provenance={}), "provenance"),
                               (chunk(speed_scale=0.5), "action_speed_unsupported")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                validate_ee_chunk(action)


class FakeArm:
    def __init__(self, name):
        self.arm_name, self.type = name + "_arm", "target"


class EEChunkWorkerCheck(unittest.TestCase):
    def adapter(self, directory, *, ik_fail=(), step_limit=100, success_at=None, state_keys=None):
        adapter = object.__new__(RoboDojoAdapter)
        adapter.np, adapter.control_hz, adapter.directory = np, 25, Path(directory)
        adapter.control_step = adapter.evaluator_actions = 0
        adapter.versions, adapter.observation_profile = {}, "rgb_proprio_calibrated_v1"
        adapter.goals = adapter.last_dispatched_goals = {"left": 1., "right": 1.}
        arms = [FakeArm("left"), FakeArm("right")]
        self.native_calls, self.ik_calls = [], []

        def solve_ik(target_pose, env_idx, robot, trans="world"):
            self.ik_calls.append(robot.arm_name)
            side = robot.arm_name.split("_")[0]
            return {"status": "Failure" if side in ik_fail else "Success", "joint_value": [0.] * 6}

        adapter.robot = SimpleNamespace(robot_list=arms, robot_key=[], solve_ik=solve_ik)
        adapter.env = SimpleNamespace(end_flag=[False], success=[False], take_action_cnt=[0], step_lim=step_limit)

        def take_action(action):
            # Mirrors eval_env.take_action for the "ee" action type: per-arm native IK.
            self.assertFalse(adapter.env.end_flag[0], "no native call may follow terminal")
            self.native_calls.append(action)
            for arm in arms:
                adapter.robot.solve_ik(target_pose=action[arm.arm_name.split("_")[0] + "_ee_pose"], env_idx=0, robot=arm)
            adapter.env.take_action_cnt[0] += 1
            adapter.control_step += 10
            count = adapter.env.take_action_cnt[0]
            if count == success_at or count == step_limit:
                adapter.env.end_flag[0] = True
                adapter.env.success[0] = count == success_at
        adapter.env.take_action = take_action
        adapter.state = lambda: {"arms": {side: {"q_rad": [0.] * 6} for side in ("left", "right")}}
        adapter.observe = lambda: {"robot_state": adapter.state()}
        keys = state_keys or ("left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state")

        def get_obs(env_idx_list):
            shade = adapter.control_step % 255
            return [dict(vision={camera: dict(color=np.full((4, 5, 3), shade, np.uint8), depth=np.ones((4, 5)))
                                 for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                         state={key: [0.1] * (7 if key.endswith("pose") else 1) for key in keys},
                         instruction="Stack the bowls.", hidden_object_pose=[1, 2, 3])]
        adapter.obs = SimpleNamespace(render_for_capture=lambda: None, get_obs=get_obs)
        return adapter

    def execute(self, adapter, action):
        return adapter.execute(action, lambda value: None, threading.Event())

    def test_rows_use_native_ee_interface_and_capture_policy_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory)
            result = self.execute(adapter, chunk(count=4, policy_capture=True, ik_failure_stop_steps=2))
            self.assertEqual(result["status"], "completed", result.get("reason"))
            self.assertEqual(result["executed_native_steps"], 4)
            self.assertEqual(len(self.native_calls), 4)
            self.assertEqual(set(self.native_calls[0]), {"left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"})
            self.assertEqual(self.native_calls[1]["left_ee_pose"], row(0.301)["left_ee_pose"])
            self.assertEqual(result["planned_trace"]["action_space"], "native_ee_pose")
            self.assertIn("solve_ik", result["planned_trace"]["native_interface"])
            self.assertEqual(result["ee_chunk_ik"]["left"]["failed_steps"], 0)
            self.assertEqual(adapter.robot.solve_ik.__name__, "solve_ik")  # native IK entry restored
            capture = result["policy_capture"]
            self.assertEqual((capture["frames"], capture["final_sim_step"]), (4, 40))
            with np.load(capture["path"]) as data:
                self.assertEqual(data["head"].shape, (4, 4, 5, 3))
                self.assertEqual(sorted(data.files), sorted(["head", "sim_steps", "final_cam_head", "final_cam_left_wrist",
                    "final_cam_right_wrist", "state_left_ee_pose", "state_right_ee_pose", "state_left_ee_joint_state",
                    "state_right_ee_joint_state"]))
            self.assertEqual(adapter.goals, {"left": 1.0, "right": 0.5})

    def test_ik_failure_streak_stops_and_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory, ik_fail=("right",))
            result = self.execute(adapter, chunk(count=6, ik_failure_stop_steps=2))
            self.assertEqual((result["status"], result["reason"], result["failed_arm"]), ("stopped", "ik_failure_stop", "right"))
            self.assertEqual(len(self.native_calls), 2)
            self.assertEqual(result["ee_chunk_ik"]["right"], dict(failed_steps=2, unknown_steps=0, max_consecutive_failed_steps=2))
            self.assertEqual(result["stop_confirmation"]["clock"], "paused")
            # Without a limit the native interface tolerates IK failures; they are counted only.
            adapter = self.adapter(directory, ik_fail=("left",))
            result = self.execute(adapter, chunk(count=3))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["ee_chunk_ik"]["left"]["failed_steps"], 3)

    def test_native_terminal_mid_chunk_stops_without_more_native_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory, success_at=2)
            result = self.execute(adapter, chunk(count=5, policy_capture=True))
            self.assertEqual((result["status"], result["reason"]), ("stopped", "episode_ended"))
            self.assertEqual(result["executed_native_steps"], 2)
            self.assertEqual(len(self.native_calls), 2)

    def test_missing_native_policy_state_fails_honestly(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory, state_keys=("left_ee_pose", "right_ee_pose"))
            result = self.execute(adapter, chunk(count=2, policy_capture=True))
            self.assertEqual(result["status"], "failed")
            self.assertIn("native_observation_missing_policy_state", result["reason"])
            self.assertEqual(result["executed_native_steps"], 1)

    def test_gripper_outside_unit_interval_is_counted_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory)
            action = chunk(actions=[row(gripper=1.2), row(gripper=0.5)])
            result = self.execute(adapter, action)
            self.assertEqual(result["gripper_commands_outside_unit_interval_rows"], 1)
            self.assertEqual(self.native_calls[0]["left_ee_joint_state"], [1.2])
            self.assertEqual(adapter.goals["left"], 0.5)

    def test_policy_observation_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory)
            capture = adapter.policy_observation()
            self.assertEqual((capture["frames"], capture["final_sim_step"], capture["physics_advanced"]), (1, 0, False))
            self.assertEqual(self.native_calls, [])
            with np.load(capture["path"]) as data:
                self.assertNotIn("hidden_object_pose", data.files)
                self.assertFalse(any("depth" in name for name in data.files))


if __name__ == "__main__":
    unittest.main()
