"""XPolicyLab websocket client and WallX session against an in-process fake server (CPU only).

Requires msgpack, msgpack-numpy and websockets (the isolated policy-client
environment); skipped where they are absent.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HAVE_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("msgpack", "msgpack_numpy", "websockets"))

if HAVE_DEPENDENCIES:
    from xpolicylab_fake_server import FakeWallXServer
    from auriga.runtime.policy.policy_workers.xpolicylab_ws_client import (
        PolicyServerError, WallXSession, XPolicyLabClient, pack_frame, unpack_frame)
    from auriga.runtime.policy.delegate_policy import PolicyBridgeError, XPolicyLabProcessPolicy


def images(shade=0, size=(6, 8)):
    return {camera: np.full(size + (3,), shade + index, dtype=np.uint8)
            for index, camera in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist"))}


STATE = dict(left_ee_pose=np.array([0.1, 0.2, 0.9, 1, 0, 0, 0.]), right_ee_pose=np.array([0.1, -0.2, 0.9, 1, 0, 0, 0.]),
             left_ee_joint_state=np.array([1.0]), right_ee_joint_state=np.array([0.0]))


def write_capture(directory, name, heads, final_images=None):
    final_images = final_images or images()
    path = Path(directory) / f"{name}.npz"
    np.savez(path, head=np.stack(heads), sim_steps=np.arange(len(heads)),
             **{"final_" + camera: image for camera, image in final_images.items()},
             **{"state_" + key: value for key, value in STATE.items()})
    return path


@unittest.skipUnless(HAVE_DEPENDENCIES, "msgpack, msgpack_numpy and websockets are policy-client dependencies")
class CodecCheck(unittest.TestCase):
    def test_numpy_frames_round_trip_with_reference_msgpack_numpy(self):
        import msgpack
        import msgpack_numpy
        value = dict(payload=dict(a=np.arange(6, dtype=np.float32).reshape(2, 3), b=np.uint8(7), raw=b"\x00\x01"))
        reference = msgpack.unpackb(pack_frame(value), raw=False, object_hook=msgpack_numpy.decode)
        np.testing.assert_array_equal(reference["payload"]["a"], value["payload"]["a"])
        self.assertEqual(reference["payload"]["a"].dtype, np.float32)
        self.assertEqual(reference["payload"]["raw"], b"\x00\x01")
        ours = unpack_frame(msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True))
        np.testing.assert_array_equal(ours["payload"]["a"], value["payload"]["a"])
        self.assertEqual(int(ours["payload"]["b"]), 7)


@unittest.skipUnless(HAVE_DEPENDENCIES, "msgpack, msgpack_numpy and websockets are policy-client dependencies")
class ClientProtocolCheck(unittest.TestCase):
    def server(self, **options):
        server = FakeWallXServer(**options)
        self.addCleanup(server.close)
        return server

    def client(self, server, timeout_s=5.0):
        client = XPolicyLabClient(server.url, evaluation_id="eval-1", trial_id="trial-1", timeout_s=timeout_s)
        self.addCleanup(client.close)
        return client

    def test_envelope_handshake_reset_and_call_infer_schema(self):
        server = self.server()
        client = self.client(server)
        self.assertEqual(client.server_instance_id, "fake-instance")
        session = WallXSession(client, replan_steps=16)
        session.reset(task_id="stack_bowls", context_key="ctx-1")
        session.add_frames([images(0)["cam_head"]])
        result = session.infer(images(0), STATE, "Stack the bowls.")
        hello, reset, call = server.frames[:3]
        for frame in (hello, reset, call):
            self.assertEqual(set(frame), {"message_type", "message_id", "evaluation_id", "action_case_id", "trial_id",
                                          "repeat_index", "step", "sent_at", "payload"})
            self.assertEqual((frame["evaluation_id"], frame["trial_id"]), ("eval-1", "trial-1"))
        self.assertEqual([frame["message_type"] for frame in server.frames[:3]], ["hello", "reset", "call"])
        self.assertEqual(reset["payload"], {"trial_id": "trial-1"})
        self.assertEqual(call["payload"]["func_name"], "infer")
        observation = server.observations[0]
        self.assertEqual(observation["sim_frame"], 0)
        self.assertEqual(observation["task_id"], "stack_bowls")
        self.assertEqual(observation["kfmem_client_state_v1"]["rng_state"], b"\x07" * 16)
        self.assertEqual(observation["kfmem_history"], [])
        self.assertEqual(result["chunk_length"], 32)
        self.assertEqual(set(result["actions"][0]), {"left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"})
        self.assertEqual(len(result["actions"][0]["left_ee_pose"]), 7)
        self.assertEqual(result["actions"][0]["left_ee_pose"][3:], [1.0, 0.0, 0.0, 0.0])
        json.dumps(result, allow_nan=False)
        self.assertEqual(server.errors, [])

    def test_memory_switches_to_sparse_ring_and_respects_replan_advance(self):
        server = self.server()
        session = WallXSession(self.client(server), replan_steps=16)
        session.reset(task_id="stack_bowls", context_key="ctx-2")
        session.add_frames([images(0)["cam_head"]])
        session.infer(images(0), STATE, "Stack the bowls.")
        for round_index in range(4):
            session.add_frames([images(round_index + 1)["cam_head"]] * 16)
            session.infer(images(round_index + 1), STATE, "Stack the bowls.")
        self.assertEqual(server.errors, [])
        self.assertEqual([obs["sim_frame"] for obs in server.observations], [0, 16, 32, 48, 64])
        self.assertTrue(all(obs["kfmem_client_state_v1"].get("ring_sparse") for obs in server.observations[1:]))
        self.assertEqual([item["sim_frame"] for item in server.observations[-1]["kfmem_client_state_v1"]["mem"]], [0])
        self.assertEqual([frame["step"] for frame in server.frames if frame["message_type"] == "call"], [0, 1, 2, 3, 4])

    def test_server_error_frame_and_invalid_chunk_poison_the_connection(self):
        for fault, error_type in (("error", PolicyServerError), ("bad_actions", ValueError)):
            with self.subTest(fault=fault):
                server = self.server(fault=fault)
                client = self.client(server)
                session = WallXSession(client)
                session.reset(task_id="stack_bowls", context_key="ctx-3")
                session.add_frames([images()["cam_head"]])
                with self.assertRaises(error_type):
                    session.infer(images(), STATE, "Stack the bowls.")
                if fault == "error":
                    self.assertTrue(client.closed)
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        client.reset()

    def test_timeout_closes_without_resending(self):
        server = self.server(delay_s=1.5)
        client = self.client(server, timeout_s=0.5)
        session = WallXSession(client)
        session.reset(task_id="stack_bowls", context_key="ctx-4")
        session.add_frames([images()["cam_head"]])
        with self.assertRaisesRegex(TimeoutError, "timeout"):
            session.infer(images(), STATE, "Stack the bowls.")
        self.assertTrue(client.closed)
        self.assertEqual([frame["message_type"] for frame in server.frames], ["hello", "reset", "call"])

    def test_rejects_non_websocket_url_and_bad_replan(self):
        with self.assertRaisesRegex(ValueError, "websocket"):
            XPolicyLabClient("http://127.0.0.1:1", evaluation_id="e", trial_id="t")
        with self.assertRaisesRegex(ValueError, "replan"):
            WallXSession(object(), replan_steps=49)


@unittest.skipUnless(HAVE_DEPENDENCIES, "msgpack, msgpack_numpy and websockets are policy-client dependencies")
class ProcessBridgeCheck(unittest.TestCase):
    def test_stdio_bridge_reads_captures_and_returns_chunk(self):
        server = FakeWallXServer()
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as directory:
            policy = XPolicyLabProcessPolicy(server.url, python=sys.executable, timeout_s=5,
                                             log_path=Path(directory) / "client.log")
            try:
                self._exercise(policy, server, directory)
            finally:
                policy.close()

    def _exercise(self, policy, server, directory):
            self.assertEqual(policy.metadata["status"], "ready")
            self.assertEqual(policy.metadata["server_instance_id"], "fake-instance")
            policy.reset(task_id="stack_bowls", context_key="ctx-5")
            first = write_capture(directory, "first", [images()["cam_head"]])
            result = policy.infer([first], "Stack the bowls.")
            self.assertEqual((result["sim_frame"], result["chunk_length"]), (0, 32))
            chunk = write_capture(directory, "chunk", [images(3)["cam_head"]] * 16)
            result = policy.infer([chunk], "Stack the bowls.")
            self.assertEqual(result["sim_frame"], 16)
            self.assertEqual(server.errors, [])
            # A missing capture is a client error: the process is poisoned, and the next reset reconnects.
            with self.assertRaises(PolicyBridgeError):
                policy.infer([Path(directory) / "missing.npz"], "Stack the bowls.")
            self.assertFalse(policy.alive)
            policy.reset(task_id="stack_bowls", context_key="ctx-6")
            self.assertTrue(policy.alive)

    def test_unreachable_server_fails_start(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PolicyBridgeError):
                XPolicyLabProcessPolicy("ws://127.0.0.1:9", python=sys.executable, timeout_s=2,
                                        log_path=Path(directory) / "client.log")


if __name__ == "__main__":
    unittest.main()
