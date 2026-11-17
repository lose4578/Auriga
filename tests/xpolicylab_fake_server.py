"""In-process stand-in for an XPolicyLab websocket policy server running the WallX adapter.

It decodes frames with the reference ``msgpack`` + ``msgpack_numpy`` codec (as
XPolicyLab ``client_server/ws/protocol/codec.py`` does) and enforces the WallX
client-held keyframe-memory rules of ``wall_x/serving/robodojo/memory_protocol.py``
that a client can violate: dense first request, history continuity, bounded
advance, sparse ring coverage and causal memory ids. No model, no GPU.
"""
import threading
import time

import numpy as np

import msgpack
import msgpack_numpy
from websockets.sync.server import serve

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
RING_PLAN = dict(span=60, window=48, num_candidates=4, select_stride=16, short_term=[2, 4], selection=True)


def needed_ring_frames(plan, now, last_select_frame):
    oldest = max(0, now - plan["span"])
    needed = {now}
    if plan["selection"] and now - last_select_frame >= plan["select_stride"]:
        start = max(0, now - plan["window"])
        span = max(1, now - start)
        needed.update(min(max(start + int((i + 0.5) * span / plan["num_candidates"]), oldest), now)
                      for i in range(plan["num_candidates"]))
    taps, stride = plan["short_term"]
    needed.update(max(oldest, now - k * stride) for k in range(taps, 0, -1))
    return sorted(needed)


class FakeWallXServer:
    def __init__(self, *, chunk=32, replan=16, fault=None, delay_s=0.0):
        self.chunk, self.replan, self.fault, self.delay_s = chunk, replan, fault, delay_s
        self.instance_id = "fake-instance"
        self.frames, self.observations, self.errors = [], [], []
        self.server = serve(self._handle, "127.0.0.1", 0, compression=None, max_size=None)
        self.url = "ws://127.0.0.1:%d" % self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.thread.join(timeout=5)

    @staticmethod
    def _pack(value):
        return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)

    def _reply(self, frame, message_type, payload):
        return self._pack(dict(message_type=message_type, message_id=frame["message_id"],
                               evaluation_id=frame["evaluation_id"], action_case_id=frame.get("action_case_id"),
                               trial_id=frame.get("trial_id"), repeat_index=frame.get("repeat_index"),
                               step=frame.get("step", 0), sent_at="2026-11-12T00:00:00+00:00", payload=payload))

    def _handle(self, websocket):
        for raw in websocket:
            frame = msgpack.unpackb(raw, raw=False, strict_map_key=False, object_hook=msgpack_numpy.decode)
            self.frames.append(frame)
            kind = frame["message_type"]
            try:
                if kind == "hello":
                    reply = self._reply(frame, "hello_ack", dict(ok=True, server="xpolicylab_policy_server",
                                                                 server_instance_id=self.instance_id))
                elif kind == "reset":
                    reply = self._reply(frame, "reset_result", dict(ok=True, result=dict(
                        version=1, prev_sim_frame=None, mem_frame_ids=[], last_sel_frame=0, consec_skip=0,
                        dump_count=0, rng_state=b"\x07" * 16)))
                elif kind == "call" and frame["payload"].get("func_name") == "infer":
                    if self.delay_s:
                        time.sleep(self.delay_s)
                    if self.fault == "error":
                        raise ValueError("synthetic infer failure")
                    reply = self._reply(frame, "call_result", dict(ok=True, result=self._infer(frame["payload"]["obs"]),
                                                                   latency_ms=1.0))
                elif kind == "close":
                    continue
                else:
                    raise ValueError("unsupported request " + kind)
            except Exception as error:
                self.errors.append(str(error))
                reply = self._reply(frame, "error", dict(code="call_failed", message=str(error), details={}))
            websocket.send(reply)

    def _infer(self, obs):
        self.observations.append(obs)
        for camera in CAMERAS:
            image = obs["vision"][camera]["color"]
            assert isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3, camera
        for key, size in (("left_ee_pose", 7), ("right_ee_pose", 7), ("left_ee_joint_state", 1), ("right_ee_joint_state", 1)):
            assert np.asarray(obs["state"][key]).shape == (size,), key
        assert isinstance(obs["instruction"], str) and obs["instruction"].strip()
        assert isinstance(obs["task_id"], str) and obs["task_id"]
        now, state = obs["sim_frame"], obs["kfmem_client_state_v1"]
        assert state["version"] == 1 and isinstance(state["context_key"], str)
        assert isinstance(state["rng_state"], bytes)
        previous = state["prev_sim_frame"]
        ring = [item["sim_frame"] for item in state["ring"]]
        if state.get("ring_sparse"):
            assert previous is not None and not obs.get("kfmem_history")
            assert now - previous <= self.replan, "advance beyond replan_steps"
            assert ring[-1] == now and set(needed_ring_frames(RING_PLAN, now, state["last_sel_frame"])) <= set(ring)
        else:
            history = [item["sim_frame"] for item in obs.get("kfmem_history", [])]
            expected = list(range(1, now + 1)) if previous is None else list(range(previous + 1, now + 1))
            assert history == expected, (history, expected)
            assert ring == list(range(max(0, now - 60), now + 1)), "first request must send the full ring"
        for item in state["ring"] + state["mem"]:
            assert item["face_view"].dtype == np.uint8
        pose = np.asarray(obs["state"]["left_ee_pose"], dtype=np.float64)
        actions = []
        for step in range(self.chunk):
            row = {}
            for side in ("left", "right"):
                position = (pose[:3] + 0.001 * (step + 1)).astype(np.float32)
                row[side + "_ee_pose"] = np.concatenate([position, np.asarray([1, 0, 0, 0], np.float32)])
                row[side + "_ee_joint_state"] = np.asarray([0.5], np.float32)
            actions.append(row)
        if self.fault == "bad_actions":
            actions[0].pop("right_ee_pose")
        memory = [0] if now > 0 else []
        return dict(actions=actions, kfmem_state=dict(version=1, prev_sim_frame=now, mem_frame_ids=memory,
                    last_sel_frame=now, consec_skip=0, dump_count=state["dump_count"], rng_state=b"\x08" * 16,
                    ring_plan=dict(RING_PLAN)))
