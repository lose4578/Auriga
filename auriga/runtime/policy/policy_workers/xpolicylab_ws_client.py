"""Generic XPolicyLab websocket policy client, plus the WallX (WALL-OSS) session on top of it.

Run in an interpreter that has ``numpy``, ``msgpack`` and ``websockets`` (>=12,
sync client). Harness launches it by file path with ``--stdio`` and exchanges
one JSON object per line; this module never loads a model, never touches the
simulator and never executes actions. Execution stays with Harness.

Wire protocol (XPolicyLab ``client_server/ws``; see docs/design/policy-delegation.md):

* one websocket binary message per frame, msgpack map with ``message_type``,
  ``message_id``, ``evaluation_id``, ``action_case_id``, ``trial_id``,
  ``repeat_index``, ``step``, ``sent_at`` and ``payload``;
* numpy arrays use the msgpack-numpy map ``{nd, type, kind, shape, data}``;
* request/response pairs ``hello``/``hello_ack``, ``reset``/``reset_result``,
  ``call``/``call_result``, ``trial_end``/``trial_end_ack``; failures arrive as
  ``error`` with ``{code, message, details}``.

WallX is driven exactly like its official ``deploy.eval_one_episode``: ``reset``
returns the client-held keyframe-memory state; each request is a ``call`` of
``infer`` whose observation carries ``vision``/``state``/``instruction`` plus the
``kfmem_client_state_v1`` memory fields; the response returns the full action
chunk and the updated ``kfmem_state``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
from urllib.parse import urlsplit
import uuid

RESPONSES = {"hello": "hello_ack", "reset": "reset_result", "call": "call_result",
             "infer": "infer_result", "trial_end": "trial_end_ack", "heartbeat": "heartbeat_ack"}
POLICY_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = ("left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state")
ACTION_DIMENSIONS = {"left_ee_pose": 7, "right_ee_pose": 7, "left_ee_joint_state": 1, "right_ee_joint_state": 1}
#: Raw head frames the WallX memory window may read (memory_client.RING_FRAMES).
RING_FRAMES = 60
#: Frames one request may advance (wall_x memory_protocol.MAX_ADVANCE).
MAX_ADVANCE = 48
_CONTEXT_KEY = re.compile(r"[A-Za-z0-9_-]{1,128}")


class PolicyServerError(RuntimeError):
    """An ``error`` frame from the policy server: ``code`` is the XPolicyLab ErrorCode."""

    def __init__(self, code, message, details=None):
        super().__init__(f"policy_server_error:{code}: {message}")
        self.code, self.details = code, details or {}


# --------------------------------------------------------------------------- codec
def _encode_numpy(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OV":
            raise TypeError("object/structured numpy arrays are not supported")
        return {b"nd": True, b"type": value.dtype.str, b"kind": b"", b"shape": list(value.shape),
                b"data": np.ascontiguousarray(value).tobytes()}
    if isinstance(value, (np.bool_, np.number)):
        return {b"nd": False, b"type": value.dtype.str, b"data": value.tobytes()}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _decode_numpy(value):
    import numpy as np

    if value.get(b"nd") is True:
        if value.get(b"kind") not in (b"", None, ""):
            raise ValueError("object/structured numpy arrays are not supported")
        return np.frombuffer(value[b"data"], dtype=np.dtype(value[b"type"])).reshape(value[b"shape"]).copy()
    if value.get(b"nd") is False and b"data" in value:
        return np.frombuffer(value[b"data"], dtype=np.dtype(value[b"type"]))[0]
    return value


def pack_frame(frame):
    import msgpack
    return msgpack.packb(frame, default=_encode_numpy, use_bin_type=True)


def unpack_frame(data):
    import msgpack
    value = msgpack.unpackb(data, raw=False, strict_map_key=False, object_hook=_decode_numpy)
    if not isinstance(value, dict):
        raise ValueError("policy_frame_must_be_map")
    return value


# --------------------------------------------------------------------------- transport
class XPolicyLabClient:
    """Synchronous single-request-in-flight client for the XPolicyLab ws server.

    No automatic reconnect or resend: a timeout or transport failure closes the
    connection, so a late response can never be attributed to a later request.
    """

    def __init__(self, url, *, evaluation_id, trial_id, action_case_id=None, timeout_s=60.0,
                 connect=None):
        parts = urlsplit(url)
        if parts.scheme not in {"ws", "wss"} or not parts.hostname:
            raise ValueError("policy_server_requires_websocket_url")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("policy_timeout_must_be_positive_and_finite")
        if connect is None:
            from websockets.sync.client import connect
        self.url, self.timeout_s = url, float(timeout_s)
        self.evaluation_id, self.trial_id, self.action_case_id = evaluation_id, trial_id, action_case_id
        self.closed = False
        self.connection = connect(url, compression=None, max_size=None, open_timeout=self.timeout_s,
                                  close_timeout=min(self.timeout_s, 5.0))
        try:
            self.hello = self.request("hello", {})["payload"]
            self.server_instance_id = self.hello.get("server_instance_id")
        except BaseException:
            self.close()
            raise

    def request(self, message_type, payload, *, step=0, timeout_s=None):
        if self.closed:
            raise RuntimeError("policy_server_connection_closed")
        expected = RESPONSES[message_type]
        message_id = uuid.uuid4().hex
        frame = dict(message_type=message_type, message_id=message_id, evaluation_id=self.evaluation_id,
                     action_case_id=self.action_case_id, trial_id=self.trial_id, repeat_index=None,
                     step=int(step), sent_at=datetime.now(timezone.utc).isoformat(), payload=payload)
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else timeout_s)
        try:
            self.connection.send(pack_frame(frame))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"policy_server_timeout_waiting_for_{expected}")
                try:
                    raw = self.connection.recv(timeout=remaining)
                except TimeoutError as error:
                    raise TimeoutError(f"policy_server_timeout_waiting_for_{expected}") from error
                if not isinstance(raw, (bytes, bytearray)):
                    raise ValueError("policy_server_sent_text_frame")
                response = unpack_frame(bytes(raw))
                if response.get("message_id") != message_id:
                    # Only one request is ever in flight on this connection.
                    raise ValueError("policy_server_response_id_mismatch")
                if response.get("message_type") == "error":
                    error = response.get("payload") or {}
                    raise PolicyServerError(error.get("code", "internal"), error.get("message", ""), error.get("details"))
                if response.get("message_type") != expected:
                    raise ValueError(f"policy_server_expected_{expected}_got_{response.get('message_type')}")
                if not isinstance(response.get("payload"), dict):
                    raise ValueError("policy_server_payload_must_be_map")
                return response
        except BaseException:
            self.close()
            raise

    def reset(self):
        return self.request("reset", {"trial_id": self.trial_id})["payload"].get("result")

    def call(self, func_name, obs=None, *, step=0):
        payload = {"func_name": func_name}
        if obs is not None:
            payload["obs"] = obs
        return self.request("call", payload, step=step)["payload"].get("result")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.connection.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- WallX keyframe memory
def needed_ring_frames(plan, now, last_select_frame):
    """Ring frames the server reads at ``now`` (XPolicyLab policy/WallX/memory_client.py)."""
    oldest = max(0, now - int(plan["span"]))
    needed = {now}
    if plan["selection"] and now - last_select_frame >= int(plan["select_stride"]):
        start = max(0, now - int(plan["window"]))
        span = max(1, now - start)
        count = int(plan["num_candidates"])
        needed.update(min(max(start + int((i + 0.5) * span / count), oldest), now) for i in range(count))
    taps, stride = (int(v) for v in plan["short_term"])
    needed.update(max(oldest, now - k * stride) for k in range(taps, 0, -1))
    return sorted(needed)


def _ring_plan(value):
    keys = {"span", "window", "num_candidates", "select_stride", "short_term", "selection"}
    if not isinstance(value, dict) or set(value) != keys or not isinstance(value["selection"], bool):
        raise ValueError("invalid_ring_plan")
    if not isinstance(value["short_term"], (list, tuple)) or len(value["short_term"]) != 2:
        raise ValueError("invalid_ring_plan")
    numbers = [value[k] for k in ("span", "window", "num_candidates", "select_stride")] + list(value["short_term"])
    if any(type(v) is not int or v < 0 for v in numbers) or value["span"] > RING_FRAMES:
        raise ValueError("invalid_ring_plan")
    return dict(value)


class KeyframeMemoryState:
    """Client-held WallX memory for one delegation (``kfmem_client_state_v1``).

    Mirrors XPolicyLab ``policy/WallX/memory_client.KFMemClientState``: raw head
    frames are kept here, each request ships the frames the server will read and
    the server returns the retained memory frame ids.
    """

    def __init__(self, context_key, *, capacity=22):
        if not isinstance(context_key, str) or not _CONTEXT_KEY.fullmatch(context_key):
            raise ValueError("invalid_context_key")
        self.context_key, self.capacity = context_key, capacity
        self.frames, self.ring_plan = {}, None
        self.mem_frame_ids, self.mem_boxes = [], {}
        self.prev_sim_frame, self.rng_state = None, None
        self.last_sel_frame = self.consec_skip = self.dump_count = 0

    def reset(self, state):
        if (not isinstance(state, dict) or state.get("version") != 1 or state.get("prev_sim_frame") is not None
                or state.get("mem_frame_ids") != []):
            raise ValueError("invalid_kfmem_reset_state")
        self.__init__(self.context_key, capacity=self.capacity)
        self.rng_state = state.get("rng_state")

    def record_frame(self, frame, image):
        import numpy as np

        image = np.asarray(image)
        if type(frame) is not int or frame < 0:
            raise ValueError("frame_must_be_nonnegative_integer")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise ValueError("frame_image_must_be_uint8_hwc3")
        self.frames[frame] = np.ascontiguousarray(image).copy()

    def request_state(self, now):
        if type(now) is not int or now < 0 or (self.prev_sim_frame is not None and now <= self.prev_sim_frame):
            raise ValueError("request_frame_must_advance")
        sparse = self.ring_plan is not None
        ring = (needed_ring_frames(self.ring_plan, now, self.last_sel_frame) if sparse
                else list(range(max(0, now - RING_FRAMES), now + 1)))
        if any(frame not in self.frames for frame in ring + self.mem_frame_ids):
            raise ValueError("dense_raw_frames_or_memory_frames_missing")
        state = {"version": 1, "prev_sim_frame": self.prev_sim_frame,
                 "ring": [{"sim_frame": f, "face_view": self.frames[f]} for f in ring],
                 "mem": [{"sim_frame": f, "face_view": self.frames[f]} for f in self.mem_frame_ids],
                 "mem_boxes": {str(f): list(box) for f, box in self.mem_boxes.items()},
                 "last_sel_frame": self.last_sel_frame, "consec_skip": self.consec_skip,
                 "dump_count": self.dump_count, "rng_state": self.rng_state, "context_key": self.context_key}
        if sparse:
            state["ring_sparse"] = True
        return state

    def history(self, now):
        if self.ring_plan is not None:
            return []
        first = 1 if self.prev_sim_frame is None else self.prev_sim_frame + 1
        return [{"sim_frame": f, "face_view": self.frames[f]} for f in range(first, now + 1)]

    def accept_result(self, state):
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("invalid_kfmem_state_version")
        now, ids = state.get("prev_sim_frame"), state.get("mem_frame_ids")
        if type(now) is not int or now not in self.frames or (self.prev_sim_frame is not None and now <= self.prev_sim_frame):
            raise ValueError("invalid_kfmem_response_anchor")
        if (not isinstance(ids, list) or len(ids) > self.capacity or len(set(ids)) != len(ids)
                or any(type(f) is not int or f not in self.frames or f >= now for f in ids)):
            raise ValueError("invalid_causal_memory_frame_ids")
        boxes = {}
        for key, box in (state.get("mem_boxes") or {}).items():
            if not isinstance(key, str) or not key.isdecimal() or int(key) not in ids:
                raise ValueError("memory_box_must_reference_retained_frame")
            if (not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(x) is not int or not 0 <= x <= 1000 for x in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("invalid_memory_box")
            boxes[int(key)] = list(box)
        counters = {}
        for name in ("last_sel_frame", "consec_skip", "dump_count"):
            value = state.get(name)
            if type(value) is not int or value < 0 or (name == "last_sel_frame" and value > now):
                raise ValueError("invalid_kfmem_counter:" + name)
            counters[name] = value
        rng = state.get("rng_state")
        if rng is not None and (not isinstance(rng, bytes) or not 0 < len(rng) <= 1 << 20):
            raise ValueError("invalid_rng_state")
        if state.get("ring_plan") is not None:
            self.ring_plan = _ring_plan(state["ring_plan"])
        self.prev_sim_frame, self.mem_frame_ids, self.mem_boxes, self.rng_state = now, list(ids), boxes, rng
        for name, value in counters.items():
            setattr(self, name, value)
        keep = set(ids) | set(range(max(0, now - RING_FRAMES), now + 1))
        self.frames = {f: image for f, image in self.frames.items() if f in keep}


def validate_actions(actions, *, minimum):
    """The WallX chunk: dicts of native EE keys, finite, at least ``minimum`` rows."""
    import numpy as np

    if not isinstance(actions, (list, tuple)) or len(actions) < minimum:
        raise ValueError(f"policy_chunk_requires_at_least_{minimum}_actions")
    rows = []
    for action in actions:
        if not isinstance(action, dict) or set(action) != set(ACTION_DIMENSIONS):
            raise ValueError("policy_action_keys_mismatch")
        row = {}
        for key, size in ACTION_DIMENSIONS.items():
            value = np.asarray(action[key], dtype=np.float64).reshape(-1)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError("policy_action_invalid_entry:" + key)
            row[key] = [float(x) for x in value]
        rows.append(row)
    return rows


class WallXSession:
    """One delegation's WallX episode: reset once, then infer at each replan boundary.

    Frames are numbered from 0 at the delegation start; every executed native
    step adds one head frame, matching the official per-step ``record_frame``.
    """

    def __init__(self, client, *, replan_steps=16, memory_capacity=22):
        if type(replan_steps) is not int or not 1 <= replan_steps <= MAX_ADVANCE:
            raise ValueError("replan_steps_must_be_1_to_48")
        self.client, self.replan_steps, self.memory_capacity = client, replan_steps, memory_capacity
        self.memory, self.task_id, self.frame, self.inferences = None, None, -1, 0

    def reset(self, *, task_id, context_key):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("wallx_requires_task_id")
        self.memory = KeyframeMemoryState(context_key, capacity=self.memory_capacity)
        self.memory.reset(self.client.reset())
        self.task_id, self.frame, self.inferences = task_id, -1, 0

    def add_frames(self, heads):
        if self.memory is None:
            raise RuntimeError("wallx_session_not_reset")
        for image in heads:
            self.frame += 1
            self.memory.record_frame(self.frame, image)

    def infer(self, images, state, instruction):
        import numpy as np

        if self.memory is None or self.frame < 0:
            raise RuntimeError("wallx_infer_requires_recorded_frames")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("wallx_requires_nonempty_instruction")
        now = self.frame
        observation = {
            "vision": {camera: {"color": np.ascontiguousarray(images[camera], dtype=np.uint8)} for camera in POLICY_CAMERAS},
            "state": {key: np.asarray(state[key], dtype=np.float64).reshape(-1) for key in STATE_KEYS},
            "instruction": instruction, "task_id": self.task_id, "sim_frame": now,
            "kfmem_client_state_v1": self.memory.request_state(now), "kfmem_history": self.memory.history(now)}
        started = time.perf_counter()
        result = self.client.call("infer", observation, step=self.inferences)
        elapsed = time.perf_counter() - started
        if not isinstance(result, dict) or not isinstance(result.get("kfmem_state"), dict):
            raise ValueError("wallx_infer_must_return_actions_and_kfmem_state")
        rows = validate_actions(result.get("actions"), minimum=self.replan_steps)
        self.memory.accept_result(result["kfmem_state"])
        self.inferences += 1
        return dict(actions=rows, chunk_length=len(rows), sim_frame=now, inference_seconds=elapsed,
                    memory_frame_ids=list(self.memory.mem_frame_ids))


# --------------------------------------------------------------------------- stdio bridge
def load_capture(path):
    """Read one worker capture npz: head frames and the final three-camera RGB + EE state."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        heads = list(data["head"])
        images = {camera: data["final_" + camera] for camera in POLICY_CAMERAS}
        state = {key: data["state_" + key] for key in STATE_KEYS}
    return heads, images, state


def serve(session, client, lines, emit):
    for line in lines:
        request = json.loads(line)
        operation = request.get("op")
        if operation == "reset":
            session.reset(task_id=request["task_id"], context_key=request["context_key"])
            emit(dict(status="reset", server_instance_id=client.server_instance_id))
        elif operation == "infer":
            images = state = None
            for path in request["capture_paths"]:
                heads, images, state = load_capture(path)
                session.add_frames(heads)
            if images is None:
                raise ValueError("wallx_infer_requires_a_capture")
            emit(dict(status="inference_completed", **session.infer(images, state, request["instruction"])))
        elif operation == "close":
            emit(dict(status="closed"))
            return
        else:
            raise ValueError(f"unsupported_operation:{operation}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--memory-capacity", type=int, default=22)
    parser.add_argument("--evaluation-id", default="auriga")
    parser.add_argument("--trial-id", default="auriga-delegation")
    parser.add_argument("--stdio", action="store_true", required=True)
    args = parser.parse_args(argv)
    out = sys.stdout

    def emit(value):
        out.write(json.dumps(value, allow_nan=False) + "\n")
        out.flush()

    client = None
    try:
        sys.stdout = sys.stderr  # third-party prints must never corrupt the JSON channel
        client = XPolicyLabClient(args.server_url, evaluation_id=args.evaluation_id, trial_id=args.trial_id,
                                  timeout_s=args.timeout)
        session = WallXSession(client, replan_steps=args.replan_steps, memory_capacity=args.memory_capacity)
        emit(dict(status="ready", protocol="xpolicylab-ws", policy_adapter="WallX",
                  server=client.hello.get("server"), server_instance_id=client.server_instance_id,
                  replan_steps=args.replan_steps))
        serve(session, client, sys.stdin, emit)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit(dict(status="error", error=f"{type(error).__name__}: {error}",
                  code=getattr(error, "code", None)))
        return 1
    finally:
        sys.stdout = out
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
