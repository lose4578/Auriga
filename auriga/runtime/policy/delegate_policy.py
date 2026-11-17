"""Harness-side owner of the isolated XPolicyLab websocket client process.

The client (``policy_workers/xpolicylab_ws_client.py``) runs in its own
interpreter and only converts native policy captures into XPolicyLab requests.
This wrapper starts it, bounds every exchange by a wall-clock deadline and
poisons the process after any timeout or error, so a late reply can never be
consumed as the answer to a later request. A poisoned process is replaced by a
fresh connection at the next delegation start, never mid-delegation.
"""
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

CLIENT = Path(__file__).with_name("policy_workers") / "xpolicylab_ws_client.py"


class PolicyBridgeError(RuntimeError):
    """The client reported a failure; ``timeout`` distinguishes a missed deadline."""

    def __init__(self, message, *, timeout=False):
        super().__init__(message)
        self.timeout = timeout


class XPolicyLabProcessPolicy:
    def __init__(self, server_url, *, python=None, timeout_s=60.0, replan_steps=16, log_path,
                 evaluation_id="auriga", trial_id="auriga-delegation", start=True):
        if not isinstance(server_url, str) or not server_url.startswith(("ws://", "wss://")):
            raise ValueError("wall_oss_server_url_requires_websocket_url")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("policy_timeout_must_be_positive_and_finite")
        self.server_url, self.timeout_s, self.replan_steps = server_url, float(timeout_s), replan_steps
        self.python = str(python) if python else sys.executable
        self.log_path, self.evaluation_id, self.trial_id = Path(log_path), evaluation_id, trial_id
        self.process, self.lines, self.metadata, self.log = None, None, None, None
        self.lock = threading.Lock()
        if start:
            self.start()

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        """Connect a fresh client process and wait for the server handshake."""
        self.close()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a")
        command = [self.python, str(CLIENT), "--stdio", "--server-url", self.server_url,
                   "--timeout", str(self.timeout_s), "--replan-steps", str(self.replan_steps),
                   "--evaluation-id", self.evaluation_id, "--trial-id", self.trial_id]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, encoding="utf-8", env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.lines = queue.Queue()
        process, lines = self.process, self.lines

        def pump():
            # A dedicated reader keeps deadlines portable (no select() on pipes).
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, name="xpolicylab-client-reader", daemon=True).start()
        try:
            # Connect plus handshake: one bounded open and one bounded hello.
            self.metadata = self._receive(2 * self.timeout_s + 10, expected="ready")
        except BaseException:
            self.close()
            raise
        return self.metadata

    def _receive(self, timeout, *, expected):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise PolicyBridgeError("policy_client_response_timeout", timeout=True) from None
        if line is None:
            self.close()
            raise PolicyBridgeError("policy_client_exited_before_response")
        result = json.loads(line)
        if result.get("status") == "error":
            self.close()
            message = result.get("error") or "policy_client_error"
            raise PolicyBridgeError(message, timeout="Timeout" in message or "timeout" in message)
        if result.get("status") != expected:
            self.close()
            raise PolicyBridgeError("policy_client_status_mismatch")
        return result

    def _exchange(self, request, expected, timeout):
        with self.lock:
            if not self.alive:
                raise PolicyBridgeError("policy_client_not_running")
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except OSError as error:
                self.close()
                raise PolicyBridgeError("policy_client_pipe_closed") from error
            return self._receive(timeout, expected=expected)

    def reset(self, *, task_id, context_key):
        """Start a new policy episode (fresh keyframe memory); reconnects a poisoned client first."""
        if not self.alive:
            self.start()
        return self._exchange(dict(op="reset", task_id=task_id, context_key=context_key), "reset", self.timeout_s + 10)

    def infer(self, capture_paths, instruction):
        result = self._exchange(dict(op="infer", capture_paths=[str(path) for path in capture_paths],
                                     instruction=instruction), "inference_completed", self.timeout_s + 10)
        return result

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write(json.dumps(dict(op="close")) + "\n")
                    process.stdin.flush()
                except OSError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        if self.log is not None:
            self.log.close()
            self.log = None
