"""Lifecycle of native policy-input captures written by the robot worker.

A capture is an uncompressed npz in the worker's private run directory holding
head-camera RGB of consecutive executed steps and the final three-camera RGB and
end-effector state. It is an input of an external policy only: it never enters
the Store, a receipt, or an actor/program projection, and it is deleted once the
policy bridge has consumed it.
"""
from pathlib import Path


def capture_path(capture):
    path = capture.get("path") if isinstance(capture, dict) else None
    if not isinstance(path, str) or not path.endswith(".npz"):
        raise ValueError("invalid_policy_capture")
    return Path(path)


def discard_policy_capture(capture):
    """Best-effort removal; a leftover file is private worker scratch, never evidence."""
    try:
        capture_path(capture).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


def capture_summary(capture):
    """Observer-safe description without the private file path."""
    return {key: capture.get(key) for key in ("frames", "final_sim_step") if isinstance(capture, dict)}
