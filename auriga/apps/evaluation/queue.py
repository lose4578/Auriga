"""Run benchmark cases on available GPUs, retrying confirmed native OOM failures."""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import threading
import time
import uuid

from auriga.apps.library.scoring import load_effective_summary
from auriga.cortex.agents.factory import agent_backend, backend_requirements, evidence_client
from auriga.cortex.session.pidfd import pidfd_open, pidfd_send_signal
from auriga.core.config import EXTERNAL_PERCEPTION_FIELDS, MOTION_THRESHOLD_FIELDS, AgentConfig, Config, QueueConfig, cli_overrides
from auriga.core.presentation.actor_payload import ACTOR_PAYLOAD_PROFILES, DEFAULT_ACTOR_PAYLOAD_PROFILE
from auriga.core.observation_policy import ACTOR_PROFILES, has_calibration
from auriga.apps.library.experiments import atomic_json
from auriga.apps.evaluation.provenance import CHECKOUT_LAYOUTS, bind_sources, checkout_environment, checkout_layout, verify_source
from auriga.core import legacy_names as legacy

from .retry import (retry_policy, retry_delay, retry_run_dir, logical_id,
                    oom_evidence, cleanup_confirmed, logical_tasks, recover_oom, recovered_command)


def eligible_gpu_sample(sample, indices):
    """Restrict admission candidates, retaining the unfiltered observation."""
    if indices is None:
        return sample
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("eligible_gpu_indices must be a nonempty list of unique nonnegative integers")
    return {**sample, "gpus": [gpu for gpu in sample.get("gpus", []) if gpu["index"] in indices],
            "all_observed_gpu_indices": [gpu["index"] for gpu in sample.get("gpus", [])],
            "eligible_gpu_indices": list(indices)}


def sample_gpus():
    result = {"gpus": [], "processes": None, "error": None, "sample_started_monotonic": time.monotonic()}
    # Credit observed process memory before sampling free memory: startup growth
    # between queries must not appear both free and already reserved.
    try:
        probe = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        processes = []
        for line in probe.stdout.splitlines():
            gpu_uuid, pid, used = (part.strip() for part in line.split(","))
            pid, used = int(pid), int(used)
            if not gpu_uuid.startswith("GPU-") or pid <= 0 or used < 0: raise ValueError("Invalid GPU process row")
            processes.append(dict(uuid=gpu_uuid, pid=pid, used_mib=used))
        result["processes"] = processes
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["process_error"] = str(error)  # Unknown attribution keeps the entire lease reserved.
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.free",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        result["raw"] = probe.stdout
        for line in probe.stdout.splitlines():
            try:
                index, uuid, utilization, free = (part.strip() for part in line.split(","))
                index, utilization, free = int(index), int(utilization), int(free)
                if index < 0 or not uuid.startswith("GPU-") or not 0 <= utilization <= 100 or free < 0: raise ValueError()
                result["gpus"].append(dict(index=index, uuid=uuid, utilization=utilization, free_mib=free))
            except ValueError:
                result["error"] = "Some GPU rows are unavailable or invalid; excluded from selection."
    except (OSError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    result["observed_at"] = datetime.now(timezone.utc).isoformat()
    return result


CASE_MEMORY_MIB = QueueConfig().memory_per_case_mib
GPU_HEADROOM_MIB = QueueConfig().gpu_headroom_mib
SAMPLE_INTERVAL_S = 90
ADMISSION_RETRY_INTERVAL_S = 5
INITIAL_LAUNCH_LIMIT = 30
WINDOW_LAUNCH_LIMIT = 30
COMPLETION_CHECK_INTERVAL_S = .5
SHUTDOWN_GRACE_S = 300
STARTUP_SHUTDOWN_GRACE_S = 950
FORCED_TERM_GRACE_S = 10
FORCED_KILL_GRACE_S = 5
GPU_LOCK_DIR = Path.home() / ".cache/auriga/gpu-locks"
EXPERIMENT_SETTINGS = Path.home() / ".config/auriga/experiments.json"
WAKE_THREAD_FILE = Path.home() / ".config/auriga/completion-thread"


def experiment_environment(retry_settings=None):
    """Adapt legacy settings or frozen path metadata; never copy credentials."""
    environment = os.environ.copy()
    if retry_settings is not None and retry_settings.get("source") == "unified_config":
        for key, variable in (("codex_home", "CODEX_HOME"), ("price_card", "AURIGA_PRICE_CARD")):
            value = retry_settings.get(key)
            if value is not None:
                if not isinstance(value, str) or not Path(value).is_absolute():
                    raise ValueError("Invalid frozen experiment setting: " + key)
                environment[variable] = value
        return environment, dict(retry_settings)
    if retry_settings is not None and retry_settings.get("configured") is False:
        return environment, {"configured": False, "recovery_binding": "inherited_environment_unrecorded"}
    if retry_settings is not None:
        path = retry_settings.get("settings_file")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("OOM retry requires the original absolute experiment settings_file")
        environment["AURIGA_EXPERIMENT_CONFIG"] = path
    settings = Path(environment.get("AURIGA_EXPERIMENT_CONFIG", str(EXPERIMENT_SETTINGS))).expanduser()
    if not settings.exists() and "AURIGA_EXPERIMENT_CONFIG" not in environment:
        return environment, {"configured": False}
    value = json.loads(settings.read_text())
    if not isinstance(value, dict) or set(value) - {"codex_home", "price_card"}:
        raise ValueError("Experiment settings only support codex_home and price_card paths")
    selected = {"configured": True, "settings_file": str(settings.resolve())}
    for key, variable in (("codex_home", "CODEX_HOME"), ("price_card", "AURIGA_PRICE_CARD")):
        if key not in value: continue
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError("Experiment setting must be an absolute path: " + key)
        path = Path(value[key]).resolve()
        required = [path / "config.toml", path / "auth.json"] if key == "codex_home" else [path]
        if not all(item.is_file() for item in required):
            raise ValueError("Experiment setting files are missing: " + key)
        environment[variable] = str(path)
        selected[key] = str(path)
    if retry_settings is not None:
        if any(selected.get(key) != retry_settings.get(key) for key in ("configured", "settings_file", "codex_home", "price_card")):
            raise ValueError("OOM retry experiment settings differ from the original attempt")
        selected["recovery_binding"] = "original_settings_paths; credential_contents_not_archived"
    return environment, selected


def ranked_gpus(previous, current, minimum_free_mib=CASE_MEMORY_MIB + GPU_HEADROOM_MIB):
    """Memory is mandatory in both samples; prefer idle, then lower utilization."""
    prior = {(gpu["index"], gpu["uuid"]): gpu for gpu in previous["gpus"] if gpu["free_mib"] >= minimum_free_mib}
    eligible = [gpu for gpu in current["gpus"] if gpu["free_mib"] >= minimum_free_mib
                and (gpu["index"], gpu["uuid"]) in prior]
    return sorted(eligible, key=lambda gpu: (
        not (gpu["utilization"] == prior[gpu["index"], gpu["uuid"]]["utilization"] == 0),
        gpu["utilization"], -gpu["free_mib"], gpu["index"]))


def select_gpu(previous, current):
    return next(iter(ranked_gpus(previous, current)), None)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"ppid": int(fields[1]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def process_snapshot(pid):
    """Missing is exited; unreadable is unknown and must not release a lease."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {"pid": pid, "ppid": int(fields[1]), "start_ticks": int(fields[19]), "state": fields[0]}


def identity_exited(identity):
    if type(identity.get("pid")) is not int or identity["pid"] <= 0:
        return False
    try:
        current = process_snapshot(identity["pid"])
    except (OSError, ValueError, IndexError):
        return False
    return (current is None or current["state"] in {"Z", "X"}
            or (type(identity.get("start_ticks")) is int and current["start_ticks"] != identity["start_ticks"]))


def quarantine_exited(lease):
    identities = lease.get("remaining_processes")
    return isinstance(identities, list) and bool(identities) and all(identity_exited(item) for item in identities)


class GatewayCleanup:
    """Keep pidfds for this gateway's descendants, including separate sessions."""
    def __init__(self, process):
        self.process = process
        self.root_identity = process_identity(process.pid)
        if self.root_identity and self.root_identity["ppid"] != os.getpid():
            self.root_identity = None
        self.targets, self.errors = {}, []
        self.capture()

    def capture(self):
        if self.root_identity is None:
            return
        try:
            root = process_snapshot(self.process.pid)
            if root is None or root["start_ticks"] != self.root_identity["start_ticks"]:
                return
            snapshots = {root["pid"]: root}
            for path in Path("/proc").glob("[0-9]*/stat"):
                try:
                    item = process_snapshot(int(path.parent.name))
                    if item is not None: snapshots[item["pid"]] = item
                except (OSError, ValueError, IndexError):
                    continue  # Never infer ownership from an unreadable process.
            owned = {root["pid"]}
            while True:
                added = {pid for pid, item in snapshots.items() if item["ppid"] in owned
                         and item["start_ticks"] >= root["start_ticks"]} - owned
                if not added: break
                owned.update(added)
            for pid in owned:
                if pid in self.targets: continue
                item = snapshots[pid]
                descriptor = None
                try:
                    descriptor = pidfd_open(pid)
                    # Check after opening: neither a reused PID nor a replacement
                    # gateway may grant ownership of an unrelated process.
                    current, current_root = process_snapshot(pid), process_snapshot(root["pid"])
                    if (current is None or current["start_ticks"] != item["start_ticks"]
                            or current_root is None or current_root["start_ticks"] != root["start_ticks"]):
                        os.close(descriptor)
                        continue
                    self.targets[pid] = {"pid": pid, "start_ticks": item["start_ticks"], "fd": descriptor}
                except ProcessLookupError:
                    if descriptor is not None: os.close(descriptor)
                except (OSError, ValueError, IndexError) as error:
                    if descriptor is not None: os.close(descriptor)
                    self.errors.append(f"capture {pid}: {type(error).__name__}: {error}")
                    self.targets[pid] = {"pid": pid, "start_ticks": item["start_ticks"], "fd": None}
        except (OSError, ValueError, IndexError) as error:
            self.errors.append(f"capture gateway: {type(error).__name__}: {error}")

    def remaining(self):
        result = []
        for item in self.targets.values():
            try:
                exited = (bool(select.select([item["fd"]], [], [], 0)[0]) if item["fd"] is not None
                          else identity_exited(item))
            except (OSError, ValueError):
                exited = False
            if not exited: result.append(item)
        if self.process.poll() is None and self.process.pid not in {item["pid"] for item in result}:
            result.append({"pid": self.process.pid, "start_ticks": (self.root_identity or {}).get("start_ticks"), "fd": None})
        return result

    def wait(self, deadline):
        try:
            self.process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return False
        while remaining := self.remaining():
            timeout = deadline - time.monotonic()
            if timeout <= 0: return False
            descriptors = [item["fd"] for item in remaining if item["fd"] is not None]
            # No model polling. Unknown /proc-only identities get a bounded recheck.
            select.select(descriptors, [], [], min(timeout, 1))
        return True

    def signal(self, sig):
        for item in self.remaining():
            try:
                if item["fd"] is not None:
                    pidfd_send_signal(item["fd"], sig)
                elif item["pid"] == self.process.pid and self.process.poll() is None:
                    # Popen owns the unreaped child and checks waitpid before kill;
                    # this fallback never applies to a descendant or a PID lookup.
                    self.process.send_signal(sig)
                else:
                    self.errors.append(f"signal {item['pid']}: no verified pidfd; stop unconfirmed")
            except ProcessLookupError:
                pass
            except OSError as error:
                self.errors.append(f"signal {item['pid']}: {type(error).__name__}: {error}")

    def close(self):
        for item in self.targets.values():
            if item["fd"] is not None: os.close(item["fd"])


def attributed_memory(gpu_uuid, processes, leases):
    """Only credit visible memory whose process ancestry matches a live gateway."""
    roots = {}
    for lease in leases:
        identity = process_identity(lease.get("pid"))
        if identity and identity["start_ticks"] == lease.get("start_ticks"):
            roots[lease["pid"]] = lease["id"]
    measured = {}
    for process in processes or []:
        if process["uuid"] != gpu_uuid: continue
        pid, seen = process["pid"], set()
        while pid > 1 and pid not in seen:
            if pid in roots:
                key = roots[pid]
                measured[key] = measured.get(key, 0) + process["used_mib"]
                break
            seen.add(pid)
            identity = process_identity(pid)
            if not identity: break
            pid = identity["ppid"]
    return measured


class GPUReservation:
    """Shared legacy lock and per-case lease survive a queue crash in its gateway."""
    def __init__(self, shared, lease, metadata, admission_lock, admission):
        self.shared, self.lease = shared, lease
        self.metadata, self.admission_lock, self.admission = metadata, admission_lock, admission

    def filenos(self):
        return self.shared.fileno(), self.lease.fileno()

    def bind(self, pid):
        with self.admission_lock.open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            self.metadata.update(pid=pid, start_ticks=(process_identity(pid) or {}).get("start_ticks"))
            self.lease.seek(0)
            self.lease.write(json.dumps(self.metadata))
            self.lease.truncate()
            self.lease.flush()

    def close(self):
        # Do not LOCK_UN or unlink: a live gateway may hold the inherited fds.
        self.lease.close()
        self.shared.close()

    def quarantine(self, remaining):
        """Persist admission protection even after the queue closes its fds."""
        with self.admission_lock.open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            self.metadata.update(cleanup_stop_unconfirmed=True, remaining_processes=remaining)
            self.lease.seek(0)
            self.lease.write(json.dumps(self.metadata))
            self.lease.truncate()
            self.lease.flush()
            os.fsync(self.lease.fileno())


def reserve_gpu(gpu, memory_mib=CASE_MEMORY_MIB, headroom_mib=GPU_HEADROOM_MIB, baseline=None):
    """Serialize cross-queue admission, reserving startup memory before Popen."""
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu["uuid"]): raise ValueError("Invalid GPU UUID")
    GPU_LOCK_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    shared = (GPU_LOCK_DIR / (gpu["uuid"] + ".lock")).open("a")
    try: fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        shared.close()
        return None, {"reason": "legacy_exclusive_reservation"}
    guard_path = GPU_LOCK_DIR / (gpu["uuid"] + ".admission.lock")
    reservation = None
    try:
        with guard_path.open("a") as guard:
            try: fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: return None, {"reason": "concurrent_admission"}
            leases = []
            for path in GPU_LOCK_DIR.glob(gpu["uuid"] + ".lease-*.json"):
                with path.open("r+") as stream:
                    try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError: leases.append(json.load(stream))
                    else:
                        lease = json.load(stream)
                        if lease.get("cleanup_stop_unconfirmed") and not quarantine_exited(lease):
                            leases.append(lease)
                        else:
                            path.unlink()  # No owner, and no unconfirmed stopped process.
            latest = sample_gpus()
            current = next((item for item in latest["gpus"] if (item["index"], item["uuid"]) == (gpu["index"], gpu["uuid"])), None)
            if current is None: return None, {"reason": "gpu_unavailable", "sample": latest}
            baseline = latest if baseline is None else baseline
            initial = next((item for item in baseline["gpus"] if (item["index"], item["uuid"]) == (gpu["index"], gpu["uuid"])), None)
            if initial is None: return None, {"reason": "gpu_missing_from_periodic_sample", "sample": latest}

            def shadow_in(sample):
                # A lease created after this sample has no resident memory in it.
                # In particular, never credit its new allocation against old free memory.
                existing = [item for item in leases if item["admitted_at"] <= sample.get("sample_started_monotonic", -math.inf)]
                measured = attributed_memory(gpu["uuid"], sample.get("processes"), existing)
                return sum(max(0, item["memory_mib"] - measured.get(item["id"], 0)) for item in leases)

            shadow, initial_shadow = shadow_in(latest), shadow_in(baseline)
            latest_free = current["free_mib"] - shadow
            initial_free = initial["free_mib"] - initial_shadow
            effective_free = min(initial_free, latest_free)
            now = time.monotonic()
            recent = sum(now - item["admitted_at"] < SAMPLE_INTERVAL_S for item in leases)
            utilization = max(initial["utilization"], gpu["utilization"], current["utilization"])
            default_limit = INITIAL_LAUNCH_LIMIT
            launch_limit = int(os.environ.get("AURIGA_QUEUE_LAUNCH_LIMIT") or default_limit)
            admission = {"sample": latest, "gpu": current, "active_leases": len(leases),
                         "shadow_reserved_mib": shadow, "latest_effective_free_mib": latest_free,
                         "baseline_gpu": initial, "baseline_observed_at": baseline.get("observed_at"),
                         "baseline_shadow_reserved_mib": initial_shadow, "baseline_effective_free_mib": initial_free,
                         "effective_free_mib": effective_free,
                         "case_memory_mib": memory_mib, "gpu_headroom_mib": headroom_mib,
                         "recent_launches": recent, "launches_per_interval": launch_limit,
                         "utilization_for_admission": utilization}
            if effective_free < memory_mib + headroom_mib:
                return None, {**admission, "reason": "insufficient_unreserved_memory"}
            if recent >= launch_limit:
                return None, {**admission, "reason": "startup_rate_limit"}
            metadata = {"id": uuid.uuid4().hex, "memory_mib": memory_mib, "admitted_at": now}
            lease = (GPU_LOCK_DIR / (gpu["uuid"] + ".lease-" + metadata["id"] + ".json")).open("x+")
            try:
                fcntl.flock(lease, fcntl.LOCK_EX)
                lease.write(json.dumps(metadata))
                lease.flush()
                reservation = GPUReservation(shared, lease, metadata, guard_path, admission)
            except BaseException:
                lease.close()
                raise
            return reservation, admission
    finally:
        if reservation is None: shared.close()


def agent_settings(config, backend):
    """Frozen agent selection for the queue record: backend and the upstream env-var NAME only, never a value."""
    result = {"agent_backend": backend}
    section = getattr(config.agent, backend, None)
    if section is not None:
        result["agent_upstream_api_key_env"] = section.upstream.api_key_env
    return result


def evidence_path_warnings(launch_configs):
    """Queue warnings about the evidence helper path of one manifest (design 04 §9.6 fairness rule).

    ``launch_configs``: the effective launch ``Config`` of every case, in case order. Cases that enable an evidence
    feature (``features.visual_review_model`` or ``features.self_briefing``) must all use the same effective evidence
    client (``"codex_agent"`` or ``"relay"``); otherwise one ``comparison_evidence_path_mismatch`` warning lists the
    case indices per client and the backends involved. Returns a list (empty when consistent). Never refuses: a
    deliberate Codex native-versus-relay comparison is its own single-variable experiment.
    """
    clients, backends = {}, set()
    for index, config in enumerate(launch_configs):
        if config.features.visual_review_model or config.features.self_briefing:
            clients.setdefault(evidence_client(config), []).append(index)
            backends.add(agent_backend(config))
    if len(clients) < 2:
        return []
    return [{"kind": "comparison_evidence_path_mismatch", "evidence_clients": dict(sorted(clients.items())),
             "agent_backends": sorted(backends),
             "message": "cases enabling visual review or self-briefing use different evidence clients; set "
                        "agent.evidence.client=\"relay\" on every arm of a harness comparison or declare the "
                        "difference as a confound"}]


def case_launch_config(case, gpu, config=None):
    """Effective unified config of one benchmark ``case`` launched on ``gpu`` over ``config`` (default ``Config()``),
    as the gateway will resolve it; the case must already have passed ``command``'s field validation."""
    tags = case["tag"] if isinstance(case["tag"], list) else [case["tag"]]
    launch_case = {**case, "tag": tags, "benchmark": True, "gpu": gpu, "category": "benchmark"}
    launch_case.pop("experiment_condition", None)
    return (config if config is not None else Config()).with_overrides(cli_overrides(launch_case))


def command(case, gpu, *, config=None, config_path=None):
    required = {"cwd", "run_dir", "task", "seed", "layout_id", "title", "tag"}
    if not isinstance(case, dict) or required - case.keys() or case.keys() - required - {"experiment_condition", "agent_backend", "model", "effort", "codex_home", "task_notes", "task_notes_source", "pi05", "visual_review_model", "actor_payload_profile", "image_feedback_policy", "enable_wait", "tracking_guard", "joint_replan", "arm_queue", "motion_cost_preview", "airborne_handoff", "handoff_preference", "action_trajectory_delivery", "bounded_continuation", "motion_safety_guard", "trajectory_preview", "manipulation_protocol", "task_scheduling", "operation_candidates", "preview_intent", "recovery_evidence", "motion_execution_mode", "default_motion_speed_scale", "completion_guidance", "observation_profile", "motion_review", "grasp_memory_dir", "disabled_tools", "record_video", "export_video", "terminal", "vision_compare", "robotic_view", "stereo_guide", "default_task_notes", "depth_skills", "depth_auto_feedback", "depth_event_payload", "external_perception", "vggt_height", "vggt_height_mode", "vggt_endpoint", "vggt_support_z_m", "vggt_timeout_s", "object_table", "object_table_max_objects", "object_table_max_height_m", "object_table_head_exclude_px", "scene3d", "scene3d_voxel_m", "scene3d_z_above_m", "scene3d_z_below_m", "object_table_observe", "scene3d_observe", "obs3d_timeout_s", "obs3d_max_objects", "obs3d_action_overlay", "pose_skill", "held_perception"}:
        raise ValueError("Each case requires cwd, run_dir, task, seed, layout_id, title, tag; official native limits are automatic; unsupported run options are rejected")
    for name in ("cwd", "run_dir", "task", "title"):
        if not isinstance(case[name], str) or not case[name].strip(): raise ValueError("Invalid " + name)
    if "experiment_condition" in case and (not isinstance(case["experiment_condition"], str)
            or not case["experiment_condition"].strip()):
        raise ValueError("Invalid experiment_condition: must be a nonempty string")
    if "codex_home" in case and (not isinstance(case["codex_home"], str)
            or not Path(case["codex_home"]).is_absolute()):
        raise ValueError("Invalid codex_home: must be an absolute path")
    for name in ("model", "effort", "agent_backend"):
        if name in case and (type(case[name]) is not str or not case[name].strip()):
            raise ValueError("Invalid " + name + ": must be a nonempty string")
    for name in ("cwd", "run_dir"):
        if not Path(case[name]).is_absolute(): raise ValueError(name + " must be absolute")
    layout = checkout_layout(case["cwd"])
    if layout is None: raise ValueError("cwd lacks a gateway entry point")
    run = Path(case["run_dir"])
    if run.exists() or run.is_symlink(): raise ValueError("run_dir already exists: " + str(run))
    for name, minimum in (("seed", 0), ("layout_id", 0)):
        if type(case[name]) is not int or case[name] < minimum: raise ValueError("Invalid " + name)
    for name in ("pi05", "enable_wait", "tracking_guard", "joint_replan", "arm_queue", "motion_cost_preview", "airborne_handoff", "bounded_continuation", "motion_safety_guard", "trajectory_preview", "manipulation_protocol", "task_scheduling", "operation_candidates", "preview_intent", "recovery_evidence", "motion_review", "record_video", "export_video", "terminal", "vision_compare", "robotic_view", "stereo_guide", "default_task_notes", "depth_skills", "depth_auto_feedback", "depth_event_payload", "external_perception", "vggt_height", "object_table", "scene3d", "object_table_observe", "scene3d_observe", "obs3d_action_overlay", "pose_skill", "held_perception"):
        if name in case and type(case[name]) is not bool: raise ValueError("Invalid " + name)
    if "handoff_preference" in case and case["handoff_preference"] not in ("surface", "airborne"):
        raise ValueError("Invalid handoff_preference")
    if "motion_execution_mode" in case and (type(case["motion_execution_mode"]) is not str
            or case["motion_execution_mode"] not in ("resampled", "feedback")):
        raise ValueError("Invalid motion_execution_mode")
    if "grasp_memory_dir" in case and (not isinstance(case["grasp_memory_dir"], str)
            or not Path(case["grasp_memory_dir"]).is_absolute()):
        raise ValueError("Invalid grasp_memory_dir: must be an absolute path")
    if "disabled_tools" in case and (not isinstance(case["disabled_tools"], list)
            or any(not isinstance(name, str) or not name.strip() for name in case["disabled_tools"])):
        raise ValueError("Invalid disabled_tools: expected a list of tool names")
    # Standalone commands need the same cross-feature checks as configured queues,
    # without reading an unrelated ambient configuration or starting a gateway.
    base_config = config if config is not None else Config()
    defaults = base_config.features
    profile = case.get("observation_profile", defaults.observation_profile)
    if profile not in ACTOR_PROFILES:
        raise ValueError("Invalid observation_profile")
    if case.get("actor_payload_profile", DEFAULT_ACTOR_PAYLOAD_PROFILE) not in ACTOR_PAYLOAD_PROFILES:
        raise ValueError("Invalid actor_payload_profile")
    # Omitted guidance inherits the unified config (default: state-aware hook).
    if "completion_guidance" in case and case["completion_guidance"] not in ("none", "initial", "hook", "both"):
        raise ValueError("Invalid completion_guidance")
    if case.get("visual_review_model") not in (None, "gpt-6.1-sol"): raise ValueError("Invalid visual_review_model")
    tags = case["tag"] if isinstance(case["tag"], list) else [case["tag"]]
    if any(not isinstance(tag, str) or not tag.strip() for tag in tags): raise ValueError("Invalid tag")
    if "task_notes" in case or "task_notes_source" in case:
        if any(not isinstance(case.get(key), str) or not case[key].strip() for key in ("task_notes", "task_notes_source")) or len(case["task_notes"]) > 8000:
            raise ValueError("Public task_notes and task_notes_source must be provided together")
    case_launch_config(case, gpu, base_config).validate_launch()
    args = [sys.executable, "-u", "-m", layout["module"], "--benchmark", "--observation-profile", profile,
            "--gpu", str(gpu), "--category", "benchmark"]
    for key in ("run_dir", "task", "seed", "layout_id", "title", "agent_backend", "model", "effort", "codex_home", "task_notes", "task_notes_source", "actor_payload_profile", "completion_guidance", "motion_execution_mode", "default_motion_speed_scale", "image_feedback_policy", "handoff_preference", "action_trajectory_delivery", "vggt_height_mode", "vggt_endpoint", "vggt_support_z_m", "vggt_timeout_s", "object_table_max_objects", "object_table_max_height_m", "object_table_head_exclude_px", "scene3d_voxel_m", "scene3d_z_above_m", "scene3d_z_below_m", "obs3d_timeout_s", "obs3d_max_objects"):
        if key in case: args.extend(["--" + key.replace("_", "-"), str(case[key])])
    if "visual_review_model" in case:
        args.extend(["--visual-review-model", case["visual_review_model"]] if case["visual_review_model"]
                    else ["--no-visual-review-model"])
    if "disabled_tools" in case:
        args.extend(["--disabled-tools", *case["disabled_tools"]])
    for key in ("pi05", "tracking_guard", "joint_replan", "arm_queue", "motion_cost_preview", "airborne_handoff", "bounded_continuation", "motion_safety_guard", "trajectory_preview", "manipulation_protocol", "task_scheduling", "operation_candidates", "preview_intent", "recovery_evidence", "record_video", "export_video", "terminal", "vision_compare", "robotic_view", "stereo_guide", "default_task_notes", "depth_skills", "depth_auto_feedback", "depth_event_payload", "external_perception", "vggt_height", "object_table", "scene3d", "object_table_observe", "scene3d_observe", "obs3d_action_overlay", "pose_skill", "held_perception"):
        if key in case:
            args.append("--" + ("" if case[key] else "no-") + key.replace("_", "-"))
    args.append("--enable-wait" if case.get("enable_wait", defaults.enable_wait) else "--no-enable-wait")
    args.append("--motion-review" if case.get("motion_review", defaults.motion_review) and has_calibration(profile) else "--no-motion-review")
    if "grasp_memory_dir" in case:
        args.extend(["--grasp-memory-dir", case["grasp_memory_dir"]])
    if config_path is not None:
        args.extend(["--config", str(config_path)])
    if not tags: args.append("--clear-tags")
    for tag in tags: args.extend(["--tag", tag])
    return args


def current_manifest(manifest):
    """Manifests written before the rename keep their configuration under the legacy key."""
    if legacy.MANIFEST_CONFIG_KEY in manifest and "auriga_config" not in manifest:
        manifest = {("auriga_config" if key == legacy.MANIFEST_CONFIG_KEY else key): value
                    for key, value in manifest.items()}
    return manifest


def queue_config(manifest, path=None):
    """Resolve once; persisted documents never re-read ambient environment."""
    manifest = current_manifest(manifest)
    if path is None and "auriga_config" in manifest:
        return Config.from_dict(manifest["auriga_config"])
    environment, _ = experiment_environment()
    return Config.load(path, environ=environment)


def configured_manifest(manifest, config):
    """Case/manifest settings override shared defaults without mutating inputs."""
    queue = config.to_dict()["queue"]
    merged = {key: queue[key] for key in ("memory_per_case_mib", "gpu_headroom_mib", "oom_retry")}
    merged["max_parallel"] = queue["max_parallel"] or QueueConfig().max_parallel
    merged.update(manifest)
    retry_policy(manifest)  # Keep invalid manifest values out of the merge.
    merged["oom_retry"] = {**queue["oom_retry"], **manifest.get("oom_retry", {})}
    effective = {key: merged[key] for key in ("max_parallel", "memory_per_case_mib", "gpu_headroom_mib", "oom_retry") if key in merged}
    merged["auriga_config"] = config.with_overrides({"queue": effective}).to_dict()
    return merged


def require_config_gateway(cwd):
    """Check frozen checkouts before reserving a GPU; never execute their code."""
    try:
        tree = ast.parse((Path(cwd) / (checkout_layout(cwd) or CHECKOUT_LAYOUTS[0])["gateway"]).read_text())
        accepts_config = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument" and any(isinstance(arg, ast.Constant)
                and arg.value == "--config" for arg in node.args) for node in ast.walk(tree))
        loads_config = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "Config" for node in ast.walk(tree))
    except (OSError, SyntaxError) as error:
        raise ValueError("unified_config_requires_supported_gateway: " + str(cwd)) from error
    if not accepts_config or not loads_config:
        raise ValueError("unified_config_requires_supported_gateway: " + str(cwd)
                         + "; launch new queues from a checkout that supports --config")


def runtime_config_schema(cwd):
    """Read a checkout's config field names without importing or executing it."""
    path = Path(cwd) / (checkout_layout(cwd) or CHECKOUT_LAYOUTS[0])["config"]
    if not path.is_file():
        # Small synthetic gateway fixtures do not carry the source tree.  The
        # gateway contract check above still protects those callers.
        return None
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ValueError("unified_config_schema_unreadable: " + str(cwd)) from error
    schemas = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("Config"):
            continue
        names = {item.target.id for item in node.body
                 if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
        if names:
            schemas[node.name] = names
    if "Config" not in schemas:
        raise ValueError("unified_config_schema_unreadable: " + str(cwd))
    return schemas


def project_config_for_runtimes(config_document, schemas, omissions=None):
    """Project the shared snapshot onto a frozen schema intersection.

    Only the observed sandbox field is allowed to disappear.  New schema drift
    must fail closed until its default and semantics have been reviewed.
    """
    if not schemas:
        return config_document
    defaults = Config.from_dict({}).to_dict()

    def class_for_field(name):
        return "".join(part.capitalize() for part in name.split("_")) + "Config"

    def default_at(path):
        value = defaults
        for part in path:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def project(value, class_name, path):
        declared = [schema.get(class_name) for schema in schemas]
        if any(fields is None for fields in declared):
            raise ValueError("unified_config_schema_missing:" + class_name)
        allowed = set.intersection(*(set(fields) for fields in declared))
        result = {}
        for key, part in value.items():
            if key not in allowed:
                field_path = ".".join(path + [key])
                omittable = {
                    "features.sandbox_enabled",
                    "features.robotic_view",
                    "features.stereo_guide",
                    "features.lean_compact",
                    "analysis",
                    *(f"features.{name}" for name in MOTION_THRESHOLD_FIELDS),
                    *(f"features.{name}" for name in EXTERNAL_PERCEPTION_FIELDS),
                }
                if field_path in omittable and part == default_at(path + [key]):
                    if omissions is not None:
                        omissions.append({"key": field_path, "value": part,
                                          "reason": "target_schema_omits_default"})
                    continue
                raise ValueError("unified_config_incompatible:" + field_path)
            child = class_for_field(key)
            if isinstance(part, dict) and all(child in schema for schema in schemas):
                part = project(part, child, path + [key])
            result[key] = part
        return result

    return project(config_document, "Config", [])


def set_max_parallel(output, value):
    """Request a live scheduling change; the queue records its application."""
    if type(value) is not int or value < 1:
        raise ValueError("max_parallel must be a positive integer")
    output = output.resolve()
    state = json.loads((output / "state.json").read_text())
    if not isinstance(state, dict) or not isinstance(state.get("scheduling"), dict):
        raise ValueError("Queue is not running with live max_parallel control support")
    scheduling = state.get("scheduling", {})
    control = output / "control.json"
    identity = process_identity(scheduling.get("queue_pid"))
    if (state.get("status") != "running" or scheduling.get("control_path") != str(control)
            or not control.is_file() or identity is None
            or identity["start_ticks"] != scheduling.get("queue_start_ticks")):
        raise ValueError("Queue is not running with live max_parallel control support")
    atomic_json(control, {"max_parallel": value})


def run_queue(manifest, output, wait_timeout_s=None, *, recovery=None):
    manifest = current_manifest(manifest)
    config = queue_config(manifest) if not recovery else None
    if config is not None:
        manifest = configured_manifest(manifest, config)
        config = Config.from_dict(manifest["auriga_config"])
    if wait_timeout_s is None:
        wait_timeout_s = config.queue.wait_timeout_s if config is not None else 1800
    if not math.isfinite(wait_timeout_s) or wait_timeout_s <= 0: raise ValueError("wait-timeout-s must be positive")
    if config is not None:
        config = config.with_overrides({"queue": {"wait_timeout_s": wait_timeout_s}})
        manifest = {**manifest, "auriga_config": config.to_dict()}
        frozen_settings = {"configured": True, "source": "unified_config",
                           "codex_home": config.model.codex_home, "price_card": config.model.price_card}
        if config.agent != AgentConfig():
            frozen_settings.update(agent_settings(config, config.agent.backend))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases: raise ValueError("Manifest requires a nonempty cases list")
    cases = list(cases)  # Appended attempts must not rewrite the input manifest.
    policy = retry_policy(manifest)
    # New queues resolve the default above; recovery keeps its historical fallback.
    max_parallel = manifest.get("max_parallel", len(cases))
    eligible_indices = manifest.get("eligible_gpu_indices")
    eligible_gpu_sample({"gpus": []}, eligible_indices)
    if type(max_parallel) is not int or max_parallel < 1: raise ValueError("max_parallel must be a positive integer")
    memory_mib = manifest.get("memory_per_case_mib", CASE_MEMORY_MIB)
    headroom_mib = manifest.get("gpu_headroom_mib", GPU_HEADROOM_MIB)
    for name, value, minimum in (("memory_per_case_mib", memory_mib, 1), ("gpu_headroom_mib", headroom_mib, 0)):
        if type(value) is not int or value < minimum: raise ValueError(f"{name} must be an integer >= {minimum}")
    validation_config = (Config.from_dict(manifest["auriga_config"])
                         if recovery and "auriga_config" in manifest else config)
    for case in cases: command(case, 0, config=validation_config)
    effective_document = config.to_dict() if config is not None else None
    compatibility = []
    if config is not None:
        for cwd in {case["cwd"] for case in cases}:
            require_config_gateway(cwd)
            schema = runtime_config_schema(cwd)
            if schema is not None:
                compatibility.append((cwd, schema))
                if "AgentConfig" not in schema and any("agent_backend" in case for case in cases if case["cwd"] == cwd):
                    # A checkout without the agent section cannot honour --agent-backend; never run it as Codex.
                    raise ValueError("unified_config_incompatible:agent")
        schemas = [schema for _, schema in compatibility]
        omissions = []
        effective_document = project_config_for_runtimes(effective_document, schemas, omissions)
        if omissions:
            manifest = {**manifest, "config_compatibility": {
                "status": "projected", "omitted_defaults": omissions,
                "target_checkouts": [cwd for cwd, _ in compatibility]}}
    # The selected agent backend must be able to run every case before any GPU is sampled or a gateway launched;
    # a checkout schema without the agent section was refused above (unified_config_incompatible:agent).
    launch_configs = [case_launch_config(case, 0, validation_config) for case in cases]
    for launch_config in launch_configs:
        backend_requirements(launch_config)
    warnings = evidence_path_warnings(launch_configs)
    source_bindings = bind_sources(cases, manifest.get("source_bindings"))
    manifest = {**manifest, "source_bindings": source_bindings,
                "source_binding_scope": (manifest.get("source_binding_scope", "legacy_recovery_current_sources_only")
                                         if recovery else "queue_start_and_each_launch")}
    entries = (recovery["entries"] if recovery else [{"case": case, "status": "not_started",
        "logical_task_id": logical_id(case["run_dir"]), "attempt_number": 1,
        "root_run_dir": case["run_dir"], "retry_status": "not_needed"} for case in cases])
    paths = [output.resolve()]
    for entry in entries:
        if recovery and "auriga_config" in manifest:
            args = recovered_command(entry, 0)
            if args.count("--config") != 1 or args.index("--config") + 1 >= len(args):
                raise ValueError("OOM recovery requires the original configuration snapshot")
            source = Path(args[args.index("--config") + 1])
            archived = manifest["auriga_config"]
            Config.from_dict(archived)  # Validate without adding newer defaults to archived evidence.
            expected = archived
            compatibility = manifest.get("config_compatibility")
            if compatibility and compatibility.get("status") == "projected":
                targets = compatibility.get("target_checkouts")
                if not isinstance(targets, list) or not targets:
                    raise ValueError("unified_config_compatibility_provenance_invalid")
                schemas = [runtime_config_schema(cwd) for cwd in targets]
                if any(schema is None for schema in schemas):
                    raise ValueError("unified_config_compatibility_provenance_invalid")
                expected = project_config_for_runtimes(archived, schemas)
            if not source.is_absolute() or json.loads(source.read_text()) != expected:
                raise ValueError("OOM recovery configuration snapshot differs from archived manifest")
        if "retry_experiment_settings" in entry:
            experiment_environment(entry["retry_experiment_settings"])
        paths.append(Path(entry["case"]["run_dir"]).resolve())
        for number in range(entry["attempt_number"] + 1, policy["max_retries"] + 2):
            path = Path(retry_run_dir(entry["root_run_dir"], number))
            if path.exists() or path.is_symlink(): raise ValueError("retry run_dir already exists: " + str(path))
            paths.append(path.resolve())
    if any(a == b or a in b.parents or b in a.parents for index, a in enumerate(paths) for b in paths[index + 1:]):
        raise ValueError("Queue output and case run directories must be distinct and non-nested")
    output.mkdir(parents=True, exist_ok=False)
    config_path = output.resolve() / "effective_config.json" if config is not None else None
    if config_path is not None:
        config_path.write_text(json.dumps(effective_document, indent=2) + "\n")
    scheduler_cfg = manifest.get("scheduler") if isinstance(manifest.get("scheduler"), dict) else {}
    raw_sample_interval = scheduler_cfg.get("sample_interval_s", manifest.get("sample_interval_s", SAMPLE_INTERVAL_S))
    if (isinstance(raw_sample_interval, bool) or not isinstance(raw_sample_interval, (int, float))
            or not math.isfinite(raw_sample_interval) or raw_sample_interval < 0):
        raise ValueError("sample_interval_s must be a non-negative finite number")
    sample_interval_s = int(raw_sample_interval) if float(raw_sample_interval).is_integer() else float(raw_sample_interval)
    control_path = output.resolve() / "control.json"
    atomic_json(control_path, {"max_parallel": max_parallel})
    scheduling = {"max_parallel": max_parallel, "sample_interval_s": sample_interval_s,
                  "control_path": str(control_path), "queue_pid": os.getpid(),
                  "queue_start_ticks": (process_identity(os.getpid()) or {}).get("start_ticks"),
                  "admission_retry_interval_s": ADMISSION_RETRY_INTERVAL_S,
                  "completion_check_interval_s": COMPLETION_CHECK_INTERVAL_S,
                  "refill_on_case_exit": True, "admission_baseline": "fresh_for_each_refill_attempt",
                  "oom_retry": policy,
                  "memory_per_case_mib": memory_mib, "gpu_headroom_mib": headroom_mib,
                  "gpu_scope": "explicit_gpu_indices" if eligible_indices is not None else "all_nvidia_smi_gpus",
                  "eligible_gpu_indices": eligible_indices, "same_gpu_concurrency": True,
                  "initial_launch_limit": INITIAL_LAUNCH_LIMIT, "window_launch_limit": WINDOW_LAUNCH_LIMIT,
                  "launches_per_gpu_per_interval": {"default": INITIAL_LAUNCH_LIMIT},
                  "burst_capacity": "minimum of fresh admission baseline and pre-launch free memory, each minus its own unallocated leases",
                  "priority": ["two_consecutive_zero_utilization", "lower_current_utilization", "more_free_memory"],
                  "allow_busy_gpu_with_free_memory": True, "gpu_lock_dir": str(GPU_LOCK_DIR),
                  "reservation_scope": "same-user cooperating queues; external jobs may share the GPU"}
    (output / "config.json").write_text(json.dumps({"manifest": manifest, "wait_timeout_s": wait_timeout_s,
        **scheduling, "python": sys.executable, "recovery": recovery}, indent=2) + "\n")
    stop = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    state = {"status": "running", "scheduling": scheduling, "cases": entries,
             "prior_attempts": recovery["prior_attempts"] if recovery else []}
    if warnings:
        state["warnings"] = warnings
    pending = list(range(len(cases)))
    ready_at = {index: time.monotonic() + max(0, entry.get("retry_not_before", 0) - time.time())
                for index, entry in enumerate(entries)}

    def save_state():
        state["logical_tasks"] = logical_tasks(state["prior_attempts"] + state["cases"])
        temporary = output / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "state.json")

    def record(kind, **value):
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "kind": kind, **value}) + "\n")
        save_state()

    last_control_error = None

    def refresh_parallelism():
        nonlocal max_parallel, next_admission, last_control_error
        raw = None
        try:
            raw = control_path.read_text()
            control = json.loads(raw)
            if (not isinstance(control, dict) or set(control) != {"max_parallel"}
                    or type(control["max_parallel"]) is not int or control["max_parallel"] < 1):
                raise ValueError("control.json requires only a positive integer max_parallel")
        except (OSError, ValueError) as error:
            failure = (raw, str(error))
            if failure != last_control_error:
                last_control_error = failure
                record("queue_control_invalid", control_path=str(control_path), error=str(error),
                       retained_max_parallel=max_parallel)
            return
        last_control_error = None
        value = control["max_parallel"]
        if value != max_parallel:
            previous, max_parallel = max_parallel, value
            scheduling["max_parallel"] = value
            if value > previous: next_admission = 0
            record("max_parallel_changed", old_max_parallel=previous, new_max_parallel=value,
                   active_cases=len(active), control_path=str(control_path))

    def sample(index, phase, trigger):
        value = eligible_gpu_sample(sample_gpus(), eligible_indices)
        record("gpu_observation", case_index=index, phase=phase, trigger=trigger, **value)
        return value

    active = {}

    def next_ready():
        return next((item for item in pending if ready_at[item] <= time.monotonic()), None)

    def retry_after_finish(index):
        entry = state["cases"][index]
        evidence = entry.get("oom_evidence")
        if not evidence:
            entry["retry_status"] = "not_needed"
            return
        entry["oom_evidence"] = evidence
        if not cleanup_confirmed(entry):
            entry["retry_status"] = "blocked_cleanup"
        elif stop.is_set() or state["status"] != "running":
            entry["retry_status"] = "cancelled"
        elif entry["attempt_number"] > policy["max_retries"]:
            entry["retry_status"] = "exhausted"
        else:
            delay = retry_delay(policy, entry["attempt_number"])
            number = entry["attempt_number"] + 1
            retry_case = {**entry["case"], "run_dir": retry_run_dir(entry["root_run_dir"], number)}
            retry_index = len(cases)
            due = time.time() + delay
            entry.update(retry_status="scheduled", retry_delay_s=delay, retry_not_before=due,
                         next_attempt_index=retry_index, next_attempt_run_dir=retry_case["run_dir"])
            cases.append(retry_case)
            state["cases"].append({"case": retry_case, "status": "not_started", "retry_status": "pending",
                "logical_task_id": entry["logical_task_id"], "attempt_number": number,
                "root_run_dir": entry["root_run_dir"], "previous_attempt_run_dir": entry["case"]["run_dir"],
                "retry_delay_s": delay, "retry_not_before": due})
            if "retry_command_template" in entry:
                state["cases"][-1]["retry_command_template"] = entry["retry_command_template"]
            state["cases"][-1]["retry_experiment_settings"] = entry.get("retry_experiment_settings", entry.get("experiment_settings"))
            pending.append(retry_index)
            ready_at[retry_index] = time.monotonic() + delay
        record("oom_retry_decision", case_index=index, logical_task_id=entry["logical_task_id"],
               attempt_number=entry["attempt_number"], retry_status=entry["retry_status"], evidence=evidence,
               delay_s=entry.get("retry_delay_s"), not_before=entry.get("retry_not_before"))

    def finish(index):
        process, lock = active[index]
        if process.returncode is None: raise RuntimeError("Gateway exit is unconfirmed")
        entry = state["cases"][index]
        try:
            entry["summary"] = json.loads((Path(entry["case"]["run_dir"]) / "summary.json").read_text())
            entry["effective_summary"] = load_effective_summary(Path(entry["case"]["run_dir"]), summary=entry["summary"])
        except (OSError, ValueError) as error: entry.update(summary=None, summary_error=str(error))
        summary = entry["summary"] or {}
        entry["oom_evidence"] = oom_evidence(entry)
        robot = (summary.get("cleanup") or {}).get("robot") or {}
        report = entry.setdefault("queue_cleanup", {})
        report.update(gateway_exit_confirmed=True, worker_exit_confirmed=robot.get("process_exit_confirmed"))
        report.setdefault("native_close_confirmed", robot.get("native_close_acknowledged"))
        uncertain = (robot.get("process_exit_confirmed") is False
                     or (entry["oom_evidence"] is not None and robot.get("process_exit_confirmed") is not True)
                     or (summary.get("cleanup_status") in {"failed", "forced_termination"}
                         and robot.get("process_exit_confirmed") is not True)
                     or (robot.get("pid") is not None and robot.get("process_exit_confirmed") is not True
                         and summary.get("cleanup_status") != "completed"))
        remaining = []
        if uncertain and robot.get("pid") is not None:
            identity = {"pid": robot["pid"], "start_ticks": robot.get("start_ticks")}
            if identity_exited(identity):
                uncertain = False
                report.update(worker_exit_confirmed=True, worker_exit_evidence="proc_identity_exited")
            else:
                remaining.append(identity)
        # An abruptly killed root may have left a detached worker without ever
        # producing a summary. No PID evidence is not proof that the GPU is free.
        if (process.returncode < 0 or process.returncode == 137) and not summary:
            uncertain = True
        if uncertain:
            lock.quarantine(remaining)
            report.update(process_exit_confirmed=False, remaining_processes=remaining)
            state.update(status="failed", cleanup_stop_unconfirmed=True)
            stop.set()
        elif report["worker_exit_confirmed"] is True:
            report["process_exit_confirmed"] = True
        active.pop(index)
        entry.update(status="stop_unconfirmed" if uncertain else "exited", returncode=process.returncode)
        lock.close()  # A quarantine survives this close; native child leases still retain their fd.
        retry_after_finish(index)
        record("case_stop_unconfirmed" if uncertain else "case_exited", case_index=index,
               returncode=entry["returncode"], summary=entry["summary"])

    index, next_sample, next_admission, deadline = 0, 0, 0, None
    window_budget, window_launched = INITIAL_LAUNCH_LIMIT, 0
    try:
        record("queue_started")
        for warning in warnings:
            record("queue_warning", warning=warning)
        while pending or active:
            completed_count = 0
            for running, (process, _) in list(active.items()):
                if process.poll() is not None:
                    deadline = None
                    finish(running)
                    completed_count += 1
            completed = completed_count > 0
            if completed:
                window_launched = max(0, window_launched - completed_count)
                next_admission = 0
            if stop.is_set():
                if state["status"] == "running": state["status"] = "interrupted"
                break
            if not pending and not active: break
            refresh_parallelism()
            now = time.monotonic()
            eligible = next_ready()
            index = eligible if eligible is not None else pending[0] if pending else len(cases)
            if active: deadline = None
            if eligible is not None and not active and deadline is None: deadline = now + wait_timeout_s
            if deadline is not None and now >= deadline:
                state["status"] = "gpu_wait_timeout"
                record("gpu_wait_timeout", case_index=index)
            periodic_due = now >= next_sample
            if periodic_due:
                window_budget = INITIAL_LAUNCH_LIMIT if next_sample == 0 else WINDOW_LAUNCH_LIMIT
                window_launched = 0
            has_slot = eligible is not None and len(active) < max_parallel and window_launched < window_budget
            refill_due = has_slot and now >= next_admission
            if (periodic_due or refill_due) and state["status"] == "running":
                trigger = "case_completed" if completed and has_slot else "periodic" if periodic_due else "capacity_retry"
                # Completion and capacity retries build a new baseline; an old
                # telemetry sample must not hide newly available memory.
                current = sample(index, "periodic" if periodic_due else "admission", trigger)
                state["last_gpu_observation"] = current
                if has_slot:
                    while next_ready() is not None and not stop.is_set():
                        refresh_parallelism()
                        if len(active) >= max_parallel or window_launched >= window_budget: break
                        index = next_ready()
                        verify_source(cases[index]["cwd"], source_bindings[str(Path(cases[index]["cwd"]).resolve())])
                        latest = sample(index, "before_launch", trigger)
                        ranked = ranked_gpus(current, latest, memory_mib + headroom_mib)
                        lock, gpu = None, None
                        for candidate in ranked:
                            lock, admission = reserve_gpu(candidate, memory_mib, headroom_mib, baseline=current)
                            if lock is not None:
                                gpu = admission["gpu"]
                                break
                            record("gpu_admission_deferred", case_index=index, gpu=candidate, admission=admission)
                        if gpu is None: break
                        refresh_parallelism()
                        if (len(active) >= max_parallel or window_launched >= window_budget or stop.is_set()
                                or (deadline is not None and time.monotonic() >= deadline)):
                            lock.close()
                            break
                        case, entry = cases[index], state["cases"][index]
                        try:
                            args = command(case, gpu["index"], config=config, config_path=config_path)
                            if "retry_command_template" in entry:
                                args = recovered_command(entry, gpu["index"])
                            environment, settings = (experiment_environment(entry["retry_experiment_settings"])
                                if "retry_experiment_settings" in entry else experiment_environment(frozen_settings))
                            if "grasp_memory_dir" in case:
                                environment["AURIGA_GRASP_MEMORY_DIR"] = case["grasp_memory_dir"]
                                settings["grasp_memory_dir"] = case["grasp_memory_dir"]
                            if "agent_backend" in case and config is not None:
                                settings.update(agent_settings(config, case["agent_backend"]))
                            if "codex_home" in case:
                                environment["CODEX_HOME"] = case["codex_home"]
                                settings["codex_home"] = case["codex_home"]
                            admission.update(classification="idle" if candidate["utilization"] == gpu["utilization"] == 0
                                             else "memory_available_shared", external_job_exclusion=False,
                                             trigger=trigger)
                            entry.update(status="starting", command=args, gpu=gpu, admission=admission,
                                         experiment_settings=settings)
                            record("case_starting", case_index=index, command=args, cwd=case["cwd"], gpu=gpu, admission=admission)
                            entry["source_sha256"] = verify_source(case["cwd"], source_bindings[str(Path(case["cwd"]).resolve())])
                            with (output / f"case-{index:03d}.log").open("x") as log:
                                process = subprocess.Popen(args, cwd=case["cwd"], stdout=log, stderr=subprocess.STDOUT,
                                                           start_new_session=True, pass_fds=lock.filenos(),
                                                           env=checkout_environment(environment, checkout_layout(case["cwd"])))
                            active[index] = (process, lock)
                            lock.bind(process.pid)
                        except BaseException:
                            if index not in active:
                                lock.close()
                                entry.update(status="not_started", not_started_reason="launch_failed")
                            raise
                        entry.update(status="running", pid=process.pid)
                        if entry["attempt_number"] > 1: entry["retry_status"] = "running"
                        record("case_started", case_index=index, pid=process.pid)
                        pending.remove(index)
                        window_launched += 1
                        deadline = None
                if periodic_due: next_sample = now + sample_interval_s
                next_admission = next_sample if window_launched >= window_budget else time.monotonic() + ADMISSION_RETRY_INTERVAL_S
                save_state()
            if state["status"] == "gpu_wait_timeout" and not active: break
            if not pending and not active: break
            wake_at = [next_sample, deadline or next_sample]
            if next_ready() is not None:
                wake_at.append(next_admission)
            elif pending:
                wake_at.append(min(ready_at[item] for item in pending))
            stop.wait(COMPLETION_CHECK_INTERVAL_S if active else max(0, min(wake_at) - time.monotonic()))
        if state["status"] == "running": state["status"] = "finished"
    except (Exception, KeyboardInterrupt) as error:
        stop.set()
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
    finally:
        def cleanup_failed(error):
            state["status"] = "failed"
            state.setdefault("cleanup_errors", []).append(f"{type(error).__name__}: {error}")

        try:
            # Evidence I/O must never prevent another owned gateway from being stopped.
            cleanup = {}
            for running, (process, _) in list(active.items()):
                tracker = GatewayCleanup(process)
                entry = state["cases"][running]
                grace = (SHUTDOWN_GRACE_S if (Path(entry["case"]["run_dir"]) / "manifest.json").is_file()
                         else STARTUP_SHUTDOWN_GRACE_S)
                cleanup[running] = (tracker, time.monotonic() + grace)
                entry["queue_cleanup"] = {"grace_s": grace, "forced_termination": None,
                                          "native_close_confirmed": None, "process_exit_confirmed": False}
                try:
                    if process.poll() is None:
                        try: process.terminate()
                        except ProcessLookupError: pass
                        record("case_stop_requested", case_index=running)
                except Exception as error: cleanup_failed(error)
            for running, (process, lock) in list(active.items()):
                tracker, deadline = cleanup[running]
                entry = state["cases"][running]
                report = entry["queue_cleanup"]
                try:
                    stopped = tracker.wait(deadline)
                    if not stopped:
                        state["cleanup_stop_unconfirmed"] = True
                        state["status"] = "failed"
                        report.update(native_close_confirmed=False, forced_termination="SIGTERM")
                        try: record("cleanup_stop_unconfirmed", case_index=running, phase="native_grace_expired")
                        except Exception as error: cleanup_failed(error)
                        tracker.capture()  # Include children created during native shutdown/startup.
                        tracker.signal(signal.SIGTERM)
                        stopped = tracker.wait(time.monotonic() + FORCED_TERM_GRACE_S)
                    if not stopped:
                        report["forced_termination"] = "SIGKILL"
                        tracker.capture()
                        tracker.signal(signal.SIGKILL)
                        stopped = tracker.wait(time.monotonic() + FORCED_KILL_GRACE_S)
                    report.update(process_exit_confirmed=stopped, gateway_exit_confirmed=process.poll() is not None)
                    if stopped:
                        finish(running)
                        if not report["forced_termination"]:
                            report["native_close_confirmed"] = ((entry.get("summary") or {}).get("cleanup", {})
                                                                .get("robot", {}).get("native_close_acknowledged"))
                except Exception as error: cleanup_failed(error)
                finally:
                    try:
                        if not report["process_exit_confirmed"] and running in active:
                            remaining = [{key: value for key, value in item.items() if key != "fd"}
                                         for item in tracker.remaining()]
                            # Empty is still unknown if cleanup itself failed before
                            # confirmation; quarantine must not silently expire.
                            report.update(remaining_processes=remaining, native_close_confirmed=False)
                            entry["status"] = "stop_unconfirmed"
                            state.update(status="failed", cleanup_stop_unconfirmed=True)
                            lock.quarantine(remaining)
                        report["errors"] = tracker.errors
                    except Exception as error: cleanup_failed(error)
                    finally:
                        try: tracker.close()
                        except Exception as error: cleanup_failed(error)
                        try: lock.close()
                        except Exception as error: cleanup_failed(error)
            for entry in state["cases"]:
                if entry["status"] == "not_started":
                    entry.setdefault("not_started_reason", state["status"])
                    if stop.is_set() and entry.get("retry_status") == "pending": entry["retry_status"] = "cancelled"
            try: record("queue_finished", status=state["status"])
            except Exception as error: cleanup_failed(error)
            # A broken events stream may still permit the atomic state snapshot.
            try: save_state()
            except Exception as error: cleanup_failed(error)
        finally:
            for sig, handler in handlers.items(): signal.signal(sig, handler)
    # Historical OOM attempts remain visible, but the queue exit code reflects
    # each logical task's last attempt; scored task failures retain gateway policy.
    latest = {entry["logical_task_id"]: entry for entry in state["cases"]}
    return 0 if state["status"] == "finished" and all(item.get("returncode") == 0 and item.get("summary") is not None
        and not item.get("oom_evidence") for item in latest.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--config", type=Path, help="Unified Auriga JSON configuration")
    parser.add_argument("--recover-oom-from", type=Path, help="New queue containing only safe OOM retries from a terminal queue")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set-max-parallel", type=int, help="Update a running queue's concurrency without starting cases")
    parser.add_argument("--wait-timeout-s", type=float)
    parser.add_argument("--wake-thread", help="Main Codex thread UUID; defaults to ~/.config/auriga/completion-thread")
    parser.add_argument("--wake-codex", help="Codex executable for completion notification; use an absolute path to bypass CLI wrappers")
    parser.add_argument("--wake-remote", help="Completion notification app-server endpoint; defaults to the user's shared socket")
    parser.add_argument("--wake-enabled", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.set_max_parallel is not None:
        if any(getattr(args, key) is not None for key in ("manifest", "config", "recover_oom_from",
                "wait_timeout_s", "wake_thread", "wake_codex", "wake_remote", "wake_enabled")):
            parser.error("--set-max-parallel accepts only --output; no launch or configuration options")
        try:
            set_max_parallel(args.output, args.set_max_parallel)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps({"status": "requested", "control_path": str(args.output.resolve() / "control.json"),
                          "max_parallel": args.set_max_parallel}))
        return 0
    if args.recover_oom_from and args.config:
        parser.error("OOM recovery retains its original configuration; --config is not accepted")
    if args.recover_oom_from:
        if args.manifest: parser.error("manifest and --recover-oom-from are mutually exclusive")
        manifest, recovery = recover_oom(args.recover_oom_from)
        config = Config.from_dict(current_manifest(manifest).get("auriga_config", {}))
    else:
        if not args.manifest: parser.error("manifest or --recover-oom-from is required")
        manifest = json.loads(args.manifest.read_text())
        config = queue_config(manifest, args.config)
    for key in ("wait_timeout_s", "wake_thread", "wake_codex", "wake_remote", "wake_enabled"):
        if getattr(args, key) is None:
            setattr(args, key, getattr(config.queue, key))
    thread_id = args.wake_thread if args.wake_enabled else None
    if args.wake_enabled and thread_id is None and WAKE_THREAD_FILE.exists():
        thread_id = str(uuid.UUID(WAKE_THREAD_FILE.read_text().strip()))
    if thread_id: uuid.UUID(thread_id)
    args.wake_thread = thread_id
    if not args.recover_oom_from:
        config = config.with_overrides({"queue": {key: getattr(args, key) for key in
            ("wait_timeout_s", "wake_thread", "wake_codex", "wake_remote", "wake_enabled")}})
        manifest = configured_manifest(manifest, config)
    if args.recover_oom_from:
        result = run_queue(manifest, args.output, args.wait_timeout_s, recovery=recovery)
    else:
        result = run_queue(manifest, args.output, args.wait_timeout_s)
    if thread_id:
        # The queue has cleaned up its gateways. Notify the existing main session;
        # no developer process, response schema, iteration cap or waiting model.
        from .completion import notify_completion
        receipt = notify_completion(args.output, thread_id, codex=args.wake_codex,
                                    remote=args.wake_remote, queue_stopped=True)
        if receipt["status"] != "sent": return result or 1
    return result


if __name__ == "__main__":
    sys.exit(main())
