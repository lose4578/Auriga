"""Offline evidence integrity, with optional native-interface acceptance.

Use --profile native for native item -> RGB -> EE -> measured feedback.
"""

import argparse
import base64
from datetime import datetime
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from auriga.core import legacy_names as legacy
from .native_evidence import codex_turn_timeline, load_native_evidence, parse_timestamp as _timestamp



def _native_audit(root, store, require_continuity=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    records = store.record_items
    events = list(_events(store))
    observations, requests, receipts, deliveries = [records(kind) for kind in ("observation", "action_request", "action_receipt", "delivery")]
    evidence = load_native_evidence(root, manifest, store, profile="audit", events=events, require_texts=require_continuity)
    failures = []
    def check(condition, reason):
        if not condition: failures.append(reason)
    def blob(ref):
        _, data = store.resolve_artifact(ref)
        return data
    check(manifest["data_origin"] == "robodojo_runtime", "not a real RoboDojo run")
    check(len(requests) >= 1, "no EE action")
    check([e["server_seq"] for e in events] == list(range(1, len(events)+1)), "event sequence gap")
    native_items = evidence.tool_item_ids
    action_summaries = []
    for identifier, request in requests.items():
        receipt = receipts.get(identifier, {})
        check(receipt.get("status") in ("completed", "failed", "stopped", "rejected"), f"unresolved action:{identifier}")
        observation_id = request["based_on_observation_id"]
        check(observation_id in observations, f"missing input observation:{identifier}")
        if receipt.get("status") == "rejected": continue
        check(any(d["observation_id"] == observation_id for d in deliveries.values()), f"input was not delivered:{identifier}")
        linked = [e for e in events if e["type"] == "action.requested" and e["correlation"].get("action_id") == identifier]
        check(len(linked) == 1 and linked[0]["correlation"].get("thread_id") == manifest["thread_id"]
              and linked[0]["correlation"].get("item_id") in native_items, f"missing native item link:{identifier}")
        check(receipt.get("result_observation_id") in observations, f"missing terminal observation:{identifier}")
        for key in ("planned_trace_ref", "dispatched_trace_ref", "actual_trace_ref"):
            check(bool(receipt.get(key)), f"missing {key}:{identifier}")
            if receipt.get(key): blob(receipt[key])
        action_summaries.append(dict(action_id=identifier, status=receipt.get("status"),
                                     executed_steps=receipt.get("executed_steps"), position_error_m=receipt.get("position_error_m"),
                                     orientation_error_rad=receipt.get("orientation_error_rad"),
                                     input_observation_id=observation_id, result_observation_id=receipt.get("result_observation_id")))
    image_hashes = set()
    program_image_hashes = set()
    for delivery in deliveries.values():
        for image in delivery["images"]:
            try:
                if image.get("reconstruction") and image.get("preprocessing") == "identity":
                    from auriga.core.storage.store import open_read_store
                    with open_read_store(root) as image_store:
                        _, data = image_store.resolve_image(image["rgb_ref"], ("actor",))
                else:
                    data = blob(image["rgb_ref"])
            except FileNotFoundError:
                data = None
                check(False, "original source image unavailable; exact source-byte audit incomplete:" + image["rgb_ref"])
            if data is not None:
                check(hashlib.sha256(data).hexdigest() == image.get("source_sha256", image["sha256"]), "source image hash differs from RGB")
            if data is not None and image.get("encoding"):
                from auriga.core.storage.images import encode_image, image_settings
                encoding = image["encoding"]
                settings = image_settings(dict(transport_format=encoding["format"],
                    **{encoding["format"] + "_quality": encoding["quality"]},
                    webp_method=encoding.get("method", 4)))
                encoded, _ = encode_image(data, settings)
                check(hashlib.sha256(encoded).hexdigest() == image["sha256"], "encoded model image hash differs from delivery")
            (program_image_hashes if delivery.get("delivery_channel")=="program_fd" or delivery.get("correlation",{}).get("program_id") else image_hashes).add(image["sha256"])
    sent_hashes = evidence.sent_image_sha256
    program_states = store.states("program_call")
    program_sent_hashes=set()
    for operation in program_states:
        for content in operation.get("result",{}).get("contentItems",[]):
            if content.get("type")=="inputImage": program_sent_hashes.add(hashlib.sha256(base64.b64decode(content["imageUrl"].split(",",1)[1])).hexdigest())
    check(program_image_hashes <= program_sent_hashes, "program input image not proved by controlled IPC result")
    check(image_hashes <= sent_hashes, "delivery not proved by actual SDK tool response")
    turns = evidence.turn_terminals
    terminal_interrupts = _owned_terminal_interrupts(None, events, store, manifest["thread_id"], evidence)
    check(any(t["status"] == "completed" for t in turns) or terminal_interrupts,
          "no completed or validly native-terminal-interrupted turn")
    if require_continuity:
        check(len({t["id"] for t in turns if t["status"] == "completed"}) >= 2, "two completed turns not proved")
        check(any(r.get("status") == "completed" for r in receipts.values()), "no completed EE action")
        check(any(marker in text for marker in ("AURIGA_CONTINUITY_OK", legacy.CONTINUITY_MARKER) for text in evidence.completed_turn_texts), "assistant continuity marker absent")
    return dict(passed=not failures, run_id=manifest["run_id"], thread_id=manifest["thread_id"], terminal_session_id=manifest.get("terminal_session_id"),
                native_completed_turns=len({t["id"] for t in turns if t["status"] == "completed"}), delivered_image_digests=len(image_hashes), program_image_digests=len(program_image_hashes), actions=action_summaries,
                native_terminal_interrupted_turns=len(terminal_interrupts),
                audit_time_source_sha256={str(p.relative_to(Path(__file__).resolve().parents[3])): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).resolve().parents[2].rglob("*.py")},
                failures=failures, evidence="interface acceptance; no evaluator task score")


def _owned_terminal_interrupts(raw_records, events, store, thread_id, evidence=None):
    """Count interrupted owner turns only when native terminal evidence caused the exact interrupt."""
    terminals = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "environment.ended" or event.get("source") != "private_evaluator":
            continue
        terminal_at = _timestamp(event.get("received_at"))
        if terminal_at is None:
            continue
        payload = event.get("payload")
        reference = payload.get("result_ref") if isinstance(payload, dict) else None
        if not reference:
            continue
        try:
            _, content = store.resolve_artifact(reference, allowed_scopes=("observer",))
            terminal = json.loads(content)
        except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(terminal, dict):
            continue
        reason, count, limit = terminal.get("reason"), terminal.get("native_action_count"), terminal.get("native_step_limit")
        success = terminal.get("success")
        if (terminal.get("ended") is True and isinstance(reason, str)
                and reason in {"native_success", "native_failure", "native_step_limit"}
                and type(success) is bool and type(count) is int and count >= 0
                and type(limit) is int and limit > 0
                and ((reason == "native_success") == success)
                and (reason != "native_failure" or count < limit)
                and (reason != "native_step_limit" or count >= limit)):
            terminals.append((event.get("server_seq"), terminal_at))

    if evidence is None:  # Codex records supplied directly (historical call shape)
        evidence = codex_turn_timeline(raw_records, events, store, thread_id)
    completed_events = evidence.interrupted_completion_seq
    turn_started = evidence.turn_started_at
    turn_completed = evidence.turn_completed
    turn_interrupted = {request["turn_id"]: request["sent_at"] for request in evidence.interrupt_requests}

    return {
        turn_id for turn_id, (status, completed_at) in turn_completed.items()
        if status == "interrupted" and completed_at is not None
        and (start_at := turn_started.get(turn_id)) is not None
        and (interrupt_at := turn_interrupted.get(turn_id)) is not None
        and start_at <= interrupt_at <= completed_at
        and turn_id in completed_events
        and any(seq is not None and seq < completed_events[turn_id] and start_at <= terminal_at <= interrupt_at
                for seq, terminal_at in terminals)
    }


def _events(store):
    cursor = 0
    while page := store.events(after=cursor):
        yield from page
        next_cursor = page[-1]["server_seq"]
        if next_cursor <= cursor:
            raise ValueError("nonadvancing_event_cursor")
        cursor = next_cursor


def _integrity_audit(root, store):
    manifest = json.loads((root / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not manifest.get("run_id"):
        raise ValueError("manifest_run_identity_missing")
    failures = []
    verified, reconstructed = set(), set()

    def artifact(reference, scopes):
        try:
            value, _ = store.artifact(reference)
            identifier = value["artifact_id"]
            if value["access_scope"] not in scopes:
                raise ValueError("artifact_access_scope_denied")
            if identifier in verified or identifier in reconstructed:
                return
            try:
                store.artifact_file(reference, scopes)
            except FileNotFoundError:
                mapping = store.read_state("image_archive", identifier)
                if value["kind"] != "rgb" or mapping is None:
                    raise
                if not isinstance(mapping, dict):
                    raise ValueError("image_archive_mapping_requires_object")
                store.resolve_image(reference, scopes)
                reconstructed.add(identifier)
            else:
                verified.add(identifier)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(str(reference) + ":" + str(error))

    def observer_references(value, *, video=False):
        if not isinstance(value, dict):
            raise ValueError("observer_evidence_requires_object")
        if video:
            videos = value.get("videos", [])
            if not isinstance(videos, list) or any(not isinstance(item, dict) for item in videos):
                raise ValueError("observer_videos_require_objects")
            for item in videos:
                artifact(item.get("video_ref"), ("observer",))
        else:
            artifact(value.get("result_ref"), ("observer",))

    event_count = 0
    for event_count, event in enumerate(_events(store), 1):
        if not isinstance(event, dict):
            raise ValueError("evidence_events_require_objects")
        if event.get("server_seq") != event_count:
            failures.append("event sequence gap:" + str(event_count))
        if event.get("type") in {"observer.evaluated", "environment.ended", "observer.video_exported"}:
            observer_references(event["payload"], video=event["type"] == "observer.video_exported")
    from auriga.apps.library.experiments import read_json
    for name in ("evaluation.json", "video.json"):
        value = read_json(root / name)
        if value is not None:
            observer_references(value, video=name == "video.json")
    for value in store.artifacts():
        artifact("artifact:" + value["artifact_id"], ("actor", "internal", "observer", "evaluator"))
    observations = store.record_items("observation")
    requests = store.record_items("action_request")
    receipts = store.record_items("action_receipt")
    if any(not isinstance(value, dict) for values in (observations, requests, receipts) for value in values.values()):
        raise ValueError("evidence_records_require_objects")
    for observation in observations.values():
        for key in ("robot_state_ref", "robot_model_ref"):
            if observation.get(key) is not None:
                artifact(observation[key], ("actor", "internal"))
        cameras = observation.get("cameras", [])
        if not isinstance(cameras, list) or any(not isinstance(camera, dict) for camera in cameras):
            raise ValueError("observation_cameras_require_objects")
        for camera in cameras:
            artifact(camera.get("rgb_ref"), ("actor",))
            for key in ("depth_ref", "calibration_ref", "pose_sample_ref"):
                if camera.get(key) is not None:
                    artifact(camera[key], ("actor", "internal"))
    for receipt in receipts.values():
        for key in ("planned_trace_ref", "dispatched_trace_ref", "actual_trace_ref", "before_state_ref"):
            if receipt.get(key) is not None:
                artifact(receipt[key], ("actor", "internal"))
    unresolved = []
    for identifier, request in requests.items():
        if request.get("based_on_observation_id") not in observations:
            failures.append("missing input observation:" + identifier)
        receipt = receipts.get(identifier)
        if not receipt or receipt.get("status") in ("unknown", "status_unknown", "running", "accepted"):
            unresolved.append(identifier)
        if receipt and receipt.get("result_observation_id") and receipt["result_observation_id"] not in observations:
            failures.append("missing terminal observation:" + identifier)
    # A recorded unknown is a fact, not a corrupt record or a successful action.
    return dict(passed=not failures, run_id=manifest["run_id"], event_count=event_count, verified_artifacts=len(verified),
                original_bytes_complete=not reconstructed and not failures, reconstructed_images=sorted(reconstructed),
                unresolved_actions=unresolved, failures=failures,
                evidence="recorded links and available artifact integrity; reconstructed images are not original bytes; no action or task success assertion")


def audit(root, require_continuity=False, *, profile="integrity"):
    """Generic evidence integrity or explicit native-interface acceptance."""
    from auriga.core.storage.store import open_read_store
    if profile not in {"integrity", "native"}:
        raise ValueError("unsupported_audit_profile")
    if require_continuity:
        profile = "native"
    root = Path(root).expanduser().resolve()
    try:
        with open_read_store(root) as store:
            result = (_native_audit(root, store, require_continuity) if profile == "native"
                      else _integrity_audit(root, store))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        result = dict(passed=False, failures=["audit_evidence_unreadable:" + str(error)],
                      error_type=type(error).__name__, evidence="incomplete evidence; no acceptance claim")
    return {**result, "profile": profile, "run_dir": str(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--profile", choices=("integrity", "native"), default="integrity")
    parser.add_argument("--require-continuity", action="store_true", help="Require native multi-turn acceptance")
    args = parser.parse_args()
    result = audit(args.run_dir, args.require_continuity, profile=args.profile)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
