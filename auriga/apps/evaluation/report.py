"""Summarize recorded pilots without starting a robot or model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import tomllib

from auriga.apps.evaluation.native_evidence import load_native_evidence
from auriga.apps.library.scoring import load_effective_summary
from auriga.core.storage.store import open_read_store
from auriga.cortex.session.usage import (TOKEN_FIELDS, RelayUsageTotal, account_usage, add_served_model, aggregate_accounting,
                                        counting_basis, load_price_card, responses_counters)


def _recovery_delivery(root, manifest, store):
    """Separate configured guidance, recorded attempts and exact native submissions."""
    from auriga.core.presentation.actor_presentation import ActorPresentation

    guidance = manifest.get("completion_guidance")
    configured = guidance.get("recovery_guidance") if isinstance(guidance, dict) else None
    version = configured.get("version") if isinstance(configured, dict) else None
    if version is not None and (not isinstance(version, str) or not version):
        raise ValueError("invalid_report_recovery_version")
    thread_id = manifest.get("thread_id")
    if thread_id is not None and (not isinstance(thread_id, str) or not thread_id):
        raise ValueError("invalid_report_main_thread")
    presentation = ActorPresentation(store)
    attempts, after = [], 0
    while True:
        events = store.events_by_type(("benchmark.actor_turn_continuation",), after=after, limit=1000)
        if not events:
            break
        for event in events:
            payload = event.get("payload")
            if (not isinstance(payload, dict) or not isinstance(payload.get("thread_id"), str)
                    or not payload["thread_id"] or not isinstance(payload.get("prompt"), str)
                    or not payload["prompt"]):
                raise ValueError("invalid_report_continuation_event")
            context = payload.get("recovery_context")
            if context is not None and not isinstance(context, dict):
                raise ValueError("invalid_report_recovery_context")
            if context is not None and context.get("version") is not None and (
                    not isinstance(context["version"], str) or not context["version"]):
                raise ValueError("invalid_report_recovery_version")
            if thread_id and payload["thread_id"] == thread_id:
                attempts.append((presentation.restore(payload["prompt"]), context))
        after = events[-1]["server_seq"]

    evidence = load_native_evidence(root, manifest, store, profile="report")
    # restore only reads saved aliases; never allocate new references
    # or rerun today's producer projection against an old archive.
    sent = [{presentation.restore(text) for text in texts} for texts in evidence.submitted_prompts]

    available = evidence.log_present and thread_id is not None
    matched, recovery_sent, versions = set(), 0, {}
    for prompt, context in attempts:
        index = next((index for index, texts in enumerate(sent) if index not in matched and prompt in texts), None)
        if index is not None:
            matched.add(index)
        if context is not None:
            entry = versions.setdefault(context.get("version"), dict(attempt_count=0, native_sent_count=0))
            entry["attempt_count"] += 1
            entry["native_sent_count"] += index is not None
            recovery_sent += index is not None
    return dict(configured_version=version,
        status="main_thread_unknown" if thread_id is None else "available" if available else "native_log_missing",
        continuation_attempt_count=len(attempts) if thread_id else None,
        recovery_attempt_count=sum(context is not None for _, context in attempts) if thread_id else None,
        native_turn_start_count=len(sent) if available else None,
        native_sent_continuation_count=len(matched) if available else None,
        native_sent_recovery_count=recovery_sent if available else None,
        unmatched_continuation_attempt_count=len(attempts)-len(matched) if available else None,
        versions=[dict(version=key, attempt_count=value["attempt_count"],
            native_sent_count=value["native_sent_count"] if available else None)
            for key, value in sorted(versions.items(), key=lambda item: item[0] or "")] if thread_id else None)


def _agent_conditions(manifest, counting=None):
    """Agent runtime of a run. Manifests without the agent_backend block predate backend selection and are Codex runs.

    ``counting``: the run's recorded counting-semantics string when the manifest block lacks it (summary
    ``model_requests.counting``). ``counting_basis`` maps equivalent strings to one comparison basis, so Codex and
    new-backend runs that count the same way report the same basis."""
    # agent_cli is the CLI version only: executable paths and hashes are machine-specific provenance and would make
    # otherwise identical conditions differ across checkouts and hosts. Codex archives fall back to manifest codex_cli.
    block = manifest.get("agent_backend") if isinstance(manifest.get("agent_backend"), dict) else {}
    kind, cli = block.get("kind"), block.get("cli")
    version = cli.get("version") if isinstance(cli, dict) else None
    kind = kind if isinstance(kind, str) and kind else "codex"
    if not isinstance(version, str) and kind == "codex" and isinstance(manifest.get("codex_cli"), str):
        version = manifest["codex_cli"]
    # Evidence helper path (design 9.6). Pre-backend Codex archives always used the Codex evidence agent; for any
    # other backend a missing record stays unknown.
    client = block.get("evidence_client", manifest.get("evidence_client"))
    if not isinstance(client, str) or not client:
        client = "codex_agent" if kind == "codex" and "agent_backend" not in manifest else None
    effort = block.get("native_effort")
    recorded = block.get("counting") if isinstance(block.get("counting"), str) else counting
    return {"agent_backend": kind, "agent_cli": version if isinstance(version, str) else None,
            "evidence_client": client, "native_effort": effort if isinstance(effort, dict) else None,
            "counting_basis": counting_basis(recorded)}


def _event_relay_usage(store):
    """Relay usage and served models of an archive that predates ``runtime/model_usage_relay`` (Codex Responses
    runs), derived from its ``model.*`` events through the same field mapping: ``(record, served_models)``, or
    ``(None, None)`` when no delivered completion was recorded."""
    kinds = ("model.model_request_finished", "model.model_call_completed", "model.model_call_finished")
    total, served, attempts, after = RelayUsageTotal(), {}, {}, 0
    while page := store.events_by_type(kinds, after=after, limit=1000):
        after = page[-1]["server_seq"]
        for event in page:
            payload, call = event["payload"], event["payload"].get("logical_call_id")
            if call is None:
                continue
            if event["type"] == "model.model_request_finished":
                attempts[call] = (responses_counters(payload.get("usage")) if payload.get("usage_known") else None,
                                  payload.get("served_model"))
            elif event["type"] == "model.model_call_completed":
                counters, model = attempts.pop(call, (None, None))
                total.add(counters)
                served = add_served_model(served, model)
            else:
                attempts.pop(call, None)
    if not total.calls:
        return None, None
    return dict(total=dict(total.total), unknown_calls=total.unknown_calls, calls=total.calls), served


def _usage_methods(native, relay):
    """Native snapshot vs relay measurement per counter: ``{field: {native, relay, delta}}`` (delta None when either
    is unknown). The two are different measurement methods and are shown side by side, never merged."""
    native = native if isinstance(native, dict) else {}
    relay = (relay or {}).get("total") if isinstance((relay or {}).get("total"), dict) else {}
    result = {}
    for key in TOKEN_FIELDS:
        a, b = native.get(key), relay.get(key)
        result[key] = dict(native=a, relay=b, delta=b - a if type(a) is int and type(b) is int else None)
    return result


def summarize(root, price_card=None):
    root = Path(root).expanduser().resolve()
    def read(name):
        path = root / name
        return json.loads(path.read_text()) if path.exists() else {}
    manifest, summary = read("manifest.json"), load_effective_summary(root, summary=read("summary.json"))
    snapshot = summary.get("token_usage")
    usage = (snapshot or {}).get("total") or {}
    counts = Counter()
    deliveries = 0
    native_images = code_mode_failures = 0
    rollouts = list((root / "private" / "codex-home" / "sessions").rglob("*.jsonl"))
    for path in rollouts:
        with path.open() as stream:
            for line in stream:
                message = json.loads(line)
                item = message.get("payload") or {}
                if message.get("type") != "response_item" or item.get("type") not in ("custom_tool_call_output", "function_call_output"): continue
                output = item.get("output")
                if isinstance(output, list): native_images += sum(isinstance(block, dict) and block.get("type") == "input_image" for block in output)
                elif isinstance(output, str) and ("SIGSEGV" in output or "code-mode host closed" in output): code_mode_failures += 1
    with open_read_store(root) as store:
        for value in store.records("tool_call"):
            counts[value["params"]["tool"]] += 1
        deliveries = sum(len(value.get("images", [])) for value in store.records("delivery")
                         if value.get("delivered_to_actor"))
        recovery = _recovery_delivery(root, manifest, store)
        model_requests = summary.get("model_requests") or store.read_state("runtime", "model_requests") or {}
        served_models = summary.get("served_models") or model_requests.get("served_models")
        relay, relay_source = summary.get("model_usage_relay"), "recorded"
        if not isinstance(relay, dict):
            relay = store.read_state("runtime", "model_usage_relay")
        if not isinstance(relay, dict) and _agent_conditions(manifest)["agent_backend"] == "codex":
            # Archives from before the relay measurement was recorded: same field mapping, from the model events.
            relay, derived = _event_relay_usage(store)
            relay_source, served_models = "derived_from_model_events", served_models or derived
        if not isinstance(relay, dict):
            relay, relay_source = None, None
    model, provider = manifest.get("model"), manifest.get("provider")
    identity_source = "manifest" if model and provider else None
    saved_accounting = summary.get("token_accounting")
    # Relay-accumulating backends label their accounting "relay.<wire>.usage.sum"; keep that label rather than
    # re-deriving it under the Codex native-snapshot source.
    saved_source = saved_accounting.get("source") if isinstance(saved_accounting, dict) else None
    source = saved_source if isinstance(saved_source, str) and saved_source.startswith("relay.") else None
    native = account_usage(snapshot, **({"source": source} if source else {}))
    # TUI startup can fail after model usage but before manifest creation. The
    # final summary still records accounting. Require this exact native snapshot,
    # rather than borrowing identity/costs from a stale or foreign thread.
    saved_matches = (isinstance(saved_accounting, dict) and native.keys() <= saved_accounting.keys()
                     and isinstance(native["thread_id"], str)
                     and bool(native["thread_id"])
                     and all(saved_accounting.get(key) == native[key] for key in
                             ("schema_version", "scope", "source", "thread_id", "turn_id", "raw_ref", "tokens", "model_context_window"))
                     and all(value is None or saved_accounting.get(key) == value
                             for key, value in (("model", model), ("provider", provider))))
    if saved_matches and (model is None or provider is None):
        model, provider = saved_accounting.get("model"), saved_accounting.get("provider")
        identity_source = "summary.token_accounting"
    if provider is None:
        # Older manifests omitted provider. Use this run's frozen configuration,
        # never the developer's current global model or credentials.
        try:
            config = tomllib.loads((root / "private/codex-home/config.toml").read_text())
            if model is not None and config.get("model") == model:
                provider = config.get("model_provider")
                identity_source = "private/codex-home/config.toml" if provider else None
        except (OSError, ValueError):
            pass
    if (saved_matches and price_card is None and manifest.get("price_card") is None
            and (saved_accounting.get("model"), saved_accounting.get("provider")) == (model, provider)):
        accounting = dict(saved_accounting)
    else:
        accounting = account_usage(snapshot, model=model, provider=provider,
                                   price_card=price_card if price_card is not None else manifest.get("price_card"),
                                   **({"source": source} if source else {}))
    accounting["identity_source"] = identity_source
    return dict(run_dir=str(root), run_id=manifest.get("run_id", root.name),
                conditions={**_agent_conditions(manifest, model_requests.get("counting")), **{key: manifest.get(key) for key in ("effort", "codex_sdk", "observation_profile", "clock_policy", "max_actions", "actor_wait_budget_s", "task_notes", "task_notes_source", "task_notes_config", "public_task_rules", "task_planning", "actor_stall_policy", "benchmark_terminal_policy", "runtime_protocol", "model_request_limit", "model_request_timeout_s", "no_progress_timeout_s", "startup_timeout_s", "pi05_enabled", "visual_review_model")}, "model": model, "provider": provider},
                task=manifest.get("task", read("experiment.json").get("task")),
                seed=manifest.get("seed"), layout_id=manifest.get("layout_id"),
                status=summary.get("status"), valid_episode=summary.get("valid_episode"),
                primary_outcome=summary.get("primary_outcome"), scored_episode=summary.get("scored_episode"),
                partial_native_result=summary.get("partial_native_result"),
                task_success=summary.get("task_success"), normalized_score=summary.get("normalized_score"),
                action_count=summary.get("action_count"), elapsed_wall_s=summary.get("elapsed_wall_s"),
                native_episode=summary.get("native_episode"), model_requests=summary.get("model_requests"),
                counting=model_requests.get("counting"), served_models=served_models or None,
                model_usage_relay=relay, model_usage_relay_source=relay_source,
                usage_methods=_usage_methods(usage, relay) if relay else None,
                policy_delegation=summary.get("policy_delegation"),
                termination_reason=summary.get("termination_reason"),
                termination_authority=summary.get("termination_authority"),
                actor_stall=summary.get("actor_stall"),
                recovery_guidance=recovery,
                task_outcome=summary.get("task_outcome"), score_status=summary.get("score_status"),
                failure_reason=summary.get("failure_reason"), scoring_policy=summary.get("scoring_policy"),
                scoring_reconciliation=summary.get("scoring_reconciliation"), native_evaluation_snapshot=summary.get("native_evaluation_snapshot"),
                visual_review_usage=summary.get("visual_review_usage"), token_accounting_scope=summary.get("token_accounting_scope"),
                token_usage=usage or None,
                token_accounting=accounting,
                uncached_input_tokens=accounting["tokens"]["uncached_input_tokens"],
                cache_hit_rate=accounting["cache_hit_rate"],
                actor_tools=dict(counts), actor_image_deliveries=deliveries,
                native_model_input_images=native_images if rollouts else None, code_mode_failures=code_mode_failures if rollouts else None,
                infrastructure_error=summary.get("infrastructure_error"), source_sha256=manifest.get("source_sha256"), source_ref=manifest.get("source_ref"))


def build_report(run_dirs, price_card=None):
    """Retain unreadable runs in the denominator and keep their usage unknown."""
    roots = [Path(root).expanduser().resolve() for root in run_dirs]
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate_run_directory")
    rows = []
    for root in roots:
        try:
            row = {**summarize(root, price_card), "report_status": "readable"}
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
            row = dict(run_dir=str(root), run_id=root.name, report_status="unreadable",
                       report_error={"code": "run_evidence_unreadable", "message": str(error),
                                     "type": type(error).__name__},
                       primary_outcome=None, task_success=None, scored_episode=None,
                       token_accounting=account_usage(None, price_card=price_card))
        rows.append(row)
    # Directory is the requested evidence unit, even if archived run IDs collide.
    identities = Counter(row["run_id"] for row in rows if row["report_status"] == "readable")
    return dict(scope="Recorded runs; missing evidence is unknown, not a task failure or zero cost",
                runs=rows, coverage=dict(requested=len(rows), readable=sum(row["report_status"] == "readable" for row in rows),
                    unreadable=sum(row["report_status"] == "unreadable" for row in rows),
                    duplicate_run_ids=sorted(key for key, count in identities.items() if count > 1)),
                outcomes=dict(Counter(row.get("primary_outcome") or "unknown" for row in rows)),
                token_accounting=aggregate_accounting([{**row, "run_id": row["run_dir"]} for row in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", help="New JSON file; existing evidence is never overwritten")
    parser.add_argument("--price-card", help="JSON price schedule; recomputes estimates without changing original runs")
    args = parser.parse_args()
    card = load_price_card(args.price_card) if args.price_card else None
    report = build_report(args.run_dirs, card)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with Path(args.output).expanduser().open("x") as stream: stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
