"""Read-only, manifest-backed experiment groups for human comparison views."""
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import re

from ...core import legacy_names as legacy
from ...core.contracts import canonical
from .experiments import RUN_TIME_FIELDS, excluded_path, read_json, utc_timestamp


CASE_FIELDS = ("case_id", "task", "variant", "eval_seed", "layout_id", "reset_seed", "max_control_steps", "control_dt_s")
SUMMARY_FIELDS = ("task_success", "normalized_score", "valid_episode", "scored_episode", "score_status", "primary_outcome", "termination_reason", "cleanup_status")
CONDITION_FIELDS = ("observation_profile", "default_task_notes", "visual_review_model", "pi05", "task_notes", "preloaded_notes", "native_memories", "cross_episode_memory", "extra_models", "extra_demonstrations", "official_task_rules", "sandbox_enabled", "agent_backend", "evidence_client", "native_effort")
PRUNED = {"private", "source", "sources", "source-snapshot", "runs", "cases", "queue", "callbacks", "logs", "blobs", "artifacts", "episode-memory", "case-memories", "operation-memory", "__pycache__", "analysis", "regression-analysis", "reports", "db-snapshot"}


def group_id(manifest, condition):
    return "group-" + hashlib.sha256(f"{manifest}#{condition}".encode()).hexdigest()[:24]


def _object(value):
    return value if isinstance(value, dict) else {}


def _scalars(value, fields):
    return {key: value[key] for key in fields if key in value and (value[key] is None or type(value[key]) in (str, bool, int)
            or type(value[key]) is float and math.isfinite(value[key]))}


def _integer(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value):
        return int(value)
    return None


def _recorded_agent(manifest):
    """Agent facts a run's own manifest recorded (``agent_backend`` block): kind, evidence client, native effort
    (a short scalar-valued object compared as canonical JSON). Absent facts are omitted, so archives from before
    backend selection record nothing and keep the configured/default reading."""
    block = _object(manifest.get("agent_backend"))
    result = _scalars(block, ("kind", "evidence_client"))
    effort = _scalars(_object(block.get("native_effort")), ("effort", "variant", "model_reasoning_effort"))
    if effort:
        result["native_effort"] = canonical(effort)
    return {("agent_backend" if key == "kind" else key): value for key, value in result.items() if isinstance(value, str) and value}


def _identity(value):
    allowed = _scalars(value, CASE_FIELDS)
    result = {key: allowed.get(key) for key in CASE_FIELDS}
    result["eval_seed"] = _integer(value.get("eval_seed", value.get("seed")))
    for key in ("layout_id", "reset_seed"):
        result[key] = _integer(value.get(key))
    complete = (isinstance(result["task"], str) and isinstance(result["variant"], str)
                and type(result["eval_seed"]) is int and type(result["layout_id"]) is int)
    result["case_key"] = canonical([result[key] for key in ("task", "variant", "eval_seed", "layout_id")]) if complete else None
    return result


def _run(entry):
    item = entry["item"]
    summary = _scalars(item["summary"], SUMMARY_FIELDS)
    summary["infrastructure_error"] = bool(item["summary"].get("infrastructure_error"))
    score = summary.get("normalized_score")
    if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
            or summary.get("scored_episode") is False or summary.get("valid_episode") is False
            or summary.get("score_status") == "unavailable" or str(summary.get("primary_outcome", "")).startswith("unscored_")
            or summary["infrastructure_error"] and summary.get("scored_episode") is not True):
        score = None
    summary["normalized_score"] = score
    if type(summary.get("task_success")) is not bool:
        summary["task_success"] = None
    return {**_scalars(item, ("id", "run_id", "view_url", "status", "created_at", "replay_available", *RUN_TIME_FIELDS)),
            "time_coverage": item["time_coverage"], "time_warnings": item["time_warnings"], "summary": summary}


def _group_times(rows):
    runs = {run["id"]: run for row in rows for run in row["attempts"]}.values()
    result = dict.fromkeys(RUN_TIME_FIELDS)
    for field, choose in (("started_at", min), ("finished_at", max), ("registered_at", min), ("imported_at", min)):
        dates = [run[field] for run in runs if run.get(field)]
        if dates:
            result[field] = choose(dates)
            result[field + "_source"] = f"attempts:{choose.__name__}({field})"
    result["time_coverage"] = dict(run_count=len(runs), started_count=sum(bool(run.get("started_at")) for run in runs),
                                   finished_count=sum(bool(run.get("finished_at")) for run in runs))
    result["time_warnings"] = sorted({warning for run in runs for warning in run.get("time_warnings", [])})
    return result


def _declared_case(identifier):
    """Decode the benchmark's explicit case ID, never a directory name."""
    match = re.fullmatch(r"(.+)__(standard|random)__g(\d+)__l(\d+)", identifier) if isinstance(identifier, str) else None
    return dict(case_id=identifier, task=match[1], variant=match[2], eval_seed=int(match[3]), layout_id=int(match[4])) if match else None


class ExperimentGroups:
    """Cache decoded manifests, never mutate manifests or open run databases."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}

    def _path(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        if (not path.is_relative_to(self.root) or path.resolve() != path
                or "private" in path.relative_to(self.root).parts or excluded_path(path.relative_to(self.root))):
            raise ValueError("comparison_path_denied")
        return path

    def _read(self, path, *, packaged=False, fields=None, project=None):
        path = Path(path) if packaged else self._path(path)
        if path.is_symlink():
            raise ValueError("comparison_path_denied")
        try:
            info = path.stat()
        except FileNotFoundError:
            return {}
        fingerprint = (info.st_size, info.st_mtime_ns)
        cache_key = (path, fields, project)
        cached = self.cache.get(cache_key)
        if cached is None or cached[0] != fingerprint:
            if info.st_size > 16 * 1024 * 1024:
                raise ValueError("comparison_metadata_too_large")
            value = read_json(path, {})
            cached = (fingerprint, project(value) if project else _scalars(value, fields) if fields else value)
            self.cache[cache_key] = cached
        return cached[1]

    def _manifests(self):
        benchmark = self._path(self.root / "benchmark")
        for directory, dirs, files in os.walk(benchmark, followlinks=False):
            path = Path(directory)
            if "experiment.json" in files:
                dirs.clear()
                continue
            dirs[:] = sorted(name for name in dirs if name not in PRUNED and not name.startswith((".", "source-", "queue-"))
                             and not (path / name).is_symlink() and not excluded_path(path / name))
            for name in sorted(files):
                if name == "manifest.json" or name.endswith("-manifest.json"):
                    yield path / name

    def _condition(self, case, manifest):
        for key in ("experiment_condition", "condition", "side"):
            if isinstance(case.get(key), str):
                return case[key]
        tags = case.get("tag", [])
        if not isinstance(tags, list):
            tags = []
        names = set(_object(manifest.get("experiment_panels"))) | set(_object(manifest.get("case_roles"))) | {"baseline", "baseline-control", "baseline_control", "control", "candidate", "r1-plane"}
        matches = [tag for tag in tags if isinstance(tag, str) and tag in names]
        if len(matches) == 1:
            return matches[0]
        roots = {row.get("cwd") for row in manifest["cases"] if isinstance(row, dict)}
        return str(_scalars(manifest, ("method",)).get("method") or "unspecified") if len(roots) <= 1 else "source:" + str(case.get("cwd") or "unknown")

    def _protocol(self, case, protocol, condition):
        tags = case.get("tag", [])
        case_id = case.get("case_id")
        runtime = _identity(case)

        def matches(identity):
            runtime_task = identity["task"] + ("_random" if identity["variant"] == "random" else "") if identity["case_key"] else None
            return (identity["case_key"] is not None and case.get("task") in (identity["task"], runtime_task)
                    and case.get("variant") in (None, identity["variant"])
                    and runtime["eval_seed"] == identity["eval_seed"] and runtime["layout_id"] == identity["layout_id"])

        candidates = [row for row in protocol if row.get("case_id") and (row.get("case_id") == case_id or row.get("case_id") in tags)] if isinstance(tags, list) else []
        if not candidates:
            candidates = [row for row in protocol if matches(_identity(row))]
        if not candidates:
            candidates = [row for identifier in [case_id, *(tags if isinstance(tags, list) else [])]
                          if (row := _declared_case(identifier))]
        scoped = [row for row in candidates if row.get("experiment_condition") in (None, condition)]
        identities = {_identity(row)["case_key"]: row for row in scoped or candidates}
        if len(identities) == 1:
            identity = _identity(next(iter(identities.values())))
            if matches(identity):
                return identity
        return runtime

    def _build(self, path, manifest, catalog, *, condition_aliases=None, protocols=None, execution_run_dirs=None):
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases or not all(isinstance(row, dict) for row in cases):
            return []
        protocol = manifest.get("protocol_cases", [])
        if not isinstance(protocol, list) or not all(isinstance(row, dict) for row in protocol):
            raise ValueError("comparison_invalid_protocol")
        if not protocol:
            declared = [identifier for values in _object(manifest.get("case_roles")).values() if isinstance(values, list) for identifier in values]
            protocol = [row for identifier in declared if (row := _declared_case(identifier))]
        relative = path.relative_to(self.root).as_posix()
        buckets = defaultdict(list)
        for case in cases:
            condition = self._condition(case, manifest)
            buckets[(condition_aliases or {}).get(condition, condition)].append(case)
        recovery_selection = manifest.get("recovery_selection")
        recovery_subset = (isinstance(recovery_selection, list) and bool(recovery_selection)
            and all(isinstance(row, dict) and isinstance(row.get("new_run_dir"), str) for row in recovery_selection)
            and {row["new_run_dir"] for row in recovery_selection} == {row.get("run_dir") for row in cases})
        by_path = {entry["path"]: entry for entry in catalog.entries.values()}
        groups = []
        recorded_by_path = {}
        for condition, members in buckets.items():
            group_protocol = (protocols or {}).get(condition, protocol)
            override_protocol = condition in (protocols or {})
            warnings = []
            if override_protocol:
                keys = {_identity(row)["case_key"] for row in group_protocol}
                if any(self._protocol(case, group_protocol, condition)["case_key"] not in keys for case in members):
                    group_protocol = protocol
                    override_protocol = False
                    warnings.append("registry_protocol_case_conflict")
            panel = _object(manifest.get("experiment_panels")).get(condition, _object(manifest.get("case_roles")).get(condition))
            # Explicit panels may be recovery subsets of a larger frozen protocol.
            expected = group_protocol if override_protocol else [row for row in group_protocol if (
                        row.get("case_id") in panel if isinstance(panel, list) else row.get("experiment_condition") in (None, condition))]
            if not expected and group_protocol and len(buckets) == 1:
                expected = group_protocol
            protocol_case_count = len({_identity(row)["case_key"] or row.get("case_id") for row in expected}) if expected else None
            if recovery_subset:
                # A declared rerun subset does not schedule the copied original panel again.
                expected = [self._protocol(case, group_protocol, condition) for case in members]
            rows = {}
            for index, row in enumerate(expected):
                identity = _identity(row)
                key = identity["case_key"] or f"unknown-protocol-{index}"
                rows.setdefault(key, dict(identity, attempts=[], warnings=[]))
            for index, case in enumerate(members):
                identity = self._protocol(case, group_protocol, condition)
                key = identity["case_key"] or f"unknown-run-{index}"
                row = rows.setdefault(key, dict(identity, attempts=[], warnings=[]))
                try:
                    run_path = self._path(case["run_dir"])
                    if condition in (execution_run_dirs or {}) and run_path not in execution_run_dirs[condition]:
                        continue
                    entry = by_path.get(run_path)
                    if entry:
                        catalog.get(entry["item"]["id"])
                        archived = self._read(run_path / "manifest.json", fields=("task", "seed", "layout_id"))
                        if entry["item"]["task"] != case.get("task") or any(
                                key in archived and archived[key] != case.get(key) for key in ("task", "seed", "layout_id")):
                            row["warnings"].append("archived_case_identity_conflict")
                            continue
                        attempt = _run(entry)
                        if attempt["id"] not in {run["id"] for run in row["attempts"]}:
                            row["attempts"].append(attempt)
                            recorded_by_path[attempt["id"]] = self._read(run_path / "manifest.json", project=_recorded_agent)
                except (KeyError, ValueError, TypeError):
                    row["warnings"].append("run_path_unavailable")
            for row in rows.values():
                attempts = row["attempts"]
                row["selection"] = "single" if len(attempts) == 1 else "ambiguous" if attempts else "missing"
                row["run"] = attempts[0] if len(attempts) == 1 else None
                if row["case_key"] is None:
                    row["warnings"].append("case_identity_incomplete")
            worktrees = sorted({case["cwd"] for case in members if isinstance(case.get("cwd"), str)})
            worktree = worktrees[0] if len(worktrees) == 1 else None
            source_root = manifest.get("source_root", manifest.get("source_checkout"))
            source_fields = _scalars(manifest, ("source_revision", "source_commit"))
            revision = source_fields.get("source_revision", source_fields.get("source_commit")) if source_root == worktree or source_root is None and len(buckets) == 1 else None
            times = _group_times(rows.values())
            config = _object(manifest.get("auriga_config", manifest.get(legacy.MANIFEST_CONFIG_KEY)))
            model = _object(config.get("model"))
            conditions = dict(model=_scalars(manifest, ("model",)).get("model", _scalars(model, ("name",)).get("name")),
                              reasoning_effort=_scalars(manifest, ("reasoning_effort",)).get("reasoning_effort", _scalars(model, ("effort",)).get("effort")))
            # Archives written before backend selection have no agent section and ran on Codex.
            conditions["agent_backend"] = _scalars(_object(config.get("agent")), ("backend",)).get("backend", "codex")
            conditions.update(_scalars(_object(config.get("features")), CONDITION_FIELDS))
            conditions.update(_scalars(model, ("native_memories",)))
            information = _object(manifest.get("information_conditions"))
            conditions.update(_scalars(information, (*CONDITION_FIELDS, "model")))
            if "effort" in _scalars(information, ("effort",)):
                conditions["reasoning_effort"] = information["effort"]
            for field in CONDITION_FIELDS:
                values = {canonical(case[field]): case[field] for case in members if field in case and type(case[field]) in (str, int, bool, float, type(None))}
                if len(values) == 1:
                    conditions[field] = next(iter(values.values()))
                elif len(values) > 1:
                    conditions[field] = None
            # Evidence helper path (design 9.6): a Codex arm on the default client uses the Codex evidence agent,
            # every other arm the relay. Derived from the FINAL backend (a case may override the configured one)
            # unless the manifest declared it explicitly.
            if "evidence_client" not in conditions and not any("evidence_client" in case for case in members):
                client = _object(_object(config.get("agent")).get("evidence")).get("client")
                backend = conditions["agent_backend"]
                conditions["evidence_client"] = None if backend is None else (
                    "codex_agent" if backend == "codex" and client in (None, "native") else "relay")
            # What the runs themselves recorded is the true provenance: a value shared by every recorded attempt
            # replaces the configured reading, disagreeing attempts make the condition unknown (None), and runs
            # without a record (archives from before backend selection) keep the configured/default reading.
            recorded = [recorded_by_path[run["id"]] for row in rows.values() for run in row["attempts"] if run["id"] in recorded_by_path]
            for field in ("agent_backend", "evidence_client", "native_effort"):
                values = {entry[field] for entry in recorded if field in entry}
                if len(values) > 1:
                    conditions[field] = None
                    warnings.append(f"recorded_{field}_differs_between_attempts")
                elif values:
                    value = next(iter(values))
                    if conditions.get(field) not in (None, value) and field != "native_effort":
                        warnings.append(f"recorded_{field}_differs_from_configured")
                    conditions[field] = value
            groups.append(dict(id=group_id(relative, condition), label=f"{path.parent.name} · {condition}",
                manifest=relative, condition=condition, benchmark_group=_scalars(manifest, ("benchmark_group",)).get("benchmark_group"),
                method=_scalars(manifest, ("method",)).get("method"), comparison_scope=_scalars(manifest, ("comparison_scope",)).get("comparison_scope"),
                series=_scalars(manifest, ("series",)).get("series"),
                iteration=_scalars(manifest, ("iteration",)).get("iteration"), parent_group_id=None, baseline_group_id=None,
                created_at=utc_timestamp(manifest.get("created_at")), prepared_at=utc_timestamp(manifest.get("prepared_at")), **times,
                source=dict(manifest=relative, worktree=worktree, worktrees=worktrees, branch=None, revision=revision), conditions=conditions,
                case_count=len(rows), protocol_case_count=protocol_case_count,
                scope="recovery_subset" if recovery_subset else "protocol" if group_protocol else "recorded_cases",
                protocol_denominator_known=bool(group_protocol), cases=list(rows.values()),
                warnings=warnings + ([] if group_protocol else ["protocol_denominator_unknown"]),
                relationship_evidence=[], recovery_of_group_id=None, note="补跑子集；不代表整组重新运行" if recovery_subset else None,
                recovery=_scalars(_object(manifest.get("recovery")), ("kind", "attempt_number", "primary_attempt_preserved"))))
        baseline = [group for group in groups if group["condition"] in {"baseline", "baseline-control", "baseline_control", "control"}]
        if len(baseline) == 1:
            for group in groups:
                if group is not baseline[0]:
                    group["baseline_group_id"] = baseline[0]["id"]
                    group["relationship_evidence"] = [relative + ":case tags"]
        return groups

    def snapshot(self, catalog):
        groups, errors, references = [], [], []
        metadata_entries, registered_paths = [], set()
        aliases, protocols = defaultdict(dict), defaultdict(dict)
        manifest_copies, execution_run_dirs = set(), defaultdict(dict)
        execution_groups = {}
        registries = [(Path(__file__).with_name("experiment_group_relations.json"), True), (self.root / "service" / "experiment-groups.json", False)]
        for path, packaged in registries:
            try:
                entries = self._read(path, packaged=packaged).get("groups", [])
                if not isinstance(entries, list):
                    raise ValueError("comparison_invalid_registry")
            except (OSError, ValueError, TypeError, AttributeError):
                errors.append(dict(path=path.name, error="comparison_registry_unavailable"))
                continue
            for index, metadata in enumerate(entries):
                try:
                    if not isinstance(metadata, dict) or not isinstance(metadata.get("manifest"), str) or not isinstance(metadata.get("condition"), str):
                        raise ValueError("comparison_invalid_registry_entry")
                    if packaged and not (self.root / metadata["manifest"]).exists():
                        continue
                    manifest_path = self._path(metadata["manifest"])
                    if manifest_path.suffix != ".json":
                        raise ValueError("comparison_invalid_registry_manifest")
                    relative = manifest_path.relative_to(self.root).as_posix()
                    declared_aliases = metadata.get("condition_aliases", [])
                    if not isinstance(declared_aliases, list) or not all(isinstance(value, str) for value in declared_aliases):
                        raise ValueError("comparison_invalid_condition_aliases")
                    copies = metadata.get("manifest_copies", [])
                    if not isinstance(copies, list):
                        raise ValueError("comparison_invalid_manifest_copies")
                    if "execution_run_dirs" in metadata:
                        run_dirs = metadata["execution_run_dirs"]
                        if not isinstance(run_dirs, list) or not all(isinstance(value, str) for value in run_dirs):
                            raise ValueError("comparison_invalid_execution_run_dirs")
                        run_paths = {self._path(value) for value in run_dirs}
                    executions = metadata.get("execution_groups", [])
                    if not isinstance(executions, list):
                        raise ValueError("comparison_invalid_execution_groups")
                    execution_references = []
                    for execution in executions:
                        if not isinstance(execution, dict) or not isinstance(execution.get("manifest"), str) or not isinstance(execution.get("condition"), str):
                            raise ValueError("comparison_invalid_execution_group")
                        execution_path = self._path(execution["manifest"])
                        if execution_path.suffix != ".json":
                            raise ValueError("comparison_invalid_execution_group")
                        execution_references.append((execution_path, group_id(execution_path.relative_to(self.root).as_posix(), execution["condition"])))
                    for relation in ("parent", "baseline", "recovery_of"):
                        target = metadata.get(relation)
                        if target is not None and (not isinstance(target, dict) or not isinstance(target.get("manifest"), str) or not isinstance(target.get("condition"), str)):
                            raise ValueError("comparison_invalid_registry_relation")
                    if "protocol_cases" in metadata:
                        declared_protocol = metadata["protocol_cases"]
                        if not isinstance(declared_protocol, list) or not all(isinstance(row, dict) for row in declared_protocol):
                            raise ValueError("comparison_invalid_registry_protocol")
                        keys = [_identity(row)["case_key"] for row in declared_protocol]
                        if None in keys or len(keys) != len(set(keys)):
                            raise ValueError("comparison_invalid_registry_protocol")
                        if any(_identity(declared)["case_key"] != _identity(row)["case_key"] for row in declared_protocol
                               if (declared := _declared_case(row.get("case_id")))):
                            raise ValueError("comparison_conflicting_registry_protocol")
                    metadata_entries.append({**metadata, "manifest": relative})
                    registered_paths.add(manifest_path)
                    aliases[relative].update({alias: metadata["condition"] for alias in declared_aliases})
                    if "protocol_cases" in metadata:
                        protocols[relative][metadata["condition"]] = declared_protocol
                    if "execution_run_dirs" in metadata:
                        execution_run_dirs[relative][metadata["condition"]] = run_paths
                    if "execution_groups" in metadata:
                        execution_groups[group_id(relative, metadata["condition"])] = [identifier for _, identifier in execution_references]
                        registered_paths.update(reference for reference, _ in execution_references)
                    for copy_index, reference in enumerate(copies):
                        try:
                            if not isinstance(reference, str):
                                raise ValueError("comparison_invalid_manifest_copy")
                            copy_path = self._path(reference)
                            if copy_path.suffix != ".json" or copy_path == manifest_path or not copy_path.is_file():
                                raise ValueError("comparison_invalid_manifest_copy")
                            original = self._read(manifest_path)
                            if not isinstance(original, dict) or not isinstance(original.get("cases"), list):
                                raise ValueError("comparison_invalid_manifest_copy")
                            if canonical(self._read(copy_path)) != canonical(original):
                                raise ValueError("comparison_manifest_copy_conflict")
                            manifest_copies.add(copy_path)
                        except (OSError, ValueError, TypeError):
                            errors.append(dict(path=path.name, entry=index, copy=copy_index, error="comparison_manifest_copy_unavailable"))
                except (OSError, ValueError, TypeError, KeyError):
                    errors.append(dict(path=path.name, entry=index, error="comparison_registry_entry_unavailable"))
        try:
            paths = (set(self._manifests()) - manifest_copies) | registered_paths
        except (OSError, ValueError):
            paths = registered_paths
            errors.append(dict(path="benchmark", error="comparison_root_unavailable"))
        for path in sorted(paths):
            try:
                manifest = self._read(path)
                if not isinstance(manifest, dict):
                    raise ValueError("comparison_invalid_manifest")
                relative = path.relative_to(self.root).as_posix()
                found = self._build(path, manifest, catalog, condition_aliases=aliases[relative], protocols=protocols[relative],
                                    execution_run_dirs=execution_run_dirs[relative])
                groups.extend(found)
                for relation in ("baseline", "parent"):
                    reference = manifest.get(relation + "_manifest")
                    if isinstance(reference, str):
                        try:
                            relative = self._path(reference).relative_to(self.root).as_posix()
                            references.extend((group, relation, relative) for group in found)
                        except ValueError:
                            pass
            except (OSError, ValueError, TypeError, KeyError):
                errors.append(dict(path=path.relative_to(self.root).as_posix(), error="comparison_manifest_unavailable"))
        indexed = {group["id"]: group for group in groups}
        for group, relation, relative in references:
            targets = [candidate for candidate in groups if candidate["manifest"] == relative]
            if len(targets) == 1 and targets[0]["id"] != group["id"]:
                group[relation + "_group_id"] = targets[0]["id"]
                group["relationship_evidence"].append(group["manifest"] + ":" + relation + "_manifest")
        for metadata in metadata_entries:
            identifier = group_id(metadata["manifest"], metadata["condition"])
            group = indexed.get(identifier)
            if group is None:
                continue
            group.update(_scalars(metadata, ("label", "series", "iteration", "note", "name_zh", "purpose", "series_name_zh",
                                            "role", "baseline_reason")))
            nature = metadata.get("experiment_nature")
            if isinstance(nature, list) and all(isinstance(value, str) for value in nature):
                group["experiment_nature"] = list(nature)
            status = metadata.get("relation_status")
            if isinstance(status, dict):
                group["relation_status"] = {key: value for key in ("parent", "baseline")
                                            if isinstance(value := status.get(key), str)
                                            and value in {"confirmed", "unresolved", "not_applicable"}}
            group["source"].update(_scalars(_object(metadata.get("source")), ("branch", "revision")))
            if "evidence" in metadata:
                evidence = metadata["evidence"]
                group["relationship_evidence"] = [value for value in evidence if isinstance(value, str)] if isinstance(evidence, list) else []
            for relation in ("parent", "baseline", "recovery_of"):
                if relation not in metadata:
                    continue
                group["warnings"] = [warning for warning in group["warnings"] if warning != relation + "_group_unavailable"]
                target = metadata[relation]
                target_id = group_id(target["manifest"], target["condition"]) if target else None
                group[relation + "_group_id"] = target_id if target_id in indexed else None
                if target_id is not None and target_id not in indexed:
                    group["warnings"].append(relation + "_group_unavailable")
        groups = self._merge_executions(groups, execution_groups, errors)
        grouped = {run["id"] for group in groups for case in group["cases"] for run in case["attempts"]}
        groups.sort(key=lambda group: (group["started_at"] or "", group["id"]), reverse=True)
        return dict(groups=groups, errors=errors, ungrouped_count=len(set(catalog.entries) - grouped))

    def _merge_executions(self, groups, declarations, errors):
        indexed = {group["id"]: group for group in groups}
        neighbors, owners = defaultdict(set), defaultdict(list)
        for owner, members in declarations.items():
            for member in members:
                neighbors[owner].add(member)
                neighbors[member].add(owner)
                owners[member].append(owner)
        replacements, aliases, visited = {}, {}, set()
        for start in neighbors:
            if start in visited:
                continue
            component, pending = set(), [start]
            while pending:
                identifier = pending.pop()
                if identifier not in component:
                    component.add(identifier)
                    pending.extend(neighbors[identifier] - component)
            visited.update(component)
            roots = component - owners.keys()
            try:
                if any(identifier not in indexed for identifier in component):
                    raise ValueError("execution_group_unavailable")
                if any(len(owners[identifier]) > 1 for identifier in component):
                    raise ValueError("execution_group_multiple_owners")
                if len(roots) != 1:
                    raise ValueError("execution_group_cycle")
                root = next(iter(roots))
                order, pending = [], [root]
                while pending:
                    identifier = pending.pop()
                    if identifier in order:
                        raise ValueError("execution_group_cycle")
                    order.append(identifier)
                    pending.extend(reversed(declarations.get(identifier, [])))
                base = indexed[root]
                base_keys = {row["case_key"] for row in base["cases"]}
                rows, run_cases = {}, {}
                known_conditions, known_benchmark = {}, None
                for identifier in order:
                    member = indexed[identifier]
                    if (known_benchmark is not None and member.get("benchmark_group") is not None
                            and known_benchmark != member["benchmark_group"]):
                        raise ValueError("execution_group_identity_conflict")
                    known_benchmark = member.get("benchmark_group") or known_benchmark
                    for key, value in member["conditions"].items():
                        if value is not None and key in known_conditions and value != known_conditions[key]:
                            raise ValueError("execution_group_conditions_conflict")
                        if value is not None:
                            known_conditions[key] = value
                    for case in member["cases"]:
                        key = case["case_key"]
                        if key is None or key != _identity(case)["case_key"]:
                            raise ValueError("execution_case_identity_conflict")
                        if base["protocol_denominator_known"] and key not in base_keys:
                            raise ValueError("execution_protocol_conflict")
                        row = rows.setdefault(key, {**case, "attempts": [], "warnings": list(case["warnings"])})
                        for field in ("reset_seed", "max_control_steps", "control_dt_s"):
                            if row.get(field) is not None and case.get(field) is not None and row[field] != case[field]:
                                raise ValueError("execution_protocol_conflict")
                            if row.get(field) is None and case.get(field) is not None:
                                row[field] = case[field]
                        row["warnings"] = list(dict.fromkeys([*row["warnings"], *case["warnings"]]))
                        for attempt in case["attempts"]:
                            previous = run_cases.setdefault(attempt["id"], key)
                            if previous != key:
                                raise ValueError("execution_run_identity_conflict")
                            if attempt["id"] not in {run["id"] for run in row["attempts"]}:
                                summary = dict(attempt["summary"])
                                if summary.get("score_status") in {"unavailable", "invalid", "unscored"}:
                                    summary["normalized_score"] = None
                                row["attempts"].append({**attempt, "summary": summary, "execution_group_id": identifier})
                for row in rows.values():
                    attempts = row["attempts"]
                    scored = next((run for run in attempts if run["summary"].get("normalized_score") is not None), None)
                    row["run"] = scored or (attempts[-1] if attempts else None)
                    row["selection"] = "first_scored" if scored else "latest_unscored" if attempts else "missing"
                    row["selection_reason"] = ("按登记批次顺序保留首个正式得分，包含零分；后续结果不替换。" if scored
                        else "尚无正式得分，显示登记顺序中最后一次尝试的状态和录像。" if attempts else "协议场景尚无执行记录。")
                summary_fields = ("id", "name_zh", "label", "purpose", "series", "series_name_zh", "iteration", "role", "manifest",
                    "condition", "created_at", "prepared_at", *RUN_TIME_FIELDS, "time_coverage", "time_warnings", "case_count", "protocol_case_count", "protocol_denominator_known", "scope", "source", "conditions",
                    "experiment_nature", "relation_status", "note", "baseline_reason", "relationship_evidence",
                    "parent_group_id", "baseline_group_id", "recovery_of_group_id", "warnings")
                replacements[root] = {**base, "cases": list(rows.values()), "case_count": len(rows), "id_aliases": order[1:],
                    **_group_times(rows.values()),
                    "execution_groups": [{key: indexed[identifier][key] for key in summary_fields if key in indexed[identifier]} for identifier in order]}
                aliases.update({identifier: root for identifier in order[1:]})
            except ValueError as error:
                errors.append(dict(path="execution_groups", groups=sorted(component), error="comparison_execution_merge_conflict", reason=str(error)))
        result = [replacements.get(group["id"], group) for group in groups if group["id"] not in aliases]
        for group in result:
            for relation in ("parent_group_id", "baseline_group_id", "recovery_of_group_id"):
                if group.get(relation) in aliases:
                    target = aliases[group[relation]]
                    group[relation] = None if target == group["id"] else target
        return result
