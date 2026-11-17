"""P4-1: mixed-backend benchmark integration (design 04 sections 4.6, 5.5, 7.3, 9.6).

Synthetic manifests, run archives and events only: no queue process, gateway, GPU or model.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_agent_mcp_session import IDENTITY, Case as McpCase, FakeProxy, StubSession

from auriga.apps.evaluation.report import build_report, summarize
from auriga.apps.library.experiment_comparisons import ExperimentGroups
from auriga.apps.library.experiments import Catalog, atomic_json, register_run
from auriga.core.config import Config
from auriga.core.storage.store import Store
from auriga.cortex.session.codex import CodexSession, MODEL_COMPLETION_COUNTING
from auriga.cortex.session.usage import (COUNTING_EQUIVALENCE, RelayUsageTotal, add_served_model, counting_basis,
                                         responses_counters)
from scripts import benchmark_queue as queue

CODEX_COUNTING = "delivered_usable_actor_helper_compact_responses_v2"
NEW_COUNTING = "delivered_usable_generation_completions_v1"
BACKENDS = ("codex", "claude_code", "opencode")


def responses_usage(input_tokens, output_tokens, cached=0, reasoning=0):
    return dict(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens,
                input_tokens_details=dict(cached_tokens=cached), output_tokens_details=dict(reasoning_tokens=reasoning))


class MixedManifestTests(unittest.TestCase):
    """One queue manifest holding Codex, Claude Code and OpenCode cases, 2 seeds each."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.checkout = self.root / "checkout"
        (self.checkout / "auriga/apps/gateway").mkdir(parents=True)
        (self.checkout / "auriga/core").mkdir(parents=True)
        (self.checkout / "auriga/apps/gateway/cli.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
        self.cases = [dict(cwd=str(self.checkout), run_dir=str(self.root / f"run-{backend}-{seed}"), task="stack_bowls",
                           seed=seed, layout_id=0, title=f"{backend} seed {seed}", tag=[backend],
                           experiment_condition=backend, agent_backend=backend, model="shared-model", effort="high")
                      for backend in BACKENDS for seed in (0, 1)]

    def test_mixed_manifest_validates_and_every_case_carries_its_backend_flag(self):
        manifest = queue.configured_manifest({"cases": self.cases}, Config())
        self.assertEqual(len(manifest["cases"]), 6)
        self.assertEqual(len({case["run_dir"] for case in manifest["cases"]}), 6)
        for case in manifest["cases"]:
            args = queue.command(case, 0, config=Config())
            for flag, value in (("--agent-backend", case["agent_backend"]), ("--model", "shared-model"),
                                ("--effort", "high"), ("--seed", str(case["seed"]))):
                self.assertEqual(args[args.index(flag) + 1], value, (case["agent_backend"], flag))
        # The shared config is never mutated by a case's backend: backend is a per-case condition.
        self.assertNotIn("agent", manifest["auriga_config"])

    def test_frozen_settings_record_each_backend_and_only_env_names(self):
        for backend in BACKENDS:
            settings = queue.agent_settings(Config(), backend)
            self.assertEqual(settings["agent_backend"], backend)
            self.assertNotIn("sk-", json.dumps(settings))

    def test_evidence_path_warning_when_backends_mix_evidence_clients(self):
        def config(backend, client="native"):
            return Config.from_dict({"features": {"self_briefing": True},
                                     "agent": {"backend": backend, "evidence": {"client": client}}})
        mixed = queue.evidence_path_warnings([config("codex"), config("claude_code"), config("opencode")])
        self.assertEqual([item["kind"] for item in mixed], ["comparison_evidence_path_mismatch"])
        self.assertEqual(mixed[0]["agent_backends"], ["claude_code", "codex", "opencode"])
        self.assertEqual(mixed[0]["evidence_clients"], {"codex_agent": [0], "relay": [1, 2]})
        self.assertEqual(queue.evidence_path_warnings(
            [config("codex", "relay"), config("claude_code"), config("opencode")]), [])


class ServedModelAndRelayUsageTests(unittest.TestCase):
    def test_counting_basis_treats_equivalent_strings_as_one_basis(self):
        self.assertEqual(counting_basis(CODEX_COUNTING), counting_basis(NEW_COUNTING))
        self.assertEqual(counting_basis(NEW_COUNTING), "all_delivered_usable_generation_completions")
        self.assertEqual(set(COUNTING_EQUIVALENCE.values()), {"all_delivered_usable_generation_completions"})
        # Undeclared strings are their own basis; missing stays unknown.
        self.assertEqual(counting_basis("custom_v1"), "custom_v1")
        self.assertIsNone(counting_basis(None))
        self.assertIsNone(counting_basis(""))

    def test_relay_total_makes_unknown_sticky_and_seeds_only_the_same_thread(self):
        total = RelayUsageTotal()
        total.add(dict(input_tokens=10, output_tokens=2))
        record = total.add(dict(input_tokens=5, output_tokens=1, cached_input_tokens=0, cache_write_input_tokens=0,
                                reasoning_output_tokens=0, total_tokens=6), thread_id="t")
        self.assertEqual((record["total"]["input_tokens"], record["calls"], record["unknown_calls"]), (15, 2, 1))
        self.assertIsNone(record["total"]["cached_input_tokens"])
        seeded = RelayUsageTotal(record, "t")
        self.assertEqual(seeded.calls, 2)
        self.assertEqual(RelayUsageTotal(record, "other").calls, 0)
        self.assertEqual(add_served_model(add_served_model({}, "m"), None), {"m": 1, "unknown": 1})

    def test_responses_counters_use_the_account_responses_mapping(self):
        counters = responses_counters(responses_usage(100, 20, cached=40, reasoning=5))
        self.assertEqual((counters["input_tokens"], counters["cached_input_tokens"], counters["output_tokens"],
                          counters["reasoning_output_tokens"], counters["total_tokens"]), (100, 40, 20, 5, 120))
        self.assertIsNone(counters["cache_write_input_tokens"])
        self.assertTrue(all(value is None for value in responses_counters(None).values()))


class McpServedModelsTests(McpCase):
    def test_served_models_aggregate_delivered_completions_and_survive_resume(self):
        session = self.session()
        session._begin_turn()
        session.model_proxy = FakeProxy([])
        session._relay_used, session._relay_attempts, session._attempt_usage = 0, 0, {}
        usage = {"input_tokens": 10, "output_tokens": 3}
        for index, served in enumerate(("alias-1", "alias-1", None, "alias-2"), 1):
            call = f"t:{index}"
            session.model_proxy.attempt_count = index
            session.model_proxy.request_count = min(index, 3)
            session._model_event(dict(kind="model_request_finished", logical_call_id=call, usage=usage,
                                      usage_known=True, served_model=served))
            if index == 4:   # an undelivered call (failed) must not be counted as served
                session._model_event(dict(kind="model_call_finished", logical_call_id=call))
            else:
                session._model_event(dict(kind="model_call_completed", logical_call_id=call,
                                          native_thread_id="thread-1"))
        requests = self.store.read_state("runtime", "model_requests")
        self.assertEqual(requests["served_models"], {"alias-1": 2, "unknown": 1})
        self.assertEqual(requests["completed_calls"], 3)
        self.assertEqual(self.store.read_state("runtime", "model_usage_relay")["calls"], 3)
        resumed = self.session()
        resumed._start_relay = None
        self.store.set_state("runtime", "model_requests", dict(requests, counting=resumed.counting))
        with mock.patch("auriga.cortex.agents.mcp_session.ModelRequestProxy") as proxy:
            proxy.return_value.start.return_value = None
            try:
                type(resumed)._start_relay(resumed, "http://127.0.0.1:1", upstream_auth_headers={},
                                           request_projection=lambda request: request, default_thread_id="thread-1",
                                           image_settings=dict(reuse_exact_images=False, trim_oversized_requests=False,
                                                               trim_max_inline_images=1))
            except Exception:   # only the seeded counter is under test, not the full relay construction
                pass
        self.assertEqual(resumed._served_models, {"alias-1": 2, "unknown": 1})


class CodexRelayRecordTests(unittest.TestCase):
    """CodexSession._prepare wires the relay measurement and the served-model ledger into the proxy callback."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = Store(self.root, dict(IDENTITY, run_id="codex-run"))
        self.addCleanup(self.store.close)
        home = self.root / "fixture-config"
        home.mkdir()
        (home / "config.toml").write_text('model_provider = "fixture"\n[model_providers.fixture]\nname = "fixture"\n'
                                          'base_url = "http://127.0.0.1:1"\n')
        self.session = CodexSession(self.root, self.store, lambda value: {},
                                    config=Config.from_dict({"model": {"codex_home": str(home)}}))
        self.session.thread_id, self.session.turn_id, self.session.state = "thread-owned", "turn-owned", "running"
        self.addCleanup(self.session.close)

    def callback(self):
        with mock.patch("auriga.cortex.session.codex.ModelRequestProxy", side_effect=RuntimeError("reached")) as proxy:
            with self.assertRaisesRegex(RuntimeError, "reached"):
                self.session._prepare()
        self.session.model_proxy = SimpleNamespace(request_count=0, attempt_count=0, max_retries=12,
                                                   transport_blocked=False, blocked_error=None,
                                                   transport_recovery_summary=dict(policy=None, rounds=0))
        self.addCleanup(setattr, self.session, "model_proxy", None)
        return proxy.call_args.kwargs["on_event"]

    def deliver(self, emit, call, usage, served, completed=True):
        emit(dict(kind="model_request_finished", logical_call_id=call, usage=usage, usage_known=usage is not None,
                  served_model=served))
        emit(dict(kind="model_call_completed" if completed else "model_call_finished", logical_call_id=call))

    def test_codex_runs_record_relay_usage_and_served_models(self):
        emit = self.callback()
        self.deliver(emit, "c1", responses_usage(100, 20, cached=40, reasoning=5), "served-a")
        self.deliver(emit, "c2", responses_usage(50, 10, cached=10, reasoning=1), "served-a")
        self.deliver(emit, "c3", responses_usage(7, 7), "served-b", completed=False)   # not delivered: not counted
        relay = self.store.read_state("runtime", "model_usage_relay")
        self.assertEqual((relay["calls"], relay["total"]["input_tokens"], relay["total"]["cached_input_tokens"],
                          relay["total"]["output_tokens"], relay["total"]["reasoning_output_tokens"]),
                         (2, 150, 50, 30, 6))
        # Codex does not report cache writes: that counter (and so the all-fields unknown count) stays unknown.
        self.assertIsNone(relay["total"]["cache_write_input_tokens"])
        requests = self.store.read_state("runtime", "model_requests")
        self.assertEqual(requests["served_models"], {"served-a": 2})
        self.assertEqual(requests["counting"], MODEL_COMPLETION_COUNTING)
        self.assertEqual(counting_basis(requests["counting"]), counting_basis(NEW_COUNTING))

    def test_resume_without_relay_record_keeps_earlier_calls_unknown_and_seeded_ledger_continues(self):
        self.store.set_state("runtime", "model_requests", dict(completed_calls=5, http_attempt_count=5,
                             counting=MODEL_COMPLETION_COUNTING, served_models={"old": 5}))
        emit = self.callback()
        self.deliver(emit, "c1", responses_usage(10, 2), "new")
        relay = self.store.read_state("runtime", "model_usage_relay")
        self.assertEqual((relay["calls"], relay["unknown_calls"]), (6, 6))
        self.assertIsNone(relay["total"]["input_tokens"])
        self.assertEqual(self.store.read_state("runtime", "model_requests")["served_models"], {"old": 5, "new": 1})


def make_run(root, *, manifest, summary, state=None, events=()):
    store = Store(root, dict(run_id=manifest.get("run_id", root.name), episode_id="e", env_instance_id="i",
                             env_generation=1))
    try:
        for (kind, identifier), value in (state or {}).items():
            store.set_state(kind, identifier, value)
        for kind, payload in events:
            store.event(kind, payload, source="private_model_transport")
    finally:
        store.close()
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def relay_state(calls, inp, out, thread="t"):
    total = dict(input_tokens=inp, cached_input_tokens=0, cache_write_input_tokens=0, output_tokens=out,
                 reasoning_output_tokens=0, total_tokens=inp + out)
    return dict(total=total, last=total, model_context_window=None, thread_id=thread, turn_id="u", raw_ref=None,
                unknown_calls=0, calls=calls)


class ReportArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)

    def run_dir(self, name, backend, **extra):
        """A finished run of ``backend`` as the matching session leaves it behind."""
        root = self.base / name
        root.mkdir()
        counting = CODEX_COUNTING if backend == "codex" else NEW_COUNTING
        relay = relay_state(3, 300, 30)
        block = dict(kind=backend, cli=dict(version="9.9", executable="/x", sha256="ab"), counting=counting,
                     evidence_client="codex_agent" if backend == "codex" else "relay",
                     native_effort=dict(variant="high") if backend == "opencode" else dict(effort="high"))
        manifest = dict(run_id=name, model="configured-model", provider="provider-a", seed=0, layout_id=0,
                        agent_backend=block, **extra)
        summary = dict(status="completed", agent_backend=backend, token_usage=relay,
                       served_models={"served-model": 3}, model_usage_relay=relay,
                       model_requests=dict(counting=counting, completed_calls=3, served_models={"served-model": 3}))
        return make_run(root, manifest=manifest, summary=summary)

    def test_each_backend_reports_conditions_usage_served_models_and_counting_basis(self):
        report = build_report([self.run_dir(f"run-{backend}", backend) for backend in BACKENDS])
        rows = {row["conditions"]["agent_backend"]: row for row in report["runs"]}
        self.assertEqual(set(rows), set(BACKENDS))
        for backend, row in rows.items():
            conditions = row["conditions"]
            self.assertEqual((conditions["agent_cli"], conditions["model"]), ("9.9", "configured-model"))
            self.assertEqual(conditions["evidence_client"], "codex_agent" if backend == "codex" else "relay")
            self.assertEqual(conditions["counting_basis"], "all_delivered_usable_generation_completions")
            self.assertEqual(row["served_models"], {"served-model": 3})
            self.assertEqual(row["model_usage_relay_source"], "recorded")
            self.assertEqual(row["model_usage_relay"]["calls"], 3)
            self.assertEqual(row["usage_methods"]["input_tokens"], dict(native=300, relay=300, delta=0))
        # The raw strings stay visible even though the basis is one.
        self.assertEqual({row["counting"] for row in rows.values()}, {CODEX_COUNTING, NEW_COUNTING})
        self.assertEqual(rows["opencode"]["conditions"]["native_effort"], {"variant": "high"})
        self.assertEqual(rows["claude_code"]["conditions"]["native_effort"], {"effort": "high"})

    def test_relay_accounting_source_label_is_kept_not_rewritten_to_codex(self):
        root = self.run_dir("run-cc", "claude_code")
        summary = json.loads((root / "summary.json").read_text())
        from auriga.cortex.session.usage import account_usage
        saved = account_usage(summary["token_usage"], model="configured-model", provider="provider-a",
                              source="relay.anthropic_messages.usage.sum")
        saved.update(thread_id="t", turn_id="u")
        (root / "summary.json").write_text(json.dumps(dict(summary, token_accounting=saved)))
        self.assertEqual(summarize(root)["token_accounting"]["source"], "relay.anthropic_messages.usage.sum")

    def test_pre_refactor_codex_archive_reads_as_codex_and_derives_relay_usage_from_events(self):
        root = self.base / "old-codex"
        root.mkdir()
        events = []
        for index, (usage, served) in enumerate(((responses_usage(100, 20, 40, 5), "served-a"),
                                                 (responses_usage(60, 10, 0, 0), "served-a")), 1):
            events.append(("model.model_request_finished", dict(logical_call_id=f"c{index}", usage=usage,
                                                                 usage_known=True, served_model=served)))
            events.append(("model.model_call_completed", dict(logical_call_id=f"c{index}")))
        events.append(("model.model_request_finished", dict(logical_call_id="c3", usage=None, usage_known=False)))
        make_run(root, manifest=dict(run_id="old", codex_sdk="0.160.0", codex_cli="0.160.0", model="m", provider="p"),
                 summary=dict(status="completed", model_requests=dict(counting=CODEX_COUNTING, completed_calls=2)),
                 events=events)
        row = summarize(root)
        conditions = row["conditions"]
        self.assertEqual((conditions["agent_backend"], conditions["agent_cli"], conditions["evidence_client"]),
                         ("codex", "0.160.0", "codex_agent"))
        self.assertEqual(conditions["counting_basis"], "all_delivered_usable_generation_completions")
        self.assertEqual(row["model_usage_relay_source"], "derived_from_model_events")
        self.assertEqual((row["model_usage_relay"]["calls"], row["model_usage_relay"]["total"]["input_tokens"],
                          row["served_models"]), (2, 160, {"served-a": 2}))

    def test_archive_without_any_relay_evidence_reports_unknown_not_zero(self):
        root = self.base / "bare"
        root.mkdir()
        make_run(root, manifest=dict(run_id="bare", agent_backend=dict(kind="opencode")), summary=dict(status="unknown"))
        row = summarize(root)
        self.assertIsNone(row["model_usage_relay"])
        self.assertIsNone(row["usage_methods"])
        self.assertIsNone(row["served_models"])
        self.assertIsNone(row["conditions"]["evidence_client"])
        self.assertIsNone(row["conditions"]["counting_basis"])


class LibraryComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = self.root / "benchmark" / "panel" / "manifest.json"
        self.row = dict(case_id="pick__standard__g0__l0", task="pick", variant="standard", eval_seed=0, layout_id=0)

    def register(self, name, recorded):
        run = self.root / "benchmark" / "panel" / "runs" / name
        register_run(run, category="benchmark", task="pick", run_id=name)
        atomic_json(run / "summary.json", dict(normalized_score=1, task_success=True, valid_episode=True,
                                               scored_episode=True, status="completed"))
        manifest = dict(task="pick", seed=0, layout_id=0)
        if recorded is not None:
            manifest["agent_backend"] = recorded
        atomic_json(run / "manifest.json", manifest)
        return run

    def groups(self, specs, config=None, **extra):
        cases = []
        for condition, (name, recorded, override) in specs.items():
            run = self.register(name, recorded)
            cases.append(dict(run_dir=str(run), cwd="/source/x", task="pick", seed=0, layout_id=0,
                              experiment_condition=condition, tag=[self.row["case_id"]], **override))
        manifest = dict(benchmark_group="t", method="paired", protocol_cases=[self.row], cases=cases, **extra)
        if config is not None:
            manifest["auriga_config"] = config
        atomic_json(self.path, manifest)
        catalog = Catalog(self.root)
        catalog.scan()
        return {g["condition"]: g for g in ExperimentGroups(self.root).snapshot(catalog)["groups"]}

    def test_baseline_and_candidate_differing_only_in_backend(self):
        groups = self.groups({
            "baseline": ("old-codex", None, {}),   # archive from before backend selection
            "candidate": ("cc", dict(kind="claude_code", evidence_client="relay", native_effort=dict(effort="high")),
                          {"agent_backend": "claude_code"})}, config=Config().to_dict())
        base, candidate = groups["baseline"]["conditions"], groups["candidate"]["conditions"]
        self.assertEqual((base["agent_backend"], base["evidence_client"]), ("codex", "codex_agent"))
        self.assertEqual((candidate["agent_backend"], candidate["evidence_client"]), ("claude_code", "relay"))
        self.assertEqual(candidate["native_effort"], '{"effort":"high"}')
        self.assertEqual(groups["candidate"]["baseline_group_id"], groups["baseline"]["id"])
        differing = {key for key in {*base, *candidate} if base.get(key) != candidate.get(key)}
        self.assertEqual(differing, {"agent_backend", "evidence_client", "native_effort"})
        self.assertEqual(groups["candidate"]["warnings"], [])

    def test_baseline_label_pairs_the_backend_arm_with_the_codex_arm(self):
        groups = self.groups({"baseline": ("b", None, {}),
                              "cand": ("c", dict(kind="opencode", evidence_client="relay"), {"agent_backend": "opencode"})})
        self.assertEqual(groups["cand"]["baseline_group_id"], groups["baseline"]["id"])
        self.assertEqual(groups["baseline"]["conditions"]["agent_backend"], "codex")
        self.assertEqual(groups["cand"]["conditions"]["agent_backend"], "opencode")

    def test_codex_arm_on_relay_evidence_is_its_own_condition(self):
        config = Config.from_dict({"agent": {"evidence": {"client": "relay"}}}).to_dict()
        groups = self.groups({"baseline": ("b", None, {}), "cand": ("c", None, {"agent_backend": "claude_code"})},
                             config=config)
        self.assertEqual(groups["baseline"]["conditions"]["evidence_client"], "relay")
        self.assertEqual(groups["cand"]["conditions"]["evidence_client"], "relay")

    def test_recorded_backend_is_the_true_provenance_and_conflicts_are_flagged(self):
        groups = self.groups({"baseline": ("b", dict(kind="opencode"), {})}, config=Config().to_dict())
        group = groups["baseline"]
        self.assertEqual(group["conditions"]["agent_backend"], "opencode")
        self.assertIn("recorded_agent_backend_differs_from_configured", group["warnings"])


if __name__ == "__main__":
    unittest.main()
