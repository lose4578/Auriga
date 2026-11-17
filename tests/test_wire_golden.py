"""Byte-identity of the WireAdapter extraction against fixtures recorded at c4ff16a.

The fixtures were captured by running ``tests/wire_golden_fixture.py`` against the
unmodified c4ff16a source tree (see that module's docstring); this test replays the same
scenarios through the current relay and egress scrub.
"""
import json
import unittest

from auriga.cortex.session.codex import DYNAMIC_TOOL_NAMESPACE
from auriga.cortex.session.wire.openai_responses import present_responses_request

import wire_golden_fixture as golden


class ForwardingGoldenTest(unittest.TestCase):
    maxDiff = 4000

    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads(golden.FORWARDING_FIXTURE.read_text())
        cls.current = json.loads(json.dumps(golden.capture_forwarding(), sort_keys=True, ensure_ascii=False))

    def test_fixture_comes_from_pre_extraction_revision(self):
        self.assertEqual(self.recorded["captured_from"], {"revision": "c4ff16a"})
        self.assertEqual(set(self.current), set(self.recorded["scenarios"]))
        self.assertEqual(len(self.current), len(golden.scenarios()))

    def test_forwarded_upstream_bodies_are_byte_identical(self):
        for name, recorded in self.recorded["scenarios"].items():
            with self.subTest(scenario=name):
                current = self.current[name]["upstream"]
                self.assertEqual([item["body"] for item in current], [item["body"] for item in recorded["upstream"]])
                self.assertEqual([(item["path"], item["headers"]) for item in current],
                                 [(item["path"], item["headers"]) for item in recorded["upstream"]])

    def test_client_visible_answers_are_byte_identical(self):
        for name, recorded in self.recorded["scenarios"].items():
            with self.subTest(scenario=name):
                self.assertEqual(self.current[name]["client"], recorded["client"])

    def test_evidence_events_and_relay_state_are_unchanged(self):
        for name, recorded in self.recorded["scenarios"].items():
            with self.subTest(scenario=name):
                self.assertEqual(self.current[name]["state"], recorded["state"])
                self.assertEqual([event["kind"] for event in self.current[name]["events"]],
                                 [event["kind"] for event in recorded["events"]])
                self.assertEqual(self.current[name]["events"], recorded["events"])

    def test_scenarios_cover_the_required_paths(self):
        scenarios = self.recorded["scenarios"]
        kinds = lambda name: [event["kind"] for event in scenarios[name]["events"]]
        self.assertIn("model_images_trimmed", kinds("image_count_trim_and_exact_reuse"))
        self.assertIn("model_images_reused", kinds("image_count_trim_and_exact_reuse"))
        self.assertIn("model_images_trimmed", kinds("image_body_bytes_trim"))
        self.assertEqual(scenarios["compaction_route_epoch"]["state"]["image_context_epoch"], "<transport>:1")
        self.assertEqual(scenarios["budget_denial"]["client"][1]["status"], 429)
        self.assertEqual(scenarios["thread_cancelled_denial"]["client"][0]["status"], 409)
        self.assertEqual(scenarios["failed_terminal_then_transport_blocked_denial"]["client"][1]["status"], 409)
        self.assertIn("model_request_retry", kinds("retryable_503_then_success"))
        self.assertIn("model_transport_recovered", kinds("transport_recovery_after_503"))
        self.assertEqual(len(scenarios["image_rejected_then_trimmed_retry"]["upstream"]), 2)
        bodies = scenarios["image_rejected_then_trimmed_retry"]["upstream"]
        self.assertLess(bodies[1]["body"]["bytes"], bodies[0]["body"]["bytes"])
        for name in ("failed_terminal_then_transport_blocked_denial", "incomplete_terminal", "worker_failed_terminal"):
            self.assertNotIn("unsafe", scenarios[name]["client"][0]["body"]["text"])


class PresentResponsesGoldenTest(unittest.TestCase):
    maxDiff = 4000

    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads(golden.PRESENTATION_FIXTURE.read_text())

    def test_codex_present_model_request_matches_pre_extraction_output(self):
        self.assertEqual(self.recorded["captured_from"], {"revision": "c4ff16a"})
        current = json.loads(json.dumps(golden.capture_presentation(), sort_keys=True, ensure_ascii=False))
        self.assertEqual(set(current), set(self.recorded["cases"]))
        for name, recorded in self.recorded["cases"].items():
            with self.subTest(case=name):
                self.assertEqual(current[name], recorded)

    def test_present_responses_request_alone_matches_old_scrub(self):
        def present(session, request):
            return present_responses_request(request, session.presentation, deferred_tools=session.deferred_tools,
                                             namespace=DYNAMIC_TOOL_NAMESPACE)
        current = json.loads(json.dumps(golden.capture_presentation(present), sort_keys=True, ensure_ascii=False))
        for name, recorded in self.recorded["cases"].items():
            if name.startswith("lean_projection"):
                continue  # lean projection is applied by CodexSession after the scrub
            recorded = {key: value for key, value in recorded.items() if key != "dumps"}
            with self.subTest(case=name):
                self.assertEqual(current[name], recorded)

    def test_planted_metadata_is_scrubbed_and_protocol_fields_are_untouched(self):
        request = golden.planted_request()
        output = self.recorded["cases"]["planted_every_field"]["output"]
        items = {item.get("type"): item for item in request["input"]}
        projected = {item.get("type"): item for item in output["input"]}
        for scrubbed in (output["instructions"], output["input"][0]["content"][0]["text"],
                         projected["reasoning"]["summary"][0]["text"], projected["function_call"]["arguments"],
                         projected["custom_tool_call"]["input"], projected["custom_tool_call_output"]["output"],
                         json.dumps(projected["tool_search_call"]["arguments"]),
                         output["tools"][0]["description"], output["tools"][2]["description"]):
            self.assertNotIn(golden.DIGEST, scrubbed)
        self.assertNotIn(golden.ARTIFACT, json.dumps(output["input"]))
        self.assertIn("artifact:ref_", output["instructions"])
        for kind, field in (("reasoning", "encrypted_content"), ("reasoning", "id"), ("function_call", "call_id"),
                            ("function_call", "id"), ("compaction", "encrypted_content"), ("compaction", "id"),
                            ("custom_tool_call", "call_id"), ("tool_search_call", "call_id")):
            self.assertEqual(projected[kind][field], items[kind][field], (kind, field))
        self.assertEqual(output["input"][1]["content"][1], request["input"][1]["content"][1])
        for key in ("model", "reasoning", "include", "prompt_cache_key", "store", "stream", "metadata"):
            self.assertEqual(output[key], request[key], key)
        self.assertEqual(output["tools"][2]["format"], request["tools"][2]["format"])

    def test_scrub_is_deterministic_for_repeated_requests(self):
        from pathlib import Path
        import tempfile
        from auriga.core.storage.store import Store
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "run", dict(run_id="golden", episode_id="episode",
                                                        env_instance_id="environment", env_generation=1))
            try:
                session = golden.fake_session(store, Path(directory), {})
                first = present_responses_request(golden.planted_request(), session.presentation,
                                                  deferred_tools=session.deferred_tools, namespace="auriga")
                second = present_responses_request(golden.planted_request(), session.presentation,
                                                   deferred_tools=session.deferred_tools, namespace="auriga")
                self.assertEqual(json.dumps(first), json.dumps(second))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
