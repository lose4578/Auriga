"""OpenCode reasoning effort (review T11 task 2): declared model variants per wire and the launch gate.

``config.effort_variants(wire)`` is the model ``variants`` map of the generated provider; ``model.effort`` selects one
per prompt (``prompt_async`` ``variant``). The session-level checks (prompt/steer carry the variant, the served
declaration must match) are in ``tests/test_agent_opencode_session_unit.py``; the request bodies the real binary
sends per wire and tier are in ``tests/test_agent_opencode_effort_smoke.py``.
"""
import unittest
from unittest import mock

from auriga.core.config import Config, ConfigError
from auriga.cortex.agents import factory
from auriga.cortex.agents.opencode.config import (EFFORT_CHOICES, EFFORT_WIRE_FIELDS, OpenCodeConfigError, build_config,
                                                  effort_variants)

QUIET = {"self_briefing": False, "visual_review_model": None}
ANTHROPIC = {"wire_api": "anthropic_messages", "base_url": "http://127.0.0.1:9", "api_key_env": "K"}
RESPONSES = {"wire_api": "openai_responses", "base_url": "http://127.0.0.1:9", "api_key_env": "K"}


class EffortVariantsTests(unittest.TestCase):
    def test_anthropic_messages_variants(self):
        variants = effort_variants("anthropic_messages")
        self.assertEqual(tuple(variants), ("low", "medium", "high", "xhigh", "max"))
        for name, options in variants.items():
            self.assertEqual(options, {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": name})
        self.assertEqual(EFFORT_WIRE_FIELDS["anthropic_messages"], "output_config.effort")

    def test_openai_responses_variants(self):
        variants = effort_variants("openai_responses")
        self.assertEqual(tuple(variants), ("none", "minimal", "low", "medium", "high", "xhigh"))
        for name, options in variants.items():
            self.assertEqual(options, {"reasoningEffort": name, "reasoningSummary": "auto",
                                       "include": ["reasoning.encrypted_content"]})
        self.assertEqual(EFFORT_WIRE_FIELDS["openai_responses"], "reasoning.effort")

    def test_choices_match_the_variants_and_other_wires_are_refused(self):
        for wire, choices in EFFORT_CHOICES.items():
            self.assertEqual(tuple(effort_variants(wire)), choices)
        with self.assertRaisesRegex(OpenCodeConfigError, "^opencode_effort_unsupported_wire:openai_chat$"):
            effort_variants("openai_chat")

    def test_variants_are_independent_copies(self):
        first = effort_variants("anthropic_messages")
        first["high"]["thinking"]["type"] = "changed"
        self.assertEqual(effort_variants("anthropic_messages")["high"]["thinking"]["type"], "adaptive")

    def test_the_generated_model_declares_them_without_reasoning_or_options(self):
        for wire in EFFORT_CHOICES:
            config = build_config(model="m", wire_api=wire, relay_base_url="http://127.0.0.1:1",
                                  context_limit_tokens=1000, output_limit_tokens=100, instructions_path="/x/AURIGA.md",
                                  variants=effort_variants(wire))
            entry = config["provider"]["relay"]["models"]["m"]
            self.assertEqual(entry["variants"], effort_variants(wire))
            self.assertIs(entry["reasoning"], False)      # OpenCode's model-id variant heuristics stay off
            self.assertNotIn("options", entry)


class EffortGateTests(unittest.TestCase):
    def requirements(self, effort, upstream):
        config = Config.from_dict({"features": QUIET, "model": {"effort": effort},
                                   "agent": {"backend": "opencode", "opencode": {"upstream": upstream}}})
        with mock.patch.object(factory, "opencode_runtime_problem", return_value=None) as probe:
            try:
                return factory.backend_requirements(config)
            finally:
                self.probed = probe.called

    def test_each_wire_accepts_exactly_its_tiers(self):
        all_efforts = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
        for name, upstream, wire in (("anthropic", ANTHROPIC, "anthropic_messages"),
                                     ("responses", RESPONSES, "openai_responses"),
                                     ("codex_home", {"source": "codex_home"}, "openai_responses")):
            for effort in all_efforts:
                with self.subTest(wire=name, effort=effort):
                    if effort in EFFORT_CHOICES[wire]:
                        self.assertIsNone(self.requirements(effort, upstream))
                        self.assertTrue(self.probed)
                    else:
                        with self.assertRaisesRegex(ConfigError, f"^agent_effort_unsupported:opencode:{effort}: "):
                            self.requirements(effort, upstream)
                        self.assertFalse(self.probed)     # refused before the runtime probe

    def test_an_incomplete_upstream_is_reported_before_the_effort(self):
        with self.assertRaisesRegex(ConfigError, "^opencode_upstream_incomplete"):
            self.requirements("ultra", {})


if __name__ == "__main__":
    unittest.main()
