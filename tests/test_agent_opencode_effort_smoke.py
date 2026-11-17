"""Real-binary check of OpenCode reasoning effort per wire and tier (review T11 task 2).

Drives the pinned ``opencode serve`` 1.19.5 named by ``AURIGA_OPENCODE_EXECUTABLE`` through the real relay against a
fake model (FakeAnthropicServer / FakeResponsesServer on loopback; no real model, no network). For every tier of
``config.EFFORT_CHOICES[wire]`` one turn is run after ``set_effort(tier)``: the main agent thinks and delegates with
the native ``task`` tool (a subagent session), the answer after the task reports a usage above the compaction
threshold (native auto-compaction runs), and the turn ends with a final text. Every request of that turn - main,
subagent and compaction - must carry the tier in the wire's body field:

* anthropic_messages: ``output_config.effort = <tier>`` and ``thinking = {type: adaptive, display: summarized}``
  (the bundled ``@ai-sdk/anthropic`` 3.0.111 accepts ``xhigh``);
* openai_responses: ``reasoning = {effort: <tier>, summary: auto}`` and ``include`` has
  ``reasoning.encrypted_content``.

It also pins what the binary does with earlier reasoning in a tool loop with ``reasoning: false`` in the model
entry: the thinking block (Anthropic, with its signature) / the reasoning item (Responses, with its encrypted content)
of the step that called the tool is sent back in the next request. ``max_tokens`` / ``max_output_tokens`` equals
``agent.opencode.output_limit_tokens``. Skipped (with the reason) unless the variable names the pinned binary and
``rg`` is installed.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support.fake_model_servers import (FakeAnthropicServer, FakeResponsesServer, Step, ToolCall, Usage, text_step)

from auriga.core.config import Config
from auriga.core.storage.store import Store
from auriga.cortex.agents import factory
from auriga.cortex.agents.opencode.config import EFFORT_CHOICES

EXECUTABLE = os.environ.get("AURIGA_OPENCODE_EXECUTABLE")
PROBLEM = (factory.opencode_runtime_problem(EXECUTABLE) if EXECUTABLE else
           "set AURIGA_OPENCODE_EXECUTABLE to the opencode-ai@1.19.5 platform binary")
if PROBLEM is None and shutil.which("rg") is None:
    PROBLEM = "ripgrep (rg) is not installed"
IDENTITY = dict(run_id="effort", episode_id="ep", env_instance_id="env", env_generation=1)
KEY_ENV = "AURIGA_SMOKE_OPENCODE_EFFORT_KEY"
OUTPUT_LIMIT = 8000
CONTEXT_LIMIT = 40000        # usable = 40000 - 8000; the after-task answer reports more and triggers compaction
SIGNATURE = "fake-signature-effort"


class Router:
    """Classifies each request (``compaction``: no tools declared - OpenCode's summary call passes none, and the title
    agent is disabled; ``child``: a session header other than the main session; else ``main``) and answers: main -> think + ``task`` call, then after the tool result a text with
    an overflowing usage, then the final text; child -> a short report; compaction -> a summary."""

    def __init__(self, wire):
        self.wire, self.main_session, self.seen = wire, None, []

    def __call__(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        blob = json.dumps(body)
        kind = ("compaction" if not body.get("tools") else
                "child" if request.header("x-opencode-session-id") != self.main_session else "main")
        self.seen.append((kind, request))
        if kind == "compaction":
            return text_step("SUMMARY of the work so far")
        if kind == "child":
            return text_step("child report")
        if '"tool_result"' in blob or '"function_call_output"' in blob:
            return Step(text="AFTER-TASK", usage=Usage(input_tokens=CONTEXT_LIMIT))
        if "DELEGATE" in last_user_text(body, self.wire):
            return Step(thinking="I should delegate.", thinking_signature=SIGNATURE,
                        tool_calls=[ToolCall("task", {"description": "explore", "prompt": "look around",
                                                      "subagent_type": "general"})])
        return text_step("FINAL")


def last_user_text(body, wire):
    items = body.get("messages") if wire == "anthropic_messages" else body.get("input")
    for item in reversed(items or []):
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        return " ".join(block.get("text", "") for block in content or [] if isinstance(block, dict))
    return ""


def effort_field(body, wire):
    """The tier the request carries in the wire's field, plus the companion fields (None when absent)."""
    if wire == "anthropic_messages":
        return (body.get("output_config") or {}).get("effort"), body.get("thinking"), body.get("max_tokens")
    reasoning = body.get("reasoning") or {}
    return reasoning.get("effort"), (reasoning.get("summary"), body.get("include")), body.get("max_output_tokens")


@unittest.skipIf(PROBLEM is not None, f"OpenCode runtime unavailable: {PROBLEM}")
class OpenCodeEffortSmoke(unittest.TestCase):
    maxDiff = None

    def run_wire(self, wire, server_class, model):
        from auriga.cortex.agents.opencode.session import OpenCodeSession
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        router = Router(wire)
        server = server_class([], default=router).start()
        self.addCleanup(server.close)
        environment = mock.patch.dict(os.environ, {KEY_ENV: "host-secret-effort"})
        environment.start()
        self.addCleanup(environment.stop)
        store = Store(root, IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {
                "executable": EXECUTABLE, "context_limit_tokens": CONTEXT_LIMIT, "output_limit_tokens": OUTPUT_LIMIT,
                "upstream": {"wire_api": wire, "base_url": server.base_url, "api_key_env": KEY_ENV}}},
            "model": {"name": model, "effort": "low"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        self.assertIsNone(factory.backend_requirements(config))
        session = OpenCodeSession(root, store, lambda params: {"success": True, "contentItems": []}, config=config)
        self.addCleanup(session.close)
        session.start([], "Effort smoke instructions.")
        router.main_session = session.thread_id
        self.assertEqual(session.effort_choices, EFFORT_CHOICES[wire])
        for index, tier in enumerate(EFFORT_CHOICES[wire]):
            with self.subTest(wire=wire, tier=tier):
                router.seen.clear()
                session.set_effort(tier)
                session.run_turn(f"DELEGATE ({tier})")
                self.assertTrue(session.turn_done.wait(120), "turn did not end")
                self.assertEqual(session.state, "completed", session.turn_result)
                self.assertEqual(session.completed_final_answer(), "FINAL")
                kinds = [kind for kind, _ in router.seen]
                self.assertTrue({"main", "child", "compaction"} <= set(kinds), kinds)
                for kind, request in router.seen:
                    if kind == "compaction":          # OpenCode's summary prompt (new or update form)
                        self.assertIn("<conversation>", json.dumps(request.json))
                for kind, request in router.seen:
                    effort, companion, limit = effort_field(request.json, wire)
                    self.assertEqual(effort, tier, (kind, request.json))
                    if wire == "anthropic_messages":
                        self.assertEqual(companion, {"type": "adaptive", "display": "summarized"}, kind)
                    else:
                        self.assertEqual(companion, ("auto", ["reasoning.encrypted_content"]), kind)
                    self.assertEqual(limit, OUTPUT_LIMIT, kind)
                if index == 0:
                    self.assert_reasoning_replayed(wire, [request for kind, request in router.seen if kind == "main"])
        native = session.provenance()["native_effort"]
        self.assertEqual((native["variant"], native["applied"], native["channel"]),
                         (EFFORT_CHOICES[wire][-1], True, "prompt_variant"))
        report = session.close()
        self.assertTrue(report["clean_shutdown"], report)

    def assert_reasoning_replayed(self, wire, main_requests):
        """The step that thought and called ``task`` is replayed with its reasoning in the next main request."""
        after_tool = next(request.json for request in main_requests
                          if '"tool_result"' in json.dumps(request.json) or '"function_call_output"' in
                          json.dumps(request.json))
        if wire == "anthropic_messages":
            assistant = [message for message in after_tool["messages"] if message.get("role") == "assistant"][-1]
            thinking = [block for block in assistant["content"] if block.get("type") == "thinking"]
            self.assertEqual([(block.get("thinking"), block.get("signature")) for block in thinking],
                             [("I should delegate.", SIGNATURE)])
        else:
            reasoning = [item for item in after_tool["input"] if item.get("type") == "reasoning"]
            self.assertEqual([item.get("encrypted_content") for item in reasoning], [SIGNATURE])

    def test_anthropic_messages_every_tier(self):
        self.run_wire("anthropic_messages", FakeAnthropicServer, "claude-effort-model")

    def test_openai_responses_every_tier(self):
        self.run_wire("openai_responses", FakeResponsesServer, "gpt-effort-model")


if __name__ == "__main__":
    unittest.main()
