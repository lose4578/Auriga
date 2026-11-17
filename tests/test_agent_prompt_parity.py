"""Instruction parity of the first actor request across agent backends (work package P4-3).

Each real-binary class captures, once, the first actor generation request its backend sends upstream for the same
benchmark episode start (``support.prompt_capture``: real Workbench instruction/prompt/catalogue assembly, real
session, relay and pinned binary, fake loopback model) and pins what
``docs/development/reports/agent-backend-prompt-parity.md`` records:

* the Workbench ``instructions`` reach the model exactly once and byte-identical in the backend's channel (Codex:
  a developer message part; Claude Code: the tail of the second ``system`` block after the preserved preset;
  OpenCode: its system/developer text after the neutralised ``Instructions:`` header), followed on MCP backends by
  exactly the one tool-name-prefix sentence;
* the first user turn is the Workbench prompt, byte-identical;
* registry tool descriptions reach the model unchanged, and the per-backend schema transformations stay within the
  recorded sets.

Two catalogue defects found by the first version of this review are fixed and now asserted: the MCP ``tools/list``
projection removes top-level ``anyOf``/``oneOf``/``allOf`` (``task_plan``; Claude Code 2.1.321 silently drops such
tools) and Claude Code runs with ``CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH`` raised (default 4096 truncated
``geometry_query``, ``program_run``, ``robot_action``).

Skipped with the reason when a pinned runtime is absent (Codex: ``requires_codex_binary``; Claude Code:
``requires_claude_cli``; OpenCode: ``requires_opencode`` plus ``rg``). No real model, no network; the sandbox is off.
Run: ``PYTHONPATH=tests mamba run -n Auriga python -m unittest tests.test_agent_prompt_parity -v``.
"""
import shutil
import unittest
from types import SimpleNamespace

from support.prompt_capture import analyze, capture, text_blocks
from support.requirements import opencode_problem, requires_claude_cli, requires_codex_binary

from auriga.apps.gateway.workbench import Workbench
from auriga.core.config import Config
from auriga.cortex.agents.mcp_server import TOP_LEVEL_COMBINATORS, mcp_tool_catalog

# Words that would make the shared instructions backend-specific (they must name no harness and no tool prefix).
BACKEND_WORDS = ("Codex", "Claude", "OpenCode", "opencode", "tool_search", "mcp__", "auriga_")
# Schema keywords a client may drop on its wire (validation bounds, examples, conditionals); recorded, not corrected:
# the generated field guide in every description carries bounds, units and defaults (design 04 R3, P1-10).
RESPONSES_DROPPED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
                     "minItems", "maxItems", "uniqueItems", "minProperties", "pattern", "default", "examples",
                     "const", "dependentRequired", "prefixItems", "if", "then", "else", "format"}
# OpenCode on Responses rewrites const as a one-value enum and spells out implicit object/array shapes.
RESPONSES_ADDED = {"type", "properties", "enum", "additionalProperties"}
# Codex 0.160.0 additionally drops nested property descriptions and a nested oneOf of its two largest parameter
# schemas (robot_action, geometry_query.request); the tool-level description with the field guide stays intact.
CODEX_DROPPED = RESPONSES_DROPPED | {"description", "oneOf", "type"}


def actor_catalogue():
    """The registry actor catalogue of a default run (``Workbench.actor_tools`` without a Harness)."""
    return Workbench.actor_tools(SimpleNamespace(config=Config()))


class RegistrySchemaPortability(unittest.TestCase):
    # Anthropic tool input schemas do not accept a top-level anyOf/oneOf/allOf (Claude Code 2.1.321 drops the tool,
    # OpenCode on anthropic_messages forwards it to the provider). The registry keeps task_plan's top-level allOf; the
    # MCP transport projects it away and the rule stays in the description (tests/test_tool_registry_prose_bounds.py).
    def test_no_mcp_published_tool_schema_has_a_top_level_combinator(self):
        published = mcp_tool_catalog(actor_catalogue(), SimpleNamespace(present_text=str, present_data=dict))
        offending = {tool["name"]: sorted(key for key in TOP_LEVEL_COMBINATORS if key in tool["inputSchema"])
                     for tool in published}
        self.assertEqual({name: keys for name, keys in offending.items() if keys}, {})


class PromptParity:
    """Mixin: one capture per class; ``channel`` = (location, role) of the instructions block, ``first_turn`` =
    location of the first user turn."""
    backend = None
    channel = None
    first_turn = None
    native_identity = None      # first characters of the native base prompt

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.captured = capture(cls.backend)
        cls.facts = analyze(cls.captured)

    def tool(self, name):
        return next(tool for tool in self.facts["tools"] if tool["name"] == name)

    def test_instructions_delivered_once_byte_identical(self):
        facts = self.facts["instructions"]
        self.assertEqual(facts["occurrences"], 1, facts)
        self.assertTrue(facts["delivered_identical"], facts)
        self.assertEqual((facts["location"], facts["role"]), self.channel)

    def test_instructions_name_no_backend(self):
        text = self.captured.instructions.replace(self.captured.run_root, "<run>")   # sandbox off: the source path
        for word in BACKEND_WORDS:
            self.assertNotIn(word, text)

    def test_naming_note_is_the_only_addition(self):
        blocks = {location: text for location, _, text in text_blocks(self.captured)}
        text = blocks[self.facts["instructions"]["location"]]
        start = text.index(self.captured.instructions) + len(self.captured.instructions)
        if self.captured.naming_note:
            self.assertTrue(text[start:].startswith("\n\n" + self.captured.naming_note + "\n"), text[start:start + 300])
            self.assertEqual(sum(block.count(self.captured.naming_note) for block in blocks.values()), 1)
        else:
            self.assertEqual(text[start:], "")

    def test_first_user_turn_is_the_workbench_prompt(self):
        turn = self.facts["first_user_turn"]
        self.assertTrue(turn["identical"], turn)
        self.assertEqual((turn["location"], turn["role"]), (self.first_turn, "user"))

    def test_native_base_prompt_preserved(self):
        first = text_blocks(self.captured)[0][2]
        self.assertTrue(first.startswith(self.native_identity), first[:200])

    def assert_descriptions_identical(self, names):
        for name in names:
            tool = self.tool(name)
            self.assertTrue(tool["present"], name)
            self.assertTrue(tool["description_identical"], (name, tool["published_description_chars"],
                                                            tool["registry_description_chars"]))

    def assert_schema_changes_within(self, dropped, added):
        for tool in self.facts["tools"]:
            if tool["present"]:
                # MCP transport projection: top-level combinators are removed before the client sees the schema
                projected = set(TOP_LEVEL_COMBINATORS) if tool["top_level_combinators"] else set()
                self.assertLessEqual(set(tool["schema_removed"]), dropped | projected, tool["name"])
                self.assertLessEqual(set(tool["schema_added"]), added, tool["name"])


@requires_codex_binary
class CodexPromptParity(PromptParity, unittest.TestCase):
    backend = "codex"
    channel = ("input[2].content[0]", "developer")   # second developer message, first part; no `instructions` field
    first_turn = "input[6].content[0]"                # after Codex's own <environment_context> user message
    native_identity = "You are Codex"

    def test_eager_tools_unchanged_deferred_tools_behind_tool_search(self):
        """Design 3.8: registry tools with deferLoading reach Codex models only through native tool_search."""
        eager = [tool["name"] for tool in self.facts["tools"] if not tool["deferred"]]
        self.assert_descriptions_identical(eager)
        self.assertEqual({self.tool(name)["namespace"] for name in eager}, {"functions"})
        self.assertEqual([tool["name"] for tool in self.facts["tools"] if tool["deferred"] and tool["present"]], [])
        self.assertIn("tool_search", [tool["name"] for tool in self.facts["native_tools"]])
        self.assert_schema_changes_within(CODEX_DROPPED, {"type", "properties"})


@requires_claude_cli
class ClaudeCodePromptParity(PromptParity, unittest.TestCase):
    backend = "claude_code"
    channel = ("system[1]", "system")   # --append-system-prompt-file: tail of the preset's second system block
    first_turn = "messages[0]"
    native_identity = "You are Claude Code"

    def test_instructions_follow_the_preserved_preset(self):
        facts = self.facts["instructions"]
        self.assertGreater(facts["native_before_chars"], 0)
        self.assertEqual(facts["native_after_chars"], len("\n\n" + self.captured.naming_note + "\n"))

    def test_published_schemas_are_the_registry_schemas(self):
        for tool in self.facts["tools"]:
            if tool["present"]:
                self.assertTrue(tool["schema_portable_identical"], tool["name"])

    def test_no_tool_search_tool(self):
        self.assertNotIn("ToolSearch", [tool["name"] for tool in self.facts["native_tools"]])

    # Defect (P4-3 report, "task_plan schema"): the CLI drops MCP tools whose input schema has a top-level
    # anyOf/oneOf/allOf unless a server-side flag enables its normaliser; task_plan never reaches the model, although
    # every physical action requires a submitted plan.
    def test_every_registry_tool_is_published(self):
        self.assertEqual([tool["name"] for tool in self.facts["tools"] if not tool["present"]], [])

    # Defect (P4-3 report, "description cap"): CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH defaults to 4096 characters;
    # longer MCP descriptions end in "... [truncated]" (geometry_query, program_run, robot_action lose most of their
    # generated field guide).
    def test_tool_descriptions_unchanged(self):
        self.assert_descriptions_identical([tool["name"] for tool in self.facts["tools"] if tool["present"]])


OPENCODE_PROBLEM = opencode_problem() or (None if shutil.which("rg") else "ripgrep (rg) is not installed")


@unittest.skipIf(OPENCODE_PROBLEM is not None, f"pinned OpenCode runtime unavailable: {OPENCODE_PROBLEM}")
class OpenCodeAnthropicPromptParity(PromptParity, unittest.TestCase):
    backend = "opencode_anthropic"
    channel = ("system[0]", "system")
    first_turn = "messages[0].content[0]"
    native_identity = "You are OpenCode"

    def test_instructions_header_is_neutralised(self):
        self.assertEqual(self.facts["instructions"]["header"], "Instructions:")

    def test_every_tool_published_with_unchanged_description_and_schema(self):
        self.assert_descriptions_identical([tool["name"] for tool in self.facts["tools"]])
        for tool in self.facts["tools"]:
            self.assertTrue(tool["schema_portable_identical"], tool["name"])


@unittest.skipIf(OPENCODE_PROBLEM is not None, f"pinned OpenCode runtime unavailable: {OPENCODE_PROBLEM}")
class OpenCodeResponsesPromptParity(PromptParity, unittest.TestCase):
    backend = "opencode_responses"
    channel = ("input[0]", "developer")
    first_turn = "input[1].content[0]"
    native_identity = "You are OpenCode"

    def test_instructions_header_is_neutralised(self):
        self.assertEqual(self.facts["instructions"]["header"], "Instructions:")

    def test_every_tool_published_with_unchanged_description(self):
        self.assert_descriptions_identical([tool["name"] for tool in self.facts["tools"]])
        self.assert_schema_changes_within(RESPONSES_DROPPED, RESPONSES_ADDED)


if __name__ == "__main__":
    unittest.main()
