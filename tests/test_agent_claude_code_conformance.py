"""Claude Code binding of the shared backend conformance suite (design 04 §10.2; plan 07 §4.2).

Drives the pinned bundled Claude Code CLI (``claude-agent-sdk==0.2.180``, CLI 2.1.321) through the real relay
(``ModelRequestProxy`` + ``AnthropicMessagesWire(client="claude_code")``; upstream = ``FakeAnthropicServer`` on
loopback) and the real ``AurigaMcpServer``. No real model and no network: the CLI's HOME/CLAUDE_CONFIG_DIR/TMPDIR
are inside the temporary run directory and its only credential is the relay placeholder. Skipped with the reason
when the pinned runtime is absent (``requires_claude_cli``).

Run: ``PYTHONPATH=tests mamba run -n Auriga python -m unittest tests.test_agent_claude_code_conformance -v``.
Item 22 (330 s provider delay) additionally needs ``AURIGA_SLOW_TESTS=1``; item 28 (sandbox) needs a host where
bubblewrap namespaces work (Linux; skipped elsewhere, a failure with ``AURIGA_REQUIRE_SANDBOX_TESTS=1``).
"""
import concurrent.futures
import contextlib
import time
import unittest
from unittest import mock

from support.agent_conformance import (MODEL, UPSTREAM_KEY_ENV, AgentBackendConformance, AnthropicMessagesView,
                                       INSTRUCTIONS_MARKER)
from support.fake_model_servers import FakeAnthropicServer, ToolCall, Usage
from support.requirements import requires_claude_cli

from auriga.core.config import Config
from auriga.cortex.agents import factory

# Text Claude Code appends to the last user message of its own auto-compaction summary request (spike S5).
COMPACTION_MARK = "create a detailed summary"
# Request header that only subagent requests carry (probe R4-0 §3).
SUBAGENT_HEADER = "x-claude-code-agent-id"
# Init-message name -> request name of native tools that differ between the two (probe R4-0 §1).
INIT_TO_REQUEST_NAMES = {"Task": "Agent"}


@requires_claude_cli
class ClaudeCodeConformance(AgentBackendConformance, unittest.TestCase):
    backend_kind = "claude_code"
    fake_model_class = FakeAnthropicServer
    view = AnthropicMessagesView()
    tool_prefix = "mcp__auriga__"
    expect_steer = True
    expect_native_wakeup = True
    expect_subagents = True
    forbidden_tool_names = frozenset({"WebSearch", "WebFetch", "AskUserQuestion", "EnterPlanMode", "ExitPlanMode"})
    identity_fields = ("sdk_version", "cli_version", "cli_executable", "cli_sha256", "native", "resumed")
    native_call_id_source = "native_tool_use_id"
    # A local slash command: no generation call, but a burst of count_tokens requests (probe R4-0 §1).
    auxiliary_prompt = "/context"
    auxiliary_routes = ("/v1/messages/count_tokens",)
    # Claude Code adds "[Image: source: <private cache path>]" after every MCP image (spike S2); egress rewrites it.
    private_markers = ("[Image: source:",)
    # A main answer reporting ~90k prompt tokens against a 100k auto-compact window (spike S5 compacted at 75k with
    # the same window). Measured in T6: the CLI compacts after the SECOND consecutive such answer, before its next
    # model request, never after the first one.
    compaction_usage = Usage(input_tokens=2000, output_tokens=200, cache_read_input_tokens=88000)
    compaction_trigger_steps = 2

    def make_session(self, run_dir, store, tool_handler, fake_model, *, model=MODEL, compact=False, sandbox=False):
        from auriga.cortex.agents.claude_code.session import ClaudeCodeSession
        claude = {"upstream": {"base_url": fake_model.base_url, "api_key_env": UPSTREAM_KEY_ENV,
                               "auth_scheme": "x-api-key"}}
        if compact:
            claude["auto_compact_window_tokens"] = 100000
        config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": claude},
                                   "model": {"name": model, "effort": "low"},
                                   "features": {"self_briefing": False, "visual_review_model": None,
                                                "sandbox_enabled": sandbox}})
        if sandbox:   # item 28: the launch gate the gateway and the queue apply accepts the sandboxed config
            factory.backend_requirements(config)
        return ClaudeCodeSession(run_dir, store, tool_handler, config=config)

    def classify_request(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        if not body.get("tools"):
            return "helper"
        if COMPACTION_MARK in self.view.last_user_text(body):
            return "compaction"
        if request.header(SUBAGENT_HEADER):
            return "subagent"
        if INSTRUCTIONS_MARKER in self.view.system_text(body):
            return "main"
        return "helper"

    # native tool calls in Claude Code's vocabulary
    def shell_call(self, command, *, background=False):
        arguments = {"command": command, "description": "Run a conformance shell step"}
        if background:
            arguments["run_in_background"] = True
        return ToolCall("Bash", arguments)

    def write_call(self, path, content):
        return ToolCall("Write", {"file_path": str(path), "content": content})

    def subagent_call(self, prompt):
        return ToolCall("Agent", {"description": "Inspect the scene", "prompt": prompt,
                                  "subagent_type": "general-purpose", "run_in_background": False})

    # provenance readers
    def runtime_record(self, provenance):
        from auriga.cortex.agents.claude_code import options
        self.assertEqual(provenance["sdk"], {"name": options.SDK_PACKAGE, "version": options.SDK_VERSION})
        cli = provenance["cli"]
        return dict(version=cli["version"], executable=cli["executable"], sha256=cli["sha256"],
                    pinned_version=options.CLI_VERSION)

    def denied_tools(self, provenance):
        return {INIT_TO_REQUEST_NAMES.get(item["name"], item["name"])
                for item in provenance["native_tools"]["disabled"]}

    def init_tool_names(self, provenance):
        return {INIT_TO_REQUEST_NAMES.get(name, name) for name in provenance["native_tools"]["enabled"] or ()}

    def interrupted_result(self, result):
        return result.is_error is True and "interrupted" in result.text.lower() and "unknown" in result.text.lower()

    def await_history_persisted(self, session, native_call_id, timeout=15):
        """The CLI writes its session JSONL (``private/claude-config/projects/*/<session>.jsonl``) about a second
        after the first message; a CLI killed before that leaves nothing to resume (``resume_native_session_missing``,
        measured T6). Wait until the file records the model's tool_use id."""
        projects = session.root / "private" / "claude-config" / "projects"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for path in projects.glob(f"*/{session.thread_id}.jsonl"):
                if native_call_id in path.read_text(errors="replace"):
                    return
            time.sleep(0.05)
        self.fail(f"the CLI never persisted the tool call {native_call_id} to its session history")

    # fault injection
    @contextlib.contextmanager
    def definite_rejection(self, session):
        """The pre-flight connection check fails: nothing is written to the CLI (plan 07 §2.8 row 3)."""
        with mock.patch.object(session, "_connected", return_value=False):
            yield

    @contextlib.contextmanager
    def unknown_admission(self, session):
        """The prompt's stdin write is scheduled but never acknowledged (a wedged transport): unknown admission
        after ``submit_timeout_s``. The write itself never happens, so any later model call would be a retry."""
        def stalled(coroutine):
            coroutine.close()
            return concurrent.futures.Future()
        with mock.patch.object(session, "_submit", side_effect=stalled), \
                mock.patch.object(session, "submit_timeout_s", 0.5):
            yield

    @contextlib.contextmanager
    def route_outside_traffic(self, proxy_url):
        """Every proxy variable of the CLI environment points at ``proxy_url`` (loopback stays direct)."""
        from auriga.cortex.agents.claude_code import options
        values = {name: proxy_url for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy",
                                               "all_proxy")}
        values.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        with mock.patch.dict(options._DISABLE_FLAGS, values):
            yield

    @contextlib.contextmanager
    def default_tool_set(self):
        """Launch without the host denylist and without the web-fetch kill switch: the CLI's own default set."""
        from auriga.cortex.agents.claude_code import options
        flags = {key: value for key, value in options._DISABLE_FLAGS.items() if key != "CLAUDE_CODE_DISABLE_WEB_FETCH"}
        with mock.patch.object(options, "DISALLOWED_TOOLS", ()), \
                mock.patch.dict(options._DISABLE_FLAGS, flags, clear=True):
            yield


if __name__ == "__main__":
    unittest.main()
