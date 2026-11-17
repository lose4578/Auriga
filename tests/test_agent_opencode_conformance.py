"""OpenCode binding of the shared backend conformance suite (design 04 §10.2; plan 07 §4.2), on both relay wires.

Drives the pinned ``opencode serve`` 1.19.5 named by ``AURIGA_OPENCODE_EXECUTABLE`` through the real relay
(``ModelRequestProxy`` + the ``opencode`` client profile of ``AnthropicMessagesWire`` / ``OpenAIResponsesWire``;
upstream = ``FakeAnthropicServer`` / ``FakeResponsesServer`` on loopback) and the real ``AurigaMcpServer``. One small
``TestCase`` per wire (``OpenCodeAnthropicConformance``, ``OpenCodeResponsesConformance``). No real model and no
network: OpenCode's XDG dirs, HOME and TMPDIR are inside the temporary run directory and its only credential is the
relay placeholder. Skipped with the reason unless the variable names the pinned binary (``requires_opencode``) and
``rg`` is installed (OpenCode refuses to start without it).

Run: ``AURIGA_OPENCODE_EXECUTABLE=<opencode-darwin-arm64>/bin/opencode PYTHONPATH=tests mamba run -n Auriga python -m
unittest tests.test_agent_opencode_conformance -v``. Item 22 (330 s provider delay) additionally needs
``AURIGA_SLOW_TESTS=1``.

OpenCode-specific behaviour pinned on top of the shared items (spikes 06 S6-S9, probe 09 §8, review T11):

* 9: the relay tombstone of the main session is in place before every ``POST /abort`` (relay block first).
* 10: the definite rejection is a real HTTP 400 of the running server (payload validation of ``prompt_async``).
* 17/18/26: the effort ``variant`` reaches every generation request - main, compaction and subagent - in the wire's
  body field (``output_config.effort`` / ``reasoning.effort``).
* 20: a steer joins at the next step boundary: it is the newest user input of the request right after the step it
  was sent during, next to that step's tool result.
* 21: after a restart the dangling tool call reaches the model as OpenCode's own ``[Tool execution was interrupted]``
  error result.
* 23: deferred delivery - OpenCode records the relay's verdict as an HTTP 400 ``APIError`` (not retryable): the
  first and only bytes it received were the final error status, never a 200 stream.
* 25 (b) is an expected failure: a diagnosed provenance defect (comment at the override).
* 27 is skipped (OpenCode never starts a turn by itself, ``capabilities.native_wakeup=False``).
* 28 (sandbox) runs where bubblewrap namespaces work (Linux): the server is reachable only through the inbound unix
  socket (no host TCP listener on its port) on top of the shared checks.
"""
import contextlib
import json
import os
import shutil
import time
import unittest
from unittest import mock

from support.agent_conformance import (MODEL, UPSTREAM_KEY_ENV, AgentBackendConformance, AnthropicMessagesView,
                                       INSTRUCTIONS_MARKER, OpenAIResponsesView, body_text)
from support.fake_model_servers import FakeAnthropicServer, FakeResponsesServer, ToolCall, Usage
from support.requirements import OPENCODE_EXECUTABLE_ENV, requires_opencode

from auriga.core.config import Config
from auriga.cortex.agents import factory
from auriga.cortex.agents.opencode.client import OpenCodeClient, OpenCodeTransportError
from auriga.cortex.agents.opencode.session import OpenCodeSession

EXECUTABLE = os.environ.get(OPENCODE_EXECUTABLE_ENV)
RIPGREP_PROBLEM = None if shutil.which("rg") else "ripgrep (rg) is not installed (OpenCode refuses to start without it)"
SESSION_HEADER = "x-opencode-session-id"     # OpenCode's per-session request header (spike S6.2)
COMPACTION_MARK = "<conversation>"           # OpenCode's tool-free summary request serialises the history in it
INTERRUPTED_TEXT = "[Tool execution was interrupted]"   # OpenCode's repair of a dangling tool call (spike S8)
EFFORT = "low"
# Auto-compaction: OpenCode compacts once a finished step reports tokens.total >= context - output (spike S9).
CONTEXT_LIMIT, OUTPUT_LIMIT = 40000, 8000


class ResponsesServer(FakeResponsesServer):
    """``FakeResponsesServer`` that reports ``total_tokens: null`` when a scripted counter is invalid (negative):
    the stock fake would fold the invalid value into a plausible total. The pinned client requires a number for
    ``input_tokens`` (a null one fails its stream parse and is re-requested, measured), so the unknown-counter case
    of item 16 reports an invalid number instead."""

    def _usage(self, u):
        usage = super()._usage(u)
        if min(u.input_tokens, u.cache_read_input_tokens, u.output_tokens) < 0:
            usage["total_tokens"] = None
        return usage


class RecordingClient(OpenCodeClient):
    """The real client; every ``abort`` records whether the relay thread of the aborted session was tombstoned
    (``ModelRequestProxy.cancelled_threads``) when the abort was sent."""

    def __init__(self, session, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session, self.aborts = session, []

    def abort(self, session_id, *, timeout_s=None):
        proxy = self.session.model_proxy
        self.aborts.append(dict(session_id=session_id,
                                tombstoned=proxy is not None and session_id in proxy.cancelled_threads))
        return super().abort(session_id, timeout_s=timeout_s)


@requires_opencode
@unittest.skipIf(RIPGREP_PROBLEM is not None, RIPGREP_PROBLEM)
class _OpenCodeConformance(AgentBackendConformance):
    """Binding shared by both wires; a subclass names the wire (``view``, ``fake_model_class``, ``auth_scheme``)."""
    backend_kind = "opencode"
    auth_scheme = None
    effort_field = None          # body field of the effort variant on this wire
    tool_prefix = "auriga_"
    expect_steer = True
    expect_native_wakeup = False
    expect_subagents = True
    forbidden_tool_names = frozenset({"webfetch", "websearch", "codesearch", "question"})
    identity_fields = ("cli_version", "cli_executable", "cli_sha256", "native", "resumed")
    native_call_id_source = None          # OpenCode sends no call id over MCP: callIdSource is "mcp_request"
    # OpenCode renders the instructions file as "Instructions from: <private path>"; egress keeps the path out.
    private_markers = ("Instructions from:", "private/opencode")
    compaction_usage = Usage(input_tokens=CONTEXT_LIMIT)
    compaction_trigger_steps = 1

    def setUp(self):
        self._sessions = []
        super().setUp()

    def make_session(self, run_dir, store, tool_handler, fake_model, *, model=MODEL, compact=False, sandbox=False):
        opencode = {"executable": EXECUTABLE, "upstream": {
            "wire_api": self.view.name, "base_url": fake_model.base_url, "api_key_env": UPSTREAM_KEY_ENV,
            "auth_scheme": self.auth_scheme, "provider": "conformance"}}
        if compact:
            opencode.update(context_limit_tokens=CONTEXT_LIMIT, output_limit_tokens=OUTPUT_LIMIT)
        config = Config.from_dict({"agent": {"backend": "opencode", "opencode": opencode},
                                   "model": {"name": model, "effort": EFFORT},
                                   "features": {"self_briefing": False, "visual_review_model": None,
                                                "sandbox_enabled": sandbox}})
        if sandbox:   # item 28: the launch gate the gateway and the queue apply accepts the sandboxed config
            factory.backend_requirements(config)
        holder = []

        def client_factory(base_url, password, unix_socket=None):
            return RecordingClient(holder[0], base_url, password, timeout_s=holder[0].request_timeout_s,
                                unix_socket=unix_socket)
        session = OpenCodeSession(run_dir, store, tool_handler, config=config, client_factory=client_factory)
        holder.append(session)
        self._sessions.append(session)
        return session

    def classify_request(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        if not self.view.tools(body):
            return "compaction" if COMPACTION_MARK in body_text(request) else "helper"
        mains = {session.thread_id for session in list(self._sessions) if session.thread_id}
        native = request.header(SESSION_HEADER)
        if native and native not in mains:
            return "subagent"
        if INSTRUCTIONS_MARKER in self.view.system_text(body):
            return "main"
        return "helper"

    # native tool calls in OpenCode's vocabulary (build agent, non-gpt model id: bash/write/task)
    def shell_call(self, command, *, background=False):
        """``bash``; ``background``: the command detaches with ``&`` (OpenCode's bash tool has no background mode)."""
        if background:
            command = f"{command} >/dev/null 2>&1 &"
        return ToolCall("bash", {"command": command, "description": "Run a conformance shell step"})

    def write_call(self, path, content):
        return ToolCall("write", {"filePath": str(path), "content": content})

    def subagent_call(self, prompt):
        return ToolCall("task", {"description": "Inspect the scene", "prompt": prompt, "subagent_type": "general"})

    # provenance readers
    def runtime_record(self, provenance):
        from auriga.cortex.agents.opencode.config import OPENCODE_VERSION, probe_cli_version
        self.assertIsNone(provenance["sdk"])
        cli = provenance["cli"]
        self.assertEqual(probe_cli_version(cli["executable"]), cli["version"], "recorded version != the binary's")
        return dict(version=cli["version"], executable=cli["executable"], sha256=cli["sha256"],
                    pinned_version=OPENCODE_VERSION)

    def denied_tools(self, provenance):
        return {item["name"] for item in provenance["native_tools"]["disabled"]}

    def init_tool_names(self, provenance):
        return set(provenance["native_tools"]["enabled"] or ())

    def interrupted_result(self, result):
        """OpenCode's own repair text; flagged as an error where the wire has a flag."""
        return INTERRUPTED_TEXT in result.text and self.view.flagged_error(result) is not False

    def unknown_usage(self):
        """An input counter reported as an invalid number: the pinned client refuses a null ``input_tokens`` on
        both wires (stream parse failure, re-requested), the relay treats a negative counter as unknown."""
        return Usage(input_tokens=-1, output_tokens=6)

    def await_history_persisted(self, session, native_call_id, timeout=15):
        """Wait until OpenCode's own session history (``GET /session/{id}/message``, its database) has the model's
        tool call ``native_call_id`` as a tool part."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for message in session.client.messages(session.thread_id):
                if any(part.get("callID") == native_call_id for part in message.get("parts") or ()):
                    return
            time.sleep(0.05)
        self.fail(f"OpenCode never recorded the tool call {native_call_id} in its history")

    # fault injection
    @contextlib.contextmanager
    def definite_rejection(self, session):
        """The prompt reaches the running server with a ``variant`` of the wrong type: OpenCode's payload validation
        answers HTTP 400 (a definite rejection, nothing admitted; measured)."""
        client = session.client

        def rejected(session_id, text=None, **_options):
            client._request("POST", client._session_path(session_id) + "/prompt_async",
                            {"parts": [{"type": "text", "text": text}], "variant": 0}, want_body=False)
            raise AssertionError("OpenCode admitted a prompt with an invalid variant")
        with mock.patch.object(client, "prompt_async", rejected):
            yield

    @contextlib.contextmanager
    def unknown_admission(self, session):
        """``prompt_async`` times out without an answer (an ``OSError`` of the client: admission unknown). The
        request is never written, so any later model call would be a retry."""
        with mock.patch.object(session.client, "prompt_async", side_effect=OpenCodeTransportError("TimeoutError")):
            yield

    @contextlib.contextmanager
    def route_outside_traffic(self, proxy_url):
        """Every proxy variable of the server environment points at ``proxy_url`` (loopback stays direct)."""
        from auriga.cortex.agents.opencode import config as opencode_config
        values = {name: proxy_url for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy",
                                               "all_proxy")}
        values.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        original = opencode_config.build_launch

        def build_launch(*args, **kwargs):
            launch = original(*args, **kwargs)
            launch.env.update(values)
            return launch
        with mock.patch.object(opencode_config, "build_launch", build_launch):
            yield

    @contextlib.contextmanager
    def default_tool_set(self):
        """Launch without the host's permission denials: the build agent's own default tool set."""
        from auriga.cortex.agents.opencode import config as opencode_config
        with mock.patch.dict(opencode_config._DENIED_TOOLS, {}, clear=True):
            yield

    # -- OpenCode-specific checks on top of the shared items ----------------------------------------------------
    def _effort(self, request):
        node = request.json if isinstance(request.json, dict) else {}
        for key in self.effort_field.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        return node

    def assert_effort(self, requests, kind):
        self.assertTrue(requests, f"no {kind} request")
        self.assertEqual([self._effort(request) for request in requests], [EFFORT] * len(requests),
                         f"a {kind} request lacks {self.effort_field}={EFFORT}")

    def test_interrupt_model_and_tool(self):
        """Item 9 + relay block first: every ``POST /abort`` of the main session was sent with its thread already
        tombstoned in the relay."""
        super().test_interrupt_model_and_tool()
        (session,) = self._sessions
        aborts = [abort for abort in session.client.aborts if abort["session_id"] == session.thread_id]
        self.assertGreaterEqual(len(aborts), 2, aborts)          # one per interrupt (plus any repeated abort)
        self.assertTrue(all(abort["tombstoned"] for abort in aborts), aborts)

    def test_provenance(self):
        """Item 17 + the effort variant: provenance names it; every main request carries it."""
        super().test_provenance()
        (session,) = self._sessions
        native = session.provenance()["native_effort"]
        self.assertEqual((native["variant"], native["applied"], native["channel"], native["wire_field"]),
                         (EFFORT, True, "prompt_variant", self.effort_field))
        self.assert_effort(self.main_requests(), "main")

    def test_compaction_changes_epoch(self):
        """Item 18 + the effort variant on the native compaction request and on every main request."""
        super().test_compaction_changes_epoch()
        self.assert_effort(self.router.requests["compaction"], "compaction")
        self.assert_effort(self.main_requests(), "main")

    def test_steer_joins_turn(self):
        """Item 20 + OpenCode's join point: each steer is the newest user input of the request right after the step
        it was sent during (the gated model call, then the slow tool), next to that step's tool result."""
        super().test_steer_joins_turn()
        mains = self.main_requests()
        firsts = [next(index for index, request in enumerate(mains) if marker in body_text(request))
                  for marker in ("STEER-ONE-20", "STEER-TWO-20")]
        self.assertEqual(firsts, [1, 2], "a steer did not join at the next step boundary")
        for index, marker, output in ((1, "STEER-ONE-20", "scene: one red cube at x=1"),
                                      (2, "STEER-TWO-20", "slow motion finished")):
            self.assertIn(marker, self.view.last_user_text(mains[index].json))
            self.assertIn(output, self.server.find_tool_results(mains[index])[-1].text)

    def test_truncated_upstream_no_storm(self):
        """Item 23 + deferred delivery: OpenCode's message error is the relay's HTTP 400 (not retryable)."""
        super().test_truncated_upstream_no_storm()
        (session,) = self._sessions
        errors = [message["info"]["error"] for message in session.client.messages(session.thread_id)
                  if message["info"].get("role") == "assistant" and message["info"].get("error")]
        self.assertEqual(len(errors), 1, errors)
        (error,) = errors
        self.assertEqual((error["name"], error["data"]["statusCode"], error["data"]["isRetryable"]),
                         ("APIError", 400, False), error)
        self.assertEqual(json.loads(error["data"]["responseBody"])["error"]["type"], "invalid_request_error")

    def test_subagent_tool_call(self):
        """Item 26 + the effort variant on every subagent request."""
        super().test_subagent_tool_call()
        self.assert_effort(self.router.requests["subagent"], "subagent")

    def test_sandbox_namespace(self):
        """Item 28 + the inbound direction: the host reached the sandboxed server only through its unix socket."""
        original = OpenCodeSession._prepare_sandbox
        inbound = []

        def recording(session, port):
            prefix, env, unix_socket = original(session, port)
            inbound.append((port, unix_socket))
            return prefix, env, unix_socket
        with mock.patch.object(OpenCodeSession, "_prepare_sandbox", recording):
            super().test_sandbox_namespace()
        (session,) = self._sessions
        ((port, unix_socket),) = inbound
        self.assertEqual(session.client.unix_socket, unix_socket)
        self.assertTrue(unix_socket.endswith(f"/inbound/{port}.sock"), unix_socket)
        self.assertIn("--unshare-net", session.process.args)


class OpenCodeAnthropicConformance(_OpenCodeConformance, unittest.TestCase):
    fake_model_class = FakeAnthropicServer
    view = AnthropicMessagesView()
    auth_scheme = "x-api-key"
    effort_field = "output_config.effort"


class OpenCodeResponsesConformance(_OpenCodeConformance, unittest.TestCase):
    fake_model_class = ResponsesServer
    view = OpenAIResponsesView()
    auth_scheme = "bearer"
    effort_field = "reasoning.effort"


if __name__ == "__main__":
    unittest.main()
