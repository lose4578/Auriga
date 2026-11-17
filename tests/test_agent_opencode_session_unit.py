"""OpenCodeSession against the scripted fake ``opencode serve`` (plan 07 §3, §3.9 failure paths, §4.1; design R4.3).

No OpenCode binary and no model: ``tests/support/fake_opencode.FakeOpenCodeServer`` answers the HTTP API and drives
``/event``; the "server process" is a tiny executable script (``--version`` prints the pin, ``serve`` sleeps, and can
start a child in its own session to model OpenCode's detached bash-tool shells), so start/close exercise the real
launch builder, version probe and process-tree teardown. The relay points at a closed port (no test sends a model
request) and the MCP endpoint is the real ``AurigaMcpServer`` driven by FakeMcpClient.
"""
import json
import os
import signal
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from support.fake_mcp_client import FakeMcpClient
from support.fake_opencode import PASSWORD, FakeOpenCodeServer

from auriga.core.config import Config, ConfigError
from auriga.core.storage.store import Store
from auriga.cortex.agents import factory
from auriga.cortex.agents.base import AgentCapabilityError, AgentRejected, AgentTransportUnknown
from auriga.cortex.agents.isolation import VersionMismatch, process_tree_snapshot
from auriga.cortex.agents.opencode.client import OpenCodeClient, OpenCodeTransportError
from auriga.cortex.agents.opencode.config import OpenCodeConfigError
from auriga.cortex.agents.opencode.session import OpenCodeSession, naming_note

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
KEY_ENV = "AURIGA_TEST_OPENCODE_UPSTREAM_KEY"
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
TOOLS = [{"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": SCHEMA},
         {"type": "function", "name": "move", "description": "Stateful motion.", "inputSchema": SCHEMA}]
ABORTED = {"name": "MessageAbortedError", "data": {"message": "Aborted"}}
API_400 = {"name": "APIError", "data": {"message": "provider text that must not be stored", "statusCode": 400}}

SERVE_MODES = {
    "sleep": "time.sleep(3600)",
    "exit": "sys.exit(3)",
    # A bash tool call that leaves `cmd &` and a setsid child behind and returns at once: the shell exits, both are
    # re-parented away from the server tree before any refresh can see them; their pids land in the cwd (actor dir).
    "orphans": ("subprocess.run(['/bin/sh', '-c', 'sleep 3600 >/dev/null 2>&1 & echo $! > orphan.pid; '"
                " + sys.executable + ' -c \"import os, time; os.setsid(); time.sleep(3600)\" >/dev/null 2>&1 &"
                " echo $! > detached.pid'], start_new_session=True)\ntime.sleep(3600)"),
    # Like OpenCode's bash tool: a child in its own session and process group, then the server keeps running.
    "detached": ("subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)'], start_new_session=True,"
                 " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\ntime.sleep(3600)"),
}


def fake_binary(directory, *, version="1.19.5", serve="sleep"):
    """An executable standing in for the OpenCode platform binary (``--version`` / ``serve``)."""
    path = Path(directory) / f"opencode-{version}-{serve}"
    path.write_text(f"#!{sys.executable}\nimport subprocess, sys, time\n"
                    f"if sys.argv[1:2] == ['--version']:\n    print({version!r})\n    sys.exit(0)\n"
                    f"{SERVE_MODES[serve]}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def relay_post(session, thread):
    """POST one Anthropic Messages generation request to the session's relay as OpenCode would (session header
    ``x-opencode-session-id``); returns ``(status, body)``. Used only where the relay must refuse before
    forwarding (the upstream of these tests is a closed port)."""
    import http.client
    from urllib.parse import urlsplit
    url = urlsplit(session.model_proxy.base_url)
    body = json.dumps({"model": "claude-test-model", "max_tokens": 8, "stream": True, "tools": [
        {"name": "auriga_look", "description": "d", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "hi"}]}).encode()
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    try:
        connection.request("POST", "/v1/messages", body=body, headers={
            "Content-Type": "application/json", "x-opencode-session-id": thread, "x-api-key": "auriga-local-relay",
            "anthropic-version": "2023-06-01"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class ScriptedFake(FakeOpenCodeServer):
    """The support fake plus four test knobs: ``events_blocked`` (``/event`` answers 503, so events are lost),
    ``events_muted`` (connected streams stay open but receive nothing: a silently lost stream), ``abort_status`` (``POST /abort`` answers this HTTP status) and ``after_snapshot``, a one-shot hook run right after
    the next ``GET /session/{id}/message`` answer was sent (on the fake's handler thread, with the fake's lock held:
    it may push events and edit messages, but must not wait for another request)."""
    events_blocked = events_muted = False
    abort_status = None
    after_snapshot = None
    # GET /experimental/tool?provider&model (offered tools, incl. the internal ``invalid``) and GET /agent (merged permission rulesets), as 1.19.5 answers them
    # for the generated config (shapes from opencode-openapi.json; only the build agent matters).
    tool_ids = ["invalid", "bash", "codesearch", "edit", "glob", "grep", "question", "read", "task", "todowrite", "webfetch",
                "websearch", "write"]
    agents = [{"name": "build", "mode": "primary", "permission": [
        {"permission": "*", "pattern": "*", "action": "allow"},
        *({"permission": name, "pattern": "*", "action": "deny"}
          for name in ("webfetch", "websearch", "codesearch", "question"))]}]

    def push(self, event_type, properties=None):
        if self.events_muted:
            return {"type": event_type, "properties": properties or {}}
        return super().push(event_type, properties)

    def _event_stream(self, handler):
        if self.events_blocked:
            return self._json(handler, 503, {"error": "unavailable"})
        return super()._event_stream(handler)

    def abort_session(self, session_id):
        if self.abort_status is None:
            super().abort_session(session_id)

    def _handle(self, handler):
        if handler.command == "GET" and handler.path.split("?")[0] in ("/experimental/tool", "/agent"):
            with self.lock:
                self.requests.append({"method": "GET", "path": handler.path, "body": None, "headers": {}})
            if handler.path.startswith("/experimental/tool"):
                return self._json(handler, 200, [{"id": name, "description": "d", "parameters": {}}
                                                 for name in self.tool_ids])
            return self._json(handler, 200, self.agents)
        if self.abort_status is not None and handler.command == "POST" and handler.path.endswith("/abort"):
            length = int(handler.headers.get("Content-Length") or 0)
            if length:
                handler.rfile.read(length)
            with self.lock:
                self.requests.append({"method": "POST", "path": handler.path, "body": None, "headers": {}})
            return self._json(handler, self.abort_status, {"name": "UnknownError"})
        return super()._handle(handler)

    def _json(self, handler, status, payload):
        snapshot = handler.command == "GET" and handler.path.endswith("/message") and isinstance(payload, list)
        super()._json(handler, status, payload)
        hook = self.after_snapshot if snapshot else None
        if hook is not None:
            self.after_snapshot = None
            hook()

    # ---- scripted conversation helpers -------------------------------------------------------------------------
    def last_user(self, session_id):
        with self.lock:
            return [m for m in self.sessions[session_id]["messages"] if m["info"]["role"] == "user"][-1]

    def assistant(self, session_id, parent=None, **info):
        parent = parent or self.last_user(session_id)["info"]["id"]
        return self._add_message(session_id, "assistant", parent, modelID="fake-model", providerID="relay",
                                 agent="build", mode="build", **info)

    def part(self, message, part):
        return self._add_part(message, part)

    def update_part(self, part, **fields):
        with self.lock:
            part.update(fields)
        self.push("message.part.updated", {"sessionID": part["sessionID"], "part": part})

    def update_info(self, message, **fields):
        with self.lock:
            message["info"].update(fields)
        self.push("message.updated", {"sessionID": message["info"]["sessionID"], "info": message["info"]})

    def go_idle_silently(self, session_id):
        with self.lock:
            self.busy.pop(session_id, None)


class SessionCase(unittest.TestCase):
    maxDiff = None
    client_timeout = 2.0

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name).resolve()
        self.root = self.dir / "run"
        self.root.mkdir()
        self.store = Store(self.root, IDENTITY)
        self.addCleanup(self.store.close)
        environment = mock.patch.dict(os.environ, {KEY_ENV: "host-secret"})
        environment.start()
        self.addCleanup(environment.stop)
        self.fake = ScriptedFake().start()
        self.addCleanup(self.fake.close)
        self.executable = fake_binary(self.dir)
        self.handler_calls, self.disconnects, self.resets = [], [], []
        self.clients = []
        # GET /config/providers of the fake: by default it serves the model variants the session's generated config
        # declares (as the real 1.19.5 server does, verified against the binary); a test sets this to serve others.
        self.served_variants = None

    # ---- helpers --------------------------------------------------------------------------------------------
    def config(self, *, executable=None, effort="high", upstream=None, **opencode):
        return Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {
                "executable": executable or self.executable,
                "upstream": upstream or {"wire_api": "anthropic_messages", "base_url": "http://127.0.0.1:9",
                                         "api_key_env": KEY_ENV, "auth_scheme": "x-api-key", "provider": "fake"},
                **opencode}},
            "model": {"name": "claude-test-model", "effort": effort},
            "features": {"self_briefing": False, "visual_review_model": None}})

    def client_factory(self, base_url, password):
        client = OpenCodeClient(self.fake.url, PASSWORD, timeout_s=self.client_timeout)
        self.clients.append((base_url, password))
        launch = self.session.launch                          # built just before the client
        model = launch.config["model"].split("/", 1)[1]
        entry = launch.config["provider"]["relay"]["models"][model]
        variants = self.served_variants if self.served_variants is not None else entry.get("variants", {})
        self.fake.providers = {"providers": [{"id": "relay", "models": {model: dict(entry, variants=variants)}}],
                               "default": {"relay": model}}
        return client

    def tool_handler(self, params):
        self.handler_calls.append(params)
        return {"success": True, "contentItems": [{"type": "inputText", "text": "ok " + params["tool"]}]}

    def make(self, config=None, *, launch_options=None, client_factory=None, **overrides):
        # The --version probe of the stand-in binary is exercised by the version tests only (it costs a process).
        options = dict(ripgrep="/bin/echo", managed_config_paths=[], check_version=False)
        options.update(launch_options or {})
        session = OpenCodeSession(self.root, self.store, self.tool_handler, config=config or self.config(),
                                  client_factory=client_factory or self.client_factory, launch_options=options)
        session.on_disconnect = lambda: self.disconnects.append(session.state)
        session.on_context_reset = self.resets.append
        session.tool_parallel_hints = {"look": True, "move": False}
        session.tool_parallel_safe = lambda tool, arguments: tool == "look"
        session.watchdog_s = 30.0
        for key, value in overrides.items():
            setattr(session, key, value)
        self.session = session
        self.addCleanup(session.close)
        return session

    def started(self, resume=None, *, launch_options=None, **overrides):
        session = self.make(launch_options=launch_options, **overrides)
        session.start(TOOLS, "Be careful.", resume)
        return session

    def events(self, kind):
        return self.store.events_by_type([kind], limit=1000)

    def raw(self, channel):
        path = self.root / "agent-raw.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [record for record in records if record["channel"] == channel]

    def rows(self, kind=None, thread=None):
        return [row for row in self.store.states(self.store.TRANSCRIPT_KIND)
                if (kind is None or row["type"] == kind) and (thread is None or row.get("thread_id") == thread)]

    def prompts(self):
        return [item for item in self.fake.requests if item["path"].endswith("/prompt_async")]

    def finish(self, session, timeout=5.0):
        self.assertTrue(session.turn_done.wait(timeout), "turn did not end")
        return session.state

    def running_turn(self, session, text="Do it."):
        turn = session.run_turn(text)
        self.assertTrue(wait_until(lambda: self.fake.busy.get(session.thread_id)), "prompt never admitted")
        return turn

    def mcp(self, session):
        client = FakeMcpClient(session.bridge.url, headers=session.bridge.headers)
        client.handshake()
        return client


class StartTests(SessionCase):
    def test_fresh_start_writes_identity_launch_and_threads(self):
        session = self.started(launch_options=dict(check_version=True))
        self.assertEqual(session.state, "idle")
        thread = session.thread_id
        self.assertIn(thread, self.fake.sessions)
        identity = self.store.read_state("runtime", "agent")
        self.assertEqual((identity["backend"], identity["thread_id"], identity["cli_version"], identity["wire_api"],
                          identity["counting"], identity["clean_shutdown"], identity["resumed"], identity["provider"]),
                         ("opencode", thread, "1.19.5", "anthropic_messages",
                          "delivered_usable_generation_completions_v1", False, False, "fake"))
        self.assertEqual(identity["pid"], session.process.pid)
        (role, process), = session.owned_processes()
        self.assertEqual((role, process.poll()), ("opencode", None))
        created = [item for item in self.fake.requests if item["method"] == "POST" and item["path"] == "/session"]
        self.assertEqual([item["body"] for item in created], [{"title": "auriga-actor"}])
        paths = [item["path"] for item in self.fake.requests]
        self.assertEqual(paths[:6], ["/global/health", "/doc",
                                     "/experimental/tool?provider=relay&model=claude-test-model", "/agent",
                                     "/config/providers", "/session"])
        # Launch: relay + MCP + read-only instructions file and config dirs; no host credential in the environment.
        launch = session.launch
        config = json.loads(launch.env["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(config["provider"]["relay"]["options"]["baseURL"], session.model_proxy.base_url + "/v1")
        self.assertEqual(config["mcp"]["auriga"]["url"], session.bridge.url)
        self.assertEqual(config["model"], "relay/claude-test-model")
        text = launch.instructions_path.read_text()
        self.assertEqual(text, "Be careful.\n\n" + naming_note() + "\n")
        self.assertEqual(stat.S_IMODE(launch.instructions_path.stat().st_mode), 0o444)
        self.assertNotIn("host-secret", json.dumps(launch.env))
        self.assertEqual(launch.argv[1:4], ("serve", "--hostname", "127.0.0.1"))
        self.assertEqual(self.clients[0][0], "http://127.0.0.1:" + launch.argv[-1])
        self.assertEqual(launch.env["OPENCODE_SERVER_PASSWORD"], self.clients[0][1])
        self.assertEqual(sorted(thread.name for thread in session._threads),
                         ["opencode-events", "opencode-reconcile", "opencode-tree", "opencode-watch"])
        registered = self.store.events_by_type(["model.model_thread_registered"])
        self.assertEqual([event["payload"]["native_thread_id"] for event in registered], [thread])
        self.assertEqual(session.model_proxy.default_thread_id, thread)
        self.assertTrue(self.fake.wait_subscribers(1))

    def test_provenance_records_runtime_routes_and_effort(self):
        session = self.started()
        block = session.provenance()
        self.assertEqual((block["kind"], block["cli"]["version"], block["wire_api"], block["native_model_id"],
                          block["counting"], block["isolation"], block["context_delivery"], block["subagents"]),
                         ("opencode", "1.19.5", "anthropic_messages", "claude-test-model",
                          "delivered_usable_generation_completions_v1", "disabled", "full", True))
        # Read back from the running server: the build agent's registry minus its permission denials.
        self.assertEqual(block["native_tools"]["enabled"], ["bash", "edit", "glob", "grep", "read", "task",
                                                            "todowrite", "write"])
        self.assertEqual({item["name"] for item in block["native_tools"]["disabled"]},
                         {"webfetch", "websearch", "codesearch", "question"})
        self.assertEqual(len(block["route_list_sha256"]), 64)
        self.assertEqual(block["native_effort"], dict(
            variant="high", applied=True, channel="prompt_variant",
            provider_options={"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "high"},
            declared_variants=["low", "medium", "high", "xhigh", "max"], wire_field="output_config.effort"))
        self.assertEqual((block["output_limit_tokens"], block["context_limit_tokens"]), (32000, 200000))
        self.assertEqual(block["prompt_channel"]["instructions_channel"], "config_instructions_file")
        self.assertEqual(block["prompt_channel"]["tool_name_prefix"], "auriga_")
        self.assertEqual(block["tool_transport"]["read_only_tools"], ["look"])
        self.assertNotIn(PASSWORD, json.dumps(block))
        self.assertTrue(block["capabilities"]["steer"])
        unstarted = OpenCodeSession(self.root, self.store, None, config=self.config()).provenance()
        self.assertEqual((unstarted["kind"], unstarted["wire_api"]), ("opencode", "anthropic_messages"))

    def test_start_twice_is_refused(self):
        session = self.started()
        with self.assertRaisesRegex(ValueError, "^session_already_started$"):
            session.start(TOOLS, "again")

    def assert_start_fails(self, session, error, pattern):
        with self.assertRaisesRegex(error, pattern):
            session.start(TOOLS, "Be careful.")
        self.assertEqual(session.state, "closed")
        self.assertEqual(session.owned_processes(), [])
        if session.process is not None:
            self.assertTrue(wait_until(lambda: session.process.poll() is not None))
        self.assertIsNone(self.store.read_state("runtime", "agent"))
        self.assertEqual(self.disconnects, [])

    def test_runtime_and_launch_failures_close_everything(self):
        cases = (
            (dict(config=self.config(executable=fake_binary(self.dir, version="1.19.4")),
                  launch_options=dict(check_version=True)), VersionMismatch, "^opencode_version_mismatch"),
            (dict(launch_options=dict(managed_config_paths=[str(self.dir)])), OpenCodeConfigError,
             "^opencode_managed_config_present$"),
            (dict(launch_options=dict(ripgrep="/nonexistent/rg")), OpenCodeConfigError, "^opencode_ripgrep_missing$"),
            # The health check keeps failing (nothing listens) until the process has exited.
            (dict(config=self.config(executable=fake_binary(self.dir, serve="exit")), startup_timeout_s=20.0,
                  client_factory=lambda url, password: OpenCodeClient("http://127.0.0.1:9", PASSWORD, timeout_s=0.2)),
             RuntimeError, "^opencode_exited_during_startup:3$"),
        )
        for options, error, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assert_start_fails(self.make(**options), error, pattern)

    def test_health_route_and_variant_failures_kill_the_server(self):
        for setup, error, pattern in (
                (lambda: setattr(self.fake, "version", "1.19.6"), VersionMismatch, "^opencode_version_mismatch"),
                (lambda: setattr(self.fake, "routes", self.fake.routes[:-1]), OpenCodeConfigError,
                 "^opencode_route_list_mismatch$"),
                (lambda: setattr(self.fake, "agents", [{"name": "plan", "permission": []}]), OpenCodeConfigError,
                 "^opencode_tool_ids_invalid$"),
                (lambda: setattr(self, "served_variants", {"low": {}, "max": {}}), OpenCodeConfigError,
                 "^opencode_variants_mismatch$")):
            with self.subTest(pattern=pattern):
                saved = (self.fake.version, list(self.fake.routes), self.served_variants, self.fake.agents)
                setup()
                try:
                    session = self.make()
                    self.assert_start_fails(session, error, pattern)
                    self.assertIsNotNone(session.process)
                finally:
                    self.fake.version, self.fake.routes, self.served_variants, self.fake.agents = saved

    def test_startup_timeout(self):
        closed = OpenCodeClient("http://127.0.0.1:9", PASSWORD, timeout_s=0.2)
        session = self.make(client_factory=lambda url, password: closed, startup_timeout_s=0.3)
        self.assert_start_fails(session, RuntimeError, "^opencode_startup_timeout$")

    def test_configuration_failures_before_any_process(self):
        cases = (
            (self.config(executable="/nonexistent/opencode"), OpenCodeConfigError, "^opencode_executable_invalid$"),
            (Config.from_dict({"agent": {"backend": "opencode"}}), ValueError, "^opencode_executable_not_set"),
            (self.config(upstream={"base_url": "http://127.0.0.1:9", "api_key_env": KEY_ENV}), ValueError,
             "^opencode_upstream_incomplete"),
            (self.config(upstream={"wire_api": "openai_chat", "base_url": "http://127.0.0.1:9", "api_key_env": KEY_ENV}),
             ValueError, "^opencode_upstream_wire_unsupported:openai_chat"),
            (self.config(upstream={"wire_api": "anthropic_messages", "base_url": "http://127.0.0.1:9",
                                   "api_key_env": "AURIGA_TEST_UNSET_KEY"}), ValueError, "^opencode_upstream_key_missing"),
        )
        for config, error, pattern in cases:
            with self.subTest(pattern=pattern):
                session = self.make(config)
                self.assert_start_fails(session, error, pattern)
                self.assertIsNone(session.process)

    def test_close_before_start_and_in_replay(self):
        session = self.make()
        report = session.close()
        self.assertEqual((session.state, report["clean_shutdown"]), ("closed", False))
        viewer = self.make()
        viewer.enter_replay("ses_recorded")
        viewer.close()
        self.assertEqual((viewer.state, viewer.thread_id, viewer.owned_processes()), ("closed", "ses_recorded", []))


class TurnTests(SessionCase):
    def test_admission_returns_before_completion_and_idle_ends_the_turn(self):
        session = self.started()
        turn = session.run_turn("Do the task.")
        self.assertEqual((session.state, session.turn_id), ("running", turn))
        self.assertFalse(session.turn_done.is_set())
        prompt, = self.prompts()
        self.assertEqual(prompt["body"], {"parts": [{"type": "text", "text": "Do the task."}],
                                          "variant": "high"})  # no system field; the effort as the variant
        self.fake.finish_turn(session.thread_id, "FINAL")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.turn_result, {"id": turn, "status": "completed", "error": None})
        self.assertEqual(session.completed_final_answer(), "FINAL")
        verdict, = self.events("agent.turn_verdict")
        self.assertEqual((verdict["payload"]["status"], verdict["payload"]["reason"]), ("completed", "answered"))
        user, = self.rows("userMessage")
        self.assertEqual((user["turn_id"], user.get("native_id")), (turn, self.fake.last_user(session.thread_id)["info"]["id"]))
        command, = self.raw("control")
        self.assertEqual(command["message"]["command"], "submit_prompt")
        self.assertEqual(self.disconnects, [])
        # A second turn on the same session.
        second = session.run_turn("Again.")
        self.fake.finish_turn(session.thread_id, "SECOND")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_id, session.completed_final_answer()), (second, "SECOND"))

    def test_run_turn_refusals(self):
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("x")
        session.start(TOOLS, "Be careful.")
        self.running_turn(session)
        with self.assertRaisesRegex(ValueError, "^turn_already_active$"):
            session.run_turn("y")
        self.assertEqual(len(self.prompts()), 1)

    def test_prompt_4xx_is_a_definite_rejection_that_restores_the_previous_turn(self):
        session = self.started()
        first = session.run_turn("First.")
        self.fake.finish_turn(session.thread_id, "ONE")
        self.finish(session)
        result = session.turn_result
        self.fake.prompt_script.append(("status", 400))
        with self.assertRaises(AgentRejected) as raised:
            session.run_turn("Refused.")
        self.assertEqual((raised.exception.code, raised.exception.message), (400, "opencode_prompt_rejected"))
        self.assertEqual((session.state, session.turn_id, session.turn_result), ("completed", first, result))
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(session.completed_final_answer(), "ONE")
        rejected, = self.events("agent.turn_start_rejected")
        self.assertEqual(rejected["payload"]["code"], 400)
        withdrawn = [row for row in self.rows("userMessage") if row.get("withdrawn")]
        self.assertEqual([row["content"][0]["text"] for row in withdrawn], ["Refused."])
        self.assertEqual(self.disconnects, [])
        # The session stays usable.
        session.run_turn("Third.")
        self.fake.finish_turn(session.thread_id, "THREE")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "THREE")

    def assert_unknown_admission(self, session, script, pattern):
        self.fake.prompt_script.append(script)
        with self.assertRaisesRegex(AgentTransportUnknown, pattern):
            session.run_turn("Maybe admitted.")
        self.assertEqual(session.state, "disconnected")
        self.assertEqual(session.turn_result["status"], "failed")
        self.assertTrue(session.turn_result["error"]["message"].startswith("upstream_connection_lost: "))
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(self.disconnects, ["disconnected"])
        self.assertEqual(len(self.prompts()), 1)          # never retried
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("again")
        lost, = self.events("agent.connection_lost")
        self.assertFalse(lost["payload"]["history_complete"])

    def test_prompt_5xx_is_unknown_admission(self):
        self.assert_unknown_admission(self.started(), ("status", 500), "^opencode_prompt_unknown:status_500$")

    def test_prompt_timeout_is_unknown_admission(self):
        self.fake.hang_s = 1.5
        self.client_timeout = 0.4
        self.assert_unknown_admission(self.started(), "hang", "^opencode_prompt_unknown:OpenCodeTransportError$")

    def test_dropped_prompt_connection_is_unknown_admission(self):
        self.assert_unknown_admission(self.started(), "drop", "^opencode_prompt_unknown:OpenCodeTransportError$")

    def test_relay_denial_and_provider_failure_fail_the_turn_without_disconnect(self):
        session = self.started()
        session.run_turn("Go.")
        self.fake.finish_turn(session.thread_id, "", error=API_400)
        self.assertEqual(self.finish(session), "failed")
        self.assertEqual(session.turn_result["error"], {"message": "opencode_turn_failed:APIError:400"})
        self.assertIsNone(session.completed_final_answer())
        self.assertEqual(self.disconnects, [])
        self.assertNotIn("provider text", json.dumps(self.rows()))

    def test_second_idle_after_the_late_error_write_is_not_dropped(self):
        """OpenCode emits session.error + idle BEFORE it writes the message error; a second idle follows the write.
        The reconcile at the first idle sees an unfinished answer; the second idle must still be reconciled."""
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        message = self.fake.assistant(sid)                    # no finish, no error yet

        def write_error_then_idle():
            self.fake.update_info(message, error=API_400)
            self.fake.push("session.idle", {"sessionID": sid})

        self.fake.after_snapshot = write_error_then_idle      # runs right after the first snapshot was served
        self.fake.go_idle_silently(sid)
        self.fake.push("session.error", {"sessionID": sid, "error": API_400})
        self.fake.push("session.status", {"sessionID": sid, "status": {"type": "idle"}})
        self.fake.push("session.idle", {"sessionID": sid})
        self.assertEqual(self.finish(session, 3.0), "failed")
        self.assertEqual(session.turn_result["error"]["message"], "opencode_turn_failed:APIError:400")
        self.assertIsNone(self.fake.after_snapshot)
        snapshots = [item for item in self.fake.requests if item["path"] == f"/session/{sid}/message"]
        self.assertGreaterEqual(len(snapshots), 3)            # history at start + both idles

    def test_idle_without_a_terminal_answer_fails_the_turn_as_incomplete(self):
        session = self.started(watchdog_s=0.2, incomplete_gap_s=0.3)
        session._guard.min_gap_s = 0.3
        self.running_turn(session)
        self.fake.set_idle(session.thread_id)                 # idle, but no assistant answer ever written
        self.assertEqual(self.finish(session, 5.0), "failed")
        self.assertEqual(session.turn_result["error"], {"message": "opencode_turn_incomplete"})
        verdict, = self.events("agent.turn_verdict")
        self.assertEqual(verdict["payload"]["reason"], "incomplete")
        self.assertEqual(self.disconnects, [])

    def test_unrelated_session_events_are_ignored(self):
        session = self.started()
        other = self.fake.add_session("other")
        self.fake.push("message.updated", {"sessionID": other, "info": {"id": "msg_x", "sessionID": other,
                                                                         "role": "assistant", "parentID": "msg_u"}})
        self.fake.push("message.part.updated", {"sessionID": other, "part": {
            "id": "prt_x", "messageID": "msg_x", "sessionID": other, "type": "text", "text": "foreign"}})
        self.fake.push("session.idle", {"sessionID": other})
        self.running_turn(session)
        time.sleep(0.2)
        self.assertEqual(session.state, "running")
        self.assertFalse([row for row in self.rows() if row.get("thread_id") == other])


class SteerTests(SessionCase):
    def test_steer_joins_the_running_turn(self):
        session = self.started()
        turn = self.running_turn(session)
        self.assertEqual(session.steer("Also check the left side."), turn)
        first, second = self.prompts()
        self.assertEqual(second["body"]["parts"], [{"type": "text", "text": "Also check the left side."}])
        self.fake.finish_turn(session.thread_id, "BOTH")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "BOTH")
        rows = self.rows("userMessage")
        self.assertEqual([(row["turn_id"], row.get("steer", False), bool(row.get("native_id"))) for row in rows],
                         [(turn, False, True), (turn, True, True)])
        self.assertEqual([record["message"]["command"] for record in self.raw("control")], ["submit_prompt", "steer"])
        self.assertEqual(len(self.events("agent.turn_steered")), 1)

    def test_steer_requires_a_running_turn(self):
        session = self.started()
        with self.assertRaisesRegex(ValueError, "^no_active_turn_to_steer$"):
            session.steer("nobody listens")
        self.assertEqual(self.prompts(), [])

    def test_steer_4xx_is_rejected_and_the_turn_continues_without_it(self):
        session = self.started()
        self.running_turn(session)
        self.fake.prompt_script.append(("status", 400))
        with self.assertRaises(AgentRejected) as raised:
            session.steer("refused text")
        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(session.state, "running")
        withdrawn = [row for row in self.rows("userMessage") if row.get("withdrawn")]
        self.assertEqual([row["content"][0]["text"] for row in withdrawn], ["refused text"])
        self.fake.finish_turn(session.thread_id, "ORIGINAL")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "ORIGINAL")
        self.assertEqual(self.disconnects, [])

    def test_steer_transport_failure_disconnects(self):
        session = self.started()
        self.running_turn(session)
        self.fake.prompt_script.append(("status", 503))
        with self.assertRaisesRegex(AgentTransportUnknown, "^opencode_steer_unknown:status_503$"):
            session.steer("lost")
        self.assertEqual((session.state, self.disconnects), ("disconnected", ["disconnected"]))

    def test_steer_written_while_a_reconcile_holds_the_final_snapshot_keeps_the_turn(self):
        """The reconcile at the end of the first answer fetched its snapshot just before a steer was admitted: the
        anchor re-read under the session lock sees the unmatched steer, so the turn keeps running for its answer.
        (The test holds the session's projector lock to park that reconcile between its snapshot and its verdict.)"""
        session = self.started()
        sid = session.thread_id
        turn = self.running_turn(session)
        served = threading.Event()
        self.fake.after_snapshot = served.set
        session._projector_lock.acquire()
        try:
            self.fake.finish_turn(sid, "FIRST")               # idle -> reconcile fetches the final snapshot
            self.assertTrue(served.wait(5))
            self.assertEqual(session.steer("STEER-LATE"), turn)   # admitted after the snapshot was taken
        finally:
            session._projector_lock.release()
        time.sleep(0.3)
        self.assertEqual(session.state, "running")
        self.assertEqual(self.events("agent.turn_completed"), [])
        self.fake.finish_turn(sid, "STEER-ANSWER")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_id, session.completed_final_answer()), (turn, "STEER-ANSWER"))
        self.assertEqual(len(self.events("agent.turn_completed")), 1)
        phases = {row["text"]: row["phase"] for row in self.rows("agentMessage")}
        self.assertEqual(phases, {"FIRST": "commentary", "STEER-ANSWER": "final_answer"})

    def test_a_reconcile_during_a_submission_never_finishes_the_turn(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        with session.lock:
            session._submitting += 1                          # a steer request is on its way
        self.fake.finish_turn(sid, "EARLY")
        time.sleep(0.3)
        self.assertEqual(session.state, "running")
        with session.lock:
            session._submitting -= 1
        session._request_reconcile("test")
        self.assertEqual(self.finish(session), "completed")


class InterruptTests(SessionCase):
    def test_interrupt_aborts_and_ends_the_turn_interrupted(self):
        session = self.started()
        self.running_turn(session)
        with mock.patch.object(session.model_proxy, "cancel_pending", wraps=session.model_proxy.cancel_pending) as cancel:
            session.interrupt()
        cancel.assert_called_once_with("turn_interrupted", threads=(session.thread_id,))
        self.assertEqual(self.finish(session), "interrupted")
        self.assertEqual(session.turn_result["status"], "interrupted")
        aborts = [item["path"] for item in self.fake.requests if item["path"].endswith("/abort")]
        self.assertEqual(aborts, [f"/session/{session.thread_id}/abort"])
        self.assertEqual([record["message"]["command"] for record in self.raw("control")],
                         ["submit_prompt", "interrupt"])
        self.assertEqual(self.disconnects, [])
        # The next turn reopens the relay thread first and runs normally.
        with mock.patch.object(session.model_proxy, "reopen_thread", wraps=session.model_proxy.reopen_thread) as reopen:
            session.run_turn("Next.")
        reopen.assert_called_once_with(session.thread_id)
        self.fake.finish_turn(session.thread_id, "AFTER")
        self.assertEqual(self.finish(session), "completed")

    def test_interrupt_blocks_the_relay_thread_before_the_native_abort(self):
        """Probe R4-0: cancel_pending (with the session's thread tombstoned even when no model call is in flight)
        must happen BEFORE POST /abort, so a model call the agent still makes after the abort is denied."""
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        order = []
        real_cancel = session.model_proxy.cancel_pending

        def cancel(reason, **keywords):
            order.append(("cancel_pending", reason, tuple(keywords.get("threads", ())),
                          len([item for item in self.fake.requests if item["path"].endswith("/abort")])))
            return real_cancel(reason, **keywords)

        with mock.patch.object(session.model_proxy, "cancel_pending", side_effect=cancel):
            session.interrupt()
        self.assertEqual(order, [("cancel_pending", "turn_interrupted", (sid,), 0)])   # no abort sent before it
        self.assertEqual([item["path"] for item in self.fake.requests if item["path"].endswith("/abort")],
                         [f"/session/{sid}/abort"])
        self.assertIn(sid, session.model_proxy.cancelled_threads)
        self.assertEqual(self.finish(session), "interrupted")
        # A model call the agent makes after the abort is denied by the relay, never forwarded (upstream is closed).
        status, body = relay_post(session, sid)
        self.assertEqual(status, 400)
        rejected = lambda: self.store.events_by_type(["model.model_thread_request_rejected"])  # noqa: E731
        self.assertTrue(wait_until(rejected))                 # relay events arrive through its callback worker
        self.assertEqual(rejected()[-1]["payload"]["native_thread_id"], sid)
        # The next turn lifts the tombstone before anything is sent.
        session.run_turn("Next.")
        self.assertNotIn(sid, session.model_proxy.cancelled_threads)
        reopened = lambda: self.store.events_by_type(["model.model_thread_reopened"])  # noqa: E731
        self.assertTrue(wait_until(reopened))
        self.assertEqual([event["payload"]["native_thread_id"] for event in reopened()], [sid])

    def test_interrupt_wins_over_an_answer_that_completed_before_the_abort_landed(self):
        session = self.started()
        self.running_turn(session)
        with session.lock:
            session._submitting += 1                          # hold the verdict until the interrupt is marked
        self.fake.finish_turn(session.thread_id, "RACED")
        session.interrupt()
        with session.lock:
            session._submitting -= 1
        session._request_reconcile("test")
        self.assertEqual(self.finish(session), "interrupted")

    def test_interrupt_while_idle_sends_nothing(self):
        session = self.started()
        session.interrupt()
        self.assertFalse([item for item in self.fake.requests if item["path"].endswith("/abort")])
        self.assertEqual(self.raw("control"), [])

    def test_abort_http_failure_disconnects(self):
        session = self.started()
        self.running_turn(session)
        self.fake.abort_status = 500
        with self.assertRaisesRegex(AgentTransportUnknown, "^opencode_abort_unknown:status_500$"):
            session.interrupt()
        self.assertEqual((session.state, session.turn_result["status"]), ("disconnected", "failed"))
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(self.disconnects, ["disconnected"])


class LifecycleTests(SessionCase):
    def test_server_exit_mid_turn_disconnects_once_and_abandons_open_tools(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        message = self.fake.assistant(sid)
        self.fake.part(message, {"type": "tool", "tool": "bash", "callID": "call_b", "state": {
            "status": "running", "input": {"command": "sleep 9"}, "time": {"start": 1}}})
        self.assertTrue(wait_until(lambda: self.rows("commandExecution")))
        session.process.kill()
        self.assertEqual(self.finish(session), "disconnected")
        self.assertEqual(session.turn_result["status"], "failed")
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: ProcessLookupError")
        self.assertEqual(self.disconnects, ["disconnected"])
        row, = self.rows("commandExecution")
        self.assertEqual((row["status"], row.get("interrupted")), ("failed", True))
        lost, = self.events("agent.connection_lost")
        self.assertTrue(lost["payload"]["detail"].startswith("opencode_exited:"))
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_server_exit_while_idle_disconnects(self):
        session = self.started()
        session.process.kill()
        self.assertTrue(wait_until(lambda: session.state == "disconnected"))
        self.assertEqual(self.disconnects, ["disconnected"])
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("x")

    def test_event_stream_gap_is_reconciled_from_the_api(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        self.fake.events_blocked = True
        self.fake.drop_event_streams()
        self.assertTrue(wait_until(lambda: self.events("agent.event_stream_gap")))
        self.fake.finish_turn(sid, "UNSEEN")                  # every event of the finish is lost
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "UNSEEN")
        self.assertEqual(self.disconnects, [])
        time.sleep(0.5)                                       # several failed reconnects: one gap event
        self.assertEqual(len(self.events("agent.event_stream_gap")), 1)
        self.fake.events_blocked = False
        self.assertTrue(self.fake.wait_subscribers(1))
        # The reconnect reconciles again; a second outage is recorded again.
        self.fake.drop_event_streams()
        self.assertTrue(wait_until(lambda: len(self.events("agent.event_stream_gap")) == 2))

    def test_unsolicited_native_busy_is_recorded_once(self):
        session = self.started()
        session.run_turn("x")
        self.fake.finish_turn(session.thread_id, "done")
        self.finish(session)
        self.fake.set_busy(session.thread_id)
        self.fake.set_busy(session.thread_id)
        self.assertTrue(wait_until(lambda: self.events("agent.unsolicited_native_turn")))
        time.sleep(0.1)
        event, = self.events("agent.unsolicited_native_turn")
        self.assertEqual(event["payload"], {"host_state": "completed"})
        self.assertEqual(session.state, "completed")

    def test_watchdog_reconciles_a_silently_lost_stream(self):
        session = self.started(watchdog_s=0.3)
        sid = session.thread_id
        self.fake.events_muted = True
        self.running_turn(session)
        # The reconcile requested by the admission saw the session busy; nothing else will ask again.
        self.assertTrue(wait_until(lambda: self.fake.request_paths("GET").count("/session/status") >= 1))
        self.fake.finish_turn(sid, "QUIET")
        self.assertEqual(self.finish(session, 5.0), "completed")
        self.assertEqual(self.events("agent.turn_verdict")[0]["payload"]["signals"], ["watchdog"])

    def test_reconcile_failures_are_logged_and_retried(self):
        session = self.started(watchdog_s=0.2)
        self.running_turn(session)
        failing = mock.patch.object(session.client, "status", side_effect=OpenCodeTransportError("ConnectionResetError"))
        failing.start()
        try:
            self.fake.finish_turn(session.thread_id, "LATER")
            self.assertTrue(wait_until(lambda: len(self.events("agent.reconcile_failed")) >= 3, 5.0))
            self.assertEqual(session.state, "running")    # the server is alive: not a disconnect
        finally:
            failing.stop()
        self.assertEqual(self.finish(session, 5.0), "completed")
        counts = [event["payload"]["consecutive"] for event in self.events("agent.reconcile_failed")]
        self.assertEqual(counts[:3], [1, 2, 3])
        self.assertEqual(self.disconnects, [])

    def test_three_failed_reconciles_with_a_dead_server_disconnect(self):
        """Backup to the watch thread: the server is gone but its exit was not observed (the watch thread is
        blocked), and the API keeps failing -> disconnected after ``reconcile_failure_limit`` failures."""
        session = self.started(watchdog_s=0.1)
        self.running_turn(session)
        with mock.patch.object(session.client, "status", side_effect=OpenCodeTransportError("ConnectionRefusedError")), \
                mock.patch.object(session.process, "poll", return_value=-9):
            self.assertEqual(self.finish(session, 5.0), "disconnected")
        self.assertEqual(len(self.events("agent.reconcile_failed")), 3)
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: ProcessLookupError")
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_close_kills_detached_children_and_records_a_clean_shutdown(self):
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="detached")))
        pid = session.process.pid
        self.assertTrue(wait_until(lambda: len(process_tree_snapshot(pid).members) >= 2, 10.0))
        members = process_tree_snapshot(pid).members
        self.assertTrue(any(item.pgid != os.getpgid(pid) for item in members))   # a detached group
        session.run_turn("x")
        self.fake.finish_turn(session.thread_id, "done")
        self.finish(session)
        report = session.close()
        self.assertTrue(report["clean_shutdown"], report)
        self.assertTrue(report["backend"]["agent_processes"]["confirmed"])
        self.assertEqual(report["backend"]["abort"], "completed")   # sent whenever the server lives (no-op if idle)
        for item in members:
            self.assertFalse(alive(item.pid), f"pid {item.pid} survived close")
        self.assertEqual((session.state, session.owned_processes()), ("closed", []))
        self.assertTrue(self.store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertEqual(session.close(), report)             # idempotent
        self.assertFalse([thread for thread in session._threads if thread.is_alive()])

    def test_close_after_an_unknown_admission_still_aborts_the_native_turn(self):
        session = self.started()
        self.fake.prompt_script.append("drop")                # admitted by the server, answer lost
        with self.assertRaises(AgentTransportUnknown):
            session.run_turn("Maybe running.")
        report = session.close()
        self.assertEqual(report["backend"]["abort"], "completed")
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_close_during_a_turn_aborts_first_and_is_unclean(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        message = self.fake.assistant(sid)
        self.fake.part(message, {"type": "tool", "tool": "read", "callID": "call_r", "state": {
            "status": "running", "input": {"filePath": "a"}, "time": {"start": 1}}})
        self.assertTrue(wait_until(lambda: self.rows("mcpToolCall")))
        report = session.close()
        self.assertEqual(report["backend"]["abort"], "completed")
        self.assertEqual(report["backend"]["abandoned_items"], 1)
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(self.disconnects, [])
        self.assertIn(f"/session/{sid}/abort", [item["path"] for item in self.fake.requests])
        row, = self.rows("mcpToolCall")
        self.assertEqual((row["status"], row.get("interrupted")), ("failed", True))
        self.assertEqual(session.turn_result, None)
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])


class ToolAndContextTests(SessionCase):
    def test_responses_wire_and_subagent_context_delivery(self):
        config = self.config(upstream={"wire_api": "openai_responses", "base_url": "http://127.0.0.1:9",
                                       "api_key_env": KEY_ENV}, context_delivery="delta")
        session = self.started(config=config)
        self.assertEqual((session.wire.name, session.wire.client), ("openai_responses", "opencode"))
        provider = json.loads(session.launch.env["OPENCODE_CONFIG_CONTENT"])["provider"]["relay"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai")
        self.assertEqual(session.context_delivery, "full")    # design R4.3: subagents force full delivery
        self.assertEqual(session.provenance()["subagents"], True)
        # subagents default on (R4-2): delta is honoured only with subagents off (design R4.3).
        self.assertTrue(self.config().agent.opencode.subagents)
        self.assertEqual(self.make(self.config(context_delivery="delta")).context_delivery, "full")
        alone = self.make(self.config(context_delivery="delta", subagents=False))
        self.assertEqual(alone.context_delivery, "delta")
        alone.start(TOOLS, "Be careful.")
        block = alone.provenance()
        self.assertEqual((block["context_delivery"], block["subagents"]), ("delta", False))
        self.assertIn({"name": "task", "reason": "subagents_disabled"}, block["native_tools"]["disabled"])
        self.assertEqual(json.loads(alone.launch.env["OPENCODE_CONFIG_CONTENT"])["permission"]["task"], "deny")

    def test_codex_home_upstream_resolves_the_codex_credentials(self):
        home = self.dir / "codex-home"
        home.mkdir()
        (home / "config.toml").write_text('model_provider = "lab"\n[model_providers.lab]\n'
                                          'base_url = "http://127.0.0.1:9/v1"\nenv_key = "AURIGA_TEST_OPENCODE_UPSTREAM_KEY"\n')
        config = Config.from_dict({**self.config().to_dict(), "model": {"name": "gpt-test", "codex_home": str(home)},
                                   "agent": {"backend": "opencode", "opencode": {"executable": self.executable,
                                                                                 "upstream": {"source": "codex_home"}}}})
        session = self.make(config)
        self.assertEqual(session.wire_api, "openai_responses")
        url, headers = session._upstream()
        self.assertEqual((url, headers), ("http://127.0.0.1:9/v1", {"Authorization": "Bearer host-secret"}))
        self.assertEqual((session.wire.name, session.wire.client), ("openai_responses", "opencode"))

    def test_relay_discontinuity_changes_the_epoch_and_resets_the_context(self):
        session = self.started()
        self.assertEqual(session.model_proxy.on_context_discontinuity, session._relay_discontinuity)
        session.model_proxy.on_context_discontinuity("history_rewritten")
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["signal"]), ("relay_context_discontinuity", "history_rewritten"))
        self.assertEqual(self.resets, ["relay_context_discontinuity"])

    def test_declared_effort_variants_reach_the_launch_config(self):
        from auriga.cortex.agents.opencode.config import effort_variants
        session = self.started()
        config = json.loads(session.launch.env["OPENCODE_CONFIG_CONTENT"])
        entry = config["provider"]["relay"]["models"]["claude-test-model"]
        self.assertEqual(entry["variants"], effort_variants("anthropic_messages"))
        self.assertFalse(entry["reasoning"])                  # no OpenCode model-id heuristics
        self.assertNotIn("options", entry)
        self.assertFalse(self.events("agent.effort_not_applied"))

    def test_registry_calls_carry_the_host_turn_and_stale_calls_are_refused(self):
        session = self.started()
        client = self.mcp(session)
        stale = client.call_tool("look", {"x": 1})
        self.assertTrue(stale.result["isError"])
        self.assertEqual(self.handler_calls, [])
        turn = self.running_turn(session)
        result = client.call_tool("look", {"x": 2})
        self.assertFalse(result.result.get("isError"))
        params, = self.handler_calls
        self.assertEqual((params["threadId"], params["turnId"], params["tool"], params["callIdSource"]),
                         (session.thread_id, turn, "look", "mcp_request"))
        row = self.store.transcript_item(params["callId"])
        self.assertEqual((row["type"], row["status"], row["turn_id"]), ("dynamicToolCall", "completed", turn))

    def test_variants_carry_the_effort_per_prompt_and_steer(self):
        session = self.started()
        self.assertEqual(session.effort_choices, ("low", "medium", "high", "xhigh", "max"))
        session.run_turn("one")
        self.assertEqual(session.steer("also this"), session.turn_id)
        self.fake.finish_turn(session.thread_id)
        self.finish(session)
        session.set_effort("max")                             # the next prompt carries the new variant
        session.run_turn("two")
        self.assertEqual([item["body"].get("variant") for item in self.prompts()], ["high", "high", "max"])
        with self.assertRaisesRegex(ValueError, "^invalid_effort$"):
            session.set_effort("minimal")                     # an openai_responses tier
        native = session.provenance()["native_effort"]
        self.assertEqual((native["variant"], native["provider_options"]["effort"]), ("max", "max"))

    def test_responses_wire_declares_its_own_tiers(self):
        from auriga.cortex.agents.opencode.config import effort_variants
        config = self.config(upstream={"wire_api": "openai_responses", "base_url": "http://127.0.0.1:9",
                                       "api_key_env": KEY_ENV}, effort="minimal")
        session = self.make(config)
        self.assertEqual(session.effort_choices, ("none", "minimal", "low", "medium", "high", "xhigh"))
        session.start(TOOLS, "Be careful.")
        entry = json.loads(session.launch.env["OPENCODE_CONFIG_CONTENT"])["provider"]["relay"]["models"][
            "claude-test-model"]
        self.assertEqual(entry["variants"], effort_variants("openai_responses"))
        session.run_turn("go")
        self.assertEqual(self.prompts()[-1]["body"]["variant"], "minimal")
        native = session.provenance()["native_effort"]
        self.assertEqual((native["wire_field"], native["provider_options"]),
                         ("reasoning.effort", {"reasoningEffort": "minimal", "reasoningSummary": "auto",
                                               "include": ["reasoning.encrypted_content"]}))

    def test_a_server_serving_other_variants_refuses_the_start(self):
        from auriga.cortex.agents.opencode.config import effort_variants
        served = effort_variants("anthropic_messages")
        served["high"] = dict(served["high"], effort="medium")   # same names, different options
        self.served_variants = served
        session = self.make()
        with self.assertRaisesRegex(OpenCodeConfigError, "^opencode_variants_mismatch$"):
            session.start(TOOLS, "Be careful.")
        self.assertEqual(session.state, "closed")

    def test_model_is_fixed_at_session_start(self):
        session = self.started()
        session.set_model("claude-test-model")
        with self.assertRaisesRegex(AgentCapabilityError, "^set_model_unsupported_by_backend:opencode"):
            session.set_model("another-model")
        self.assertEqual(session.model, "claude-test-model")

    def test_native_compaction_signals_once_per_compaction(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        user = self.fake.last_user(sid)
        self.fake.part(user, {"type": "compaction", "auto": True, "overflow": False})
        self.assertTrue(wait_until(lambda: self.resets))
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["signal"], epoch["thread_id"]),
                         ("native_context_compaction", "compaction_part", sid))
        self.fake.push("session.compacted", {"sessionID": sid})          # the same compaction: no second signal
        second = self.fake.part(user, {"type": "compaction", "auto": True, "overflow": True})
        self.assertTrue(wait_until(lambda: len(self.resets) == 2))
        self.fake.push("session.compacted", {"sessionID": sid})
        self.fake.update_part(second, auto=True)                          # an update of a known part
        self.fake.finish_turn(sid, "AFTER")
        self.finish(session)
        self.assertEqual(self.resets, ["native_context_compaction"] * 2)
        self.assertEqual([event["payload"]["signal"] for event in self.events("agent.compaction")],
                         ["compaction_part", "compaction_part"])
        self.assertEqual(len(self.rows("contextCompaction")), 2)

    def test_a_compaction_missed_by_the_events_is_signalled_by_the_snapshot(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        self.fake.events_blocked = True
        self.fake.drop_event_streams()
        self.fake.part(self.fake.last_user(sid), {"type": "compaction", "auto": True})
        self.fake.finish_turn(sid, "DONE")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(self.resets, ["native_context_compaction"])


class SubagentTests(SessionCase):
    """Design R4.3: child sessions are followed, their rows hang under the task tool part, never final answer."""

    def child_turn(self, session, *, finish=True):
        sid = session.thread_id
        turn = self.running_turn(session)
        child = self.fake.add_session("explore (@general subagent)")
        self.fake.push("session.created", {"sessionID": child, "info": {"id": child, "parentID": sid}})
        message = self.fake.assistant(sid)
        task = self.fake.part(message, {"type": "tool", "tool": "task", "callID": "call_task", "state": {
            "status": "running", "input": {"description": "explore", "prompt": "look around",
                                           "subagent_type": "general"},
            "metadata": {"sessionId": child, "parentSessionId": sid}, "time": {"start": 1}}})
        child_user = self.fake._add_message(child, "user", agent="general")
        self.fake.part(child_user, {"type": "text", "text": "look around"})
        child_answer = self.fake.assistant(child, child_user["info"]["id"], finish="stop")
        self.fake.part(child_answer, {"type": "text", "text": "CHILD-REPORT", "time": {"start": 1, "end": 2}})
        self.fake.push("session.idle", {"sessionID": child})              # a child idle never ends the turn
        if finish:
            self.fake.update_part(task, state=dict(task["state"], status="completed", output="CHILD-REPORT",
                                                   time={"start": 1, "end": 2}))
            self.fake.update_info(message, finish="tool-calls")
            self.fake.finish_turn(sid, "MAIN-FINAL")
        return turn, child, task

    def test_child_rows_hang_under_the_task_part_and_are_never_final(self):
        session = self.started()
        turn, child, task = self.child_turn(session)
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "MAIN-FINAL")
        texts = self.rows("agentMessage", thread=child)
        self.assertEqual([(row["text"], row["parent_item_id"], row["turn_id"], row["phase"]) for row in texts],
                         [("CHILD-REPORT", task["id"], turn, None)])
        child_user, = self.rows("userMessage", thread=child)
        self.assertEqual((child_user["parent_item_id"], child_user.get("source")), (task["id"], "native"))
        task_row = self.store.transcript_item(task["id"])
        self.assertEqual((task_row["type"], task_row["tool"], task_row["thread_id"]),
                         ("mcpToolCall", "task", session.thread_id))
        linked, = self.events("agent.subagent_session")
        self.assertEqual(linked["payload"], {"session_id": child, "parent_session_id": session.thread_id})
        self.assertEqual(linked["correlation"]["item_id"], task["id"])

    def test_child_rows_are_recovered_from_snapshots_after_an_event_gap(self):
        session = self.started()
        self.fake.events_blocked = True
        self.fake.drop_event_streams()
        self.assertTrue(wait_until(lambda: self.events("agent.event_stream_gap")))
        turn, child, task = self.child_turn(session)
        self.assertEqual(self.finish(session), "completed")
        texts = self.rows("agentMessage", thread=child)
        self.assertEqual([(row["text"], row["parent_item_id"]) for row in texts], [("CHILD-REPORT", task["id"])])
        self.assertIn(f"/session/{child}/message", [item["path"] for item in self.fake.requests])

    def test_a_resumed_subagent_hangs_its_new_rows_under_the_new_task_part(self):
        session = self.started()
        sid = session.thread_id
        first_turn, child, first_task = self.child_turn(session)
        self.finish(session)
        second_turn = self.running_turn(session, "Continue the exploration.")
        message = self.fake.assistant(sid)
        second_task = self.fake.part(message, {"type": "tool", "tool": "task", "callID": "call_task2", "state": {
            "status": "running", "input": {"description": "explore", "prompt": "more", "subagent_type": "general",
                                           "task_id": child},
            "metadata": {"sessionId": child, "parentSessionId": sid}, "time": {"start": 3}}})
        user = self.fake._add_message(child, "user", agent="general")
        self.fake.part(user, {"type": "text", "text": "more"})
        answer = self.fake.assistant(child, user["info"]["id"], finish="stop")
        self.fake.part(answer, {"type": "text", "text": "CHILD-REPORT-2", "time": {"start": 3, "end": 4}})
        self.fake.update_part(second_task, state=dict(second_task["state"], status="completed", output="r2"))
        self.fake.update_info(message, finish="tool-calls")
        self.fake.finish_turn(sid, "MAIN-FINAL-2")
        self.assertEqual(self.finish(session), "completed")
        texts = {row["text"]: (row["parent_item_id"], row["turn_id"]) for row in self.rows("agentMessage", thread=child)}
        self.assertEqual(texts, {"CHILD-REPORT": (first_task["id"], first_turn),
                                 "CHILD-REPORT-2": (second_task["id"], second_turn)})
        self.assertEqual(session.completed_final_answer(), "MAIN-FINAL-2")

    def test_interrupt_aborts_the_main_session_then_its_children(self):
        session = self.started()
        sid = session.thread_id
        _turn, child, _task = self.child_turn(session, finish=False)
        self.assertTrue(wait_until(lambda: child in session._children and session._children[child]["projector"]))
        time.sleep(0.1)
        self.assertEqual(session.state, "running")            # the child's idle never ends the host turn
        order = []
        real_cancel = session.model_proxy.cancel_pending
        with mock.patch.object(session.model_proxy, "cancel_pending", side_effect=lambda reason, **keywords: (
                order.append((tuple(keywords.get("threads", ())),
                              [item["path"] for item in self.fake.requests if item["path"].endswith("/abort")])),
                real_cancel(reason, **keywords))):
            session.interrupt()
        self.assertEqual(order, [((sid, child), [])])         # both threads blocked before any abort
        self.assertEqual(self.finish(session), "interrupted")
        aborts = [item["path"] for item in self.fake.requests if item["path"].endswith("/abort")]
        self.assertEqual(aborts, [f"/session/{sid}/abort", f"/session/{child}/abort"])
        control = self.raw("control")[-1]["message"]
        self.assertEqual((control["command"], control["subagent_sessions"]), ("interrupt", [child]))
        # The main agent's extra model call after the abort, the child's own, and one from a child the session never
        # saw (an unregistered id falls back to the main session) are all denied.
        for thread in (sid, child, "ses_unknownChildAbc"):
            with self.subTest(thread=thread):
                self.assertEqual(relay_post(session, thread)[0], 400)
        # The next turn reopens the main and the child thread (a child can be resumed with task_id).
        session.run_turn("Resume the exploration.")
        self.assertFalse({sid, child} & set(session.model_proxy.cancelled_threads))


class ResumeTests(SessionCase):
    def first_run(self, *, clean=True):
        session = self.started()
        session.run_turn("Earlier.")
        self.fake.finish_turn(session.thread_id, "EARLIER")
        self.finish(session)
        thread = session.thread_id
        session.close()
        if not clean:
            identity = self.store.read_state("runtime", "agent")
            self.store.set_state("runtime", "agent", dict(identity, clean_shutdown=False))
        self.disconnects.clear()
        return thread

    def test_clean_resume_adopts_the_history_without_moving_it(self):
        thread = self.first_run()
        rows_before = {row["id"]: row["turn_id"] for row in self.rows()}
        session = self.started(resume=thread)
        self.assertEqual((session.state, session.thread_id), ("idle", thread))
        self.assertEqual({row["id"]: row["turn_id"] for row in self.rows()}, rows_before)
        self.assertEqual(self.resets, [])
        recovered, = self.events("agent.history_recovered")
        self.assertTrue(recovered["payload"]["clean"])
        self.assertTrue(self.store.read_state("runtime", "agent")["resumed"])
        session.run_turn("Now.")
        self.fake.finish_turn(thread, "NOW")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "NOW")
        self.assertFalse([item for item in self.fake.requests if item["method"] == "POST" and item["path"] == "/session"][1:])

    def test_unclean_resume_invalidates_the_context(self):
        thread = self.first_run(clean=False)
        self.started(resume=thread)
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual(epoch["source"], "agent_resume_after_unclean_exit")
        self.assertEqual(self.resets, ["agent_resume_after_unclean_exit"])

    def test_resume_identity_and_native_session_checks(self):
        thread = self.first_run()
        identity = self.store.read_state("runtime", "agent")
        for key, value in (("model", "other-model"), ("tools_digest", "x"), ("backend", "claude_code"),
                           ("wire_api", "openai_responses"), ("cli_version", "1.19.4"), ("thread_id", "ses_other")):
            with self.subTest(key=key):
                self.store.set_state("runtime", "agent", dict(identity, **{key: value}))
                session = self.make()
                with self.assertRaisesRegex(ValueError, f"^resume_agent_identity_conflict:{key}"):
                    session.start(TOOLS, "Be careful.", thread)
                self.assertIsNone(session.process)
        self.store.set_state("runtime", "agent", identity)
        with self.fake.lock:
            del self.fake.sessions[thread]
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^resume_native_session_missing"):
            session.start(TOOLS, "Be careful.", thread)
        self.assertEqual(session.owned_processes(), [])
        self.store.set_state("runtime", "agent", None)
        with self.assertRaisesRegex(ValueError, "^resume_agent_identity_missing"):
            self.make().start(TOOLS, "Be careful.", thread)


class ReviewRegressionTests(SessionCase):
    """Review T11 (research/review-t11.md) X1-X6 and the minor items, as regressions."""

    def aborts(self):
        return [item["path"] for item in self.fake.requests if item["path"].endswith("/abort")]

    def hold_abort(self):
        """The fake's POST /abort becomes a no-op (as OpenCode's cancel with no runner: before the loop started);
        returns a function that restores the real behaviour."""
        real = self.fake.abort_session
        self.fake.abort_session = lambda session_id: None
        return lambda: setattr(self.fake, "abort_session", real)

    def test_x1_unknown_admission_blocks_the_relay_and_interrupt_still_aborts(self):
        session = self.started()
        sid = session.thread_id
        self.fake.prompt_script.append("drop")               # admitted by the server, its answer lost
        with self.assertRaises(AgentTransportUnknown):
            session.run_turn("Maybe running.")
        self.assertEqual(session.state, "disconnected")
        self.assertIn(sid, session.model_proxy.cancelled_threads)
        self.assertEqual(relay_post(session, sid)[0], 400)    # the admitted loop's model call is denied
        session.interrupt()                                   # the server lives and the loop is busy: abort it
        self.assertEqual(self.aborts(), [f"/session/{sid}/abort"])
        self.assertEqual(self.raw("control")[-1]["message"]["host_turn"], False)
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_x2_native_busy_without_a_host_turn_is_stopped_and_closes_unclean(self):
        session = self.started()
        sid = session.thread_id
        session.run_turn("x")
        self.fake.finish_turn(sid, "done")
        self.assertEqual(self.finish(session), "completed")
        self.fake.set_busy(sid)                               # a loop with no host turn
        self.assertTrue(wait_until(lambda: self.events("agent.unsolicited_native_turn")))
        session.interrupt()
        self.assertEqual(self.aborts(), [f"/session/{sid}/abort"])
        self.assertIn(sid, session.model_proxy.cancelled_threads)
        self.assertEqual(relay_post(session, sid)[0], 400)
        self.assertEqual(session.state, "completed")          # the host turn's result is kept
        self.fake.set_busy(sid)
        self.assertTrue(wait_until(lambda: session._native_busy))
        report = session.close()
        self.assertEqual((report["clean_shutdown"], report["native_active_at_close"]), (False, True))
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])

    def test_interrupt_reads_the_live_status_when_no_event_reported_busy(self):
        session = self.started()
        sid = session.thread_id
        session.run_turn("x")
        self.fake.finish_turn(sid, "done")
        self.finish(session)
        with self.fake.lock:                                  # busy, but the event was lost
            self.fake.busy[sid] = {"type": "busy"}
        session.interrupt()
        self.assertEqual(self.aborts(), [f"/session/{sid}/abort"])
        with self.fake.lock:
            self.fake.busy.pop(sid, None)
        session.interrupt()                                   # idle and no host turn: nothing is sent
        self.assertEqual(len(self.aborts()), 1)

    def test_a_close_that_finds_the_session_busy_is_unclean(self):
        session = self.started()
        sid = session.thread_id
        session.run_turn("x")
        self.fake.finish_turn(sid, "done")
        self.finish(session)
        with self.fake.lock:                                  # busy with no event: only the live read sees it
            self.fake.busy[sid] = {"type": "busy"}
        report = session.close()
        self.assertEqual((report["clean_shutdown"], report["native_active_at_close"]), (False, True))
        self.assertEqual(report["backend"]["abort"], "completed")

    def test_x3_crash_teardown_kills_detached_children_and_is_verified_by_the_marker(self):
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="detached")),
                               tree_refresh_s=0.1)
        pid = session.process.pid
        self.assertTrue(wait_until(lambda: any(row.pgid != pid for row in session._known_tree.values()), 10.0))
        children = [row.pid for row in session._known_tree.values() if row.pid != pid]
        os.kill(pid, signal.SIGKILL)
        self.assertTrue(wait_until(lambda: session.state == "disconnected", 5.0))
        report = session.close()
        backend = report["backend"]
        self.assertTrue(backend["agent_exited_before_teardown"])
        self.assertTrue(set(children) <= set(backend["agent_processes"]["members"]), backend)
        for child in children:
            self.assertTrue(wait_until(lambda: not alive(child), 5.0), f"detached child {child} survived")
        # The children carried the session marker: the re-scan after the kill verifies the crash teardown too.
        self.assertTrue(backend["agent_processes"]["confirmed"], backend)
        self.assertNotIn("agent_process_exit_unconfirmed", backend)
        self.assertFalse(report["clean_shutdown"])                    # a crash is never a clean boundary
        self.assertTrue(session._teardown_confirmed)
        self.assertEqual(session.owned_processes(), [])                 # confirmed gone

    def test_x4_a_rejected_turn_after_an_interrupt_keeps_the_relay_blocked(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        session.interrupt()
        self.assertEqual(self.finish(session), "interrupted")
        self.assertIn(sid, session.model_proxy.cancelled_threads)
        self.fake.prompt_script.append(("status", 400))
        with self.assertRaises(AgentRejected):
            session.run_turn("again")
        self.assertEqual(session.state, "interrupted")
        self.assertIn(sid, session.model_proxy.cancelled_threads)   # put back: nothing was admitted
        self.assertEqual(relay_post(session, sid)[0], 400)
        session.run_turn("admitted")                                 # an admitted turn lifts it
        self.assertNotIn(sid, session.model_proxy.cancelled_threads)
        self.fake.finish_turn(sid, "OK")
        self.assertEqual(self.finish(session), "completed")

    def test_steer_is_refused_once_an_interrupt_was_requested(self):
        session = self.started()
        self.running_turn(session)
        self.addCleanup(self.hold_abort())                    # the turn stays running after the abort
        session.interrupt()
        self.assertEqual(session.state, "running")
        before = len(self.prompts())
        with self.assertRaisesRegex(ValueError, "^turn_interrupt_requested"):
            session.steer("too late")
        self.assertEqual(len(self.prompts()), before)
        self.assertFalse(self.events("agent.turn_steered"))

    def test_x5_interrupt_before_the_loop_starts_waits_for_a_confirming_idle(self):
        session = self.started()
        sid = session.thread_id
        session._guard.min_gap_s = 0.5
        session.run_turn("x")
        self.assertTrue(wait_until(lambda: self.fake.busy.get(sid)))
        self.fake.go_idle_silently(sid)                       # the 204 came; the loop has not started
        self.addCleanup(self.hold_abort())
        session.interrupt()
        time.sleep(0.25)
        self.assertEqual(session.state, "running")            # not declared interrupted on the first idle
        self.assertEqual(self.finish(session, 5.0), "interrupted")
        verdict, = self.events("agent.turn_verdict")
        self.assertEqual((verdict["payload"]["reason"], verdict["payload"]["message_id"]),
                         ("interrupt_requested", None))
        self.assertIn("recheck", verdict["payload"]["signals"])

    def test_x5_a_loop_that_starts_after_the_abort_is_aborted_again(self):
        session = self.started()
        sid = session.thread_id
        session._guard.min_gap_s = 2.0
        session.run_turn("x")
        self.assertTrue(wait_until(lambda: self.fake.busy.get(sid)))
        self.fake.go_idle_silently(sid)
        restore = self.hold_abort()
        session.interrupt()                                   # a no-op on OpenCode: no runner yet
        self.assertTrue(wait_until(lambda: session._idle_after_abort))
        restore()
        self.fake.set_busy(sid)                               # the loop starts now
        self.assertEqual(self.finish(session, 5.0), "interrupted")
        self.assertEqual(self.aborts(), [f"/session/{sid}/abort"] * 2)
        self.assertEqual(len(self.events("agent.interrupt_repeated")), 1)
        verdict, = self.events("agent.turn_verdict")
        self.assertIsNotNone(verdict["payload"]["message_id"])    # ended on the aborted answer
        self.assertEqual([r["message"]["command"] for r in self.raw("control")],
                         ["submit_prompt", "interrupt", "interrupt_repeat"])
        self.assertIn(sid, session.model_proxy.cancelled_threads)

    def test_x6_a_busy_event_disarms_a_stale_incomplete_guard(self):
        session = self.started()
        sid = session.thread_id
        session._guard.min_gap_s = 0.5
        session.run_turn("x")
        self.assertTrue(wait_until(lambda: session._busy_seq >= 1))   # the fake's busy event was seen
        self.fake.go_idle_silently(sid)                       # the 204 came before OpenCode turned busy
        session._request_reconcile("submitted")
        self.assertTrue(wait_until(lambda: session._guard.first_idle_at is not None))
        self.fake.set_busy(sid)                               # the loop starts: the busy event disarms the guard
        self.assertTrue(wait_until(lambda: session._guard.first_idle_at is None))
        time.sleep(0.7)                                       # longer than the guard gap
        message = self.fake.assistant(sid)                    # a step that failed: no finish, error not written yet
        self.fake.go_idle_silently(sid)
        self.fake.push("session.idle", {"sessionID": sid})    # the FIRST idle precedes the error write
        time.sleep(0.2)
        self.assertEqual(session.state, "running")
        self.fake.update_info(message, error=API_400)
        self.fake.push("session.idle", {"sessionID": sid})    # the second idle follows the write
        self.assertEqual(self.finish(session), "failed")
        self.assertEqual(session.turn_result["error"]["message"], "opencode_turn_failed:APIError:400")

    def test_a_store_failure_writing_the_verdict_event_does_not_delay_the_terminal(self):
        session = self.started()
        sid = session.thread_id
        real = session.transcript.event

        def event(kind, *args, **kwargs):
            if kind == "agent.turn_verdict":
                raise OSError("store unavailable")
            return real(kind, *args, **kwargs)

        session.transcript.event = event
        session.run_turn("x")
        self.fake.finish_turn(sid, "FINAL")
        self.assertEqual(self.finish(session, 2.0), "completed")
        self.assertEqual(session.completed_final_answer(), "FINAL")
        failures = [e["payload"] for e in self.events("agent.native_handler_error")]
        self.assertIn("turn_verdict", [failure["handler"] for failure in failures])


class DetachedChildTests(SessionCase):
    """Conformance item 15 (b) for OpenCode with a fake server: children a bash call left behind (``cmd &``, setsid)
    are re-parented out of the server's tree; close must still kill them and confirm only after verifying."""

    def orphans(self, session):
        actor = session.actor_dir
        self.assertTrue(wait_until(lambda: all((actor / f"{name}.pid").is_file() and (actor / f"{name}.pid")
                                               .read_text().strip() for name in ("orphan", "detached")), 10.0))
        pids = [int((actor / f"{name}.pid").read_text()) for name in ("orphan", "detached")]
        for pid in pids:
            self.addCleanup(lambda pid=pid: os.kill(pid, signal.SIGKILL) if alive(pid) else None)
        self.assertTrue(all(alive(pid) for pid in pids))
        return pids

    def test_close_kills_children_outside_the_server_tree(self):
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="orphans")))
        pids = self.orphans(session)
        session._refresh_tree()
        self.assertFalse(set(pids) & set(session._known_tree), "the orphans should be outside the server tree")
        report = session.close()
        for pid in pids:
            self.assertTrue(wait_until(lambda: not alive(pid), 5.0), f"detached child {pid} survived close")
        backend = report["backend"]
        self.assertTrue(set(pids) <= {item["pid"] for item in backend["agent_processes"]["marker"]["found"]}, backend)
        self.assertTrue(backend["agent_processes"]["confirmed"], backend)
        self.assertTrue(report["clean_shutdown"], report)
        marker = self.store.read_state("runtime", "agent")["instance_id"]
        self.assertEqual(session.launch.env["AURIGA_AGENT_INSTANCE"], marker)

    def test_an_unverifiable_teardown_is_never_confirmed(self):
        from auriga.cortex.agents import isolation
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="orphans")))
        pids = self.orphans(session)
        with mock.patch.object(isolation, "_marker_rows", side_effect=OSError("ps unavailable")):
            report = session.close()
        backend = report["backend"]
        self.assertTrue(backend["agent_process_exit_unconfirmed"], backend)
        self.assertFalse(report["clean_shutdown"])
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertTrue(all(alive(pid) for pid in pids))   # nothing found them; the receipt says so

    def test_resume_stops_what_the_previous_instance_left_running(self):
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="orphans")))
        thread = session.thread_id
        pids = self.orphans(session)
        session._teardown_confirmed = True
        with mock.patch.object(type(session), "_stop_backend", lambda self: None):   # a gateway that died
            session.close()
        self.assertTrue(all(alive(pid) for pid in pids))
        resumed = self.started(resume=thread)
        for pid in pids:
            self.assertTrue(wait_until(lambda: not alive(pid), 5.0), f"previous instance's child {pid} survived")
        stopped, = self.events("agent.previous_process_stopped")
        self.assertTrue(set(pids) <= set(stopped["payload"]["pids"]))
        self.assertTrue(stopped["payload"]["stop_confirmed"])
        self.assertEqual(resumed.state, "idle")


class TeardownSnapshotTests(unittest.TestCase):
    def test_shared_teardown_snapshot_rules(self):
        """``isolation.teardown_snapshot`` (both sessions reach it through ``isolation.stop_agent_tree``)."""
        from auriga.cortex.agents import isolation
        from auriga.cortex.agents.claude_code import session as claude_session
        from auriga.cortex.agents.opencode import session as opencode_session
        from auriga.cortex.agents.isolation import ProcessInfo, teardown_snapshot
        self.assertIs(claude_session.stop_agent_tree, isolation.stop_agent_tree)
        self.assertIs(opencode_session.stop_agent_tree, isolation.stop_agent_tree)
        claude_rule = teardown_snapshot
        me, own = os.getpid(), os.getpgrp()
        table = [ProcessInfo(100, me, own), ProcessInfo(101, 100, 100), ProcessInfo(102, 1, 102),
                 ProcessInfo(103, 102, 102), ProcessInfo(200, me, own), ProcessInfo(me, 1, own)]
        known = [ProcessInfo(101, 100, 100), ProcessInfo(102, 101, 102), ProcessInfo(200, 100, own),
                 ProcessInfo(300, 100, 300)]
        for alive_root, expected in ((True, [100, 101, 102, 103]), (False, [101, 102, 103])):
            with self.subTest(root_alive=alive_root):
                members = [m.pid for m in teardown_snapshot(100, alive_root, known, table).members]
                self.assertEqual(members, expected)
                self.assertEqual(members, [m.pid for m in claude_rule(100, alive_root, known, table).members])

class RelayBlockTests(unittest.TestCase):
    """``ModelRequestProxy.cancel_pending(reason, threads=...)``: the reopenable block an interrupting session puts on
    its own threads even when no call is in flight (non-Codex profiles only)."""

    def test_block_threads_tombstones_only_the_named_threads(self):
        from auriga.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire
        proxy, _events = self.proxy(AnthropicMessagesWire(client="opencode"))
        self.assertEqual(proxy.block_threads(["ses_a", "", None, "ses_a"], "turn_interrupted"), ["ses_a"])
        self.assertEqual(proxy.cancelled_threads, {"ses_a": "turn_interrupted"})
        self.assertTrue(proxy.reopen_thread("ses_a"))
        codex, _ = self.proxy(OpenAIResponsesWire())
        self.assertEqual(codex.block_threads(["thread-1"], "x"), [])
        self.assertEqual(codex.cancelled_threads, {})

    def proxy(self, wire):
        from auriga.cortex.session.model_proxy import ModelRequestProxy
        events = []
        proxy = ModelRequestProxy("http://127.0.0.1:9", wire=wire, on_event=events.append).start()
        self.addCleanup(proxy.close)
        return proxy, events

    def test_threads_are_tombstoned_reopenably_on_the_opencode_profile(self):
        from auriga.cortex.session.wire import AnthropicMessagesWire
        proxy, events = self.proxy(AnthropicMessagesWire(client="opencode"))
        proxy.cancel_pending("turn_interrupted", threads=("ses_main", "ses_child", "", None))
        self.assertEqual(proxy.cancelled_threads, {"ses_main": "turn_interrupted", "ses_child": "turn_interrupted"})
        self.assertTrue(wait_until(lambda: len([e for e in events if e["kind"] == "model_thread_cancelled"]) == 2))
        self.assertTrue(all(e["reopenable"] for e in events if e["kind"] == "model_thread_cancelled"))
        proxy.cancel_pending("again", threads=("ses_main",))            # already blocked: kept, not re-recorded
        self.assertEqual(proxy.cancelled_threads["ses_main"], "turn_interrupted")
        self.assertTrue(proxy.reopen_thread("ses_main"))
        self.assertFalse(proxy.reopen_thread("ses_main"))
        proxy.register_thread("ses_child", "actor", "ses_child")         # registering lifts it as well
        self.assertEqual(proxy.cancelled_threads, {})

    def test_the_codex_profile_ignores_threads(self):
        from auriga.cortex.session.wire import OpenAIResponsesWire
        proxy, _events = self.proxy(OpenAIResponsesWire())
        proxy.cancel_pending("turn_interrupted", threads=("thread-1",))
        self.assertEqual(proxy.cancelled_threads, {})


class FactoryTests(SessionCase):
    def test_create_session_builds_an_opencode_session_without_starting_anything(self):
        card, planning = {"input_per_million": 1.0}, object()
        session = factory.create_session(self.config(), self.root, self.store, None, price_card=card,
                                         task_planning=planning)
        self.assertIsInstance(session, OpenCodeSession)
        self.assertEqual((session.price_card, session.task_planning, session.model, session.backend_kind,
                          session.state, session.process), (card, planning, "claude-test-model", "opencode",
                                                            "not_started", None))
        session.enter_replay("ses_recorded")
        session.close()

    def test_requirements_pass_with_the_pinned_binary_and_refuse_others(self):
        self.assertIsNone(factory.backend_requirements(self.config()))
        for executable, reason in ((fake_binary(self.dir, version="1.19.4"), "opencode_version_mismatch"),
                                   ("/nonexistent/opencode", "opencode_executable_invalid"),
                                   ("relative/opencode", "opencode_executable_invalid")):
            with self.subTest(reason=reason):
                config = Config.from_dict({**self.config().to_dict(), "agent": {
                    "backend": "opencode", "opencode": {**self.config().agent.opencode.to_dict(),
                                                        "executable": executable}}}) \
                    if executable.startswith("/") else None
                if config is None:
                    self.assertEqual(factory.opencode_runtime_problem(executable), reason)
                    continue
                with self.assertRaisesRegex(ConfigError, f"^agent_backend_unavailable:opencode: {reason}; install "
                                                         "opencode-ai@1.19.5"):
                    factory.backend_requirements(config)
        self.assertEqual(factory.opencode_runtime_problem(None), "opencode_executable_not_set")

    def test_requirement_order_and_upstream(self):
        quiet = {"self_briefing": False, "visual_review_model": None}
        cases = (
            # features.sandbox_enabled is accepted (R4-8): the configuration checks after it still apply.
            ({"features": {**quiet, "sandbox_enabled": True}, "agent": {"opencode": {}}}, "opencode_upstream_incomplete"),
            ({"features": quiet, "agent": {"opencode": {}}}, "opencode_upstream_incomplete"),
            ({"features": quiet, "agent": {"opencode": {"upstream": {"wire_api": "openai_chat", "base_url": "http://x",
                                                                     "api_key_env": "K"}}}},
             "opencode_upstream_wire_unsupported:openai_chat"),
            ({"features": quiet, "agent": {"opencode": {"upstream": {"source": "codex_home",
                                                                     "wire_api": "anthropic_messages"}}}},
             "opencode_upstream_codex_home_requires_openai_responses"),
            ({"features": quiet, "agent": {"opencode": {"upstream": {"wire_api": "openai_responses",
                                                                     "base_url": "http://x", "api_key_env": "K"}}}},
             "agent_backend_unavailable:opencode"),
        )
        for document, code in cases:
            with self.subTest(code=code), mock.patch.object(factory, "opencode_runtime_problem",
                                                            return_value="opencode_executable_not_set") as probe:
                agent = {"backend": "opencode", **document.pop("agent", {})}
                config = Config.from_dict({**document, "agent": agent})
                with self.assertRaises(ConfigError) as raised:
                    factory.backend_requirements(config)
                self.assertTrue(str(raised.exception).startswith(code), str(raised.exception))
                self.assertEqual(probe.called, code.startswith("agent_backend_unavailable"))
        with mock.patch.object(factory, "opencode_runtime_problem", return_value=None):
            codex_home = Config.from_dict({"features": quiet, "agent": {"backend": "opencode", "opencode": {
                "upstream": {"source": "codex_home"}}}})
            self.assertIsNone(factory.backend_requirements(codex_home))


if __name__ == "__main__":
    unittest.main()
