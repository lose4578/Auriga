"""Official fresh-agent isolation and lifecycle; localhost fixtures only."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import tomllib
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

from auriga.cortex.session.codex import CodexSession, MODEL_PROXY_PROVIDER
from auriga.core.presentation.actor_presentation import ActorPresentation
from auriga.cortex.session.codex_evidence_agent import CodexEvidenceAgent, CONTEXT_POLICY, EVIDENCE_CONFIG, MODEL
from auriga.core.config import Config
from auriga.core.storage.images import image_url, store_image_settings
from auriga.core.storage.store import Store
from auriga.cortex.context.task_planning import TaskPlanning
from support.host_capabilities import bubblewrap_namespaces, require


SCHEMA = dict(type="object", properties=dict(assessment=dict(type="string")), required=["assessment"], additionalProperties=False)


class FakeClient:
    def __init__(self, notification, rejected, *, model=MODEL, contaminated=False):
        self.notification, self.rejected = notification, rejected
        self.model, self.contaminated = model, contaminated
        self.calls = []
        self.events = []

    def start(self): self.calls.append(("start",))
    def initialize(self): self.calls.append(("initialize",))
    def thread_start(self, parameters):
        self.calls.append(("thread/start", parameters))
        return dict(model=self.model, modelProvider=MODEL_PROXY_PROVIDER, instructionSources=["actor/AGENTS.md"] if self.contaminated else [],
                    thread=dict(id="fresh-helper", ephemeral=True, turns=[], forkedFromId=None))
    def turn_start(self, thread_id, items, parameters):
        self.calls.append(("turn/start", thread_id, items, parameters))
        self.events = [SimpleNamespace(method="turn/completed", payload=dict(threadId=thread_id,
            turn=dict(id="helper-turn", status="completed", items=[dict(id="final", type="agentMessage", phase="final_answer",
                      text=json.dumps(dict(assessment="uncertain")))])))]
        return dict(turn=dict(id="helper-turn"))
    def register_turn_notifications(self, turn_id): self.calls.append(("register", turn_id))
    def next_turn_notification(self, turn_id): return self.events.pop(0)
    def unregister_turn_notifications(self, turn_id): self.calls.append(("unregister", turn_id))
    def request(self, method, params, **kwargs): self.calls.append((method, params)); return dict(status="notLoaded")
    def turn_interrupt(self, thread_id, turn_id): self.calls.append(("turn/interrupt", thread_id, turn_id))
    def close(self): self.calls.append(("close",))


class EvidenceAgentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / "run", dict(run_id="fixture", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(self.store.close)
        self.session = SimpleNamespace(store=self.store, private=self.store.root / "private", socket_path="fixture.sock",
                                       thread_id="ACTOR_HISTORY_MUST_NOT_APPEAR", client=object(), sandbox_enabled=True)
        self.session.private.mkdir(parents=True, exist_ok=True)
        self.proxy_calls = []
        self.session.model_proxy = SimpleNamespace(
            register_thread=lambda *args:self.proxy_calls.append(("register", *args)),
            unregister_thread=lambda *args:self.proxy_calls.append(("unregister", *args)),
            cancel_thread=lambda *args:self.proxy_calls.append(("cancel", *args)))
        self.clients = []
    def factory(self, notification, rejected):
        client = FakeClient(notification, rejected)
        self.clients.append(client)
        return client

    def test_empty_workspace_owns_separate_namespace_without_actor_mounts(self):
        cwd = self.session.private / "evidence" / "workspace"
        cwd.mkdir(parents=True)
        self.session.codex_home = self.session.private / "actor-codex-home"
        self.session.codex_home.mkdir()
        source_catalog = self.session.codex_home / "actor-models.json"
        source_catalog.write_text('{"models": []}\n')
        (self.session.codex_home / "config.toml").write_text(
            'model_provider = "fixture"\n'
            'model_catalog_json = ' + json.dumps(str(source_catalog)) + '\n'
            '[features]\nplugins = false\n'
            '[model_providers.fixture]\nbase_url = "http://127.0.0.1:54321/v1"\n'
            '[mcp_servers."fixture service"]\ncommand = "host-mcp"\n'
            'args = [\n  "--flag",\n  "two words"\n]\n'
            '[mcp_servers."fixture service".env]\nMULTILINE = "line one\\nline two"\n'
            '[plugins."fixture@host"]\nenabled = true\n'
            '[marketplaces.local]\nsource_type = "local"\nsource = "/host/plugins"\n'
            '[apps."fixture@host"]\nenabled = true\n'
            '[skills.host]\npath = "/host/skills"\n'
            '[shell_environment_policy]\ninherit = "all"\n')
        self.session.native_config = tomllib.loads((self.session.codex_home / "config.toml").read_text())
        auth_path = self.session.codex_home / "auth.json"
        auth_path.write_text('{"fixture_auth":true}\n')
        self.session.cli_executable = "fixture-native"
        self.session.env = {"PATH": "/usr/bin:/bin", "HOME": "host-home", "CODEX_HOME": "host-codex-home"}
        self.session.model_proxy = SimpleNamespace(base_url="http://127.0.0.1:54321/v1")
        account_ca = self.session.private / "account-control-ca.pem"
        account_ca.write_text("fixture CA")
        self.session.account_ca_bytes = b"fixture CA"
        self.session.account_control = SimpleNamespace(server=SimpleNamespace(server_port=54322), ca_path=account_ca)
        self.session.ipc_base = self.store.root / "ipc"
        with mock.patch("auriga.cortex.session.codex_evidence_agent.CodexSandbox") as factory:
            sandbox = factory.return_value
            sandbox.prepare.return_value = (["sandbox-launch"], {})
            client = CodexEvidenceAgent(self.session)._client(lambda _:None, lambda _:None, cwd)
        self.assertEqual(factory.call_args.kwargs, {"ports": [54321, 54322], "packages": False})
        self.assertEqual(factory.call_args.args[2:4], (cwd, cwd.parent / "runtime" / "codex-home"))
        self.assertNotEqual(factory.call_args.args[1], self.session.ipc_base)
        self.assertIs(client.sandbox, sandbox)
        self.addCleanup(client.server_pool.shutdown)
        self.addCleanup(client.close)
        command = client.config.launch_args_override
        self.assertEqual(command[:3], ("sandbox-launch", str(Path("fixture-native").resolve()), "--no-daemon"))
        self.assertEqual(client.config.cwd, str(cwd))
        self.assertEqual(list(cwd.iterdir()), [])
        helper_config_path = cwd.parent / "runtime" / "codex-home" / "config.toml"
        helper_config = tomllib.loads(helper_config_path.read_text())
        self.assertEqual(helper_config["model_provider"], "fixture")
        self.assertEqual(helper_config["features"]["plugins"], False)
        self.assertEqual(helper_config["model_providers"]["fixture"]["base_url"], "http://127.0.0.1:54321/v1")
        self.assertEqual(helper_config["model_catalog_json"], str(cwd.parent / "runtime" / "codex-home" / "models.json"))
        self.assertTrue(json.loads(Path(helper_config["model_catalog_json"]).read_text())["models"])
        for key in ("mcp_servers", "plugins", "marketplaces", "apps", "skills", "shell_environment_policy"):
            self.assertNotIn(key, helper_config)
        helper_auth = cwd.parent / "runtime" / "codex-home" / "auth.json"
        self.assertFalse(helper_auth.is_symlink())
        self.assertEqual(json.loads(helper_auth.read_text()), {"OPENAI_API_KEY": "auriga-local-relay"})
        self.assertEqual(EVIDENCE_CONFIG["features.plugins"], False)
        self.assertEqual(EVIDENCE_CONFIG["features.apps"], False)

    def test_disabled_sandbox_uses_private_home_and_clean_environment_without_bubblewrap(self):
        self.session.sandbox_enabled = False
        self.session.native_config = {"model_provider": "fixture", "features": {"plugins": False},
                                      "mcp_servers": {"host-service": {"command": "host-command"}}}
        self.session.cli_executable = sys.executable
        self.session.account_ca_bytes = b"fixture CA"
        cwd = self.session.private / "evidence" / "workspace"
        cwd.mkdir(parents=True)
        with mock.patch("auriga.cortex.session.codex_evidence_agent.CodexSandbox", side_effect=AssertionError("sandbox unavailable")):
            client = CodexEvidenceAgent(self.session)._client(lambda _:None, lambda _:None, cwd)
        self.addCleanup(client.server_pool.shutdown)
        self.addCleanup(client.close)
        self.assertIsNone(client.sandbox)
        command = client.config.launch_args_override
        self.assertIn("features.shell_tool=false", command)
        self.assertIn("features.memories=false", command)
        environment_probe = command[:command.index("--no-daemon")] + ("-c", "import json,os; print(json.dumps(dict(os.environ)))")
        with mock.patch.dict(os.environ, {"AURIGA_HOST_SECRET": "host-only", "HOME": "/host-home",
                                          "CODEX_HOME": "/host-codex-home"}):
            result = subprocess.run(environment_probe, cwd=cwd, capture_output=True, text=True, timeout=5, check=True)
        environment = json.loads(result.stdout)
        home = cwd.parent / "runtime" / "codex-home"
        self.assertEqual(environment["HOME"], str(cwd))
        self.assertEqual(environment["CODEX_HOME"], str(home))
        self.assertEqual(environment["TMPDIR"], str(cwd.parent / "runtime" / "tmp"))
        self.assertEqual(environment["CODEX_CA_CERTIFICATE"], str(home / "account-control-ca.pem"))
        self.assertNotIn("AURIGA_HOST_SECRET", environment)
        self.assertEqual(json.loads((home / "auth.json").read_text()), {"OPENAI_API_KEY": "auriga-local-relay"})
        self.assertNotIn("mcp_servers", tomllib.loads((home / "config.toml").read_text()))
        self.assertFalse((cwd.parent / "runtime" / "sandbox").exists())

        CodexEvidenceAgent(self.session, client_factory=self.factory).review_evidence(
            input_items=[dict(type="text", text="Selected evidence")], output_schema=SCHEMA, request_id="unsandboxed")
        calls = self.clients[0].calls
        thread = next(call[1] for call in calls if call[0] == "thread/start")
        turn = next(call[3] for call in calls if call[0] == "turn/start")
        self.assertEqual(thread["sandbox"], "danger-full-access")
        self.assertEqual(turn["sandboxPolicy"], {"type": "dangerFullAccess"})
        self.assertEqual(thread["dynamicTools"], [])

    def test_official_fresh_thread_has_minimal_input_schema_and_shared_provider(self):
        agent = CodexEvidenceAgent(self.session, client_factory=self.factory)
        items = [dict(type="text", text="Selected evidence only"), dict(type="image", url="data:image/png;base64,fixture")]
        result = agent.review_evidence(input_items=items, output_schema=SCHEMA, request_id="review-one")
        self.assertEqual(result["answer"], dict(assessment="uncertain"))
        self.assertEqual(result["context_policy"], CONTEXT_POLICY)
        calls = self.clients[0].calls
        parameters = next(call[1] for call in calls if call[0] == "thread/start")
        self.assertTrue(parameters["ephemeral"])
        self.assertEqual(parameters["dynamicTools"], [])
        self.assertEqual(parameters["model"], MODEL)
        self.assertEqual(EVIDENCE_CONFIG["project_doc_max_bytes"], 0)
        self.assertFalse(EVIDENCE_CONFIG["features.memories"])
        self.assertFalse(EVIDENCE_CONFIG["features.code_mode.enabled"])
        self.assertEqual(parameters["environments"], [])
        self.assertEqual(parameters["selectedCapabilityRoots"], [])
        self.assertEqual(self.proxy_calls[:2], [("register", "fresh-helper", "sol_visual_review", "review-one"),
                                               ("unregister", "fresh-helper")])
        self.assertEqual(list(Path(parameters["cwd"]).iterdir()), [])
        turn = next(call for call in calls if call[0] == "turn/start")
        self.assertEqual(turn[2], items)
        self.assertEqual(turn[3]["outputSchema"], SCHEMA)
        self.assertNotIn("ACTOR_HISTORY_MUST_NOT_APPEAR", json.dumps(calls))
        self.assertFalse(any(call[0] in ("thread/fork", "thread/resume") for call in calls))
        self.assertTrue(any(call[0] == "thread/unsubscribe" for call in calls))
        # Another request obtains another client and another empty cwd.
        agent.review_evidence(input_items=items, output_schema=SCHEMA, request_id="review-two")
        self.assertEqual(len(self.clients), 2)
        second = next(call[1] for call in self.clients[1].calls if call[0] == "thread/start")
        self.assertNotEqual(parameters["cwd"], second["cwd"])

    def test_actual_turn_hides_hashes_and_roundtrips_schema_references(self):
        reference = self.store.put_artifact(b"measured evidence", "trace", "text/plain")
        fingerprint = "b" * 64
        action_id = "action-" + "c" * 24
        image = "data:image/png;base64," + "d" * 64
        items = [dict(type="text", text=json.dumps(dict(source_sha256=fingerprint,
                 action_id=action_id, reference=reference, measured_steps=7))),
                 dict(type="text", text="Evidence fingerprint=" + fingerprint),
                 dict(type="image", url=image)]
        schema = {**SCHEMA, "properties": {"assessment": dict(type="string", enum=[action_id, reference])}}

        class EchoReference(FakeClient):
            def turn_start(self, thread_id, items, parameters):
                result = super().turn_start(thread_id, items, parameters)
                value = parameters["outputSchema"]["properties"]["assessment"]["enum"][0]
                self.events[0].payload["turn"]["items"][0]["text"] = json.dumps(dict(assessment=value))
                return result

        client = EchoReference(None, None)
        agent = CodexEvidenceAgent(self.session, client_factory=lambda *_: client)
        result = agent.review_evidence(input_items=items, output_schema=schema, request_id="hash-boundary")
        turn = next(call for call in client.calls if call[0] == "turn/start")
        wire = json.dumps([turn[2][:-1], turn[3]["outputSchema"]])
        for private in (fingerprint, action_id, reference):
            self.assertNotIn(private, wire)
        self.assertEqual(turn[2][-1]["url"], image)
        self.assertEqual(result["answer"], dict(assessment=action_id))
        model_schema = turn[3]["outputSchema"]
        self.assertEqual(ActorPresentation(self.store).restore(model_schema), schema)
        self.assertEqual(json.loads(turn[2][0]["text"])["measured_steps"], 7)
        self.assertEqual(items[0]["text"], json.dumps(dict(source_sha256=fingerprint,
            action_id=action_id, reference=reference, measured_steps=7)))
        self.assertEqual(self.store.resolve_artifact(reference)[1], b"measured evidence")

    def test_contaminated_context_or_substituted_model_never_starts_a_turn(self):
        for changes in (dict(contaminated=True), dict(model="different-model")):
            with self.subTest(changes=changes):
                clients = []
                def factory(notification, rejected):
                    client = FakeClient(notification, rejected, **changes)
                    clients.append(client)
                    return client
                agent = CodexEvidenceAgent(self.session, client_factory=factory)
                with self.assertRaisesRegex(ValueError, "verified_fresh_evidence_context"):
                    agent.review_evidence(input_items=[], output_schema=SCHEMA, request_id="invalid-"+str(len(changes))+str(changes[ next(iter(changes))]))
                self.assertFalse(any(call[0] == "turn/start" for call in clients[0].calls))

    def test_pre_cancelled_request_creates_no_agent(self):
        cancel = threading.Event(); cancel.set()
        agent = CodexEvidenceAgent(self.session, client_factory=self.factory)
        with self.assertRaises(TimeoutError):
            agent.review_evidence(input_items=[], output_schema=SCHEMA, request_id="cancelled", cancel_event=cancel)
        self.assertEqual(self.clients, [])

    def test_cancellation_during_start_does_not_initialize_or_leak_client(self):
        entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
        class SlowStart(FakeClient):
            def start(self):
                entered.set()
                release.wait(2)
                super().start()
        client = SlowStart(None, None)
        errors = []
        def run():
            try:
                CodexEvidenceAgent(self.session, client_factory=lambda *_:client).review_evidence(
                    input_items=[], output_schema=SCHEMA, request_id="start-race", cancel_event=cancel)
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(entered.wait(1))
        cancel.set(); release.set(); worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(errors[0], TimeoutError)
        self.assertEqual(client.calls, [("start",), ("close",)])

    def test_abnormal_active_turn_is_cancelled_before_helper_cleanup(self):
        class UnexpectedTool(FakeClient):
            def turn_start(self, *args):
                result = super().turn_start(*args)
                self.events.insert(0, SimpleNamespace(method="item/completed", payload=dict(
                    threadId="fresh-helper", item=dict(id="tool", type="commandExecution"))))
                return result
        client = UnexpectedTool(None, None)
        with self.assertRaisesRegex(ValueError, "unexpected_tool"):
            CodexEvidenceAgent(self.session, client_factory=lambda *_:client).review_evidence(
                input_items=[], output_schema=SCHEMA, request_id="unexpected-tool")
        self.assertIn(("cancel", "fresh-helper"), self.proxy_calls)
        self.assertIn(("turn/interrupt", "fresh-helper", "helper-turn"), client.calls)
        self.assertLess(client.calls.index(("turn/interrupt", "fresh-helper", "helper-turn")), client.calls.index(("close",)))

    def test_invalid_final_json_after_terminal_does_not_cancel_completed_turn(self):
        class InvalidFinal(FakeClient):
            def turn_start(self, *args):
                result = super().turn_start(*args)
                self.events[0].payload["turn"]["items"][0]["text"] = "invalid-json"
                return result
        client = InvalidFinal(None, None)
        with self.assertRaises(ValueError):
            CodexEvidenceAgent(self.session, client_factory=lambda *_:client).review_evidence(
                input_items=[], output_schema=SCHEMA, request_id="invalid-final")
        self.assertFalse(any(call[0] == "cancel" for call in self.proxy_calls))
        self.assertFalse(any(call[0] == "turn/interrupt" for call in client.calls))


class NativeEvidenceAgentTest(unittest.TestCase):
    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_real_official_agent_thread_image_schema_and_shared_budget(self):
        self._run_native_fixture(sandbox_enabled=True)

    def test_real_official_agent_without_sandbox(self):
        with mock.patch("auriga.cortex.session.codex_sandbox.CodexSandbox.prepare", side_effect=AssertionError("sandbox unavailable")):
            self._run_native_fixture(sandbox_enabled=False)

    def _run_native_fixture(self, *, sandbox_enabled):
        requests = []
        blocked, release = threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((body, dict(self.headers)))
                index = len(requests)
                if "BLOCK_HELPER" in json.dumps(body):
                    blocked.set()
                    release.wait(5)
                text = "ACTOR_PRIVATE_ANSWER_SENTINEL" if index == 1 else json.dumps(dict(assessment="uncertain"))
                item = dict(id="msg-fixture-"+str(index), type="message", role="assistant", phase="final_answer", status="completed",
                            content=[dict(type="output_text", text=text, annotations=[])])
                response = dict(id="resp-fixture", object="response", model=body["model"], status="completed", output=[item],
                                usage=dict(input_tokens=20, output_tokens=8, total_tokens=28, input_tokens_details=dict(cached_tokens=0),
                                           output_tokens_details=dict(reasoning_tokens=0)))
                events = [dict(type="response.created", response={**response, "status":"in_progress", "output":[]}),
                          dict(type="response.output_item.added", output_index=0, item={**item, "status":"in_progress", "content":[]}),
                          dict(type="response.output_text.delta", item_id=item["id"], output_index=0, content_index=0, delta=text),
                          dict(type="response.output_item.done", output_index=0, item=item),
                          dict(type="response.completed", response=response)]
                payload = "".join("data: "+json.dumps(event)+"\n\n" for event in events).encode()
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload))); self.end_headers()
                try: self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"; source.mkdir()
            (source / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nbase_url="http://127.0.0.1:'+str(server.server_port)+'/v1"\n')
            store = Store(root / "run", dict(run_id="official-evidence", episode_id="e", env_instance_id="i", env_generation=1))
            self.addCleanup(store.close)
            planning = TaskPlanning(store)
            observation = dict(store.identity, observation_id="fixture-plan-observation")
            plan = "ACTOR_PRIVATE_PLAN_SENTINEL"
            attention = [dict(concern="ACTOR_PRIVATE_ATTENTION_SENTINEL",
                              check="Use visible co-motion to reassess retention.",
                              when="After the next useful transfer action.")]
            planning.call(dict(plan=plan, attention=attention, ledger=[]), store.identity, observation, {})
            config = Config.from_dict(dict(model=dict(codex_home=str(source)), features=dict(sandbox_enabled=sandbox_enabled)))
            class FixtureSession(CodexSession):
                def _prepare(self, *args):
                    provider = super()._prepare(*args)
                    # Background actor memory consolidation is independent of
                    # this lifecycle fixture and would add asynchronous calls.
                    path = self.codex_home / "config.toml"
                    path.write_text(path.read_text().replace("memories = true", "memories = false"))
                    return provider
            session = FixtureSession(store.root, store, lambda _: {}, config=config, task_planning=planning)
            def close_session():
                bridge = session.client._proc if session.client else None
                session.close()
                if bridge:
                    for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                        if stream and not stream.closed:
                            stream.close()
            self.addCleanup(close_session)
            session.start([], "ACTOR_PRIVATE_CONTEXT_SENTINEL")
            actor_thread = session.thread_id
            actor_config = (session.codex_home / "config.toml").read_bytes()
            session.run_turn("ACTOR_PRIVATE_QUESTION_SENTINEL")
            self.assertTrue(session.turn_done.wait(5))
            self.assertEqual(session.state, "completed")
            actor_turn = session.turn_id
            actor_body, actor_headers = requests[0]
            self.assertEqual({k.lower(): v for k, v in actor_headers.items()}.get("thread-id"), actor_thread)
            self.assertNotIn("Current task decision boundary", json.dumps(actor_body))
            actor_text = [part.get("text", "") for item in actor_body["input"] for part in item.get("content", [])]
            self.assertTrue(any(text.startswith("You are Codex, an agent based on GPT-6.") for text in actor_text))
            actor_tools = next(item["tools"] for item in actor_body["input"] if item.get("type") == "additional_tools")
            self.assertTrue(any(tool.get("name") == "exec_command" for namespace in actor_tools for tool in namespace.get("tools", [])))
            pixels = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
            url, _ = image_url(pixels, store_image_settings(store))
            evidence_hash = "abcdef0123456789" * 4
            evidence_id = "review-" + "cdef0123456789abc" * 2
            try:
                result = CodexEvidenceAgent(session).review_evidence(input_items=[dict(type="text", text="Only selected evidence fingerprint=" + evidence_hash),
                           dict(type="text", text=json.dumps(dict(receipt_sha256=evidence_hash, action_id=evidence_id, executed_steps=7))),
                           dict(type="image", url=url)], output_schema=SCHEMA, request_id="native-fixture")
            except ValueError:
                self.fail(str([event["payload"] for event in store.events() if event["type"] in ("visual.agent_turn_unverified", "visual.agent_features")]))
            self.assertEqual(result["answer"], dict(assessment="uncertain"))
            self.assertNotEqual(result["thread_id"], actor_thread)
            self.assertEqual(session.thread_id, actor_thread)
            self.assertEqual(session.state, "completed")
            self.assertEqual(session.turn_id, actor_turn)
            body, headers = next((body,headers) for body,headers in requests if {k.lower():v for k,v in headers.items()}.get("thread-id") == result["thread_id"])
            feature_state = [event["payload"] for event in store.events() if event["type"] == "visual.agent_features"]
            flags = feature_state[0]["features"]
            for name in ("multi_agent", "multi_agent_v2", "code_mode", "code_mode_host", "memories", "hooks", "goals"):
                self.assertIs(flags[name], False)
            self.assertFalse(body.get("tools"), [item.get("name", item.get("type")) for item in body.get("tools", [])])
            # Native 0.160 retains baseline capability definitions in its
            # additional_tools context item despite disabled feature switches.
            # Selected robot tools are absent; any observed tool/context item
            # invalidates the helper result in the production collector.
            self.assertNotIn("operation_verify", json.dumps(body))
            self.assertNotIn("ACTOR_HISTORY_MUST_NOT_APPEAR", json.dumps(body))
            self.assertEqual(body["model"], MODEL)
            self.assertNotIn("ACTOR_PRIVATE_CONTEXT_SENTINEL", json.dumps(body))
            self.assertNotIn("ACTOR_PRIVATE_QUESTION_SENTINEL", json.dumps(body))
            self.assertNotIn("ACTOR_PRIVATE_ANSWER_SENTINEL", json.dumps(body))
            self.assertNotIn(plan, json.dumps(body))
            self.assertNotIn(attention[0]["concern"], json.dumps(body))
            self.assertIn("Only selected evidence", json.dumps(body))
            self.assertNotIn(evidence_hash, json.dumps(body))
            self.assertNotIn(evidence_id, json.dumps(body))
            self.assertIn("input_image", json.dumps(body))
            headers = {name.lower(): value for name, value in headers.items()}
            self.assertEqual(headers.get("thread-id"), result["thread_id"], headers)
            self.assertEqual(session.model_proxy.request_count, 2)
            self.assertEqual((session.codex_home / "config.toml").read_bytes(), actor_config)
            self.assertEqual(store.read_state("runtime", "model_requests")["limit"], 180)
            owners = [event["payload"] for event in store.events() if event["type"].endswith("model_thread_registered")]
            self.assertTrue(any(item.get("source") == "sol_visual_review" and item.get("caller_request_id") == "native-fixture" for item in owners))

            errors, cancel = [], threading.Event()
            def run_cancelled():
                try:
                    CodexEvidenceAgent(session).review_evidence(input_items=[dict(type="text", text="BLOCK_HELPER")],
                        output_schema=SCHEMA, request_id="native-cancel", cancel_event=cancel)
                except Exception as error:
                    errors.append(error)
            worker = threading.Thread(target=run_cancelled)
            worker.start()
            self.assertTrue(blocked.wait(5))
            cancel.set(); worker.join(3)
            release.set()
            self.assertFalse(worker.is_alive())
            self.assertIsInstance(errors[0], TimeoutError)
            self.assertFalse(session.model_proxy.transport_blocked)
            self.assertEqual(session.thread_id, actor_thread)
            self.assertEqual(session.turn_id, actor_turn)
            self.assertEqual(session.state, "completed")
            planning.call(dict(operation="update", attention=[]), store.identity, observation, {})
            session.run_turn("Actor continues after helper cancellation")
            self.assertTrue(session.turn_done.wait(5))
            self.assertEqual(session.state, "completed")
            second_actor_body, actor_headers = requests[-1]
            self.assertEqual({k.lower(): v for k, v in actor_headers.items()}.get("thread-id"), actor_thread)
            self.assertNotIn("Current task decision boundary", json.dumps(second_actor_body))
            self.assertEqual(second_actor_body["input"][:len(actor_body["input"])], actor_body["input"])
            for helper_body, helper_headers in requests:
                if {k.lower(): v for k, v in helper_headers.items()}.get("thread-id") != actor_thread:
                    self.assertNotIn(plan, json.dumps(helper_body))
                    self.assertNotIn(attention[0]["concern"], json.dumps(helper_body))
            self.assertEqual(len(requests), 4)
            self.assertEqual(session.model_proxy.request_count, 3)
            self.assertEqual(session.model_proxy.attempt_count, 4)
