"""Golden relay scenarios for the WireAdapter extraction (P0-2).

``capture_forwarding()`` drives a real ``ModelRequestProxy`` against a scripted loopback
upstream and records, per scenario, every forwarded upstream request (path, headers, body
bytes), every client-visible answer (status, reason, headers, body bytes) and the normalised
evidence-event stream. ``capture_presentation()`` records the Responses egress scrub of
requests with planted hashes, paths and long ids. Both use only public behaviour, so the
same code captured the fixtures from the pre-extraction source (c4ff16a)::

    git archive c4ff16a | tar -x -C /tmp/orig
    PYTHONPATH=/tmp/orig mamba run -n Auriga python tests/wire_golden_fixture.py --write c4ff16a

and ``tests/test_wire_golden.py`` replays it against the current tree. Volatile values are
normalised: event clocks (``wall_time``, ``elapsed_s``, ``first_upstream_byte_s``), the
random transport id, ``Date``/``Server`` response headers, the upstream ``Host`` header and
relay keepalive frames (their bytes are pinned by the adapter unit tests).
"""
import base64
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import MethodType, SimpleNamespace
from unittest import mock

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FORWARDING_FIXTURE = FIXTURES / "wire_forwarding_golden.json"
PRESENTATION_FIXTURE = FIXTURES / "wire_present_responses_golden.json"
KEEPALIVE = b'event: auriga.keepalive\ndata: {"type":"auriga.keepalive"}\n\n'
VOLATILE_EVENT_KEYS = frozenset({"wall_time", "elapsed_s", "first_upstream_byte_s"})
DROPPED_RESPONSE_HEADERS = frozenset({"date", "server"})
INLINE_BODY_LIMIT = 24_000

CLIENT_HEADERS = (
    ("Content-Type", "application/json"), ("Accept", "text/event-stream"),
    ("Authorization", "Bearer client-token"), ("ChatGPT-Account-Id", "client-account"),
    ("session_id", "session-fixture"), ("originator", "codex_exec"), ("User-Agent", "codex-fixture/1.0"),
    ("X-Auriga-Source", "helper"), ("X-Auriga-Request-Id", "req-private"), ("X-Auriga-Deadline-Unix-Ms", "1"),
    ("Accept-Encoding", "gzip, zstd"), ("Connection", "keep-alive"), ("Cookie", "c=1"), ("x-api-key", "client-key"),
    ("OpenAI-Beta", "responses=experimental"),
)


# ------------------------------------------------------------------ request/response payloads
def body(request):
    # Non-compact separators make any re-encoding by the relay visible in the bytes.
    return json.dumps(request, separators=(", ", ": "), ensure_ascii=True).encode()


STREAM_BODY = (b'{"model": "gpt-fixture",  "stream": true, "store": false, "tools": [],\n'
               b' "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", '
               b'"text": "literal \\u4e2d and raw \xe4\xb8\xad"}]}], "reasoning": {"effort": "high"}}')
PLAIN_BODY = b'{"model": "gpt-fixture", "input": [{"role": "user", "content": "ping"}], "store": false}'
COMPACT_BODY = body(dict(model="gpt-fixture", input=[dict(type="message", role="user", content=[
    dict(type="input_text", text="compact me")])], instructions="base"))


def sse(*events):
    return 200, [("Content-Type", "text/event-stream")], b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)


def completed(index, model="gpt-fixture-served"):
    return sse(dict(type="response.created", response=dict(id=f"resp-{index}", status="in_progress")),
               dict(type="response.output_text.delta", delta="ok"),
               dict(type="response.completed", response=dict(
                   id=f"resp-{index}", model=model, status="completed",
                   output=[dict(type="message", role="assistant", content=[dict(type="output_text", text="ok")])],
                   usage=dict(input_tokens=11, input_tokens_details=dict(cached_tokens=4), output_tokens=3,
                              output_tokens_details=dict(reasoning_tokens=1), total_tokens=14))))


def json_reply(status, value, headers=()):
    return status, [("Content-Type", "application/json"), *headers], json.dumps(value).encode()


def failed_sse(status="failed", error=None):
    response = dict(id="resp-failed", model="gpt-fixture-served", status=status,
                    output=[dict(type="function_call", name="unsafe_terminal_tool", call_id="c1", arguments="{}")],
                    error=error, usage=dict(input_tokens=3, output_tokens=1))
    code, headers, payload = sse(dict(type="response." + status, response=response))
    leaked = b'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","name":"unsafe_tool"}}\n\n'
    return code, headers, leaked + payload


def image_url(seed, size):
    chunk = hashlib.sha256(seed.encode()).digest()
    return "data:image/png;base64," + base64.b64encode((chunk * (size // 32 + 1))[:size]).decode()


def image_history(steps, size, duplicate_every=0, first=0):
    items = [dict(type="message", role="user", content=[dict(type="input_text", text="Task: stack the blocks.")])]
    for i in range(first, first + steps):
        call_id = f"call_{i:03d}"
        seed = i - 1 if duplicate_every and i and i % duplicate_every == 0 else i
        items.append(dict(type="function_call", call_id=call_id, name="robot_observe", arguments="{}"))
        items.append(dict(type="function_call_output", call_id=call_id, output=[
            dict(type="input_text", text=json.dumps(dict(bundle=dict(
                observation_id=f"obs_{i}", cameras=[dict(camera_id="head"), dict(camera_id="wrist")])))),
            dict(type="input_image", image_url=image_url(f"head-{seed}", size)),
            dict(type="input_image", image_url=image_url(f"wrist-{seed}", size), detail="high")]))
        if i % 5 == 4:
            items.append(dict(type="message", role="assistant", content=[dict(type="output_text", text=f"step {i} done")]))
    return items


# ---------------------------------------------------------------------------- harness
class Upstream:
    """Scripted loopback upstream; records each request exactly as received."""

    def __init__(self):
        self.requests, self.script = [], []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *_): pass

            def do_POST(self):
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append(dict(path=self.path, body=payload, headers=[
                    [key, value] for key, value in self.headers.items() if key.lower() != "host"]))
                entry = owner.script.pop(0) if owner.script else completed(len(owner.requests))
                status, headers, reply = entry[:3]
                if len(entry) > 3:
                    time.sleep(entry[3])  # slow upstream: the relay sends keepalive frames meanwhile
                self.send_response(status)
                for key, value in headers:
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(reply)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    self.wfile.write(reply)
                    self.wfile.flush()
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)


def closed_port_url():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"http://127.0.0.1:{port}/v1"


def payload_record(data):
    record = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if len(data) <= INLINE_BODY_LIMIT:
        try:
            record["text"] = data.decode()
        except UnicodeDecodeError:
            record["base64"] = base64.b64encode(data).decode()
    return record


def finished_calls(proxy):
    with proxy.lock:
        return sum(event["kind"] == "model_call_finished" for event in proxy.events)


def call(proxy, path="/responses", data=STREAM_BODY, headers=CLIENT_HEADERS, *, method="POST", admitted=True,
         extra_headers=(), expect_keepalive=False):
    """One client request; waits for the relay's own bookkeeping before closing."""
    before = finished_calls(proxy)
    port = int(proxy.base_url.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.putrequest(method, path, skip_accept_encoding=True)
        for key, value in (*headers, *extra_headers):
            connection.putheader(key, value)
        connection.putheader("Content-Length", str(len(data)))
        connection.endheaders(data)
        response = connection.getresponse()
        received = response.read()
        record = dict(status=response.status, reason=response.reason,
                      headers=[[key, value] for key, value in response.getheaders()
                               if key.lower() not in DROPPED_RESPONSE_HEADERS],
                      body=payload_record(received.replace(KEEPALIVE, b"")))
        if expect_keepalive:
            record["keepalive_seen"] = KEEPALIVE in received
        deadline = time.monotonic() + 10
        while admitted and finished_calls(proxy) <= before and time.monotonic() < deadline:
            time.sleep(.005)
        return record
    finally:
        connection.close()


def normalize(value, transport_id):
    if isinstance(value, dict):
        return {key: normalize(item, transport_id) for key, item in value.items() if key not in VOLATILE_EVENT_KEYS}
    if isinstance(value, list):
        return [normalize(item, transport_id) for item in value]
    if isinstance(value, str):
        return value.replace(transport_id, "<transport>")
    return value


def run_scenario(module, name, *, proxy_kwargs=None, upstream_url=None, script=(), steps, setup=None,
                 worker=False, no_delay=True):
    upstream = Upstream()
    upstream.script = list(script)
    clients = []
    proxy = module.ModelRequestProxy(upstream_url or upstream.url, **(proxy_kwargs or {}))
    proxy.start()
    try:
        if setup:
            setup(proxy)
        patches = []
        if no_delay:
            patches.append(mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0))
        if worker:
            original = module.subprocess.Popen
            patches.append(mock.patch.object(module.subprocess, "Popen", side_effect=lambda *a, **k: original(*a, **k)))
        for patch in patches:
            patch.start()
        try:
            for step in steps:
                clients.append(step(proxy))
        finally:
            for patch in reversed(patches):
                patch.stop()
        with proxy.lock:
            events = [dict(event) for event in proxy.events]
            state = dict(request_count=proxy.request_count, attempt_count=proxy.attempt_count,
                         logical_call_count=proxy.logical_call_count, reserved_calls=proxy.reserved_calls,
                         blocked_error=proxy.blocked_error, image_context_epoch=proxy.image_context_epoch,
                         image_trim_watermarks={k: list(v) for k, v in sorted(proxy._image_trim_watermarks.items())},
                         image_trim_starts=dict(sorted(proxy._image_trim_starts.items())))
        transport = proxy.transport_id
    finally:
        proxy.close()
        upstream.close()
    return normalize(dict(
        upstream=[dict(path=item["path"], headers=item["headers"], body=payload_record(item["body"]))
                  for item in upstream.requests],
        client=clients, state=state, events=events), transport)


def scenarios():
    """(name, run_scenario keyword arguments) for every golden forwarding scenario."""
    register = lambda thread, source="actor": (lambda proxy: proxy.register_thread(thread, source, "req-" + thread))
    thread = lambda value: (("thread-id", value),)
    def tiered(proxy):
        proxy.service_tier = "priority"
    def cancelled(proxy):
        proxy.register_thread("thread-x", "actor", "req-x")
        proxy.cancel_thread("thread-x")
    recovery = dict(max_rounds=1, initial_delay_s=0, backoff_multiplier=1, max_delay_s=0,
                    max_episode_delay_s=1, transport_incomplete=False)
    count_request = dict(model="gpt-fixture", stream=True, input=image_history(20, 600, duplicate_every=3))
    extended = dict(count_request, input=count_request["input"] + image_history(2, 600, first=20)[1:])
    large_request = dict(model="gpt-fixture", stream=True, input=image_history(30, 90_000))
    rejected_request = dict(model="gpt-fixture", stream=True, input=image_history(12, 400))
    stream_error = 200, [("Content-Type", "text/event-stream")], (
        b'data: {"type":"error","code":"server_error","message":"fixture-private"}\n\n')
    return [
        ("slow_stream_keepalive", dict(
            script=[(*completed(1), 1.4)], steps=[lambda p: call(p, expect_keepalive=True)])),
        ("stream_completed_registered_and_unregistered", dict(
            setup=register("thread-a"), script=[completed(1), completed(2)],
            steps=[lambda p: call(p, extra_headers=thread("thread-a")),
                   lambda p: call(p, extra_headers=thread("thread-unregistered"))])),
        ("host_credentials_hooks_and_service_tier", dict(
            proxy_kwargs=dict(upstream_auth_headers={"Authorization": "Bearer host-token", "chatgpt-account-id": "host-account"},
                              request_context=lambda request, identity: dict(
                                  request, prompt_cache_key="ctx:" + identity["endpoint"] + ":" + identity["source"]),
                              request_projection=lambda request: dict(request, instructions="projected 中")),
            setup=tiered, script=[completed(1), json_reply(200, dict(object="response.compaction", output=[], usage={}))],
            steps=[lambda p: call(p, extra_headers=thread("thread-b")),
                   lambda p: call(p, "/responses/compact", COMPACT_BODY, extra_headers=thread("thread-b"))])),
        ("non_stream_completed_headers_relayed", dict(
            script=[json_reply(200, dict(id="resp-json", model="gpt-fixture-served", status="completed",
                                         output=[], usage=dict(input_tokens=2, output_tokens=1)),
                               headers=[("x-request-id", "req-up-1"), ("openai-processing-ms", "12")])],
            steps=[lambda p: call(p, data=PLAIN_BODY)])),
        ("compaction_route_epoch", dict(
            setup=register("thread-c"),
            script=[json_reply(200, dict(object="response.compaction", output=[dict(
                type="compaction", encrypted_content="opaque")], usage=dict(input_tokens=5, output_tokens=2)))],
            steps=[lambda p: call(p, "/responses/compact", COMPACT_BODY, extra_headers=thread("thread-c"))])),
        ("image_count_trim_and_exact_reuse", dict(
            proxy_kwargs=dict(image_reuse=True, max_inline_images=24), setup=register("thread-img"),
            steps=[lambda p: call(p, data=body(count_request), extra_headers=thread("thread-img")),
                   lambda p: call(p, data=body(extended), extra_headers=thread("thread-img"))])),
        ("image_body_bytes_trim", dict(
            setup=register("thread-big"),
            steps=[lambda p: call(p, data=body(large_request), extra_headers=thread("thread-big"))])),
        ("image_rejected_then_trimmed_retry", dict(
            setup=register("thread-rej"),
            script=[json_reply(400, dict(error=dict(code="image_compression_failed", message="fixture-private"))), completed(2)],
            steps=[lambda p: call(p, data=body(rejected_request), extra_headers=thread("thread-rej"))])),
        ("retryable_503_then_success", dict(
            script=[json_reply(503, dict(error=dict(code="server_error", message="fixture-private"))), completed(2)],
            steps=[lambda p: call(p)])),
        ("rate_limit_429_then_success", dict(
            script=[json_reply(429, dict(error=dict(code="rate_limit_exceeded", type="requests",
                                                    message="Rate limit. Please try again in 20ms.")),
                               headers=[("Retry-After", ".03")]), completed(2)],
            steps=[lambda p: call(p)])),
        ("terminal_missing_then_success", dict(
            script=[sse(dict(type="response.created", response=dict(id="r", status="in_progress")),
                        dict(type="response.output_text.delta", delta="x")), completed(2)],
            steps=[lambda p: call(p)])),
        ("transport_recovery_after_503", dict(
            proxy_kwargs=dict(max_retries=0, transport_recovery=recovery, recovery_gate=lambda: None),
            script=[json_reply(503, dict(error=dict(code="server_is_overloaded", message="fixture-private"))), completed(2)],
            steps=[lambda p: call(p)])),
        ("failed_terminal_then_transport_blocked_denial", dict(
            script=[failed_sse("failed", dict(code="context_length_exceeded", message="fixture-private"))],
            steps=[lambda p: call(p), lambda p: call(p, admitted=False)])),
        ("incomplete_terminal", dict(
            script=[failed_sse("incomplete")], steps=[lambda p: call(p)])),
        ("failed_json_non_stream", dict(
            script=[json_reply(200, dict(id="resp-f", model="gpt-fixture-served", status="failed",
                                         output=[dict(type="function_call", name="unsafe")],
                                         error=dict(code="invalid_prompt", message="fixture-private")))],
            steps=[lambda p: call(p, data=PLAIN_BODY)])),
        ("sse_error_event_no_retries", dict(
            proxy_kwargs=dict(max_retries=0), script=[stream_error], steps=[lambda p: call(p)])),
        ("http_400_configuration_error_stream", dict(
            script=[json_reply(400, dict(error=dict(code="model_not_found", type="invalid_request_error",
                                                    message="fixture-private")))],
            steps=[lambda p: call(p)])),
        ("http_503_model_not_found_non_stream", dict(
            script=[json_reply(503, dict(error=dict(code="model_not_found", type="new_api_error",
                                                    message="fixture-private")))],
            steps=[lambda p: call(p, data=PLAIN_BODY)])),
        ("connect_failure_relay_error", dict(
            upstream_url=closed_port_url(), proxy_kwargs=dict(max_retries=1),
            steps=[lambda p: call(p, data=PLAIN_BODY)])),
        ("budget_denial", dict(
            proxy_kwargs=dict(max_requests=1),
            steps=[lambda p: call(p), lambda p: call(p, admitted=False)])),
        ("thread_cancelled_denial", dict(
            setup=cancelled, steps=[lambda p: call(p, extra_headers=thread("thread-x"), admitted=False)])),
        ("unknown_routes", dict(
            steps=[lambda p: call(p, "/v1/responses", admitted=False),
                   lambda p: call(p, "/responses/", admitted=False),
                   lambda p: call(p, "/responses?x=1", admitted=False),
                   lambda p: call(p, data=b"", method="GET", admitted=False)])),
        ("projection_rejected_relay_error", dict(
            proxy_kwargs=dict(request_projection=lambda request: (_ for _ in ()).throw(
                ValueError("sandbox_remote_model_tool_disabled"))),
            steps=[lambda p: call(p)])),
        ("worker_stream_completed", dict(
            worker=True, setup=register("thread-w"),
            steps=[lambda p: call(p, extra_headers=thread("thread-w"))])),
        ("worker_failed_terminal", dict(
            worker=True, script=[failed_sse("failed", dict(code="server_is_overloaded", message="fixture-private"))],
            proxy_kwargs=dict(max_retries=0), steps=[lambda p: call(p)])),
    ]


def capture_forwarding(module=None):
    if module is None:
        from auriga.cortex.session import model_proxy as module
    return {name: run_scenario(module, name, **kwargs) for name, kwargs in scenarios()}


# ------------------------------------------------------------------------ egress scrub
DIGEST = "0123456789abcdef" * 4
ARTIFACT = "artifact:" + "a" * 64 + "-" + "b" * 16
LONG_OBSERVATION = "observation-" + "c" * 32
LONG_ACTION = "action-" + "d" * 24
CLOCK = "env-" + "e" * 16 + "/generation-3"


def planted_request():
    """One Responses body touching every scrubbed and every protected field."""
    namespace_tool = dict(type="function", name="fixture_read", description=f"Read sha256={DIGEST}",
                          parameters={"type": "object", "properties": {"value_sha256": {"type": "string",
                                      "description": f"digest {DIGEST}"}}})
    return {
        "model": "gpt-fixture",
        "instructions": f"Base instructions sha256={DIGEST} keep 0.765 m and {ARTIFACT}",
        "input": [
            {"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": f"dev text digest: {DIGEST} ref {ARTIFACT} clock {CLOCK}"}]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "look at " + LONG_OBSERVATION},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA" + DIGEST, "detail": "high"}]},
            {"type": "reasoning", "id": "rs_" + "f" * 40, "encrypted_content": "gAAAA" + DIGEST,
             "summary": [{"type": "summary_text", "text": f"thinking about {LONG_ACTION} hash {DIGEST}"}]},
            {"type": "function_call", "id": "fc_" + "1" * 32, "call_id": "call_" + "2" * 24, "name": "robot_observe",
             "arguments": json.dumps({"observation_id": LONG_OBSERVATION, "note": f"sha256={DIGEST}"})},
            {"type": "function_call_output", "call_id": "call_" + "2" * 24, "output": [
                {"type": "input_text", "text": json.dumps({"bundle": {"observation_id": LONG_OBSERVATION,
                 "source_sha256": DIGEST, "clock": CLOCK}, "artifact": ARTIFACT})},
                {"type": "input_image", "image_url": "data:image/png;base64,BBBB"}]},
            {"type": "custom_tool_call", "call_id": "call_3", "name": "program",
             "input": f"print('{ARTIFACT}')  # commit {DIGEST}"},
            {"type": "custom_tool_call_output", "call_id": "call_3",
             "output": f"/home/fixture/run/private/program-workspaces/w1/main.py ok digest={DIGEST} {LONG_ACTION}"},
            {"type": "additional_tools", "tools": [{"type": "tool_search", "execution": "client"}]},
            {"type": "tool_search_call", "execution": "client", "call_id": "search-1",
             "arguments": {"query": f"robot hash={DIGEST}", "limit": 2, "ref": LONG_OBSERVATION}},
            {"type": "tool_search_output", "execution": "client", "call_id": "search-1", "status": "completed",
             "tools": [{"type": "namespace", "name": "auriga", "description": f"ns {DIGEST}", "tools": [namespace_tool]}]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": f"I will act on {LONG_ACTION}"},
                {"type": "refusal", "refusal": f"no {DIGEST}"}]},
            {"type": "compaction", "encrypted_content": "opaque-" + DIGEST, "id": "cmp_" + "3" * 20},
        ],
        "tools": [
            {"type": "function", "name": "robot_move", "description": f"Move. build_sha={DIGEST}",
             "parameters": {"type": "object", "properties": {"commit": {"type": "string"},
                                                             "x_m": {"type": "number", "description": "metres"}}}},
            {"type": "namespace", "name": "auriga", "description": "deferred", "tools": [namespace_tool]},
            {"type": "custom", "name": "program", "description": f"Run code digest {DIGEST}",
             "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}},
            {"type": "local_shell"}, {"type": "shell", "environment": {"type": "local"}},
            {"type": "tool_search", "execution": "client", "description": f"search {DIGEST}"},
        ],
        "text": {"format": {"type": "json_schema", "name": "answer", "strict": True,
                            "schema": {"type": "object", "properties": {"sha256": {"type": "string"}}}},
                 "verbosity": "low"},
        "reasoning": {"effort": "high", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "thread-" + "9" * 32, "store": False, "stream": True,
        "metadata": {"note": f"sha256={DIGEST}"},
    }


def presentation_cases():
    """(name, request, session options); rejection cases record the ValueError message."""
    lean_history = dict(model="gpt-fixture", input=image_history(6, 64) + [
        {"type": "function_call_output", "call_id": "call_tail", "output": json.dumps(dict(
            receipt=dict(action_id=LONG_ACTION, status="completed", source_sha256=DIGEST)))}])
    cases = [
        ("planted_every_field", planted_request(), {}),
        ("string_input_and_instructions", {"instructions": f"sha256={DIGEST}", "input": f"user {ARTIFACT}"}, {}),
        ("lean_projection_block", lean_history, dict(lean_projection=True, lean_projection_k=1, lean_projection_block=8)),
        ("lean_projection_trailing", lean_history, dict(lean_projection=True, lean_projection_k=1, lean_projection_block=0)),
        ("request_dump", planted_request(), dict(request_dump_limit=2)),
    ]
    rejected = [
        {"tools": [{"type": "web_search"}]},
        {"tools": [{"type": "namespace", "name": "remote", "tools": [{"type": "mcp"}]}]},
        {"tools": [{"type": "shell", "environment": {"type": "container_auto"}}]},
        {"tools": [{"type": "tool_search", "execution": "server"}]},
        {"input": [{"type": "additional_tools", "tools": [{"type": "code_interpreter"}]}]},
        {"input": [{"type": "tool_search_call", "execution": "server", "arguments": {}}]},
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.org/x.png"}]}]},
        {"input": [{"role": "user", "content": [{"type": "input_file", "file_url": "http://127.0.0.1/x.txt"}]}]},
        {"input": [{"type": "tool_search_output", "execution": "client", "tools": [
            {"type": "namespace", "name": "auriga", "tools": [{"type": "function", "name": "not_registered"}]}]}]},
        {"input": [{"type": "tool_search_output", "execution": "client", "tools": [
            {"type": "namespace", "name": "other", "tools": [{"type": "function", "name": "fixture_read"}]}]}]},
    ]
    return cases + [(f"rejected_{index}", request, {}) for index, request in enumerate(rejected)]


def fake_session(store, root, options):
    """The attributes ``CodexSession._present_model_request`` reads, nothing else."""
    from auriga.core.presentation.actor_presentation import ActorPresentation
    from auriga.cortex.session.codex import CodexSession
    session = SimpleNamespace(presentation=ActorPresentation(store), deferred_tools=frozenset({"fixture_read"}),
                              lean_projection=False, lean_projection_k=1, lean_projection_block=8,
                              request_dump_limit=0, request_dump_count=0, request_dump_lock=threading.Lock(),
                              private=root / "private", store=store)
    session.__dict__.update(options)
    session._dump_request = MethodType(CodexSession._dump_request, session)
    return session


def capture_presentation(present=None):
    """``present(session, request)`` defaults to ``CodexSession._present_model_request``."""
    from auriga.core.storage.store import Store
    if present is None:
        from auriga.cortex.session.codex import CodexSession
        present = CodexSession._present_model_request
    results = {}
    for name, request, options in presentation_cases():
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = Store(root / "run", dict(run_id="golden", episode_id="episode",
                                             env_instance_id="environment", env_generation=1))
            try:
                session = fake_session(store, root, options)
                original = json.dumps(request, sort_keys=True)
                try:
                    record = dict(output=present(session, request))
                except ValueError as error:
                    record = dict(error=str(error))
                record["input_unchanged"] = json.dumps(request, sort_keys=True) == original
                dumps = sorted((root / "private" / "request-dump").glob("*.json"))
                if dumps:
                    record["dumps"] = [json.loads(path.read_text()) for path in dumps]
                results[name] = record
            finally:
                store.close()
    return results


def write_fixtures(label):
    FIXTURES.mkdir(exist_ok=True)
    from auriga.cortex.session import model_proxy
    print("capturing from", Path(model_proxy.__file__).resolve(), file=sys.stderr)
    source = dict(revision=label)
    FORWARDING_FIXTURE.write_text(json.dumps(dict(captured_from=source, scenarios=capture_forwarding()),
                                             indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    PRESENTATION_FIXTURE.write_text(json.dumps(dict(captured_from=source, cases=capture_presentation()),
                                               indent=1, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--write":
        raise SystemExit("usage: PYTHONPATH=<source tree> python tests/wire_golden_fixture.py --write <revision>")
    write_fixtures(sys.argv[2])
