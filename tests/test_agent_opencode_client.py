import json
import threading
import time
import unittest
from pathlib import Path

from auriga.cortex.agents.opencode import client as oc
from support.fake_opencode import FakeOpenCodeServer, PASSWORD

FIXTURE = Path(__file__).parent / "fixtures" / "opencode" / "events_turn_with_tool.json"


class ClientBase(unittest.TestCase):
    def setUp(self):
        self.server = FakeOpenCodeServer().start()
        self.addCleanup(self.server.close)
        self.client = oc.OpenCodeClient(self.server.url, PASSWORD, timeout_s=1.0, stream_timeout_s=3.0)
        self.addCleanup(self.client.close_streams)


class RouteTests(ClientBase):
    def test_health_and_auth(self):
        self.assertEqual(self.client.health(), {"healthy": True, "version": "1.19.5"})
        bad = oc.OpenCodeClient(self.server.url, "wrong")
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            bad.health()
        self.assertEqual(ctx.exception.status, 401)
        self.assertTrue(ctx.exception.definite_rejection)

    def test_session_lifecycle_and_prompt_body(self):
        sid = self.client.create_session("auriga-run-1")
        self.assertTrue(sid.startswith("ses"))
        self.assertEqual(self.server.sessions[sid]["title"], "auriga-run-1")
        self.assertTrue(self.client.session_exists(sid))
        self.assertFalse(self.client.session_exists("ses_missing"))
        self.client.prompt_async(sid, "hello", variant="high", agent="build", model=("relay", "m"))
        body = self.server.requests[-1]["body"]
        self.assertEqual(body, {"parts": [{"type": "text", "text": "hello"}], "agent": "build",
                                "model": {"providerID": "relay", "modelID": "m"}, "variant": "high"})
        self.assertNotIn("system", body)
        self.assertEqual(self.client.status(), {sid: {"type": "busy"}})
        self.assertEqual(self.client.messages(sid)[0]["info"]["role"], "user")
        self.assertTrue(self.client.abort(sid))
        self.assertEqual(self.client.status(), {})
        last = self.client.messages(sid)[-1]["info"]
        self.assertEqual(last["error"]["name"], "MessageAbortedError")

    def test_explicit_parts_and_argument_validation(self):
        sid = self.client.create_session("t")
        self.client.prompt_async(sid, parts=[{"type": "text", "text": "x"}])
        with self.assertRaises(ValueError):
            self.client.prompt_async(sid)
        with self.assertRaises(ValueError):
            self.client.prompt_async(sid, "a", parts=[])
        with self.assertRaises(ValueError):
            self.client.create_session("  ")
        with self.assertRaises(ValueError):
            self.client.messages("../x")

    def test_routes_and_providers(self):
        self.assertEqual(self.client.routes(), sorted(self.server.routes))
        self.assertEqual(self.client.config_providers(), {"providers": [], "default": {}})

    def test_tool_list_query_and_shape(self):
        seen = []
        tools = [{"id": "bash", "description": "d", "parameters": {}}]
        self.client._request = lambda method, path, *a, **k: (seen.append((method, path)), tools)[1]
        self.assertEqual(self.client.tool_list("relay", "a/b c"), tools)
        self.assertEqual(seen, [("GET", "/experimental/tool?provider=relay&model=a%2Fb%20c")])
        self.client._request = lambda *a, **k: {"not": "a list"}
        with self.assertRaises(oc.OpenCodeTransportError):
            self.client.tool_list("relay", "m")

    def test_base_url_validation(self):
        for url in ("https://x", "http://u:p@127.0.0.1:1", "http://127.0.0.1:1/p", "127.0.0.1"):
            with self.assertRaises(ValueError):
                oc.OpenCodeClient(url, "p")


class AdmissionTests(ClientBase):
    def test_4xx_is_definite_rejection(self):
        sid = self.client.create_session("t")
        self.server.prompt_script.append(("status", 400))
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.client.prompt_async(sid, "x")
        self.assertTrue(oc.is_definite_rejection(ctx.exception))
        self.assertEqual(ctx.exception.body["name"], "InvalidRequestError")
        self.assertEqual(self.server.sessions[sid]["messages"], [])

    def test_unknown_session_is_404_rejection(self):
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.client.prompt_async("ses_nope", "x")
        self.assertEqual(ctx.exception.status, 404)

    def test_5xx_is_not_definite(self):
        sid = self.client.create_session("t")
        self.server.prompt_script.append(("status", 500))
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.client.prompt_async(sid, "x")
        self.assertFalse(oc.is_definite_rejection(ctx.exception))

    def test_timeout_is_transport_unknown_and_not_retried(self):
        sid = self.client.create_session("t")
        self.server.hang_s = 2.0
        self.server.prompt_script.append("hang")
        with self.assertRaises(oc.OpenCodeTransportError) as ctx:
            self.client.prompt_async(sid, "x")
        self.assertIsInstance(ctx.exception, OSError)
        self.assertFalse(oc.is_definite_rejection(ctx.exception))
        self.assertEqual(self.server.request_paths("POST").count(f"/session/{sid}/prompt_async"), 1)

    def test_dropped_connection_after_admission_is_transport_unknown(self):
        sid = self.client.create_session("t")
        self.server.prompt_script.append("drop")
        with self.assertRaises(oc.OpenCodeTransportError):
            self.client.prompt_async(sid, "x")
        self.assertEqual(len(self.server.sessions[sid]["messages"]), 1)  # admitted although unacknowledged

    def test_connection_refused(self):
        dead = oc.OpenCodeClient("http://127.0.0.1:1", "p", timeout_s=0.5)
        with self.assertRaises(oc.OpenCodeTransportError):
            dead.health()


class EventTests(ClientBase):
    def _collect(self, count, timeout=5.0):
        frames, errors = [], []

        def run():
            try:
                for frame in self.client.events():
                    frames.append(frame)
                    if len(frames) >= count:
                        return
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return frames, errors, thread

    def test_real_captured_events_roundtrip(self):
        captured = json.loads(FIXTURE.read_text())
        frames, errors, thread = self._collect(len(captured))
        self.assertTrue(self.server.wait_subscribers())
        for event in captured[1:]:
            self.server.push(event["type"], event["properties"])
        thread.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(frames[0]["type"], "server.connected")
        self.assertEqual([f["type"] for f in frames[1:]], [e["type"] for e in captured[1:]])
        self.assertEqual(frames[-1]["type"], "session.idle")

    def test_comments_and_invalid_frames_skipped(self):
        frames, _errors, thread = self._collect(2)
        self.assertTrue(self.server.wait_subscribers())
        self.server.comment()
        self.server.send_raw_event(b"data: {not json\n\n")
        self.server.send_raw_event(b"event: x\n\n")
        self.server.push("server.heartbeat")
        thread.join(5)
        self.assertEqual([f["type"] for f in frames], ["server.connected", "server.heartbeat"])
        self.assertEqual(self.client.invalid_event_frames, 1)

    def test_eof_raises_stream_ended(self):
        _frames, errors, thread = self._collect(100)
        self.assertTrue(self.server.wait_subscribers())
        self.server.drop_event_streams()
        thread.join(5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], oc.OpenCodeTransportError)

    def test_event_stream_reconnects_reports_gaps_and_loses_events(self):
        stop = threading.Event()
        frames, gaps = [], []
        thread = threading.Thread(target=lambda: [frames.append(f) for f in self.client.event_stream(
            stop, on_gap=gaps.append, initial_backoff_s=0.05, max_backoff_s=0.1)], daemon=True)
        thread.start()
        self.assertTrue(self.server.wait_subscribers())
        self.server.push("session.idle", {"sessionID": "ses_a"})
        self.server.drop_event_streams()
        self.server.push("session.idle", {"sessionID": "ses_lost"})   # emitted while disconnected: gone
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (gaps and self.server.wait_subscribers(1, 0.05)):
            time.sleep(0.02)
        self.server.push("session.idle", {"sessionID": "ses_b"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(f["properties"].get("sessionID") == "ses_b" for f in frames):
            time.sleep(0.02)
        stop.set()
        self.client.close_streams()
        thread.join(5)
        ids = [f["properties"].get("sessionID") for f in frames]
        self.assertIn("ses_a", ids)
        self.assertIn("ses_b", ids)
        self.assertNotIn("ses_lost", ids)
        self.assertGreaterEqual(len(gaps), 1)
        self.assertIsInstance(gaps[0], oc.OpenCodeStreamEnded)

    def test_event_stream_http_error_is_gap_then_stops_on_event(self):
        bad = oc.OpenCodeClient(self.server.url, "wrong", timeout_s=1.0)
        stop, gaps = threading.Event(), []

        def on_gap(error):
            gaps.append(error)
            stop.set()
        self.assertEqual(list(bad.event_stream(stop, on_gap=on_gap, initial_backoff_s=0.01)), [])
        self.assertEqual(gaps[0].status, 401)

    def test_close_streams_unblocks_reader(self):
        _frames, errors, thread = self._collect(100)
        self.assertTrue(self.server.wait_subscribers())
        self.client.close_streams()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)


class RouteListTests(unittest.TestCase):
    def test_routes_from_openapi(self):
        doc = {"paths": {"/a": {"get": {}, "post": {}, "parameters": []}, "/b": {"delete": {}}}}
        self.assertEqual(oc.routes_from_openapi(doc), ["DELETE /b", "GET /a", "POST /a"])
        with self.assertRaises(ValueError):
            oc.routes_from_openapi({})


if __name__ == "__main__":
    unittest.main()
