"""D1 regression: the first startup ``GET /global/health`` of a starting OpenCode server can stall for far longer
than later requests (observed on Linux: the first request made while ``opencode serve`` is still coming up never
gets an answer until the per-request timeout expires, ~30 s). ``OpenCodeSession._await_health`` must cap each
individual poll (``health_poll_request_timeout_s``) instead of letting one stalled request consume the whole
``startup_timeout_s`` budget.

Uses the scripted fake server's ``health_stalls`` / ``health_stall_s`` option (``support.fake_opencode``) to stall
exactly the first health request; no real OpenCode binary involved.
"""
import unittest

from test_agent_opencode_session_unit import SessionCase


class StartupHealthStallRegressionTest(SessionCase):
    def test_stalled_first_health_request_does_not_exhaust_startup_budget(self):
        # Mirrors the production shape of the bug: the per-request timeout used for a health poll equals the whole
        # startup budget (30 s == 30 s in production). Here both are 2.0 s, and the fake stalls the first health
        # request for longer than that (3.0 s), so the pre-fix `self.client.health()` (no per-poll timeout) would
        # consume the full 2.0 s budget on that one request and `_await_health` would raise
        # `opencode_startup_timeout` before a second, healthy poll ever got a chance to run.
        self.client_timeout = 2.0
        self.fake.health_stalls = 1
        self.fake.health_stall_s = 3.0
        session = self.started(startup_timeout_s=2.0, health_poll_request_timeout_s=0.3, health_poll_s=0.02)
        self.assertEqual(session.state, "idle")
        # A second health poll must have happened after the stalled one timed out, well inside the 2.0 s budget.
        health_requests = [item for item in self.fake.requests if item["path"] == "/global/health"]
        self.assertGreaterEqual(len(health_requests), 2)


if __name__ == "__main__":
    unittest.main()
