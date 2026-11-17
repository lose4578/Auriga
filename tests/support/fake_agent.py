"""Scripted in-process ``AgentSession`` for gateway/supervision tests (design 04 §2.2, §10.1).

``FakeAgentSession`` follows the normative state and turn contract without any native process or model:

* ``start`` -> ``state="idle"``, a host-minted (or resumed) ``thread_id``, a ``runtime/agent`` identity record.
* ``run_turn`` scrubs the prompt like every backend (``ActorPresentation.present_text`` when a Store is attached),
  sets ``state="running"`` and clears ``turn_done``, then applies the next scripted admission outcome:

  - ``"admit"`` (default): returns a host-minted ``turn_id``; the turn stays running until the test calls
    ``complete`` / ``fail`` / ``interrupt`` / ``disconnect``;
  - ``"reject"``: definite rejection; restores ``(state, turn_id, turn_result, active_turn_model)``, sets
    ``turn_done`` and raises ``AgentRejected(code, message, busy)``;
  - ``"unknown"``: unknown admission; ``_disconnected`` (state ``disconnected``, synthetic failed ``turn_result``,
    ``on_disconnect()``, ``turn_done`` set) and raises ``AgentTransportUnknown``. Never retried.

Script outcomes with ``script_admission`` (FIFO). Every call is recorded in ``calls`` for assertions.
"""
import collections
import threading
import time

from auriga.core.contracts import new_id
from auriga.cortex.agents.base import (AgentRejected, AgentSession, AgentTransportUnknown,
                                       STATE_CLOSED, STATE_DISCONNECTED, STATE_IDLE, STATE_NOT_STARTED,
                                       STATE_RUNNING, UNCONNECTED_STATES)


class FakeAgentSession(AgentSession):
    backend_kind = "fake_agent"
    counting = "fake_agent_completions_v1"

    def __init__(self, run_dir=None, store=None, tool_handler=None, *, model="fake-model", effort="high",
                 provider="fake-provider", price_card=None, capabilities=None, effort_choices=None,
                 backend_kind=None):
        self.root, self.store, self.tool_handler = run_dir, store, tool_handler
        self.model, self.effort, self.provider, self.price_card = model, effort, provider, price_card
        self.active_turn_model = model
        if capabilities is not None:
            self.capabilities = capabilities
        if effort_choices is not None:
            self.effort_choices = tuple(effort_choices)
        if backend_kind is not None:
            self.backend_kind = backend_kind
        self.state = STATE_NOT_STARTED
        self.thread_id = self.turn_id = None
        self.turn_done = threading.Event()
        self.turn_result = None
        self.last_activity_at = time.monotonic()
        self.model_limit_reached = threading.Event()
        self.model_proxy = None
        self.isolation = "disabled"
        self.sandbox_enabled = False
        self.lock = threading.RLock()
        self.admissions = collections.deque()
        self.calls = []
        self.processes = []          # [(role, Popen-like)] returned by owned_processes()
        self.attach_spec = None      # returned by terminal_attach_spec()

    # -- scripting -------------------------------------------------------------------------------------------
    def script_admission(self, outcome, *, code="fake_rejected", message="fake rejection", busy=False):
        """Queue the outcome of the next ``run_turn``: "admit", "reject" (definite) or "unknown" (admission)."""
        if outcome not in ("admit", "reject", "unknown"):
            raise ValueError("unknown_fake_admission")
        self.admissions.append(dict(outcome=outcome, code=code, message=message, busy=busy))

    def complete(self, status="completed", error=None, final_text=None):
        """Native terminal of the current turn: ``status`` completed | interrupted | failed.

        ``final_text`` writes a completed ``agentMessage`` row with ``phase="final_answer"`` to the Store.
        """
        with self.lock:
            if final_text is not None and self.store is not None:
                identifier = new_id("item")
                self.store.set_transcript_item(identifier, dict(
                    id=identifier, thread_id=self.thread_id, turn_id=self.turn_id, type="agentMessage",
                    status="completed", phase="final_answer", text=final_text, backend=self.backend_kind))
            self.turn_result = dict(id=self.turn_id, status=status, error=error)
            self.state = status
            self.turn_done.set()

    def fail(self, message="fake turn failure"):
        self.complete("failed", dict(message=message))

    def disconnect(self, reason="fake transport lost"):
        self._disconnected(RuntimeError(reason))

    # -- AgentSession contract -------------------------------------------------------------------------------
    def start(self, tools, instructions, resume_thread_id=None):
        self.calls.append(("start", len(tools), resume_thread_id))
        self.thread_id = resume_thread_id or new_id("fake-thread")
        self.state = STATE_IDLE
        if self.store is not None:
            self.store.set_state("runtime", "agent", dict(
                backend=self.backend_kind, thread_id=self.thread_id, instance_id=new_id("fake-agent"), pid=None,
                model=self.model, provider=self.provider, effort=self.effort, isolation=self.isolation,
                counting=self.counting))
        return self.thread_id

    def run_turn(self, prompt):
        with self.lock:
            if self.state == STATE_RUNNING:
                raise ValueError("turn_already_active")
            if self.state in UNCONNECTED_STATES:
                raise ValueError("session_not_connected")
            if self.store is not None:
                from auriga.core.presentation.actor_presentation import ActorPresentation
                prompt = ActorPresentation(self.store).present_text(prompt)
            self.calls.append(("run_turn", prompt))
            previous = self.state, self.turn_id, self.turn_result, self.active_turn_model
            self.active_turn_model = self.model
            self.turn_done.clear()
            self.turn_id = self.turn_result = None
            self.state = STATE_RUNNING
            admission = self.admissions.popleft() if self.admissions else dict(outcome="admit")
            if admission["outcome"] == "reject":
                self.state, self.turn_id, self.turn_result, self.active_turn_model = previous
                self.turn_done.set()
                raise AgentRejected(admission["code"], admission["message"], busy=admission["busy"])
            if admission["outcome"] == "unknown":
                error = AgentTransportUnknown("upstream_admission_unknown")
                self._disconnected(error)
                raise error
            self.turn_id = new_id("turn")
            return self.turn_id

    def _disconnected(self, error):
        with self.lock:
            if self.state == STATE_CLOSED:
                return
            self.state = STATE_DISCONNECTED
            if not self.turn_done.is_set():
                self.turn_result = dict(id=self.turn_id, status="failed",
                                        error=dict(message="upstream_connection_lost: " + str(error)))
        try:
            if self.on_disconnect:
                self.on_disconnect()
        finally:
            self.turn_done.set()

    def steer(self, text):
        if not self.capabilities.steer:
            return super().steer(text)
        with self.lock:
            if self.state != STATE_RUNNING or not self.turn_id:
                raise ValueError("no_active_turn_to_steer")
            self.calls.append(("steer", text))
            return self.turn_id

    def interrupt(self):
        self.calls.append(("interrupt",))
        if self.state == STATE_RUNNING:
            self.complete("interrupted")

    def close(self):
        self.calls.append(("close",))
        self.state = STATE_CLOSED
        self.turn_done.set()

    def set_effort(self, effort):
        self.calls.append(("set_effort", effort))
        super().set_effort(effort)

    def set_model(self, model):
        self.calls.append(("set_model", model))
        super().set_model(model)

    def owned_processes(self):
        return [(role, process) for role, process in self.processes if process is not None]

    def terminal_attach_spec(self):
        return self.attach_spec if self.capabilities.terminal_attach else None

    def provenance(self):
        return dict(kind=self.backend_kind, model_configured=self.model, effort=self.effort,
                    counting=self.counting, capabilities=self.capabilities.to_dict())
