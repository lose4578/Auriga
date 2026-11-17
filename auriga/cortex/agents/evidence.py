"""Backend-neutral evidence client for visual review and self-briefing (design 04 §9).

``RelayEvidenceClient`` answers one evidence packet with ONE fresh, streamed, tool-free model request through the
run's own ``ModelRequestProxy``: no agent process, no home directory, no history, no memory, no tools. It offers the
constructor keywords and the ``review_evidence`` method of ``CodexEvidenceAgent`` (same failure strings), so
``VisualReviewer`` and ``self_briefing.generate`` use either without change; like the Codex helper it does not
validate the answer against the schema (the consumers do).

Because the request goes through the relay it inherits, without code here: the shared 180-call budget (charged only
for a delivered usable completion), ``source`` / ``caller_request_id`` attribution in every ``model.*`` event,
bounded retries, cooldown recovery behind the transport-recovery gate, the blocked-transport gate, no request
deadline, the activity and model-wait clocks, host-side credentials and the egress re-scrub of what the provider
sees. The private ``X-Auriga-Thread`` header names the helper's thread (``evidence-<request_id>``, registered with
the helper's source); it is never forwarded upstream.

Bindings (design §9.4): ``upstream="actor"`` sends the request on the actor's own wire, URL and credentials (the
unprefixed relay routes); any other name is a binding added with ``ModelRequestProxy.add_upstream`` and served under
``/_auriga/u/<name>``. ``bind_evidence_upstream`` adds the ``"evidence"`` binding from ``agent.evidence.upstream``.

Nothing here is model-visible except the request body itself; result fields other than ``answer`` are observer
provenance written to the run Store by the consumers.
"""
import http.client
import json
import os
from pathlib import Path
import re
import socket
import threading
import tomllib
from urllib.parse import urlsplit

from ...core.config import ConfigError
from ...core.presentation.actor_presentation import ActorPresentation
from ..session.codex_evidence_agent import BASE_INSTRUCTIONS, MODEL
from ..session.model_proxy import ACTOR_UPSTREAM
from ..session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from ..session.wire.base import EVIDENCE_UNVERIFIED
from .factory import EVIDENCE_CLIENT_RELAY

BACKEND_LABEL = "auriga_relay_direct_request"
CONTEXT_POLICY = "fresh_single_request_selected_evidence_only_v1"
DEFAULT_DEVELOPER_INSTRUCTIONS = "Use only the supplied evidence packet."   # CodexEvidenceAgent's default
EVIDENCE_UPSTREAM = "evidence"     # binding name of agent.evidence.upstream on the run's relay
THREAD_PREFIX = "evidence-"        # helper thread id = THREAD_PREFIX + request_id
CANCEL_POLL_S = 0.05               # seconds between two checks of the caller's cancel_event
CANCELLED = "visual_review_program_cancelled"
INVALID_REQUEST_ID = "sol_agent_invalid_request_id"
REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
# Wire adapter of an agent.evidence.upstream wire_api (EvidenceConfig refuses every other value).
UPSTREAM_WIRES = {OpenAIResponsesWire.name: OpenAIResponsesWire, AnthropicMessagesWire.name: AnthropicMessagesWire}


class RelayEvidenceClient:
    """One fresh, tool-free request per ``review_evidence`` call through the session's model relay.

    session: the run's agent session; ``session.store`` (run Store) and ``session.model_proxy`` (started
        ``ModelRequestProxy``) are read at call time.
    model: model name requested (str); default ``"gpt-6.1-sol"`` (the visual reviewer); self-briefing passes the
        actor's model. Returned as ``model`` (the configured value); the served model is reported separately.
    base_instructions / developer_instructions: instruction text (str); defaults are ``CodexEvidenceAgent``'s.
    source: attribution of the helper's model calls in every relay event (``"sol_visual_review"`` |
        ``"self_briefing"``).
    upstream: relay binding the request uses: ``"actor"`` (default; the actor's wire, URL and credentials) or a
        name bound with ``ModelRequestProxy.add_upstream`` (``"evidence"`` for ``agent.evidence.upstream``).
    effort: reasoning effort label sent with the request (``"high"``, as the Codex evidence agent uses).
    """
    backend_label = BACKEND_LABEL
    context_policy = CONTEXT_POLICY
    evidence_client = EVIDENCE_CLIENT_RELAY

    def __init__(self, session, *, model=MODEL, base_instructions=BASE_INSTRUCTIONS,
                 developer_instructions=DEFAULT_DEVELOPER_INSTRUCTIONS, source="sol_visual_review",
                 upstream=ACTOR_UPSTREAM, effort="high"):
        for name, value in (("model", model), ("base_instructions", base_instructions),
                            ("developer_instructions", developer_instructions), ("source", source),
                            ("upstream", upstream), ("effort", effort)):
            if not isinstance(value, str) or not value:
                raise ValueError("evidence_client_invalid_" + name)
        self.session = session
        self.model, self.source, self.upstream, self.effort = model, source, upstream, effort
        self.base_instructions, self.developer_instructions = base_instructions, developer_instructions

    @property
    def store(self):
        return getattr(self.session, "store", None)

    def _relay(self):
        """``(proxy, binding, base_url)`` of the configured binding on the session's started relay."""
        proxy = getattr(self.session, "model_proxy", None)
        if proxy is None or getattr(proxy, "base_url", None) is None:
            raise RuntimeError("evidence_client_requires_started_model_relay")
        try:
            binding = proxy.upstream_binding(self.upstream)
            base_url = proxy.upstream_base_url(self.upstream)
        except KeyError:
            raise ValueError("evidence_upstream_not_bound:" + self.upstream) from None
        if not getattr(binding.wire, "evidence_path", None):
            raise ValueError("evidence_wire_unsupported:" + str(binding.wire.name))
        return proxy, binding, base_url

    def _unverified(self, request_id, thread, wire, **details):
        """Record ``visual.agent_turn_unverified`` (sanitised identifiers only) and return the shared error."""
        try:
            self.store.event("visual.agent_turn_unverified", dict(
                request_id=request_id, thread_id=thread, backend=self.backend_label, wire_api=wire.name,
                upstream=self.upstream, source=self.source, **details), source="private_model_transport")
        except Exception:
            pass
        return ValueError(EVIDENCE_UNVERIFIED)

    def review_evidence(self, *, input_items, output_schema, request_id, cancel_event=None):
        """Answer one evidence packet; same contract as ``CodexEvidenceAgent.review_evidence``.

        input_items: ``[{"type": "text", "text": str} | {"type": "image", "url": "data:<mime>;base64,..."}]``.
        output_schema: JSON schema (dict) of the single answer. request_id: ``[A-Za-z0-9_-]{1,128}``, unique per
        call. cancel_event: optional ``threading.Event``; setting it cancels the relay call and closes the socket.

        Returns ``{model, status: "completed", answer, usage, thread_id, turn_id, served_model, context_policy,
        backend, wire_api, upstream, model_verification, runtime_policy, tool_policy}``; ``usage`` is
        ``{"total", "last", "model_context_window": None}`` (``normalize_token_usage`` shape) or None when the
        provider reported none. Raises ``ValueError``: ``sol_agent_invalid_request_id``,
        ``sol_agent_unverified_or_incomplete_turn`` (relay denial or failure, provider failure, refusal,
        truncation, lost relay connection), ``sol_agent_unexpected_tool_or_context_item``,
        ``sol_agent_requires_one_structured_final_answer`` or a JSON decoding error; ``TimeoutError
        ("visual_review_program_cancelled")`` once ``cancel_event`` is set.
        """
        if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
            raise ValueError(INVALID_REQUEST_ID)
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError(CANCELLED)
        proxy, binding, base_url = self._relay()
        wire = binding.wire
        route = wire.route("POST", wire.evidence_path)
        presentation = ActorPresentation(self.store)
        body, extra_headers = wire.build_evidence_request(
            model=self.model, base_instructions=self.base_instructions,
            developer_instructions=self.developer_instructions, input_items=presentation.present(input_items),
            output_schema=presentation.present_data(output_schema), effort=self.effort)
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        thread = THREAD_PREFIX + request_id
        proxy.register_thread(thread, self.source, request_id)
        target = urlsplit(base_url)
        connection = http.client.HTTPConnection(target.hostname, target.port, timeout=None)
        lock, done = threading.Lock(), threading.Event()
        link = dict(socket=None, cancelled=False)

        def cancelled():
            return link["cancelled"] or (cancel_event is not None and cancel_event.is_set())

        def close_socket():
            # Wakes a blocked read in the calling thread; the relay sees the client gone only after
            # cancel_thread has already cancelled the call (no blocked transport, no charge).
            with lock:
                sock = link["socket"]
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        def watch():
            while not done.wait(CANCEL_POLL_S):
                if cancel_event.is_set():
                    link["cancelled"] = True
                    try:
                        proxy.cancel_thread(thread)
                    finally:
                        close_socket()
                    return

        watcher = threading.Thread(target=watch, daemon=True) if cancel_event is not None else None
        try:
            if watcher is not None:
                watcher.start()
            try:
                connection.connect()
                with lock:
                    link["socket"] = connection.sock
                if cancelled():
                    raise TimeoutError(CANCELLED)
                connection.putrequest("POST", target.path + wire.evidence_path, skip_accept_encoding=True)
                for name, value in (("Content-Type", "application/json"), ("Accept-Encoding", "identity"),
                                    ("X-Auriga-Thread", thread), ("X-Auriga-Source", self.source),
                                    ("X-Auriga-Request-Id", request_id), *extra_headers.items(),
                                    ("Content-Length", str(len(data)))):
                    connection.putheader(name, value)
                connection.endheaders(data)
                response = connection.getresponse()
                status, raw = response.status, response.read()
            except (OSError, http.client.HTTPException) as error:
                if cancelled():
                    raise TimeoutError(CANCELLED) from None
                raise self._unverified(request_id, thread, wire, transport_error=type(error).__name__) from None
            if cancelled():
                raise TimeoutError(CANCELLED)
            keepalive = wire.keepalive_frame()
            while keepalive and raw.startswith(keepalive):
                raw = raw[len(keepalive):]
            if status != 200:
                # A relay denial (budget, blocked transport, cancelled thread) or relay error answer.
                raise self._unverified(request_id, thread, wire, http_status=status)
            try:
                complete, evidence, terminal = wire.response_evidence(raw, route)
            except (EOFError, ValueError, TypeError, AttributeError) as error:
                raise self._unverified(request_id, thread, wire, http_status=status,
                                       stream_error=type(error).__name__) from None
            if wire.completion_outcome(complete, evidence) != "usable":
                raise self._unverified(request_id, thread, wire, http_status=status,
                                       response_status=evidence.get("response_status"),
                                       stop_reason=evidence.get("stop_reason"),
                                       response_error=evidence.get("response_error"))
            try:
                text = wire.evidence_answer(terminal)
            except ValueError as error:
                if str(error) == EVIDENCE_UNVERIFIED:
                    raise self._unverified(request_id, thread, wire, http_status=status,
                                           response_status=evidence.get("response_status"),
                                           stop_reason=evidence.get("stop_reason")) from None
                raise
            answer = presentation.restore(json.loads(text))
            normalized = wire.normalize_usage(evidence.get("usage")) if evidence.get("usage_known") else None
            usage = (dict(total=normalized, last=dict(normalized), model_context_window=None)
                     if normalized is not None else None)
            return dict(model=self.model, status="completed", answer=answer, usage=usage,
                        thread_id=thread, turn_id=evidence.get("response_id"),
                        served_model=evidence.get("served_model"), context_policy=self.context_policy,
                        backend=self.backend_label, wire_api=wire.name, upstream=self.upstream,
                        model_verification="requested_model_recorded_served_model_reported",
                        runtime_policy="single_stateless_request_no_session_no_memory_v1",
                        tool_policy="no_tools_declared_reject_any_tool_output_v1")
        except Exception:
            if cancel_event is not None and cancel_event.is_set():
                raise TimeoutError(CANCELLED) from None
            raise
        finally:
            done.set()
            try:
                connection.close()
            except OSError:
                pass
            proxy.unregister_thread(thread)
            if watcher is not None:
                watcher.join(2)


# ------------------------------------------------------------------------------------- evidence upstream
def resolve_upstream(upstream, *, codex_home=None, environ=None):
    """Wire, URL and host credentials of an ``agent.evidence.upstream`` (``UpstreamConfig``).

    ``source="codex_home"``: the Codex credential source under ``codex_home`` (``model.codex_home``: its
    ``config.toml`` provider and ``auth.json``), on the OpenAI Responses wire. ``source="explicit"``: ``wire_api``
    and ``base_url`` from the section, and the key from the environment variable NAMED by ``api_key_env`` (read from
    ``environ``, default ``os.environ``; None = no credential header) sent as ``Authorization: Bearer <key>``
    (``auth_scheme="bearer"``) or ``x-api-key: <key>``. Returns ``(wire_adapter, base_url, auth_headers_or_None,
    provider_label)``; the key value never leaves this process except in the relay's upstream request headers.
    Raises ``ConfigError`` for a missing key or an unusable Codex home.
    """
    environ = os.environ if environ is None else environ
    if upstream.source == "codex_home":
        from ..session.codex import _model_provider, _upstream_auth
        if not codex_home:
            raise ConfigError("evidence_upstream_codex_home_unset: set model.codex_home")
        source = Path(codex_home).expanduser()
        try:
            document = tomllib.loads((source / "config.toml").read_text())
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigError("evidence_upstream_codex_home_unreadable: " + type(error).__name__) from None
        name, provider, url = _model_provider(document, source)
        if provider.get("wire_api", "responses") != "responses":
            raise ConfigError("evidence_upstream_codex_home_requires_responses_provider")
        return OpenAIResponsesWire(), url, _upstream_auth(provider, source) or None, upstream.provider or name
    headers = None
    if upstream.api_key_env is not None:
        key = environ.get(upstream.api_key_env)
        if not key:
            raise ConfigError(f"evidence_upstream_api_key_missing:{upstream.api_key_env}: export the key in the "
                              "gateway environment")
        headers = ({"Authorization": "Bearer " + key} if upstream.auth_scheme == "bearer" else {"x-api-key": key})
    return UPSTREAM_WIRES[upstream.wire_api](), upstream.base_url, headers, upstream.provider


def bind_evidence_upstream(session, config, *, environ=None):
    """Bind ``config.agent.evidence.upstream`` on the session's relay as ``"evidence"`` and return the binding name
    the visual reviewer's ``RelayEvidenceClient`` uses: ``"evidence"``, or ``"actor"`` when none is configured.

    Idempotent (an existing ``"evidence"`` binding is kept). Called by the gateway after ``session.start()`` and
    before the first actor turn. The binding shares the relay's budget, events, retry/recovery policy and blocked
    gate and uses only its own wire, URL and credentials; its requests pass the wire's egress walker with the run's
    ``ActorPresentation``. Records the ``model_upstream_bound`` relay event (credential header names only).
    """
    upstream = config.agent.evidence.upstream
    if upstream is None:
        return ACTOR_UPSTREAM
    proxy = getattr(session, "model_proxy", None)
    if proxy is None:
        raise RuntimeError("evidence_client_requires_started_model_relay")
    try:
        proxy.upstream_binding(EVIDENCE_UPSTREAM)
        return EVIDENCE_UPSTREAM
    except KeyError:
        pass
    wire, url, headers, _provider = resolve_upstream(upstream, codex_home=config.model.codex_home, environ=environ)
    proxy.add_upstream(EVIDENCE_UPSTREAM, wire=wire, upstream_url=url, upstream_auth_headers=headers,
                       presentation=ActorPresentation(session.store))
    return EVIDENCE_UPSTREAM
