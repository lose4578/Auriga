"""OpenAI Responses wire (Codex, OpenCode): routes, SSE terminals, errors, images and the egress scrub.

The ``codex`` profile is a behaviour-preserving extraction from ``model_proxy.py`` and
``CodexSession._present_model_request`` (c4ff16a). Every byte the relay forwards or
delivers through it is unchanged; golden fixtures in
``tests/fixtures/wire_forwarding_golden.json`` pin that.

The ``opencode`` profile (``@ai-sdk/openai`` in opencode 1.19.5, spike report 06 S6/S9) adds the
``/v1/responses`` aliases, thread key ``x-opencode-session-id`` and deferred delivery: nothing is
sent before the attempt is validated, then the complete stream or one HTTP 400
``invalid_request_error`` JSON with a neutral digit-free message (OpenCode retries a
``response.failed``/``error`` event five times and takes ``response.incomplete`` as a silent
success, so neither is ever delivered to it); every denial is the same 400 form.
"""
from copy import deepcopy
import json
import math
import re
from types import MappingProxyType, SimpleNamespace

from ...agents import context_signal
from ..usage import TOKEN_FIELDS, counter
from .base import (CODEX_PROFILE, EVIDENCE_ONE_ANSWER, EVIDENCE_TOOL_ACTIVITY, EVIDENCE_UNVERIFIED,
                   OPENCODE_DENIAL_MESSAGES, OPENCODE_PROFILE, OPENCODE_REQUEST_FAILED_MESSAGE,
                   OPENCODE_ROUTE_NOT_FOUND_MESSAGE, OPENCODE_THREAD_HEADER, Route, WireAdapter, WireReply,
                   evidence_parts, header_value, neutralize_instruction_header)

TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.failed", "response.incomplete", "error"})
# Codex times parsed SSE events, so comments do not keep its idle watchdog alive.
# Unknown event types are ignored without model output.
KEEPALIVE_FRAME = b'event: auriga.keepalive\ndata: {"type":"auriga.keepalive"}\n\n'
RELAY_ERROR_MESSAGE = "Model request failed; provider-error retry policy exhausted or not applicable."
DENIALS = MappingProxyType({
    "thread_cancelled": WireReply(409, "model thread explicitly cancelled"),
    # Blocks SDK reconnect, automatic compaction recovery and any new logical
    # request until an operator authorizes it.
    "transport_blocked": WireReply(409, "model transport blocked; manual review required"),
    "budget": WireReply(429, "episode model completion budget unavailable"),
})
# Native Responses error categories (code first, then type). Harness policy also retries
# explicit temporary overloads; auth/configuration failures stay terminal.
FAILURE_CLASSIFICATION = MappingProxyType({
    "context_length_exceeded": "context_window_exceeded",
    "insufficient_quota": "quota_exceeded", "credit_balance_exhausted": "quota_exceeded",
    "organization_spend_limit_exceeded": "quota_exceeded", "project_spend_limit_exceeded": "quota_exceeded",
    "usage_not_included": "usage_not_included", "cyber_policy": "cyber_policy",
    "bio_policy": "bio_policy", "misalignment_policy_violation": "misalignment_policy_violation",
    "invalid_api_key": "authentication_error", "authentication_error": "authentication_error",
    "permission_denied": "permission_denied", "model_not_found": "configuration_error",
    "invalid_request_error": "configuration_error", "unsupported_parameter": "configuration_error",
    "invalid_prompt": "invalid_prompt", "image_compression_failed": "retryable",
    "server_is_overloaded": "server_overloaded",
    "server_overloaded": "server_overloaded", "overloaded": "server_overloaded",
    "content_filter": "refusal", "content_policy": "refusal", "refusal": "refusal",
    "flex_unavailable": "flex_unavailable"})
RETRYABLE_CLASSIFICATIONS = frozenset({"retryable", "stream", "rate_limit", "server_overloaded"})
_OUTPUT_ITEM_TYPES = ("function_call_output", "custom_tool_call_output")
# Responses ``input`` item types that echo the model's own earlier output back (besides ``assistant``
# messages): reasoning items and the model's tool calls. Their results (``_OUTPUT_ITEM_TYPES``) are client
# input. Read only by ``history_model_output_flags`` (relay consume guard; never on the codex profile).
MODEL_OUTPUT_ITEM_TYPES = frozenset({"reasoning", "function_call", "custom_tool_call"})
_TEXT_PART_TYPES = ("input_text", "inputText", "output_text", "text")
# Egress scrub (see present_responses_request).
CLIENT_TOOL_TYPES = frozenset({"function", "custom", "namespace", "local_shell", "shell"})
SCRUBBED_TEXT_FIELDS = frozenset({"text", "content", "output", "arguments", "input",
                                  "instructions", "description", "summary", "refusal"})
SCHEMA_FIELDS = frozenset({"parameters", "inputSchema", "schema"})
# Relay-generated OpenCode answers (deferred delivery): the OpenAI error object shape.
OPENCODE_ERROR_HEADERS = MappingProxyType({"Content-Type": "application/json"})
SYSTEM_ITEM_ROLES = frozenset({"system", "developer"})
# Relay evidence requests (design §9.3). ``text.format`` mirrors what the pinned Codex app-server sends
# for a turn ``outputSchema`` (``tests/fixtures/evidence/codex_output_schema_request.json``):
# ``{"type": "json_schema", "name": ..., "schema": S, "strict": true}``.
EVIDENCE_FORMAT_NAME = "evidence_answer"
EVIDENCE_FORMAT_STRICT = True


def _openai_error(message):
    return json.dumps({"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}},
                      separators=(",", ":")).encode()


class OpenAIResponsesWire(WireAdapter):
    """``POST /responses`` and ``/responses/compact`` as sent by the pinned Codex client, or
    ``POST /v1/responses`` as sent by OpenCode.

    Thread key: the ``thread-id`` request header (codex) or ``x-opencode-session-id`` (opencode).
    Credentials: ``Authorization`` and ``ChatGPT-Account-Id``. Terminal events:
    ``response.completed|failed|incomplete`` and ``error``; only ``completed`` is a usable
    completion. Images: ``input_image`` parts with ``data:image/`` URLs inside message ``content``
    and ``function_call_output`` / ``custom_tool_call_output`` ``output`` arrays; an omitted image
    becomes an ``input_text`` part. Body limit 4.5 MB, trim target 2.4 MB (unchanged constants).
    Client profiles: ``codex`` (default; early 200 + ``auriga.keepalive`` every second, 409/429
    denials via the standard error page) and ``opencode`` (module docstring).
    """

    name = "openai_responses"
    client_profiles = MappingProxyType({"codex": CODEX_PROFILE, "opencode": OPENCODE_PROFILE})
    auth_header_names = MappingProxyType({"authorization": "Authorization", "chatgpt-account-id": "ChatGPT-Account-Id"})
    supports_service_tier = True
    body_limit_bytes = 4_500_000
    trim_target_bytes = 2_400_000
    evidence_path = "/responses"
    ROUTES = MappingProxyType({
        "/responses": Route("generation", "/responses", "/responses"),
        "/responses/compact": Route("generation", "/responses/compact", "/responses/compact", compaction=True),
    })
    # OpenCode's provider base URL ends in ``/v1`` (spike S6.1): its requests arrive at the ``/v1``
    # aliases and are forwarded to the same upstream paths as the Codex routes.
    OPENCODE_ALIASES = MappingProxyType({"/v1/responses": "/responses", "/v1/responses/compact": "/responses/compact"})

    # ------------------------------------------------------------------ request ingress
    def route(self, method, raw_path):
        """codex: exact paths only (no query string, trailing slash or ``/v1`` alias). opencode: also
        the ``/v1/responses[/compact]`` aliases, with any query string kept upstream; ``endpoint``
        is the normalised path without ``/v1`` and query."""
        if method != "POST":
            return None
        if self.client != "opencode":
            return self.ROUTES.get(raw_path)
        if not isinstance(raw_path, str):
            return None
        path, _, query = raw_path.partition("?")
        route = self.ROUTES.get(self.OPENCODE_ALIASES.get(path, path))
        if route is None or not query:
            return route
        return route._replace(upstream_path=route.upstream_path + "?" + query)

    def thread_key(self, headers, request=None):
        if self.client == "opencode":
            value = header_value(headers, OPENCODE_THREAD_HEADER)
            return value if isinstance(value, str) else ""
        return headers.get("thread-id", "")

    def is_streaming(self, request):
        return request.get("stream") is True

    def requested_model(self, request):
        return request.get("model")

    def project_request(self, request, presentation, *, registered_tools, tool_name_prefix,
                        on_unknown_part=None):
        """Egress re-scrub of a Responses body for non-Codex clients (Codex keeps its own
        ``CodexSession._present_model_request``): ``present_responses_request`` with no native
        tool search (``deferred_tools`` empty, no namespace), so a ``tool_search_output`` is rejected
        unless it names only registered tools. On the opencode profile the ``Instructions from:
        <path>`` lines of the system text (``instructions`` and the leading developer/system
        ``input`` items) become ``Instructions:`` first. ``prompt_cache_key``, ``include``,
        ``store``, ids, encrypted reasoning and media stay byte-identical. ``tool_name_prefix`` and
        ``on_unknown_part`` are accepted for the common signature (this walker has no unknown-part
        report: every key is walked).
        """
        if self.client == "opencode" and isinstance(request, dict):
            request = _neutralized_system_items(request)
        return present_responses_request(request, presentation, deferred_tools=frozenset(registered_tools),
                                         namespace=None)

    def history_view(self, request):
        """``instructions`` plus the leading developer/system ``input`` items vs the remaining
        ``input`` items, ``context_signal.openai_responses_history`` (not used for Codex: the relay
        installs no continuity tracker on the ``codex`` profile)."""
        return context_signal.openai_responses_history(request)

    def history_model_output_flags(self, request):
        """``WireAdapter.history_model_output_flags`` for Responses ``input`` items: True for an
        ``assistant`` message and for the model's own output item types (``MODEL_OUTPUT_ITEM_TYPES``:
        reasoning and tool calls), False for user/system/developer messages, tool outputs
        (``function_call_output``, ``custom_tool_call_output``) and unknown types."""
        return tuple(isinstance(item, dict) and (item.get("role") == "assistant"
                                                 or item.get("type") in MODEL_OUTPUT_ITEM_TYPES)
                     for item in self.history_view(request).items)

    # ------------------------------------------------------------------------- images
    @staticmethod
    def part_observation_labels(parts):
        """``(observation_id, camera_id)`` for each part; images take labels from the
        preceding Auriga tool-output JSON text (bundle cameras, image view or overlay)."""
        labels, queue, observation = [], [], None
        for part in parts:
            if not isinstance(part, dict):
                labels.append((None, None))
                continue
            ptype = part.get("type")
            if ptype in _TEXT_PART_TYPES:
                value = None
                try:
                    value = json.loads(part.get("text", ""))
                except (TypeError, ValueError):
                    pass
                bundle = value.get("bundle") if isinstance(value, dict) else None
                if isinstance(bundle, dict) and isinstance(bundle.get("observation_id"), str):
                    observation = bundle["observation_id"]
                    view = value.get("image_view")
                    feedback = value.get("image_feedback")
                    if isinstance(view, dict) and view.get("camera_id"):
                        queue = [view["camera_id"]]
                    elif isinstance(feedback, dict) and isinstance(feedback.get("included_camera_ids"), list):
                        queue = list(feedback["included_camera_ids"])
                    else:
                        queue = [cam.get("camera_id") for cam in bundle.get("cameras", []) if isinstance(cam, dict)]
                elif isinstance(value, dict):
                    observation = (value.get("observation_id")
                                   or (value.get("receipt") or {}).get("result_observation_id")
                                   or (value.get("observation_ids") or [None])[0])
                    overlay = value.get("trajectory_overlay") or value.get("value")
                    cam = overlay.get("camera_id") if isinstance(overlay, dict) else None
                    queue = [cam] if cam else []
                else:
                    queue, observation = [], None
                labels.append((None, None))
            elif ptype == "input_image" and str(part.get("image_url", "")).startswith("data:image/"):
                camera = queue.pop(0) if queue else None
                labels.append((observation, camera))
            else:
                labels.append((None, None))
        return labels

    @staticmethod
    def _inline_image(part):
        return isinstance(part, dict) and part.get("type") == "input_image" and str(part.get("image_url", "")).startswith("data:image/")

    def image_locations(self, request, *, include_labels=False):
        items = request.get("input") if isinstance(request, dict) else None
        if not isinstance(items, list):
            return []
        locations = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = "content" if item.get("type", "message") == "message" else (
                "output" if item.get("type") in _OUTPUT_ITEM_TYPES else None)
            parts = item.get(key) if key else None
            if not isinstance(parts, list):
                continue
            labels = self.part_observation_labels(parts) if include_labels else None
            for idx, part in enumerate(parts):
                if self._inline_image(part):
                    locations.append((parts, idx, labels[idx]) if include_labels else (parts, idx))
        return locations

    def step_image_boundaries(self, request):
        """A step ends after a message or after the last of consecutive tool outputs."""
        items = request.get("input") if isinstance(request, dict) else None
        if not isinstance(items, list):
            return [], []
        locations, boundaries = [], []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            itype = item.get("type", "message")
            key = "content" if itype == "message" else ("output" if itype in _OUTPUT_ITEM_TYPES else None)
            parts = item.get(key) if key else None
            if isinstance(parts, list):
                for idx, part in enumerate(parts):
                    if self._inline_image(part):
                        locations.append((parts, idx))
            next_type = items[i + 1].get("type") if i + 1 < len(items) and isinstance(items[i + 1], dict) else None
            if (itype not in _OUTPUT_ITEM_TYPES or next_type not in _OUTPUT_ITEM_TYPES) and locations and (
                    not boundaries or boundaries[-1] != len(locations)):
                boundaries.append(len(locations))
        return locations, boundaries

    def image_part_size(self, part):
        return len(part.get("image_url", ""))

    def placeholder_part(self, text):
        return dict(type="input_text", text=text)

    def reuse_exact_images(self, request):
        # The default part locator of reuse_exact_images is this wire's shape.
        from ....core.presentation.image_reuse import reuse_exact_images
        return reuse_exact_images(request)

    def image_rejected(self, failure_evidence):
        return failure_evidence.get("code") == "image_compression_failed"

    # ----------------------------------------------------------------- response stream
    def keepalive_frame(self):
        """``auriga.keepalive`` event (codex); None for the deferred opencode profile."""
        return None if self.client == "opencode" else KEEPALIVE_FRAME

    def terminal_sse_type(self, block):
        data = b"\n".join(line[5:].lstrip(b" ") for line in block.splitlines() if line.startswith(b"data:"))
        if not data or data == b"[DONE]":
            return None
        try:
            value = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return None
        event_type = value.get("type") if isinstance(value, dict) else None
        return event_type if event_type in TERMINAL_EVENT_TYPES else None

    def terminal_allows_reuse(self, terminal_type):
        return terminal_type == "response.completed"

    def response_evidence(self, raw, route):
        """Return authoritative response/usage; never construct output from deltas.

        A compaction route's non-streaming JSON answer is complete without a status.
        """
        compaction = route.compaction
        terminal = None
        if self.is_event_stream(raw):
            for block in raw.decode().replace("\r\n", "\n").split("\n\n"):
                data = "\n".join(line[5:].lstrip(" ") for line in block.splitlines() if line.startswith("data:"))
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("type") == "error":
                    error = event.get("error", event)
                    response = dict(status="failed", output=[], error=error)
                    return True, dict(response_id=None, served_model=None, response_status="failed",
                                      usage=None, usage_known=False,
                                      response_error=self.failed_response_error(response)), response
                if event.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                    terminal = event
                    break
            if terminal is None:
                raise EOFError("response terminal missing")
            response = terminal.get("response")
            complete = True  # A terminal is transport evidence, not necessarily a usable completion.
        else:
            response = json.loads(raw)
            complete = compaction or response.get("status") in ("completed", "incomplete", "failed")
        if not isinstance(response, dict) or (response.get("status") != "failed" and not isinstance(response.get("output"), list)):
            raise ValueError("invalid response structure")
        if terminal is not None and terminal["type"] != "response." + str(response.get("status")):
            raise ValueError("response event and status conflict")
        if not compaction:
            complete = complete and response.get("status") in ("completed", "incomplete", "failed")
        evidence = dict(response_id=response.get("id"), served_model=response.get("model"),
                        response_status=response.get("status"),
                        usage=response.get("usage"), usage_known=isinstance(response.get("usage"), dict) and bool(response["usage"]))
        if response.get("status") == "failed":
            evidence["response_error"] = self.failed_response_error(response)
        return complete, evidence, response

    def completion_outcome(self, complete, evidence):
        status = evidence.get("response_status")
        if status in ("failed", "incomplete"):
            # Neither failed nor incomplete responses may release tool deltas.
            return status
        return "usable" if complete else "not_completed"

    def failed_response_error(self, body):
        """Codex Responses error mapping; never persist the provider message.

        Based on native Responses error categories. Harness policy also retries
        explicit temporary overloads; auth/configuration failures stay terminal.
        """
        error = body.get("error")
        detail = error if isinstance(error, dict) else {}
        valid = isinstance(error, dict) and all(detail.get(k) is None or isinstance(detail[k], str)
                    for k in ("type", "code", "message", "plan_type"))
        reset = detail.get("resets_at")
        valid = valid and (reset is None or type(reset) is int and -(2**63) <= reset < 2**63)
        code = detail.get("code")
        category = ("flex_unavailable" if code == "flex_unavailable" else "stream" if not valid else
                    FAILURE_CLASSIFICATION.get(code, FAILURE_CLASSIFICATION.get(detail.get("type"),
                        "rate_limit" if code in ("rate_limit_exceeded", "slow_down") else "retryable")))
        delay = None
        if category == "rate_limit":
            match = re.search(r"(?i)try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)", detail.get("message") or "")
            if match:
                delay = float(match[1]) / (1000 if match[2].lower() == "ms" else 1)
                if not math.isfinite(delay) or delay <= 0:
                    delay = None
        def safe(value):
            return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", value) else None
        return dict(code=safe(code), type=safe(detail.get("type")),
                    code_state="unknown" if code is None else "present" if safe(code) is not None else "redacted_or_invalid",
                    classification=category, retryable=category in RETRYABLE_CLASSIFICATIONS,
                    retry_after_s=delay)

    def classify_http_failure(self, status, headers, body, *, policy):
        error = Exception("provider HTTP failure")
        error.status_code = status
        error.body = body
        # The pinned classifier only reads these attributes, never raw headers/body into logs.
        error.response = SimpleNamespace(headers={k.lower(): v for k, v in headers.items()})
        failure = policy.classify_openai_error(error)
        provider_error = self.failed_response_error(body if isinstance(body, dict) else {})
        if provider_error.get("code") == "image_compression_failed":
            failure.retryable = True
        elif provider_error.get("classification") == "configuration_error":
            # The pinned classifier reads only the status, so a relay's 503 model_not_found
            # looked transient. Resending the same model/parameters cannot succeed: fail now,
            # with no immediate retries and no transport-recovery cooldown.
            failure.retryable = False
        return failure

    def failed_payload(self, terminal, *, streaming):
        """codex: the failed/incomplete terminal with ``output: []`` (unchanged). opencode: None —
        OpenCode retries ``response.failed`` and accepts ``response.incomplete`` as success, so the
        relay answers with ``relay_error`` instead."""
        if self.client == "opencode":
            return None
        terminal = {**terminal, "output": []}
        if not streaming:
            return json.dumps(terminal).encode()
        event_type = "response." + terminal["status"]
        return ("event: " + event_type + "\ndata: " + json.dumps(dict(type=event_type, response=terminal)) + "\n\n").encode()

    def relay_error(self, *, streaming, evidence, manual_retry_allowed):
        """codex: unchanged (an ``error`` event in the open stream, or a JSON 502/HTTP-status body).
        opencode: HTTP 400 ``invalid_request_error`` JSON with a neutral message and no evidence."""
        if self.client == "opencode":
            return WireReply(400, headers=OPENCODE_ERROR_HEADERS, body=_openai_error(OPENCODE_REQUEST_FAILED_MESSAGE))
        if streaming:
            details = dict(type="upstream_error", code="model_request_failed", message=RELAY_ERROR_MESSAGE,
                           evidence=evidence, manual_retry_allowed=manual_retry_allowed)
            body = ("event: error\ndata: " + json.dumps(dict(type="error", error=details)) + "\n\n").encode()
            return WireReply(200, body=body)
        data = json.dumps(dict(error=dict(code="model_request_failed", message=RELAY_ERROR_MESSAGE,
                                          evidence=evidence, manual_retry_allowed=manual_retry_allowed))).encode()
        return WireReply((evidence or {}).get("http_status") or 502, headers={"Content-Type": "application/json"}, body=data)

    def denial(self, kind):
        """codex: 409/409/429 standard error pages (unchanged). opencode: HTTP 400
        ``invalid_request_error`` JSON (one client attempt; 409/429 are retried by OpenCode)."""
        if self.client == "opencode":
            return WireReply(400, headers=OPENCODE_ERROR_HEADERS, body=_openai_error(OPENCODE_DENIAL_MESSAGES[kind]))
        return DENIALS[kind]

    def not_found(self):
        """codex: the standard 404 page (unchanged); opencode: a neutral 404 JSON error."""
        if self.client == "opencode":
            return WireReply(404, headers=OPENCODE_ERROR_HEADERS, body=_openai_error(OPENCODE_ROUTE_NOT_FOUND_MESSAGE))
        return WireReply(404)

    def build_evidence_request(self, *, model, base_instructions, developer_instructions, input_items,
                               output_schema, effort):
        """Responses evidence request (§9.3): ``instructions`` = base instructions; first input item
        a developer message with the developer instructions; one user message with ``input_text`` /
        ``input_image`` (data URL) parts; ``text.format`` json_schema (name ``evidence_answer``,
        ``strict: true`` as the pinned Codex sends for ``outputSchema``); ``reasoning.effort``;
        ``tools: []``, ``store: false``, ``stream: true``. No extra headers."""
        content = []
        for part in evidence_parts(input_items):
            content.append(dict(type="input_text", text=part[1]) if part[0] == "text"
                           else dict(type="input_image", image_url=part[3]))
        body = dict(model=model, instructions=base_instructions,
                    input=[dict(type="message", role="developer",
                                content=[dict(type="input_text", text=developer_instructions)]),
                           dict(type="message", role="user", content=content)],
                    tools=[], text=dict(format=dict(type="json_schema", name=EVIDENCE_FORMAT_NAME,
                                                    schema=output_schema, strict=EVIDENCE_FORMAT_STRICT)),
                    reasoning=dict(effort=effort), store=False, stream=True)
        return body, {}

    def evidence_answer(self, terminal):
        """The answer of a ``response.completed`` terminal: the concatenated ``output_text`` parts of
        exactly one ``message`` output item; ``reasoning`` items are ignored; any other item type
        (function/custom/hosted tool calls, ...) or a refusal part is tool/context activity."""
        if not isinstance(terminal, dict) or terminal.get("status") != "completed":
            raise ValueError(EVIDENCE_UNVERIFIED)
        messages = []
        for item in terminal.get("output") or []:
            kind = item.get("type") if isinstance(item, dict) else None
            if kind == "reasoning":
                continue
            if kind != "message":
                raise ValueError(EVIDENCE_TOOL_ACTIVITY)
            parts = item.get("content") if isinstance(item.get("content"), list) else []
            if any(not isinstance(part, dict) or part.get("type") != "output_text" for part in parts):
                raise ValueError(EVIDENCE_TOOL_ACTIVITY)
            messages.append("".join(str(part.get("text", "")) for part in parts))
        if len(messages) != 1:
            raise ValueError(EVIDENCE_ONE_ANSWER)
        return messages[0]

    def normalize_usage(self, usage):
        """Same field mapping as ``usage.account_responses_usage``; missing details stay None."""
        usage = usage if isinstance(usage, dict) else {}
        incoming = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
        outgoing = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
        raw = dict(input_tokens=usage.get("input_tokens"), cached_input_tokens=incoming.get("cached_tokens"),
                   cache_write_input_tokens=incoming.get("cache_write_tokens", usage.get("cache_write_input_tokens")),
                   output_tokens=usage.get("output_tokens"), reasoning_output_tokens=outgoing.get("reasoning_tokens"),
                   total_tokens=usage.get("total_tokens"))
        return {key: counter(raw[key]) for key in TOKEN_FIELDS}


def _neutralized_system_items(request):
    """A shallow copy of a Responses body whose ``instructions`` and leading developer/system
    ``input`` items have OpenCode's ``Instructions from: <path>`` lines neutralised."""
    result = dict(request)
    if isinstance(result.get("instructions"), str):
        result["instructions"] = neutralize_instruction_header(result["instructions"])
    items = result.get("input")
    if isinstance(items, list):
        items = list(items)
        for index, item in enumerate(items):
            if not (isinstance(item, dict) and item.get("role") in SYSTEM_ITEM_ROLES):
                break
            content = item.get("content")
            if isinstance(content, str):
                content = neutralize_instruction_header(content)
            elif isinstance(content, list):
                content = [dict(part, text=neutralize_instruction_header(part["text"]))
                           if isinstance(part, dict) and isinstance(part.get("text"), str) else part
                           for part in content]
            items[index] = dict(item, content=content)
        result["input"] = items
    return result


def present_responses_request(request, presentation, *, deferred_tools, namespace):
    """Egress information-boundary re-scrub of one Responses body (resumed/compact input too).

    request: decoded Responses body (dict); never mutated. presentation: the session's
    ``ActorPresentation``. deferred_tools: set of Auriga tool names that native local tool
    search may reveal. namespace: the dynamic-tool namespace name those tools live in
    (Codex ``"auriga"``). Returns a new body.

    Prose at keys ``text, content, output, arguments, input, instructions, description,
    summary, refusal`` goes through ``present_text``; JSON schemas at
    ``parameters|inputSchema|schema`` through ``present_data``; structured
    ``tool_search_call.arguments`` through ``present``. Native item/call ids, encrypted
    reasoning and media are protocol data and stay byte-identical. Raises ``ValueError``
    ``sandbox_remote_model_tool_disabled`` for tool types outside
    ``{function, custom, namespace, local_shell, shell}``, non-client tool search and
    non-local shell environments; ``sandbox_remote_media_url_disabled`` for non-``data:``
    media and ``file_url``; ``native_discovered_tool_not_registered`` for a
    ``tool_search_output`` naming anything but registered deferred tools.
    """
    def check_tools(tools):
        for tool in tools:
            kind = tool.get("type")
            if kind not in CLIENT_TOOL_TYPES:
                if kind != "tool_search" or tool.get("execution") != "client":
                    raise ValueError("sandbox_remote_model_tool_disabled")
            if kind == "namespace":
                check_tools(tool.get("tools", []))
            if kind in {"local_shell", "shell"}:
                environment = tool.get("environment") or {"type": "local"}
                if environment.get("type") != "local":
                    raise ValueError("sandbox_remote_model_tool_disabled")
    check_tools(request.get("tools", []))

    def readable(value):
        if isinstance(value, list):
            return [readable(item) for item in value]
        if not isinstance(value, dict):
            return deepcopy(value)
        kind = value.get("type")
        if kind == "additional_tools":
            check_tools(value.get("tools", []))
        if kind in {"tool_search_call", "tool_search_output"}:
            if value.get("execution") != "client":
                raise ValueError("sandbox_remote_model_tool_disabled")
        if kind == "tool_search_output":
            discovered = value.get("tools", [])
            check_tools(discovered)
            for group in discovered:
                if (group.get("type") != "namespace"
                        or group.get("name") != namespace
                        or any(tool.get("type") != "function" or tool.get("name") not in deferred_tools
                               for tool in group.get("tools", []))):
                    raise ValueError("native_discovered_tool_not_registered")
        if value.get("type") in {"input_image", "inputImage", "image_url", "image"}:
            url = value.get("image_url", value.get("imageUrl", value.get("url")))
            if isinstance(url, dict):
                url = url.get("url")
            if isinstance(url, str) and not url.startswith("data:"):
                raise ValueError("sandbox_remote_media_url_disabled")
        if value.get("file_url") or value.get("fileUrl"):
            raise ValueError("sandbox_remote_media_url_disabled")
        result = {}
        for key, item in value.items():
            if key in SCHEMA_FIELDS and isinstance(item, dict):
                # A provenance-looking property is still a schema property.
                result[key] = presentation.present_data(item)
            elif kind == "tool_search_call" and key == "arguments" and isinstance(item, (dict, list)):
                result[key] = presentation.present(item)
            elif key in SCRUBBED_TEXT_FIELDS and isinstance(item, str):
                result[key] = presentation.present_text(item)
            elif isinstance(item, (dict, list)):
                result[key] = readable(item)
            else:
                result[key] = deepcopy(item)
        return result

    result = deepcopy(request)
    for key in ("instructions", "input", "tools", "text"):
        if key in result:
            result[key] = (presentation.present_text(result[key])
                           if isinstance(result[key], str) else readable(result[key]))
    return result
