"""Anthropic Messages wire (Claude Code, OpenCode): routes, SSE terminals, errors, images and the egress scrub.

Client profile ``claude_code`` (the default), measured against the real Claude Code CLI 2.1.321
(spike report 05, S1/S2/S5):

* routes: ``POST /v1/messages`` (generation; the query string, e.g. ``?beta=true``, is kept
  upstream), ``POST /v1/messages/count_tokens`` (auxiliary: it carries raw tool output, so it is
  projected like a generation body, but never charged or retried), ``GET /v1/models`` (local 404).
* thread key: the ``X-Claude-Code-Session-Id`` header (equals the host-chosen session id).
* delivery: early HTTP 200 + ``event: ping`` every ``keepalive_interval_s`` seconds while the
  relay buffers (pings before ``message_start`` are accepted by the CLI); a failure inside the
  open stream is one ``event: error`` of type ``invalid_request_error`` (single attempt).
* denials and relay errors outside a stream: HTTP 403 ``permission_error`` with
  ``x-should-retry: false`` (one attempt; a 400/422 makes the CLI re-send twice with fewer betas,
  413 maps to its image-error path). Client-visible bytes carry no digits and no identifiers.

Client profile ``opencode`` (``@ai-sdk/anthropic`` in opencode 1.19.5, spike report 06 S6/S9): same
routes and body format; thread key ``x-opencode-session-id``; *deferred delivery* (no status line,
no keepalive before the attempt is validated, then the complete stream or one HTTP 400
``invalid_request_error`` JSON with ``x-should-retry: false`` and a neutral digit-free message —
OpenCode retries a first-event stream error five times but makes one attempt on such a 400); every
denial is the same 400 form. Egress: the ``Instructions from: <path>`` line OpenCode renders for its
instruction file is replaced by ``Instructions:`` in the ``system`` text.

Images (user decision, design revision 3.1): this adapter only *locates* images for the
existing generic trim/reuse logic (``WireAdapter.trim_image_range`` and the relay's window state
machine) with the Responses thresholds; it never changes image bytes, sizes or quality.
"""
from copy import deepcopy
import json
import re
from types import MappingProxyType
import zlib

from ...agents import context_signal
from ..usage import TOKEN_FIELDS, counter
from .base import (EVIDENCE_ONE_ANSWER, EVIDENCE_TOOL_ACTIVITY, EVIDENCE_UNVERIFIED, OPENCODE_DENIAL_MESSAGES,
                   OPENCODE_PROFILE, OPENCODE_REQUEST_FAILED_MESSAGE, OPENCODE_THREAD_HEADER, ClientProfile, Route,
                   WireAdapter, WireReply, evidence_parts, header_value, neutralize_instruction_header)
from .openai_responses import OpenAIResponsesWire

GENERATION_PATH = "/v1/messages"
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
MODELS_PATH = "/v1/models"
TERMINAL_EVENT_TYPES = frozenset({"message_stop", "error"})
# Exact frame of the Anthropic API's own keepalive; the CLI ignores it while waiting.
KEEPALIVE_FRAME = b'event: ping\ndata: {"type": "ping"}\n\n'
# Stop reasons whose answer is delivered as a usable completion (when every tool_use block is
# closed and its input parses as an object; design §4.2, a closed set).
USABLE_STOP_REASONS = frozenset({"end_turn", "tool_use", "stop_sequence", "pause_turn", "refusal"})
# Stop reasons that cut generation short: usable only when the structural check passes and every
# streamed tool_use block received at least one non-blank argument byte (a call cut before its
# first argument byte would otherwise run as an all-defaults ``{}`` call). Any stop reason in
# neither set (a value this table does not know) is ``incomplete``.
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})
# Native error ``type`` → Auriga classification. ``invalid_request_error`` is refined by the
# message classes below; an unknown type string is treated like a transient provider failure.
FAILURE_CLASSIFICATION = MappingProxyType({
    "overloaded_error": "server_overloaded", "rate_limit_error": "rate_limit", "api_error": "retryable",
    "timeout_error": "retryable", "authentication_error": "authentication_error",
    "permission_error": "permission_denied", "not_found_error": "configuration_error",
    "billing_error": "quota_exceeded", "request_too_large": "image_rejected",
    "invalid_request_error": "configuration_error"})
# image_rejected is retried only after the relay advanced the image-trim window (and never when
# image trim is off), like the Responses image_compression_failed code.
RETRYABLE_CLASSIFICATIONS = frozenset({"retryable", "stream", "rate_limit", "server_overloaded", "image_rejected"})
# Message classes of invalid_request_error (read only to classify, never stored or forwarded).
_PROMPT_TOO_LONG = re.compile(r"(?i)prompt is too long|context (?:window|length) (?:exceeded|is exceeded)")
_CREDIT_BALANCE = re.compile(r"(?i)credit balance")
_IMAGE_OR_SIZE = re.compile(r"(?i)\b(?:image|images|request|body|payload)\b[^.\n]*"
                            r"\b(?:exceeds?|exceeded|too large|too big|too many)\b|\btoo many images\b")
# Relay-generated client-visible messages: no digits and none of the words OpenCode/Claude Code
# retry on (rate limit, overloaded, timeout, try again, capacity, server error, connection,
# unavailable). Checked by the adapter tests.
RESPONSE_UNUSABLE_MESSAGE = "auriga_model_response_unusable"
REQUEST_FAILED_MESSAGE = "auriga_model_request_failed"
ROUTE_NOT_FOUND_MESSAGE = "auriga_relay_route_not_found"
DENIAL_MESSAGES = MappingProxyType({
    "thread_cancelled": "auriga_model_thread_cancelled",
    "transport_blocked": "auriga_model_transport_blocked",
    "budget": "auriga_model_budget_exhausted",
    "client_retry_runaway": "auriga_client_retry_runaway",
})
NO_RETRY_HEADERS = MappingProxyType({"Content-Type": "application/json", "x-should-retry": "false"})
# Upper bound (bytes) of a gzip-decoded request body; the relay bounds the raw body the same way.
MAX_DECODED_BODY_BYTES = 128 * 1024 * 1024

# ------------------------------------------------------------------------- egress scrub tables
# Claude Code adds this text block after every MCP image inside a tool_result; the path lies in
# its private config directory. Replaced by a constant so the cache prefix stays stable.
IMAGE_SOURCE_BLOCK = re.compile(r"\[Image: source: [^\]\n]*\]")
IMAGE_SOURCE_REPLACEMENT = "[Image]"
# The same marker found anywhere in any prose the walker scrubs on the claude_code profile (merged into
# other text, followed by a newline, outside a tool_result). The path ends at the first image file
# extension followed by "]" (so a "]" inside the path does not leak its tail), else at the first "]",
# else at the end of the line (an unterminated marker).
IMAGE_SOURCE_MARKER = re.compile(
    r"\[Image: source: (?:[^\n]*?\.(?:png|jpe?g|gif|webp|bmp|tiff?|heic)\]|[^\]\n]*(?:\]|$))",
    re.IGNORECASE | re.MULTILINE)
CLIENT_TOOL_TYPES = frozenset({None, "custom"})
SERVER_TOOL_BLOCK_TYPES = frozenset({
    "server_tool_use", "web_search_tool_result", "web_fetch_tool_result", "code_execution_tool_result",
    "bash_code_execution_tool_result", "text_editor_code_execution_tool_result", "tool_search_tool_result",
    "mcp_tool_use", "mcp_tool_result", "container_upload"})
# Keys whose values the generic walker (blocks of unknown type, and thinking blocks outside assistant
# messages) copies unchanged: protocol data, media, signatures, cache markers. Every other string at
# any depth is scrubbed as prose (fail closed: an unknown key may carry host text).
GENERIC_PROTECTED_KEYS = frozenset({"id", "tool_use_id", "signature", "data", "media_type", "cache_control",
                                    "citations"})
# Signed model output, copied byte-identically only inside assistant messages.
SIGNED_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
CLAUDE_CODE_PROFILE = ClientProfile(early_stream=True, client_loop_guard=True)
# Relay evidence requests (design §9.3): native structured output (output_config.format json_schema,
# GA, no beta header), never a forced tool_choice (rejected with HTTP 400 by current Claude models).
EVIDENCE_MAX_TOKENS = 16000        # output token cap of one evidence answer (non-thinking + thinking)
EVIDENCE_API_VERSION = "2023-06-01"
EVIDENCE_USABLE_STOP = "end_turn"  # refusal / max_tokens / pause_turn / others: unverified


_header = header_value  # case-insensitive request header lookup (kept name)


def _error_json(error_type, message):
    return json.dumps({"type": "error", "error": {"type": error_type, "message": message}},
                      separators=(",", ":")).encode()


def _error_frame(error_type, message):
    return b"event: error\ndata: " + _error_json(error_type, message) + b"\n\n"


class AnthropicMessagesWire(WireAdapter):
    """``POST /v1/messages`` as sent by Claude Code (``client="claude_code"``, the default) or
    OpenCode (``client="opencode"``; see the module docstring for the profile differences).

    Credentials: ``x-api-key`` or ``Authorization`` (host-injected; client copies stripped);
    ``anthropic-version`` / ``anthropic-beta`` pass through unchanged. Request bodies may be
    gzip-encoded (decoded here; Claude Code only gzips for ``api.anthropic.com``). Terminals:
    ``message_stop`` and ``error``. Images: inline ``image`` blocks (``source.type == "base64"``)
    in message ``content`` and nested in ``tool_result.content``; an omitted image becomes a
    ``text`` block. Thresholds are the Responses values (body 4.5 MB, trim target 2.4 MB).
    """

    name = "anthropic_messages"
    client_profiles = MappingProxyType({"claude_code": CLAUDE_CODE_PROFILE, "opencode": OPENCODE_PROFILE})
    methods = frozenset({"POST", "GET"})
    auth_header_names = MappingProxyType({"x-api-key": "x-api-key", "authorization": "Authorization"})
    supports_service_tier = False
    body_limit_bytes = OpenAIResponsesWire.body_limit_bytes
    trim_target_bytes = OpenAIResponsesWire.trim_target_bytes
    # Claude Code's stream watchdog resets on at most 30 consecutive pings, so a wider spacing
    # covers a longer wait; 5 s and 10 s were both accepted by the real CLI (spike S5).
    keepalive_interval_s = 5.0
    max_decoded_body_bytes = MAX_DECODED_BODY_BYTES
    evidence_path = GENERATION_PATH

    # ------------------------------------------------------------------ request ingress
    def route(self, method, raw_path):
        """``POST /v1/messages[?query]`` → generation, ``POST /v1/messages/count_tokens[?query]`` →
        auxiliary, ``GET /v1/models[/<id>]`` → local; anything else None. The query string is
        appended to the upstream path unchanged; ``endpoint`` never contains it."""
        if not isinstance(raw_path, str):
            return None
        path, _, query = raw_path.partition("?")
        upstream = path + ("?" + query if query else "")
        if method == "POST" and path == GENERATION_PATH:
            return Route("generation", path, upstream)
        if method == "POST" and path == COUNT_TOKENS_PATH:
            return Route("auxiliary", path, upstream)
        if method == "GET" and (path == MODELS_PATH or path.startswith(MODELS_PATH + "/")):
            return Route("local", MODELS_PATH, "")
        return None

    def decode_body(self, headers, raw):
        """Identity bodies pass through; ``Content-Encoding: gzip`` is decoded (at most
        ``max_decoded_body_bytes``). Any other encoding, a corrupt or truncated gzip stream or an
        oversized result raises ``ValueError`` (the relay answers with ``relay_error``)."""
        encoding = str(_header(headers, "content-encoding") or "").strip().lower()
        if encoding in ("", "identity"):
            return raw
        if encoding != "gzip":
            raise ValueError("unsupported_request_content_encoding")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            decoded = decoder.decompress(raw, self.max_decoded_body_bytes + 1)
        except zlib.error:
            raise ValueError("invalid_gzip_request_body") from None
        if len(decoded) > self.max_decoded_body_bytes or decoder.unconsumed_tail:
            raise ValueError("decoded_request_body_too_large")
        if not decoder.eof:
            raise ValueError("invalid_gzip_request_body")
        return decoded

    def thread_key(self, headers, request=None):
        """``X-Claude-Code-Session-Id`` (claude_code) or ``x-opencode-session-id`` (opencode), any
        case, or ``""`` when absent."""
        name = OPENCODE_THREAD_HEADER if self.client == "opencode" else "x-claude-code-session-id"
        value = _header(headers, name)
        return value if isinstance(value, str) else ""

    def is_streaming(self, request):
        return isinstance(request, dict) and request.get("stream") is True

    def requested_model(self, request):
        model = request.get("model") if isinstance(request, dict) else None
        return model if isinstance(model, str) else None

    def history_view(self, request):
        """Top-level ``system`` vs every ``messages`` entry (``role:"system"`` messages included),
        ``context_signal.anthropic_messages_history``."""
        return context_signal.anthropic_messages_history(request)

    def project_request(self, request, presentation, *, registered_tools, tool_name_prefix,
                        on_unknown_part=None):
        """``present_messages_request`` with this adapter's client profile (see there)."""
        return present_messages_request(request, presentation, registered_tools=registered_tools,
                                        tool_name_prefix=tool_name_prefix, client=self.client,
                                        on_unknown_part=on_unknown_part)

    # ------------------------------------------------------------------------- images
    @staticmethod
    def _inline_image(part):
        source = part.get("source") if isinstance(part, dict) and part.get("type") == "image" else None
        return isinstance(source, dict) and source.get("type") == "base64" and isinstance(source.get("data"), str)

    @staticmethod
    def _content_lists(message):
        """The content lists of one message that may hold images, in context order: the message
        ``content`` list itself, with each ``tool_result``'s own ``content`` list expanded at the
        position of that block. Yields ``(parts, None)`` for the message list and
        ``(nested, block_index)`` for nested lists."""
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return
        yield content, None
        for index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                yield block["content"], index

    @classmethod
    def _message_images(cls, message, include_labels):
        """Inline images of one message in context order (top-level blocks interleaved with the
        images nested in each tool_result at that block's position)."""
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return []
        found = []
        top_labels = cls.part_observation_labels(content) if include_labels else None
        for index, block in enumerate(content):
            if cls._inline_image(block):
                found.append((content, index, top_labels[index]) if include_labels else (content, index))
            elif isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                nested = block["content"]
                labels = cls.part_observation_labels(nested) if include_labels else None
                for position, part in enumerate(nested):
                    if cls._inline_image(part):
                        found.append((nested, position, labels[position]) if include_labels else (nested, position))
        return found

    @classmethod
    def part_observation_labels(cls, parts):
        """``(observation_id, camera_id)`` for each part, by the Responses labelling rule: an image
        takes its labels from the preceding Auriga tool-output JSON ``text`` block. Implemented by
        handing the Responses rule an equivalent part list, so both wires label identically. The
        text block Claude Code inserts after every MCP image (``[Image: source: …]``, or ``[Image]``
        after the egress rewrite) is not tool output and is passed as a neutral part, so it does not
        end the camera sequence of a multi-camera observation."""
        def shim(part):
            if cls._inline_image(part):
                return dict(type="input_image", image_url="data:image/")
            text = part.get("text") if isinstance(part, dict) and part.get("type") == "text" else None
            if isinstance(text, str) and (text == IMAGE_SOURCE_REPLACEMENT or IMAGE_SOURCE_BLOCK.fullmatch(text)):
                return dict(type="image_source_marker")
            return part
        return OpenAIResponsesWire.part_observation_labels([shim(part) for part in parts])

    def image_locations(self, request, *, include_labels=False):
        messages = request.get("messages") if isinstance(request, dict) else None
        if not isinstance(messages, list):
            return []
        locations = []
        for message in messages:
            locations.extend(self._message_images(message, include_labels))
        return locations

    def step_image_boundaries(self, request):
        """A step is one message: Claude Code returns all tool results of one assistant turn
        (and any merged user text) in one user message, so a boundary follows every message
        that added images."""
        messages = request.get("messages") if isinstance(request, dict) else None
        if not isinstance(messages, list):
            return [], []
        locations, boundaries = [], []
        for message in messages:
            locations.extend(self._message_images(message, False))
            if locations and (not boundaries or boundaries[-1] != len(locations)):
                boundaries.append(len(locations))
        return locations, boundaries

    def image_part_size(self, part):
        source = part.get("source") if isinstance(part, dict) else None
        data = source.get("data") if isinstance(source, dict) else None
        return len(data) if isinstance(data, str) else 0

    def placeholder_part(self, text):
        return dict(type="text", text=text)

    def image_slots(self, request):
        """Part locator for ``core.presentation.image_reuse.reuse_exact_images``: yields
        ``(parts, index, inline_data)`` for every ``image`` block in context order;
        ``inline_data`` is the base64 string, or None for a non-inline source."""
        messages = request.get("messages") if isinstance(request, dict) else None
        if not isinstance(messages, list):
            return
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for index, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image":
                    yield content, index, block["source"]["data"] if self._inline_image(block) else None
                elif isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                    nested = block["content"]
                    for position, part in enumerate(nested):
                        if isinstance(part, dict) and part.get("type") == "image":
                            yield nested, position, part["source"]["data"] if self._inline_image(part) else None

    def reuse_exact_images(self, request):
        from ....core.presentation.image_reuse import reuse_exact_images
        return reuse_exact_images(request, locate=self.image_slots, text_part=self.placeholder_part)

    def image_rejected(self, failure_evidence):
        """The provider refused the request's images or size (``request_too_large``, an
        image/size ``invalid_request_error``, or HTTP 413)."""
        return (failure_evidence.get("classification") == "image_rejected"
                or failure_evidence.get("http_status") == 413)

    # ----------------------------------------------------------------- response stream
    def keepalive_frame(self):
        """``event: ping`` (claude_code); None for the deferred opencode profile (nothing is sent
        before validation)."""
        return None if self.client == "opencode" else KEEPALIVE_FRAME

    @staticmethod
    def _sse_event(block):
        """``(type, data)`` of one SSE block (bytes or str), or ``(None, None)`` without data."""
        text = block.decode() if isinstance(block, (bytes, bytearray)) else block
        name, data = None, []
        for line in text.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if not data:
            return None, None
        payload = "\n".join(data)
        if payload == "[DONE]":
            return None, None
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("invalid stream event")
        kind = value.get("type") if isinstance(value.get("type"), str) else name
        return kind, value

    def terminal_sse_type(self, block):
        try:
            kind, _ = self._sse_event(block)
        except (ValueError, UnicodeDecodeError):
            return None
        return kind if kind in TERMINAL_EVENT_TYPES else None

    def terminal_allows_reuse(self, terminal_type):
        return terminal_type == "message_stop"

    @staticmethod
    def _merge_usage(target, usage):
        if isinstance(usage, dict):
            for key, value in usage.items():
                if value is not None:
                    target[key] = value

    def _failed(self, message, error):
        """``(complete, evidence, terminal)`` for an explicit provider error terminal."""
        terminal = dict(status="failed", error=error)
        evidence = dict(response_id=message.get("id") if message else None,
                        served_model=message.get("model") if message else None,
                        response_status="failed", stop_reason=None, usage=None, usage_known=False,
                        response_error=self.failed_response_error(dict(error=error)))
        return True, evidence, terminal

    @staticmethod
    def _message_outcome(stop_reason, tool_blocks):
        """``"completed"`` or ``"incomplete"`` for a finished message (design §4.2).

        Every tool_use block must be closed and its accumulated input must parse as a JSON
        object, whatever the stop reason (a cut tool call is never released). Then a stop reason
        in ``USABLE_STOP_REASONS`` completes the answer; one in ``TRUNCATION_STOP_REASONS``
        completes it only when no streamed tool_use block ended with blank arguments (cut before
        its first argument byte); any other stop reason is ``incomplete``.
        """
        if stop_reason not in USABLE_STOP_REASONS and stop_reason not in TRUNCATION_STOP_REASONS:
            return "incomplete"
        for block in tool_blocks:
            if not block["closed"]:
                return "incomplete"
            text = "".join(block["partial"])
            if not text.strip() and block.get("streamed") and stop_reason in TRUNCATION_STOP_REASONS:
                return "incomplete"
            try:
                value = json.loads(text) if text.strip() else block["input"]
            except ValueError:
                return "incomplete"
            if not isinstance(value, dict):
                return "incomplete"
        return "completed"

    def response_evidence(self, raw, route):
        """Validate a buffered Messages answer; tool_use blocks are tracked only to check that
        their input is complete, never to build output.

        Stream: ``message_start`` … ``message_stop`` (usage merged from ``message_start`` and the
        final ``message_delta``) or an ``error`` event. A JSON body: one ``message`` object or an
        ``error`` object. Evidence: ``response_id``, ``served_model``, ``response_status``
        (``completed``/``incomplete``/``failed``), ``stop_reason``, raw ``usage``, ``usage_known``,
        ``response_error`` for failures. Terminal: ``{id, model, status, stop_reason, usage,
        context_management, content}`` or ``{status: "failed", error}``; ``content`` lists the
        message's blocks in index order as ``{type}`` (``{type: "text", text}`` for text blocks, the
        only content host-side helpers read; the relay itself always forwards the raw bytes).
        """
        if self.is_event_stream(raw):
            message, stop_reason, usage, edits = None, None, {}, None
            blocks, terminal_seen = {}, False
            for block in raw.decode().replace("\r\n", "\n").split("\n\n"):
                kind, event = self._sse_event(block)
                if kind is None or kind == "ping":
                    continue
                if kind == "error":
                    error = event.get("error", event)
                    return self._failed(message, error if isinstance(error, dict) else {})
                if kind == "message_start":
                    if message is not None or not isinstance(event.get("message"), dict):
                        raise ValueError("invalid message_start")
                    message = event["message"]
                    self._merge_usage(usage, message.get("usage"))
                    stop_reason = message.get("stop_reason")
                    continue
                if message is None:
                    raise ValueError("stream event before message_start")
                if kind == "content_block_start":
                    start = event.get("content_block")
                    index = event.get("index")
                    if not isinstance(start, dict) or type(index) is not int or index in blocks:
                        raise ValueError("invalid content_block_start")
                    blocks[index] = dict(type=start.get("type"), closed=False, partial=[], input=start.get("input"),
                                         streamed=True,
                                         text=[start["text"]] if isinstance(start.get("text"), str) else [])
                elif kind == "content_block_delta":
                    target = blocks.get(event.get("index"))
                    delta = event.get("delta")
                    if target is None or target["closed"] or not isinstance(delta, dict):
                        raise ValueError("invalid content_block_delta")
                    if delta.get("type") == "input_json_delta":
                        target["partial"].append(str(delta.get("partial_json", "")))
                    elif delta.get("type") == "text_delta":
                        target["text"].append(str(delta.get("text", "")))
                elif kind == "content_block_stop":
                    target = blocks.get(event.get("index"))
                    if target is None:
                        raise ValueError("invalid content_block_stop")
                    target["closed"] = True
                elif kind == "message_delta":
                    delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                    if delta.get("stop_reason") is not None:
                        stop_reason = delta["stop_reason"]
                    self._merge_usage(usage, event.get("usage"))
                    if isinstance(event.get("context_management"), dict):
                        edits = event["context_management"]
                elif kind == "message_stop":
                    terminal_seen = True
                    break
            if not terminal_seen:
                raise EOFError("response terminal missing")
            tool_blocks = [block for block in blocks.values() if block["type"] == "tool_use"]
            content = [dict(type=blocks[index]["type"], text="".join(blocks[index]["text"]))
                       if blocks[index]["type"] == "text" else dict(type=blocks[index]["type"])
                       for index in sorted(blocks)]
        else:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("invalid response structure")
            if body.get("type") == "error":
                error = body.get("error")
                return self._failed(None, error if isinstance(error, dict) else {})
            if body.get("type") != "message" or not isinstance(body.get("content"), list):
                raise ValueError("invalid response structure")
            message, stop_reason, usage = body, body.get("stop_reason"), {}
            self._merge_usage(usage, body.get("usage"))
            edits = body.get("context_management") if isinstance(body.get("context_management"), dict) else None
            tool_blocks = [dict(closed=True, partial=[], input=item.get("input")) for item in body["content"]
                           if isinstance(item, dict) and item.get("type") == "tool_use"]
            content = [dict(type=item.get("type"), text=str(item.get("text", ""))) if item.get("type") == "text"
                       else dict(type=item.get("type") if isinstance(item, dict) else None) for item in body["content"]]
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise ValueError("invalid stop_reason")
        status = "not_completed" if stop_reason is None else self._message_outcome(stop_reason, tool_blocks)
        terminal = dict(id=message.get("id"), model=message.get("model"), status=status, stop_reason=stop_reason,
                        usage=usage or None, context_management=edits, content=content)
        evidence = dict(response_id=message.get("id"), served_model=message.get("model"), response_status=status,
                        stop_reason=stop_reason, usage=usage or None, usage_known=bool(usage))
        return stop_reason is not None, evidence, terminal

    def completion_outcome(self, complete, evidence):
        status = evidence.get("response_status")
        if status in ("failed", "incomplete"):
            return status
        return "usable" if complete and status == "completed" else "not_completed"

    def failed_response_error(self, body):
        """Sanitised Messages error: ``type`` only (Anthropic errors carry no ``code``); the
        message is read only to tell ``invalid_request_error`` classes apart and is never kept."""
        error = body.get("error") if isinstance(body, dict) else None
        detail = error if isinstance(error, dict) else {}
        kind, message = detail.get("type"), detail.get("message")
        valid = isinstance(error, dict) and isinstance(kind, str) and (message is None or isinstance(message, str))
        if not valid:
            category = "stream"
        elif kind == "invalid_request_error" and _PROMPT_TOO_LONG.search(message or ""):
            category = "context_window_exceeded"
        elif kind == "invalid_request_error" and _CREDIT_BALANCE.search(message or ""):
            category = "quota_exceeded"
        elif kind == "invalid_request_error" and _IMAGE_OR_SIZE.search(message or ""):
            category = "image_rejected"
        else:
            category = FAILURE_CLASSIFICATION.get(kind, "retryable")

        def safe(value):
            return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", value) else None
        code = detail.get("code")
        return dict(code=safe(code), type=safe(kind),
                    code_state="unknown" if code is None else "present" if safe(code) is not None else "redacted_or_invalid",
                    classification=category, retryable=category in RETRYABLE_CLASSIFICATIONS, retry_after_s=None)

    def classify_http_failure(self, status, headers, body, *, policy):
        """Status-based retry verdict of the pinned policy (408/429/5xx incl. 529 retryable,
        ``retry-after`` honoured), refined by the Messages error class: configuration, auth,
        context-window and credit errors never retry immediately; image/size rejections (and
        HTTP 413) retry, after the relay advanced the image-trim window."""
        from types import SimpleNamespace
        error = Exception("provider HTTP failure")
        error.status_code = status
        error.body = body
        error.response = SimpleNamespace(headers={k.lower(): v for k, v in headers.items()})
        failure = policy.classify_openai_error(error)
        if not isinstance(failure, policy.InfrastructureFailure):
            return failure
        provider_error = self.failed_response_error(body if isinstance(body, dict) else {})
        category = provider_error["classification"]
        if status == 413 or category == "image_rejected":
            failure.retryable = True
        elif category in ("configuration_error", "authentication_error", "permission_denied",
                          "context_window_exceeded", "quota_exceeded"):
            failure.retryable = False
        return failure

    def failed_payload(self, terminal, *, streaming):
        """claude_code: one ``event: error`` (``invalid_request_error``, neutral message) inside the
        open stream; no content of the failed answer. None for a non-streaming answer and for the
        deferred opencode profile (the relay then sends ``relay_error``)."""
        if self.client == "opencode" or not streaming:
            return None
        return _error_frame("invalid_request_error", RESPONSE_UNUSABLE_MESSAGE)

    def relay_error(self, *, streaming, evidence, manual_retry_allowed):
        """claude_code — streaming: one ``event: error`` frame; otherwise HTTP 403
        ``permission_error`` with ``x-should-retry: false``. opencode: HTTP 400
        ``invalid_request_error`` JSON with ``x-should-retry: false`` (deferred delivery never has
        an open stream). Evidence stays private (never in client bytes)."""
        if self.client == "opencode":
            return WireReply(400, headers=NO_RETRY_HEADERS,
                             body=_error_json("invalid_request_error", OPENCODE_REQUEST_FAILED_MESSAGE))
        if streaming:
            return WireReply(200, body=_error_frame("invalid_request_error", REQUEST_FAILED_MESSAGE))
        return WireReply(403, headers=NO_RETRY_HEADERS, body=_error_json("permission_error", REQUEST_FAILED_MESSAGE))

    def denial(self, kind):
        """claude_code: HTTP 403 ``permission_error`` + ``x-should-retry: false``; opencode: HTTP
        400 ``invalid_request_error`` + ``x-should-retry: false``. Either is one client attempt."""
        if self.client == "opencode":
            return WireReply(400, headers=NO_RETRY_HEADERS,
                             body=_error_json("invalid_request_error", OPENCODE_DENIAL_MESSAGES[kind]))
        return WireReply(403, headers=NO_RETRY_HEADERS, body=_error_json("permission_error", DENIAL_MESSAGES[kind]))

    def not_found(self):
        return WireReply(404, headers={"Content-Type": "application/json"},
                         body=_error_json("not_found_error", ROUTE_NOT_FOUND_MESSAGE))

    def normalize_usage(self, usage):
        """``input = input_tokens + cache_read_input_tokens + cache_creation_input_tokens``,
        ``cached = cache_read``, ``cache_write = cache_creation``, ``output = output_tokens`` (the
        final ``message_delta`` value), ``reasoning = None`` (not reported separately),
        ``total = input + output``. A cache field that is absent counts 0 in the input sum and
        stays None itself; a reported but invalid counter makes every sum it enters None."""
        usage = usage if isinstance(usage, dict) else {}
        base = counter(usage.get("input_tokens"))
        parts = [base]
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if key in usage and usage[key] is not None:
                parts.append(counter(usage[key]))
        total_input = None if any(part is None for part in parts) else sum(parts)
        output = counter(usage.get("output_tokens"))
        return {key: value for key, value in zip(TOKEN_FIELDS, (
            total_input, counter(usage.get("cache_read_input_tokens")),
            counter(usage.get("cache_creation_input_tokens")), output, None,
            None if total_input is None or output is None else total_input + output))}

    def context_edit_applied(self, terminal):
        edits = terminal.get("context_management") if isinstance(terminal, dict) else None
        return isinstance(edits, dict) and bool(edits.get("applied_edits"))

    def build_evidence_request(self, *, model, base_instructions, developer_instructions, input_items,
                               output_schema, effort):
        """Messages evidence request (§9.3): ``system`` = one text block of base + blank line +
        developer instructions; one user message of ``text`` / base64 ``image`` blocks;
        ``output_config = {"format": {"type": "json_schema", "schema": S}, "effort": effort}`` (native
        structured output, not forced tool use); ``max_tokens`` 16000; ``stream: true``; no tools,
        no ``tool_choice``, no ``thinking`` (the model's default) and no sampling parameters. Extra
        header ``anthropic-version: 2023-06-01``."""
        content = []
        for part in evidence_parts(input_items):
            content.append(dict(type="text", text=part[1]) if part[0] == "text" else
                           dict(type="image", source=dict(type="base64", media_type=part[1], data=part[2])))
        body = dict(model=model, max_tokens=EVIDENCE_MAX_TOKENS,
                    system=[dict(type="text", text=base_instructions + "\n\n" + developer_instructions)],
                    messages=[dict(role="user", content=content)],
                    output_config=dict(format=dict(type="json_schema", schema=output_schema), effort=effort),
                    stream=True)
        return body, {"anthropic-version": EVIDENCE_API_VERSION}

    def evidence_answer(self, terminal):
        """The answer of a completed message with ``stop_reason == "end_turn"`` (``refusal``,
        ``max_tokens``, ``pause_turn`` and any other stop are unverified): the concatenated ``text``
        blocks; ``thinking`` / ``redacted_thinking`` blocks are ignored; any ``tool_use`` or
        server-tool block of a completed message is tool activity, whatever its stop reason (a
        ``tool_use`` stop included, as ``CodexEvidenceAgent`` reports any tool item); no text block at
        all is zero answers."""
        if not isinstance(terminal, dict) or terminal.get("status") != "completed":
            raise ValueError(EVIDENCE_UNVERIFIED)
        texts = []
        for block in terminal.get("content") or []:
            kind = block.get("type") if isinstance(block, dict) else None
            if kind in ("thinking", "redacted_thinking"):
                continue
            if kind != "text":
                raise ValueError(EVIDENCE_TOOL_ACTIVITY)
            texts.append(str(block.get("text", "")))
        if terminal.get("stop_reason") != EVIDENCE_USABLE_STOP:
            raise ValueError(EVIDENCE_UNVERIFIED)
        if not texts:
            raise ValueError(EVIDENCE_ONE_ANSWER)
        return "".join(texts)


# ============================================================================ egress walker
def _neutralized_system(system):
    """``system`` (string or block list) with OpenCode's instruction-file header lines neutralised."""
    if isinstance(system, str):
        return neutralize_instruction_header(system)
    if isinstance(system, list):
        return [dict(block, text=neutralize_instruction_header(block["text"]))
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
                else block for block in system]
    return system


def present_messages_request(request, presentation, *, registered_tools, tool_name_prefix, client,
                             on_unknown_part=None):
    """Egress information-boundary re-scrub of one Messages body (generation or count_tokens).

    request: decoded body (dict); never mutated. presentation: the session's
    ``ActorPresentation``. registered_tools: names of the Auriga tools the agent may call.
    tool_name_prefix: prefix the client puts before Auriga tool names (Claude Code
    ``"mcp__auriga__"``). client: profile name; ``claude_code`` enables the image-source rewrite,
    ``opencode`` the instruction-header rewrite (``Instructions from: <path>`` lines of the
    ``system`` text become ``Instructions:``).
    on_unknown_part: optional ``(type_name)`` callback for content blocks of unknown type.

    Scrubbed: ``system`` text; every message ``content`` string and ``text`` block (also
    ``role:"system"`` messages); ``tool_use.input`` (``present``); ``tool_result`` string content
    and nested blocks; ``document`` ``title``/``context``/text ``source.data``/``source.content``;
    ``search_result`` ``title``/``source``/content; client tools' ``description`` (``present_text``),
    ``input_schema`` (``present_data``) and ``input_examples`` (``present``);
    ``output_config.format.schema`` and the legacy top-level ``output_format.schema``
    (``present_data``). Blocks of unknown type, and ``thinking``/``redacted_thinking`` blocks outside
    assistant messages, go through a fail-closed generic walker: every string at any depth except
    the ``GENERIC_PROTECTED_KEYS`` values is scrubbed. Claude Code profile: every private image-source
    marker ``[Image: source: <path>]`` in any scrubbed prose (whole block, merged text, trailing
    newline, outside a tool_result) becomes ``[Image]``; image bytes are never touched.
    Untouched: ids and pairing fields, ``type``/``role``/``name``, media bytes and
    ``media_type``, ``cache_control``, ``citations``, ``thinking``/``redacted_thinking`` blocks of
    assistant messages (signed), ``metadata``, ``thinking``, ``tool_choice``, ``context_management``,
    ``max_tokens``, ``model``, ``stream``, ``eager_input_streaming`` and every other parameter.
    Raises ``ValueError``: ``sandbox_remote_model_tool_disabled`` for non-client tools, non-empty
    ``mcp_servers``, a ``container`` or server-tool blocks; ``sandbox_remote_media_url_disabled``
    for non-base64 images and url/file documents; ``native_discovered_tool_not_registered`` for a
    ``tool_reference`` naming anything but a registered tool. Deterministic and idempotent.
    """
    if not isinstance(request, dict):
        raise ValueError("model_request_not_object")
    registered = frozenset(registered_tools)
    if client == "opencode" and "system" in request:
        request = dict(request, system=_neutralized_system(request["system"]))
    rewrite_image_source = client == "claude_code"
    if request.get("mcp_servers") or request.get("container") is not None:
        raise ValueError("sandbox_remote_model_tool_disabled")

    def prose(value):
        """Scrub one prose string; on the claude_code profile every private image-source marker in it
        becomes ``[Image]`` first (text only, never image bytes)."""
        if rewrite_image_source:
            value = IMAGE_SOURCE_MARKER.sub(IMAGE_SOURCE_REPLACEMENT, value)
        return presentation.present_text(value)

    def text(value):
        return prose(value) if isinstance(value, str) else deepcopy(value)

    def unknown(kind):
        if on_unknown_part is not None:
            on_unknown_part(str(kind)[:120])

    def generic(value):
        """Key-based walker for a block of unknown type (fail closed): every string at any depth is
        scrubbed as prose except the values of ``GENERIC_PROTECTED_KEYS`` and ``type``; list items
        that are typed blocks are walked as blocks (so nested images are still checked)."""
        if isinstance(value, list):
            return [block(item) if isinstance(item, dict) and "type" in item else generic(item) for item in value]
        if not isinstance(value, dict):
            return text(value)
        result = {}
        for key, item in value.items():
            if key in GENERIC_PROTECTED_KEYS or key == "type":
                result[key] = deepcopy(item)
            elif isinstance(item, (str, dict, list)):
                result[key] = generic(item)
            else:
                result[key] = deepcopy(item)
        return result

    def content(value, *, assistant=False):
        if isinstance(value, str):
            return prose(value)
        if isinstance(value, list):
            return [block(item, assistant=assistant) if isinstance(item, dict) else text(item) for item in value]
        return deepcopy(value)

    def block(value, *, assistant=False):
        kind = value.get("type")
        if kind in SERVER_TOOL_BLOCK_TYPES or (isinstance(kind, str) and kind != "tool_result"
                                               and kind.endswith("_tool_result")):
            raise ValueError("sandbox_remote_model_tool_disabled")
        result = deepcopy(value)
        if kind == "text":
            if "text" in value:
                result["text"] = text(value["text"])
        elif kind == "image":
            source = value.get("source")
            if not isinstance(source, dict) or source.get("type") != "base64":
                raise ValueError("sandbox_remote_media_url_disabled")
        elif kind == "document":
            source = value.get("source")
            stype = source.get("type") if isinstance(source, dict) else None
            if stype not in ("base64", "text", "content"):
                raise ValueError("sandbox_remote_media_url_disabled")
            for key in ("title", "context"):
                if key in value:
                    result[key] = text(value[key])
            if stype == "text" and "data" in source:
                result["source"]["data"] = text(source["data"])
            elif stype == "content" and "content" in source:
                result["source"]["content"] = content(source["content"])
        elif kind == "search_result":
            for key in ("title", "source"):
                if key in value:
                    result[key] = text(value[key])
            if "content" in value:
                result["content"] = content(value["content"])
        elif kind in SIGNED_BLOCK_TYPES:
            if not assistant:
                # Only an assistant's own signed output is protocol data; elsewhere it is text.
                return generic(value)
        elif kind == "tool_use":
            if "input" in value:
                result["input"] = (presentation.present(value["input"])
                                   if isinstance(value["input"], (dict, list)) else text(value["input"]))
        elif kind == "tool_result":
            if "content" in value:
                result["content"] = content(value["content"])
        elif kind == "tool_reference":
            name = value.get("tool_name")
            bare = name[len(tool_name_prefix):] if isinstance(name, str) and tool_name_prefix and name.startswith(tool_name_prefix) else name
            if not isinstance(name, str) or (tool_name_prefix and not name.startswith(tool_name_prefix)) or bare not in registered:
                raise ValueError("native_discovered_tool_not_registered")
        else:
            unknown(kind)
            return generic(value)
        return result

    walked = ("system", "messages", "tools", "output_config", "output_format")
    result = {key: (None if key in walked else deepcopy(value)) for key, value in request.items()}
    system = request.get("system")
    if isinstance(system, str):
        result["system"] = prose(system)
    elif "system" in request:
        result["system"] = content(system) if isinstance(system, list) else deepcopy(system)
    messages = request.get("messages")
    if isinstance(messages, list):
        result["messages"] = [{key: content(value, assistant=message.get("role") == "assistant") if key == "content"
                               else deepcopy(value) for key, value in message.items()}
                              if isinstance(message, dict) else deepcopy(message)
                              for message in messages]
    elif "messages" in request:
        result["messages"] = deepcopy(messages)
    if "output_config" in request:
        result["output_config"] = deepcopy(request["output_config"])
    if "tools" in request and not isinstance(request["tools"], list):
        result["tools"] = deepcopy(request["tools"])
    tools = request.get("tools")
    if isinstance(tools, list):
        projected = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") not in CLIENT_TOOL_TYPES:
                raise ValueError("sandbox_remote_model_tool_disabled")
            item = deepcopy(tool)
            if isinstance(tool.get("description"), str):
                item["description"] = presentation.present_text(tool["description"])
            if isinstance(tool.get("input_schema"), dict):
                item["input_schema"] = presentation.present_data(tool["input_schema"])
            if isinstance(tool.get("input_examples"), list):
                # Example tool inputs are structured arguments, scrubbed like tool_use.input.
                item["input_examples"] = presentation.present(tool["input_examples"])
            projected.append(item)
        result["tools"] = projected
    output_config = request.get("output_config")
    output_format = output_config.get("format") if isinstance(output_config, dict) else None
    if isinstance(output_format, dict) and isinstance(output_format.get("schema"), dict):
        result["output_config"]["format"]["schema"] = presentation.present_data(output_format["schema"])
    if "output_format" in request:
        # Legacy top-level structured-output parameter (superseded by output_config.format).
        result["output_format"] = deepcopy(request["output_format"])
        legacy = request["output_format"]
        if isinstance(legacy, dict) and isinstance(legacy.get("schema"), dict):
            result["output_format"]["schema"] = presentation.present_data(legacy["schema"])
    return result
