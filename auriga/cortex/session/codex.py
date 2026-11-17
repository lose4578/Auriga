"""Pinned Python SDK over a stdio bridge to the run's shared Unix app-server."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
import signal
import select
import stat
import tempfile
from urllib.parse import urlsplit
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from importlib.metadata import version

from codex_cli_bin import bundled_codex_path
from openai_codex.client import CodexClient, CodexConfig
from openai_codex.errors import JsonRpcError
from jsonschema import validate as validate_json

from ...core.contracts import canonical, digest, new_id, utc_now
from .usage import RelayUsageTotal, account_usage, add_served_model, responses_counters
from .model_proxy import ModelRequestProxy, DEFAULT_MAX_RETRIES, RETRY_POLICY
from .account_control import AccountControlRelay, ACCOUNT_BASE_URL
from ...core.presentation.image_reuse import IMAGE_REUSE_POLICY
from ...core.storage.images import store_image_settings
from .pidfd import pidfd_open, pidfd_send_signal
from .actor_source import snapshot_actor_source
from .agent_sandbox import ACTOR_ISOLATION  # shared by every sandboxed backend (design R4.4)
from ..context.task_planning import decision_reminder_text
from ..agents.dispatch import OrderedBarrier, is_parallel_call
from ..agents.base import AgentCapabilities, AgentSession, model_effort_choices

MODEL_COMPLETION_COUNTING = "delivered_usable_actor_helper_compact_responses_v2"
MODEL_PROXY_PROVIDER = "auriga_model_proxy"
DYNAMIC_TOOL_NAMESPACE = "auriga"
CODEX_SDK_VERSION = version("openai-codex")
CODEX_CLI_VERSION = version("openai-codex-cli-bin")
_CODEX_PROTOCOL = json.loads(Path(__file__).with_name('codex_responses.json').read_text())
CODEX_PROTOCOL_VERSION = _CODEX_PROTOCOL['build']
NATIVE_RESPONSE_SCHEMAS = _CODEX_PROTOCOL['responses']


def require_codex_version():
    if CODEX_SDK_VERSION != CODEX_PROTOCOL_VERSION or CODEX_CLI_VERSION != CODEX_PROTOCOL_VERSION:
        raise RuntimeError(f'codex_version_mismatch: expected SDK and CLI {CODEX_PROTOCOL_VERSION}; '
            f'installed SDK={CODEX_SDK_VERSION}, CLI={CODEX_CLI_VERSION}. '
            'Install the repository requirements.lock.txt in the active Harness environment before starting or resuming.')


def write_direct_model_catalog(path):
    """Keep the pinned native metadata; select typed dynamic tool delivery."""
    catalog = json.loads(Path(__file__).with_name('codex_models.json').read_text())
    for model in catalog['models']:
        # Model metadata overrides feature flags, including code_mode_host=false.
        model['tool_mode'] = 'direct'
    path = Path(path)
    path.write_text(json.dumps(catalog, ensure_ascii=False) + '\n')
    return path


def _native_dynamic_tools(tools):
    """Keep registry contracts flat; native discovery requires a namespace."""
    eager, deferred = [], []
    for tool in tools:
        (deferred if tool.get("deferLoading", False) else eager).append(deepcopy(tool))
    if deferred:
        eager.append(dict(type="namespace", name=DYNAMIC_TOOL_NAMESPACE,
            description="Auriga tools for robot observations, geometry, execution evidence and task support.",
            tools=deferred))
    return eager


def _toml_value(value):
    """Encode native extension tables without changing their keys or values."""
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(key) + ' = ' + _toml_value(item) for key, item in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(_toml_value(item) for item in value) + ']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _chatgpt_login(source):
    auth_path = source / "auth.json"
    auth = json.loads(auth_path.read_text()) if auth_path.exists() else {}
    return auth.get("auth_mode") in ("chatgpt", "chatgptAuthTokens") or (
        not auth.get("auth_mode") and bool(auth.get("tokens")) and not auth.get("OPENAI_API_KEY"))


def _model_provider(config, source):
    """Resolve the upstream before replacing it with the private Responses relay."""
    name = config.get("model_provider", "openai")
    provider = dict(config.get("model_providers", {}).get(name) or {})
    if name == "openai":
        # Built-in IDs cannot be redefined in Codex 0.159.2. Preserve its auth
        # behavior when moving it to our own provider ID.
        provider.setdefault("requires_openai_auth", True)
    explicit = provider.get("base_url")
    if name == "openai":
        explicit = explicit or config.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
    if explicit:
        return name, provider, explicit
    if provider.get("requires_openai_auth") and _chatgpt_login(source):
        base = config.get("chatgpt_base_url", "https://chatgpt.com/backend-api").rstrip("/")
        upstream = base if base.endswith("/codex") else base + "/codex"
    else:
        upstream = "https://api.openai.com/v1"
    return name, provider, upstream


def _upstream_auth(provider, source):
    """Credentials stay in the host relay, never in the actor's home or env."""
    path = source / "auth.json"
    auth = json.loads(path.read_text()) if path.exists() else {}
    headers = dict(provider.get("http_headers") or {})
    for header, key in (provider.get("env_http_headers") or {}).items():
        if key in os.environ:
            headers[header] = os.environ[key]
    if any(key.lower() not in {"authorization", "chatgpt-account-id"} for key in headers):
        raise ValueError("sandbox_provider_requires_supported_auth_headers")
    key = provider.get("env_key")
    token = os.environ.get(key) if key else provider.get("experimental_bearer_token")
    if provider.get("requires_openai_auth"):
        if _chatgpt_login(source):
            tokens = auth.get("tokens") or {}
            token = tokens.get("access_token")
            if tokens.get("account_id"):
                headers["ChatGPT-Account-Id"] = tokens["account_id"]
        else:
            token = auth.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if token:
        headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
        headers["Authorization"] = "Bearer " + token
    return headers


def normalize_token_usage(value):
    """Preserve native last/total snapshots; absent counters are unknown, not zero."""
    value = value if isinstance(value, dict) else {}
    fields = {"input_tokens": "inputTokens", "cached_input_tokens": "cachedInputTokens",
              "cache_write_input_tokens": "cacheWriteInputTokens", "output_tokens": "outputTokens",
              "reasoning_output_tokens": "reasoningOutputTokens", "total_tokens": "totalTokens"}

    def counter(number):
        return number if type(number) is int and number >= 0 else None

    return {**{kind: {target: counter(value[kind].get(native)) for target, native in fields.items()}
               if isinstance(value.get(kind), dict) else None for kind in ("total", "last")},
            "model_context_window": counter(value.get("modelContextWindow"))}


class RecordedClient(CodexClient):
    # The pinned SDK owns RPC routing. These hooks retain raw messages.
    def __init__(self, config, store, tool_handler, notification_handler, request_handler=None, disconnect_handler=None,
                 *, route_notifications=True, parallel_safe=None):
        super().__init__(config, approval_handler=self._approval)
        self.store, self.tool_handler, self.notification_handler = store, tool_handler, notification_handler
        self.record_send_lock = threading.Lock()
        self.request_handler, self.disconnect_handler = request_handler, disconnect_handler
        self.route_notifications = route_notifications
        self.server_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="codex-tools")
        # (tool, arguments) -> bool from the registry contract (ToolRegistry.parallel_safe);
        # None orders every tool call.
        self.parallel_safe = parallel_safe
        self._tool_barrier = OrderedBarrier()

    def _submit_request(self, message):
        """非工具 RPC 立即并发执行；工具调用按注册表契约允许并发安全调用并发，其余按到达顺序栅栏串行。"""
        if message.get("method") != "item/tool/call":
            return self.server_pool.submit(self._reply_request, message)
        parallel = is_parallel_call(self.parallel_safe, message.get("params"))
        return self._tool_barrier.schedule(self.server_pool, parallel, self._reply_request, message)

    def _reader_loop(self):
        # The pinned SDK calls tools on its RPC reader. Dispatching them separately
        # keeps interrupts, native replies, and status RPCs responsive.
        try:
            while True:
                message = self._read_message()
                if "method" in message and "id" in message:
                    self._submit_request(message)
                elif "method" in message:
                    if self.route_notifications and isinstance(message["method"], str):
                        self._router.route_notification(self._coerce_notification(message["method"], message.get("params")))
                else:
                    self._router.route_response(message)
        except BaseException as error:
            self._router.fail_all(error)
            if self.disconnect_handler:
                self.disconnect_handler(error)

    def _reply_request(self, message):
        try:
            response = self.request_handler(message) if self.request_handler and message["method"] != "item/tool/call" else self._handle_server_request(message)
            if self.request_handler and message["method"] != "item/tool/call" and response is not None:
                response = self.request_handler(message, before_send=True)
            if response is None:  # The server already resolved this RPC.
                return
            self._write_message(dict(id=message["id"], result=response))
            if self.request_handler and message["method"] != "item/tool/call":
                self.request_handler(message, sent=True)
        except Exception as error:
            if message.get("method") == "item/tool/call":
                # Observation baselines are prepared before this RPC write.
                # Delivery is now uncertain, so later feedback must restore
                # full views even if the same app-server session survives.
                self.store.set_state("runtime", "image_context_epoch", dict(
                    source="tool_reply_delivery_uncertain", rpc_id=message.get("id"),
                    marker=new_id("image-delivery-failure")))
            self.store.event("codex.request_response_unknown", {"rpc_id": message["id"], "reason": str(error)})
            if self.request_handler and message["method"] != "item/tool/call":
                self.request_handler(message, failed=str(error))

    def _read_message(self):
        message = super()._read_message()
        self.store.raw_codex("received", message)
        if "method" in message:
            self.notification_handler(message)
        return message

    def _write_message(self, message):
        with self.record_send_lock:
            self.store.raw_codex("sent", message)
            super()._write_message(message)

    def _approval(self, method, params):
        if method == "item/tool/call":
            try:
                result = self.tool_handler(params)
            except Exception as error:
                from ...core.tool_errors import error_response
                result = error_response(error, tool=params.get("tool"), arguments=params.get("arguments"))
            from ...core.presentation.actor_presentation import ActorPresentation
            # The final native reply applies the actor's recursive metadata policy.
            return ActorPresentation(self.store).present(result)
        if method in NATIVE_RESPONSE_SCHEMAS:
            response = CodexSession._approve(method, params or {})
            validate_json(response, NATIVE_RESPONSE_SCHEMAS[method])
            return response
        if method.endswith("requestApproval"):
            return CodexSession._decline(method)
        # No second input owner. Requests remain visible as unavailable in the log.
        return {}


class CodexSession(AgentSession):
    """Codex backend: pinned Python SDK over a stdio bridge to the run's private app-server (design §2.4)."""
    backend_kind = "codex"
    counting = MODEL_COMPLETION_COUNTING
    capabilities = AgentCapabilities(steer=True, native_requests=True, terminal_attach=True,
                                     native_terminal_replay=True, fork=True, deferred_tools=True, account_relay=True,
                                     native_evidence_agent=True, sandbox=True, service_tier=True,
                                     native_memories=True, live_effort=True, compaction_signal="native+relay")
    # Codex accepts every model.effort choice of the config schema.
    effort_choices = model_effort_choices()
    # Wired by Workbench after construction: (tool, arguments) -> bool, normally
    # ToolRegistry.parallel_safe. None orders every native tool call.
    tool_parallel_safe = None

    def __init__(self, run_dir, store, tool_handler, model=None, effort=None, price_card=None, *, config=None,
                 task_planning=None):
        from ...core.config import Config
        self.config = config if config is not None else Config.load()
        self.sandbox_enabled = self.config.features.sandbox_enabled
        self.isolation = ACTOR_ISOLATION if self.sandbox_enabled else "disabled"
        self.root = Path(run_dir).resolve()
        self.store, self.tool_handler = store, tool_handler
        self.task_planning = task_planning
        from ...core.presentation.actor_presentation import ActorPresentation
        self.presentation = ActorPresentation(store)
        self.model = self.config.model.name if model is None else model
        self.active_turn_model = self.model
        self.effort = self.config.model.effort if effort is None else effort
        self.provider = None
        self.price_card = price_card
        self.image_settings = store_image_settings(store, self.config.images.to_dict())
        self.actor_dir = self.root / "actor"
        self.private = self.root / "private"
        self.codex_home = self.private / "codex-home"
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        self.ipc_runtime_dir = Path(runtime_dir) if runtime_dir else None
        if self.ipc_runtime_dir is not None and not self.ipc_runtime_dir.is_absolute():
            raise ValueError("codex_runtime_directory_must_be_absolute")
        self.ipc_base = (self.ipc_runtime_dir / "auriga" if self.ipc_runtime_dir is not None
                         else Path(tempfile.gettempdir()) / f"auriga-ipc-{os.getuid()}")
        self.ipc = self.ipc_base / hashlib.sha256(os.fsencode(self.root)).hexdigest()[:24]
        self.socket_path = self.ipc / "codex.sock"
        if len(os.fsencode(self.ipc / "sandbox-owner.sock")) > 107:
            raise ValueError("codex_ipc_path_too_long: set XDG_RUNTIME_DIR to a shorter private runtime directory")
        self.ipc_prepared = False
        self.process = self.client = None
        self.thread_id = self.turn_id = None
        self.turn_done = threading.Event()
        self.turn_result = None
        self.state = "not_started"
        self.last_activity_at = time.monotonic()
        self.pending = {}
        self.pending_lock = threading.RLock()
        self.native_connection_id = new_id("native-connection")
        self.resolved_requests = set()
        self.closing = False
        self.on_disconnect = None
        self.model_proxy = None
        self.transport_recovery_gate = None
        self.account_control = None
        self.sandbox = None
        self.command_prefix = []
        self.model_limit_reached = threading.Event()
        self.projection_lock = threading.RLock()
        self.live_revision = self.turn_revision = 0
        self.item_revisions = {}
        self.response_schemas = deepcopy(NATIVE_RESPONSE_SCHEMAS)
        self.deferred_tools = frozenset()
        features = self.config.features
        self.lean_projection = getattr(features, "lean_projection", False)
        self.lean_projection_k = getattr(features, "lean_projection_k", 1)
        self.lean_projection_block = getattr(features, "lean_projection_block", 8)
        # Block-mode decision reminder slots (features.lean_projection), per actor thread.
        self.lean_reminder_slots = {}
        self.lean_reminder_thread = None
        self.lean_reminder_lock = threading.Lock()
        self.request_dump_limit = getattr(features, "lean_request_dump", 0)
        self.request_dump_count = 0
        self.request_dump_lock = threading.Lock()

    def _task_decision_context(self, request, identity):
        """Carry advisory actor planning state into context compaction or block projection."""
        block_reminders = (
            identity.get("endpoint") == "/responses"
            and bool(getattr(self, "lean_projection", False))
            and bool(getattr(self, "lean_projection_block", 0))
        )
        if (self.task_planning is None or not self.thread_id
                or identity.get("native_thread_id") != self.thread_id
                or identity.get("source") != "actor"
                or (identity.get("endpoint") != "/responses/compact" and not block_reminders)):
            return request
        context = self.task_planning.decision_context(self.store.identity)
        if context is None and not self.task_planning.required:
            return request
        # Carry the latest plan/attention/ledger snapshot only when compacting
        # an over-limit context (or at fixed block slots when lean_projection
        # block mode is explicitly enabled), so default /responses calls remain
        # pure prefix extensions and preserve the upstream prompt cache.
        reminder = decision_reminder_text(context)
        result = deepcopy(request)
        items = result.get("input", [])
        if isinstance(items, str):
            items = [dict(role="user", content=items)]
        if not isinstance(items, list):
            raise ValueError("invalid_model_request_input")
        def is_decision_reminder(item):
            if not isinstance(item, dict) or item.get("role") != "assistant":
                return False
            if item.get("type") != "message" or not isinstance(item.get("content"), list):
                return False
            parts = item["content"]
            if len(parts) != 1 or not isinstance(parts[0], dict):
                return False
            text = parts[0].get("text")
            if parts[0].get("type") != "output_text" or not isinstance(text, str):
                return False
            # Remove both current and pre-v7 Harness markers when resuming an
            # archived request, while preserving ordinary assistant messages.
            return any(text.startswith(prefix) for prefix in (
                "Current task decision boundary\nThe saved plan and attention",
                "Current task decision context\nThe saved plan and attention"))
        # Keep actor-authored judgments out of the developer instruction layer.
        items = [item for item in items if not is_decision_reminder(item)]
        message = dict(type="message", role="assistant", content=[dict(type="output_text", text=reminder)])
        if block_reminders:
            # Block projection: same message, fixed slots that are never the last
            # item, so each request in a block extends the previous one and ends
            # with a tool output (see auriga/core/presentation/lean_projection.py).
            from ...core.presentation.lean_projection import place_block_reminders
            with self.lean_reminder_lock:
                if self.lean_reminder_thread != self.thread_id:
                    self.lean_reminder_thread, self.lean_reminder_slots = self.thread_id, {}
                items = place_block_reminders(items, message, self.lean_projection_block,
                                              self.lean_reminder_slots, planned=context is not None)
        else:
            items.append(message)
        result["input"] = items
        return result

    def _present_model_request(self, request):
        """Project readable Responses history, including resumed/compact input.

        Native item/call IDs, encrypted reasoning and media are protocol data.
        Only text and tool schemas share the actor's presentation policy
        (``wire.openai_responses.present_responses_request``).
        """
        from .wire.openai_responses import present_responses_request
        result = present_responses_request(request, self.presentation, deferred_tools=self.deferred_tools,
                                           namespace=DYNAMIC_TOOL_NAMESPACE)
        if getattr(self, "lean_projection", False) and isinstance(result.get("input"), list):
            from ...core.presentation.lean_projection import project_request
            result = project_request(result, self.lean_projection_k, getattr(self, "lean_projection_block", 0))
        if getattr(self, "request_dump_limit", 0) and isinstance(request.get("input"), list):
            self._dump_request(request, result)
        return result

    def _dump_request(self, original, forwarded):
        """Diagnostics (features.lean_request_dump): bounded private body copies."""
        with self.request_dump_lock:
            if self.request_dump_count >= self.request_dump_limit:
                return
            self.request_dump_count += 1
            number = self.request_dump_count
        def strip(value):
            if isinstance(value, list):
                return [strip(item) for item in value]
            if isinstance(value, dict):
                return {key: ("<data-url sha256=" + hashlib.sha256(item.encode()).hexdigest()[:16]
                              + " chars=" + str(len(item)) + ">"
                              if isinstance(item, str) and item.startswith("data:") else strip(item))
                        for key, item in value.items()}
            return value
        try:
            body = json.dumps(forwarded, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            folder = self.private / "request-dump"
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            (folder / f"{number:03d}.json").write_text(json.dumps(dict(
                index=number, forwarded_body_bytes=len(body), forwarded_body_sha256=hashlib.sha256(body).hexdigest(),
                original=strip(original), forwarded=strip(forwarded)), ensure_ascii=False))
        except Exception as error:
            self.store.event("codex.request_dump_failed", {"error_type": type(error).__name__})

    def _request_id(self, message):
        return "native-" + digest([self.native_connection_id, message["id"]])[:24]

    def _server_request(self, message, sent=False, failed=None, before_send=False):
        identifier = self._request_id(message)
        if before_send:
            with self.pending_lock:
                entry = self.pending.get(identifier)
                if identifier in self.resolved_requests or self.closing or self.state == "disconnected" or (entry and entry.get("expired")):
                    return None
                if entry:
                    return entry["response"]
                return self._decline(message["method"]) if message["method"] in self.response_schemas else {}
        if sent or failed:
            with self.pending_lock:
                entry = self.pending.get(identifier)
                if entry:
                    info = self.store.read_state("native_request", identifier)
                    info.update(status="expired" if entry.get("expired") else "unknown" if failed else "resolved",
                                response_digest=digest(entry["response"]), response_sent_at=utc_now(), response_error=failed)
                    self.store.set_state("native_request", identifier, info)
                    self.store.event("codex.request_resolved", info, {k:info[k] for k in ("thread_id", "turn_id", "item_id") if info.get(k)})
                    entry["sent"].set()
            return
        method, params = message["method"], message.get("params") or {}
        if method not in self.response_schemas:
            self.store.event("codex.unsupported_request", {"rpc_id": message["id"], "method": method})
            return {}
        if params.get("threadId") != self.thread_id or (self.turn_id and params.get("turnId") != self.turn_id):
            self.store.event("codex.request_rejected", {"rpc_id":message["id"], "reason":"request_identity_mismatch"})
            return self._decline(method)
        response = self._approve(method, params)
        validate_json(response, self.response_schemas[method])
        info = dict(request_id=identifier, rpc_id=message["id"], method=method, params=params, status="responding",
                    thread_id=params.get("threadId"), turn_id=params.get("turnId"), item_id=params.get("itemId"),
                    created_at=utc_now(), response_policy="unattended", response_digest=digest(response),
                    reason="unattended_no_human_input" if method == "item/tool/requestUserInput" else "unattended_permission_granted")
        entry = dict(event=threading.Event(), sent=threading.Event(), method=method, params=params,
                     response=response, automatic=True)
        with self.pending_lock:
            self.pending[identifier] = entry
            if self.closing or self.state == "disconnected" or identifier in self.resolved_requests:
                entry["expired"] = True
                info.update(status="expired", reason="server_request_resolved" if identifier in self.resolved_requests
                            else "session_closed_or_disconnected")
            self.store.set_state("native_request", identifier, info)
            self.store.event("codex.request_expired" if entry.get("expired") else "codex.request_automatic", info)
            entry["event"].set()
            return None if entry.get("expired") else response

    def _resolve_request(self, params):
        """End only this connection's exact RPC, including before worker admission."""
        rpc_id = params.get("requestId")
        if params.get("threadId") != self.thread_id or type(rpc_id) not in (str, int):
            return
        identifier = self._request_id({"id": rpc_id})
        with self.pending_lock:
            self.resolved_requests.add(identifier)
            entry = self.pending.get(identifier)
            if entry:
                info = self.store.read_state("native_request", identifier)
                if info["status"] in ("pending", "responding"):
                    entry["expired"] = True
                    info.update(status="expired", reason="server_request_resolved")
                    self.store.set_state("native_request", identifier, info)
                    self.store.event("codex.request_expired", info)
                entry["event"].set()
                entry["sent"].set()

    @staticmethod
    def _decline(method):
        if method == "item/tool/requestUserInput": return {"answers": {}}
        if method == "item/permissions/requestApproval": return {"permissions": {}, "scope": "turn"}
        return {"decision": "decline"}

    @staticmethod
    def _approve(method, params):
        if method == "item/permissions/requestApproval":
            return {"permissions": deepcopy(params.get("permissions", {})), "scope": "session", "strictAutoReview": False}
        if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
            return {"decision": "acceptForSession"}
        return CodexSession._decline(method)

    def reply(self, identifier, response):
        with self.pending_lock:
            entry = self.pending.get(identifier)
            info = self.store.read_state("native_request", identifier)
            if not entry or not info or entry.get("automatic") or info["status"] in ("expired", "unknown", "resolved"):
                raise ValueError("native_request_expired_or_not_owned")
            if entry["response"] is not None:
                if digest(entry["response"]) != digest(response):
                    raise ValueError("native_response_content_conflict")
                return info
            if not isinstance(response, dict): raise ValueError("native_response_requires_object")
            try:
                validate_json(response, self.response_schemas[entry["method"]])
            except Exception as error:
                raise ValueError("invalid_native_response:" + str(error).splitlines()[0]) from error
            if entry["method"] == "item/tool/requestUserInput":
                question_ids = {question["id"] for question in entry["params"].get("questions", [])}
                if set(response["answers"]) - question_ids:
                    raise ValueError("unknown_native_question_id")
            entry["response"] = response
            info.update(status="responding", response_digest=digest(response))
            self.store.set_state("native_request", identifier, info)
            entry["event"].set()
        entry["sent"].wait(5)
        return self.store.read_state("native_request", identifier)

    def expire_requests(self, reason):
        with self.pending_lock:
            for info in self.store.states("native_request"):
                if info["request_id"] not in self.pending and info["status"] in ("pending", "responding"):
                    info.update(status="expired",reason=reason)
                    self.store.set_state("native_request",info["request_id"],info)
                    self.store.event("codex.request_expired",info)
            for identifier, entry in self.pending.items():
                info = self.store.read_state("native_request", identifier)
                if info["status"] in ("pending", "responding"):
                    entry.update(expired=True, response=self._decline(entry["method"]))
                    info.update(status="expired", reason=reason)
                    self.store.set_state("native_request", identifier, info)
                    self.store.event("codex.request_expired", info)
                    entry["event"].set()
                    entry["sent"].set()

    def _disconnected(self, error):
        if not self.closing:
            self.state = "disconnected"
            if not self.turn_done.is_set():
                self.turn_result = {"status": "failed", "error": {"message": "upstream_connection_lost: " + str(error)}}
            try:
                self.store.event("codex.connection_lost", {"reason": str(error), "history_complete": False})
                self.expire_requests("upstream_connection_lost")
            finally:
                try:
                    if self.on_disconnect: self.on_disconnect()
                finally:
                    self.turn_done.set()

    def _prepare_ipc(self):
        # Linux sockaddr_un paths are bounded independently of the experiment archive path.
        if self.ipc_runtime_dir is not None:
            info = self.ipc_runtime_dir.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("unsafe_codex_runtime_directory")
        for path in (self.ipc_base, self.ipc):
            path.mkdir(mode=0o700, exist_ok=True)
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("unsafe_codex_ipc_directory")
            path.chmod(0o700)
        self.ipc_prepared = True

    def _transport_recovery_block(self):
        gate = self.transport_recovery_gate
        return "recovery_gate_unavailable" if gate is None else gate()

    def _prepare(self, expected_identity=None):
        self._prepare_ipc()
        previous = self.store.read_state("runtime", "codex")
        if previous and previous.get("isolation") != self.isolation:
            raise ValueError("resume_isolation_changed: use a fresh run when changing sandbox mode")
        if self.codex_home.is_symlink() or any((self.codex_home / item).is_symlink()
                for item in ("config.toml", "models.json", "auth.json", "account-control-ca.pem")):
            raise ValueError("sandbox_home_symlink: use a fresh run")
        # Earlier full-access runs copied host extension credentials and resources.
        if previous is None and any((self.codex_home / item).exists()
                                    for item in (".credentials.json", "skills", "plugins")):
            raise ValueError("sandbox_home_contains_host_extensions: use a fresh run")
        for path in [self.actor_dir, self.codex_home, self.private / "tmp"]:
            path.mkdir(parents=True, exist_ok=True)
        self.private.chmod(0o700)
        source = Path(self.config.model.codex_home).expanduser()
        config = tomllib.loads((source / "config.toml").read_text())
        provider_name, provider, upstream_url = _model_provider(config, source)
        if expected_identity is not None and expected_identity != (self.model, provider_name):
            raise ValueError("resume_usage_identity_conflict: cumulative usage requires the same model/provider")
        if self.sandbox:
            self.sandbox.close()
            self.sandbox = None
        for relay in (self.model_proxy, self.account_control):
            if relay:
                relay.close()
        self.model_proxy = self.account_control = None
        needs_account_control = provider.get("requires_openai_auth") and _chatgpt_login(source)
        upstream_auth = _upstream_auth(provider, source)
        if needs_account_control and config.get("chatgpt_base_url", ACCOUNT_BASE_URL).rstrip("/") not in (
                ACCOUNT_BASE_URL, "https://chatgpt.com"):
            raise ValueError("account_control_requires_default_chatgpt_backend")
        model_catalog = write_direct_model_catalog(self.codex_home / 'models.json')
        lines = [f'model = {json.dumps(self.model)}', f'model_provider = {json.dumps(MODEL_PROXY_PROVIDER)}',
                 f'model_catalog_json = {json.dumps(str(model_catalog))}',
                 f'model_reasoning_effort = {json.dumps(self.effort)}', 'approval_policy = "never"',
                 # OS isolation is controlled by features.sandbox_enabled.
                 'sandbox_mode = "danger-full-access"', 'web_search = "disabled"',
                 # Selected local preferences are fixed harness defaults, not
                 # inherited from future edits to the user's Codex config.
                 'model_verbosity = "high"', 'service_tier = "default"',
                 'suppress_unstable_features_warning = true',
                 # Native dynamic tools keep image parts typed through model input.
                 '[features]', 'code_mode_host = false', 'shell_tool = true', 'unified_exec = true',
                 'browser_use = false', 'browser_use_external = false', 'computer_use = false',
                 'apps = false', 'plugins = false', 'artifact = false', 'image_generation = false',
                 'goals = true', 'hooks = true',
                 'memories = ' + str(self.config.model.native_memories).lower(), 'network_proxy = false',
                 'mentions_v2 = true', 'multi_agent = true', 'prevent_idle_sleep = true',
                 'guardian_approval = true', 'multi_agent_v2 = true', 'fast_mode = false',
                 'context_management.experimental_mode = true',
                 '[features.code_mode]', 'enabled = false',
                 # Native memory extraction/consolidation otherwise defaults to a built-in model the
                 # provider may not serve; its exhausted retries end the whole episode.
                 '[memories]', f'extract_model = {json.dumps(self.model)}',
                 f'consolidation_model = {json.dumps(self.model)}']
        if provider.get("wire_api", "responses") != "responses":
            raise ValueError("official_model_limits_require_responses_provider")
        previous = self.store.read_state("runtime", "model_requests", {})
        if previous and previous.get("counting") != MODEL_COMPLETION_COUNTING:
            raise ValueError("resume_model_budget_semantics_changed: choose a fresh run")
        used = previous.get("completed_calls", 0)
        previous_attempts = previous.get("http_attempt_count", 0)
        if used >= 180:
            raise ValueError("episode_model_request_limit_exhausted")
        self.model_limit_reached.clear()
        # A resumed run keeps the recovery policy it started with (none if it predates one).
        recovery = ((previous.get("transport_recovery") or {}).get("policy") if previous
                    else self.config.model.transport_recovery.to_dict())
        self.store.set_state("runtime", "model_image_policy", dict(
            enabled=self.image_settings["reuse_exact_images"], policy=IMAGE_REUSE_POLICY,
            compaction="unchanged", oversized_trim=self.image_settings["trim_oversized_requests"],
            max_inline_images=self.image_settings["trim_max_inline_images"]))
        # Relay-measured usage and served model of every delivered completion (actor + helper + compaction), the
        # same measurement every backend records (design 5.5, 4.6); a run resumed from before this record keeps
        # its earlier calls as unknown rather than silently dropping them.
        relay_seed = self.store.read_state("runtime", "model_usage_relay", None)
        relay_usage, served_models = RelayUsageTotal(relay_seed), dict(previous.get("served_models") or {})
        if used and not isinstance(relay_seed, dict):
            relay_usage.total, relay_usage.calls = dict.fromkeys(relay_usage.total), used
            relay_usage.unknown_calls = used
        attempts, relay_lock = {}, threading.Lock()
        def request_event(event):
            kind, call = event.get("kind"), event.get("logical_call_id")
            with relay_lock:
                if kind == "model_request_finished" and call is not None:
                    attempts[call] = (responses_counters(event.get("usage")) if event.get("usage_known") else None,
                                      event.get("served_model"))
                elif kind in ("model_call_completed", "model_call_finished") and call is not None:
                    counters, served = attempts.pop(call, (None, None))
                    if kind == "model_call_completed":
                        served_models.update(add_served_model(served_models, served))
                        self.store.set_state("runtime", "model_usage_relay",
                            relay_usage.add(counters, thread_id=self.thread_id))
            if event.get("endpoint") == "/responses/compact" and event.get("kind") == "model_call_started":
                # Compaction can remove previously delivered RGB from actor context.
                # Invalidate on attempt, even if it fails, to prefer extra evidence.
                self.store.set_state("runtime", "image_context_epoch", event.get("logical_call_id"))
            value = dict(completed_calls=used + self.model_proxy.request_count,
                         http_attempt_count=previous_attempts + self.model_proxy.attempt_count,
                         counting=MODEL_COMPLETION_COUNTING, limit=180,
                         timeout_s=None, max_retries=DEFAULT_MAX_RETRIES, retry_policy=RETRY_POLICY,
                         transport_blocked=self.model_proxy.transport_blocked,
                         transport_error=self.model_proxy.blocked_error,
                         transport_recovery=self.model_proxy.transport_recovery_summary,
                         served_models=dict(served_models), last_event=event)
            self.store.set_state("runtime", "model_requests", value)
            self.store.event("model." + event["kind"], event, source="private_model_transport")
        self.model_proxy = ModelRequestProxy(upstream_url,
            upstream_auth_headers=upstream_auth,
            max_requests=180-used, request_timeout_s=None, max_retries=DEFAULT_MAX_RETRIES, on_event=request_event,
            image_reuse=self.image_settings["reuse_exact_images"],
            image_trim=self.image_settings["trim_oversized_requests"],
            max_inline_images=self.image_settings["trim_max_inline_images"],
            transport_recovery=recovery, recovery_gate=self._transport_recovery_block,
            request_context=self._task_decision_context,
            request_projection=self._present_model_request,
            on_limit=lambda _event: self.model_limit_reached.set(),
            metrics_path=self.private / (new_id("model-requests") + ".jsonl")).start()
        if needs_account_control:
            # Native account/read performs workspace discovery independently of
            # model Responses. Keep that read-only control path narrowly routed.
            account_control = AccountControlRelay(self.private / "account-control",
                self.codex_home / "account-control-ca.pem", upstream_auth_headers=upstream_auth)
            try:
                self.account_control = account_control.start()
            except BaseException:
                account_control.close()
                self.model_proxy.close()
                self.model_proxy = None
                raise
            lines.insert(0, f'chatgpt_base_url = {json.dumps(self.account_control.base_url)}')
        # Native Codex exposes a finite integer idle timeout, not a disabled
        # sentinel. Use TOML's largest positive integer (about 292 million years)
        # and keepalive events; neither proxy nor socket imposes a request limit.
        # Do not serialize source headers, environment keys or upstream URLs to the actor.
        provider = dict(name=provider_name, requires_openai_auth=True)
        provider.update(base_url=self.model_proxy.base_url, wire_api="responses", supports_websockets=False,
                        request_max_retries=0, stream_max_retries=0, stream_idle_timeout_ms=2**63-1)
        provider.setdefault("name", provider_name)
        lines.append(f'[model_providers.{MODEL_PROXY_PROVIDER}]')
        for key, value in provider.items():
            lines.append(f'{key} = {json.dumps(value)}')
        config_text = "\n".join(lines) + "\n"
        self.native_config = tomllib.loads(config_text)
        (self.codex_home / "config.toml").write_text(config_text)
        (self.codex_home / "config.toml").chmod(0o600)
        (self.codex_home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "auriga-local-relay"}))
        (self.codex_home / "auth.json").chmod(0o600)
        # Match the SDK's pinned native runtime, regardless of PATH wrappers or
        # whether the caller activated the Python environment in its shell.
        executable = bundled_codex_path().resolve()
        self.cli_executable = executable
        self.env = {"PATH": os.defpath, "LANG": "C.UTF-8"}
        self.env.update(HOME=str(self.actor_dir), CODEX_HOME=str(self.codex_home), TMPDIR=str(self.private / "tmp"))
        if self.account_control:
            self.env["CODEX_CA_CERTIFICATE"] = str(self.account_control.ca_path)
        self.account_ca_bytes = self.account_control.ca_path.read_bytes() if (
            self.account_control and self.account_control.ca_path.is_file()) else None
        source_snapshot = snapshot_actor_source(self.private / "actor-source")
        self.command_prefix = []
        if self.sandbox_enabled:
            from .codex_sandbox import CodexSandbox
            ports = [urlsplit(self.model_proxy.base_url).port]
            if self.account_control:
                ports.append(urlsplit(self.account_control.base_url).port)
            self.sandbox = CodexSandbox(self.private / "sandbox", self.ipc, self.actor_dir,
                self.codex_home, executable, ports=ports, packages=True,
                read_paths=[(source_snapshot, Path("/harness"))])
            self.command_prefix, overrides = self.sandbox.prepare()
            self.env.update(overrides)
        self.command = self.command_prefix + [str(executable), "--no-daemon", "app-server", "--listen", f"unix://{self.socket_path}"]
        return provider_name

    def _stop_previous_owned_processes(self, previous):
        """A crashed gateway may leave its native app-server alive; never attach to it."""
        self._prepare_ipc()
        uri=f"unix://{self.socket_path}"
        runtime=self.store.read_state("runtime","app_server") or {}
        candidates=[((previous or {}).get("app_server_pid"), (previous or {}).get("socket_uri") or uri),
                    (runtime.get("pid"), runtime.get("socket_uri") or uri)]
        terminal_file=self.root/"terminal.json"
        if terminal_file.exists():
            terminal=json.loads(terminal_file.read_text())
            candidates.append((terminal.get("pid"), terminal.get("socket_uri") or uri))
        for pid, owned_uri in set(candidates):
            if pid is None: continue
            try:
                descriptor=pidfd_open(pid)
            except ProcessLookupError: continue
            try:
                arguments=Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
                if owned_uri not in arguments: continue  # PID reused by unrelated work.
                pidfd_send_signal(descriptor,signal.SIGTERM)
                if not select.select([descriptor],[],[],5)[0]:
                    pidfd_send_signal(descriptor,signal.SIGKILL)
                    if not select.select([descriptor],[],[],5)[0]: raise RuntimeError("previous_owned_codex_stop_unconfirmed")
                self.store.event("codex.previous_process_stopped",{"pid":pid,"stop_confirmed":True})
            except (FileNotFoundError,ProcessLookupError): pass
            finally: os.close(descriptor)
        self.socket_path.unlink(missing_ok=True)

    def start(self, tools, instructions, resume_thread_id=None):
        if self.process and self.process.poll() is None: raise ValueError("session_already_started")
        require_codex_version()
        previous = self.store.read_state("runtime", "codex")
        if previous and previous.get("isolation") != self.isolation:
            raise ValueError("resume_isolation_changed: use a fresh run when changing sandbox mode")
        if resume_thread_id and previous and previous.get("tools_digest") != digest(tools):
            raise ValueError("resume_tool_registry_conflict")
        # A dead app-server can leave admitted/queued SDK tool workers behind.
        # Finish or cancel that owner's work before recycling connection state.
        if self.client is not None:
            self.close()
        self.deferred_tools = frozenset(tool["name"] for tool in tools if tool.get("deferLoading", False))
        self.closing = False
        with self.pending_lock:
            self.native_connection_id = new_id("native-connection")
            self.pending.clear()
            self.resolved_requests.clear()
        for info in self.store.states("native_request"):
            if info["status"] in ("pending", "responding"):
                info.update(status="expired", reason="app_server_restarted")
                self.store.set_state("native_request", info["request_id"], info)
        self._stop_previous_owned_processes(previous)
        expected_identity = None
        if resume_thread_id:
            saved_config = tomllib.loads((self.codex_home / "config.toml").read_text())
            saved_provider = saved_config.get("model_provider")
            if saved_provider == MODEL_PROXY_PROVIDER:
                saved_provider = (previous or {}).get("provider")
            expected_identity = (saved_config.get("model"), saved_provider)
        provider_name = self._prepare(expected_identity)
        self.provider = provider_name
        self.store.set_state("runtime", "price_card", self.price_card)
        self.log = (self.private / "app-server.log").open("a")
        self.process = subprocess.Popen(self.command, env=self.env, stdin=subprocess.DEVNULL,
                                        stdout=self.log, stderr=self.log, cwd=self.actor_dir)
        self.store.set_state("runtime","app_server",{"pid":self.process.pid,"socket_uri":f"unix://{self.socket_path}"})
        deadline = time.monotonic() + 20
        while not (self.sandbox.repair_socket_alias(self.socket_path) if self.sandbox else self.socket_path.is_socket()):
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"app-server startup failed; inspect {self.private / 'app-server.log'}")
            time.sleep(0.05)
        connection_id = self.native_connection_id
        # A delayed old reader/worker must not resolve a recycled RPC in a new
        # connection, even when resume retains the same thread identity.
        def notification(message):
            if self.native_connection_id == connection_id:
                self._notification(message)
        def request(message, **kwargs):
            if self.native_connection_id == connection_id:
                return self._server_request(message, **kwargs)
        def disconnected(error):
            if self.native_connection_id == connection_id:
                self._disconnected(error)
        def tool(params):
            if (self.native_connection_id != connection_id or self.closing
                    or self.state in {"disconnected", "closed"}):
                raise ValueError("native_tool_call_stale_connection")
            namespace = DYNAMIC_TOOL_NAMESPACE if params.get("tool") in self.deferred_tools else None
            if params.get("namespace") != namespace:
                raise ValueError("native_tool_call_namespace_mismatch")
            return self.tool_handler(params)
        self.client = RecordedClient(CodexConfig(
            launch_args_override=(sys.executable, "-m", "auriga.cortex.session.codex", "proxy",
                                  str(self.sandbox.host_socket_path if self.sandbox else self.socket_path)),
            client_name="auriga", client_title="Auriga", cwd=str(Path(__file__).resolve().parents[3])),
            self.store, tool, notification, request, disconnected,
            route_notifications=False, parallel_safe=self.tool_parallel_safe)
        self.client.start()
        initialized = self.client.initialize()
        if getattr(self, "lean_projection", False):
            from ..context.actor_instructions import LEAN_PROJECTION_BLOCK_INSTRUCTIONS, LEAN_PROJECTION_INSTRUCTIONS
            instructions = instructions + "\n" + (LEAN_PROJECTION_BLOCK_INSTRUCTIONS if getattr(self, "lean_projection_block", 0)
                                                  else LEAN_PROJECTION_INSTRUCTIONS)
        parameters = dict(model=self.model, modelProvider=MODEL_PROXY_PROVIDER, cwd=str(self.actor_dir),
                                                approvalPolicy="never", sandbox="danger-full-access",
                                                developerInstructions=self.presentation.present_text(instructions),
                                                config={"model_reasoning_effort": self.effort},
                                                allowProviderModelFallback=False)
        if resume_thread_id:
            parameters.pop("allowProviderModelFallback")
            started = self.client.thread_resume(resume_thread_id, parameters)
        else:
            parameters["dynamicTools"] = self._present_model_request(dict(tools=_native_dynamic_tools(tools)))["tools"]
            started = self.client.thread_start(parameters)
        self.thread_id = started.thread.id
        if self.model_proxy is not None:
            self.model_proxy.register_thread(self.thread_id, "actor", self.thread_id)
        self.state = "idle"
        instance_id = new_id("appserver")
        self.store.set_state("runtime", "codex", dict(thread_id=self.thread_id, app_server_instance_id=instance_id,
                                                  app_server_pid=self.process.pid, socket_uri=f"unix://{self.socket_path}",
                                                  model=self.model, provider=self.provider, effort=self.effort, sdk_version=CODEX_SDK_VERSION,
                                                  cli_version=CODEX_CLI_VERSION, cli_executable=str(self.cli_executable),
                                                  initialization=initialized.model_dump(by_alias=True, mode="json"),
                                                  isolation=self.isolation, tools_digest=digest(tools)))
        # Backend-neutral identity (design §5.2), mirroring runtime/codex; observer data only.
        self.store.set_state("runtime", "agent", dict(
            backend=self.backend_kind, thread_id=self.thread_id, instance_id=instance_id, pid=self.process.pid,
            model=self.model, provider=self.provider, effort=self.effort, isolation=self.isolation,
            tools_digest=digest(tools), instructions_digest=digest(parameters["developerInstructions"]),
            counting=MODEL_COMPLETION_COUNTING, wire_api="openai_responses", sdk_version=CODEX_SDK_VERSION,
            cli_version=CODEX_CLI_VERSION, cli_executable=str(self.cli_executable), cli_sha256=None,
            native=dict(codex_home="private/codex-home", socket_uri=f"unix://{self.socket_path}")))
        if resume_thread_id:
            self.recover_history()
        return self.thread_id

    def recover_history(self):
        with self.projection_lock:
            watermark=self.live_revision
        thread = self.client.thread_read(self.thread_id, include_turns=True).thread.model_dump(by_alias=True, mode="json")
        recovery_ref=self.store.put_artifact(canonical(thread).encode(),"codex_history_snapshot","application/json","observer")
        for turn in thread.get("turns", []):
            for item in turn.get("items", []):
                self._project_item(item, turn["id"], item.get("status", "completed" if turn["status"] != "inProgress" else "unknown"), raw_ref=recovery_ref, recovered=True, watermark=watermark)
        with self.projection_lock:
            turns = thread.get("turns", [])
            if turns and self.turn_revision <= watermark:
                turn = turns[-1]
                self.turn_id = turn["id"]
                self.state = "running" if turn["status"] == "inProgress" else turn["status"]
                if self.state != "running":
                    self.turn_result = turn
                    self.turn_done.set()
        self.store.event("codex.history_recovered", {"turns": len(turns), "delta_history_complete": False,"raw_ref":recovery_ref}, {"thread_id": self.thread_id})
        return thread

    def _project_item(self, item, turn_id, status, raw_ref=None, recovered=False, watermark=None):
        with self.projection_lock:
            return self._project_item_locked(item, turn_id, status, raw_ref, recovered, watermark)

    def _project_item_locked(self, item, turn_id, status, raw_ref=None, recovered=False, watermark=None):
        identifier = item.get("id")
        if not identifier: return
        previous = self.store.read_state("codex_item", identifier, {})
        if recovered and self.item_revisions.get(identifier,0) > (watermark or 0):
            return
        if not recovered:
            self.live_revision += 1
            self.item_revisions[identifier]=self.live_revision
        refs = previous.get("raw_refs", []) + ([raw_ref] if raw_ref else [])
        previous.update(item)
        previous.update(id=identifier, thread_id=self.thread_id, turn_id=turn_id, status=item.get("status",status), raw_refs=refs)
        self.store.set_state("codex_item", identifier, previous)

    def _notification(self, message):
        with self.projection_lock:
            return self._notification_locked(message)

    def _notification_locked(self, message):
        method = message["method"]
        params = message.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        correlation = {target: params[key] for key, target in [("threadId", "thread_id"), ("turnId", "turn_id"), ("callId", "tool_call_id")] if params.get(key)}
        if isinstance(params.get("item"), dict) and params["item"].get("id"):
            correlation["item_id"] = params["item"]["id"]
        owned = self.thread_id is not None and params.get("threadId") == self.thread_id
        if owned and (method in {"item/started", "item/completed", "item/tool/call"}
                      or (method.startswith("item/") and method.endswith(("Delta", "/delta"))
                          and isinstance(params.get("delta"), str) and params["delta"])):
            self.last_activity_at = time.monotonic()
        if method == "serverRequest/resolved":
            self._resolve_request(params)
        if (owned and method in ("item/started", "item/completed")
                and isinstance(params.get("item"), dict)
                and params["item"].get("type") == "contextCompaction"):
            # Local/window compaction uses ordinary /responses, not necessarily
            # /responses/compact. Invalidate synchronously on the native event
            # before later tool feedback can rely on images that were discarded.
            self.store.set_state("runtime", "image_context_epoch", dict(
                source="native_context_compaction", thread_id=self.thread_id,
                item_id=params["item"].get("id"), phase=method))
        if owned and method in ("turn/started", "turn/completed"):
            self.live_revision += 1
            self.turn_revision = self.live_revision
        if method == "turn/started" and owned:
            self.turn_id = params["turn"]["id"]
            correlation["turn_id"] = self.turn_id
            self.state = "running"
        if method == "turn/completed" and owned and (params.get("turn") or {}).get("id") == self.turn_id:
            self.turn_result = params["turn"]
            self.state = params["turn"]["status"]
            self.turn_done.set()
        if method == "item/tool/call":
            # callId is the native dynamic tool item identity in this fixed build.
            correlation["item_id"] = params.get("callId")
        ref = self.store.put_artifact(canonical(message).encode(), "codex_event", "application/json", "observer")
        if owned and isinstance(params.get("item"), dict):
            self._project_item(params["item"], params.get("turnId"), "completed" if method == "item/completed" else "running", ref)
        elif owned and params.get("itemId"):
            item = self.store.read_state("codex_item", params["itemId"], dict(id=params["itemId"]))
            if isinstance(params.get("delta"), str):
                key = "aggregatedOutput" if "outputDelta" in method else "text"
                item[key] = (item.get(key) or "") + params["delta"]
            self._project_item(item, params.get("turnId"), "running", ref)
        payload = {"raw_ref": ref, "method": method}
        if owned and method == "thread/tokenUsage/updated":
            usage = dict(normalize_token_usage(params.get("tokenUsage")), thread_id=params.get("threadId"),
                         turn_id=params.get("turnId"), raw_ref=ref)
            self.store.set_state("runtime", "token_usage", usage)
            accounting = account_usage(usage, model=self.active_turn_model or self.model, provider=self.provider, price_card=self.price_card)
            self.store.set_state("runtime", "token_accounting", accounting)
            if usage["turn_id"]:
                self.store.set_state("token_usage", usage["turn_id"], usage)
                self.store.set_state("token_accounting", usage["turn_id"], accounting)
            payload["token_usage"] = usage
            payload["token_accounting"] = accounting
        self.store.event("codex." + method, payload, correlation, source="codex_app_server")

    def run_turn(self, prompt):
        with self.projection_lock:
            if self.state == "running":
                raise ValueError("turn_already_active")
            if self.state in ("disconnected", "unknown", "closed"):
                raise ValueError("session_not_connected")
            prompt = self.presentation.present_text(prompt)
            previous = self.state, self.turn_id, self.turn_result, self.active_turn_model
            revision = self.turn_revision
            turn_model, turn_effort = self.model, self.effort
            self.active_turn_model = turn_model
            self.turn_done.clear()
            self.turn_id = None
            self.turn_result = None
            self.state = "running"
        try:
            result = self.client.turn_start(self.thread_id, [dict(type="text", text=prompt)],
                                            dict(model=turn_model, effort=turn_effort))
        except JsonRpcError as error:
            with self.projection_lock:
                # A later native event is stronger evidence than this rejection.
                restored = self.state == "running" and self.turn_id is None and self.turn_revision == revision
                if restored:
                    self.state, self.turn_id, self.turn_result, self.active_turn_model = previous
                    self.turn_done.set()
                self.store.event("codex.turn_start_rejected", dict(code=error.code, message=error.message,
                    state_restored=restored), {"thread_id": self.thread_id})
            raise
        except Exception as error:
            # Missing acknowledgement may conceal an admitted turn. Stop robot
            # admission and require recovery instead of making the turn retryable.
            if self.state != "disconnected":
                self._disconnected(error)
            raise
        with self.projection_lock:
            if self.turn_id is None:
                self.turn_id = result.turn.id
        return result.turn.id

    def steer(self, text):
        """Add operator text to the ACTIVE turn (native ``turn/steer``), like typing into Codex mid-turn.

        Rejected unless a turn is running; it never starts a turn and never touches robot admission.
        """
        with self.projection_lock:
            if self.state != "running" or not self.turn_id:
                raise ValueError("no_active_turn_to_steer")
            turn_id = self.turn_id
        text = self.presentation.present_text(text)
        result = self.client.turn_steer(self.thread_id, turn_id, [dict(type="text", text=text)])
        self.store.event("codex.turn_steered", {"turn_id": turn_id, "chars": len(text)},
                         {"thread_id": self.thread_id, "turn_id": turn_id})
        return result.turn_id

    def owned_processes(self):
        """``[("codex_bridge", SDK stdio bridge), ("codex", app-server)]``, omitting processes not created yet."""
        client = getattr(self, "client", None)
        processes = (("codex_bridge", getattr(client, "_proc", None)), ("codex", getattr(self, "process", None)))
        return [(role, process) for role, process in processes if process is not None]

    def terminal_attach_spec(self):
        """Read-only ``codex resume --remote`` TUI on the run's app-server socket, in the actor's sandbox prefix.

        Pins the TUI to this session's current model and effort. None before ``_prepare`` resolved the native CLI.
        """
        executable = getattr(self, "cli_executable", None)
        if executable is None:
            return None
        return dict(cwd=self.actor_dir,
                    command_prefix=self.command_prefix + [str(executable),
                        '-c', 'model=' + json.dumps(self.model),
                        '-c', 'model_reasoning_effort=' + json.dumps(self.effort)],
                    env=self.env, attach_uri=f"unix://{self.socket_path}", thread_id=self.thread_id)

    def evidence_agent(self, **options):
        """Codex's native evidence helper (``CodexEvidenceAgent``) bound to this session, or the backend-neutral
        ``RelayEvidenceClient`` when ``agent.evidence.client == "relay"`` (design §9.6); ``options`` pass through."""
        from ..agents.factory import EVIDENCE_CLIENT_RELAY, evidence_client
        if evidence_client(getattr(self, "config", None), "codex") == EVIDENCE_CLIENT_RELAY:
            from ..agents.evidence import RelayEvidenceClient
            return RelayEvidenceClient(self, **options)
        from .codex_evidence_agent import CodexEvidenceAgent
        return CodexEvidenceAgent(self, **options)

    def provenance(self):
        """Manifest ``agent_backend`` block for Codex (design §7.3); values come from the pinned constants and
        this session's resolved settings. Unknown values (not computed for Codex) are None."""
        from ..agents.factory import evidence_client
        executable = getattr(self, "cli_executable", None)
        capabilities = getattr(self, "capabilities", None) or CodexSession.capabilities
        return dict(kind="codex", sdk=CODEX_SDK_VERSION,
                    cli=dict(version=CODEX_CLI_VERSION, executable=str(executable) if executable is not None else None,
                             sha256=None),
                    wire_api="openai_responses", upstream_provider=getattr(self, "provider", None),
                    model_configured=getattr(self, "model", None), effort=getattr(self, "effort", None),
                    native_effort=dict(model_reasoning_effort=getattr(self, "effort", None)),
                    isolation=getattr(self, "isolation", None), counting=MODEL_COMPLETION_COUNTING,
                    tool_transport=dict(kind="codex_dynamic_tools", tool_name_prefix="",
                                        deferred_tools=("native_namespace" if getattr(self, "deferred_tools", None)
                                                        else "none"),
                                        deferred_namespace=DYNAMIC_TOOL_NAMESPACE),
                    prompt_channel=dict(instructions_channel="developer_instructions", native_base_prompt="preserved"),
                    evidence_client=evidence_client(getattr(self, "config", None), "codex"),
                    capabilities=capabilities.to_dict())

    def interrupt(self):
        self.expire_requests("turn_interrupted")
        if self.model_proxy:
            self.model_proxy.cancel_pending("turn_interrupted")
        if self.state == "running" and self.turn_id:
            try:
                self.client.turn_interrupt(self.thread_id, self.turn_id)
            except JsonRpcError as error:
                if error.code != -32600 or "no active turn to interrupt" not in str(error.message).lower():
                    raise

    def close(self):
        self.closing = True
        self.expire_requests("session_closed")
        if self.account_control:
            self.account_control.close()
        if self.model_proxy:
            self.model_proxy.close()
        if self.client:
            # A failed bridge may leave an SDK writer blocked on a full stdin pipe.
            if self.client._proc and self.client._proc.poll() is None:
                self.client._proc.terminate()
            try:
                self.client.close()
            except (BrokenPipeError, OSError):
                pass
            if hasattr(self.client, "server_pool"):
                self.client.server_pool.shutdown(wait=True, cancel_futures=True)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        if self.sandbox:
            self.sandbox.close()
        if hasattr(self, "log"):
            self.log.close()
        if hasattr(self, "daemon_socket"):
            self.daemon_socket.unlink(missing_ok=True)
        if self.ipc_prepared:
            self.socket_path.unlink(missing_ok=True)
            self.ipc.rmdir()
            self.ipc_prepared = False
        self.state = "closed"
        self.turn_done.set()


async def proxy(path):
    """The pinned Unix transport is WebSocket, SDK stdio is JSONL."""
    import aiohttp
    reader = asyncio.StreamReader(limit=32 * 1024 * 1024)
    await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=path)) as client:
        async with client.ws_connect("http://localhost/", max_msg_size=32 * 1024 * 1024) as ws:
            async def send():
                while line := await reader.readline():
                    await ws.send_str(line.decode().rstrip("\n"))
                await ws.close()

            async def receive():
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        sys.stdout.write(message.data + "\n"); sys.stdout.flush()
            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(proxy(sys.argv[2]))
