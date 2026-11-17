"""Backend-neutral transcript writer (design 5.1, 5.3, 5.5).

Rows go into the existing ``codex_item`` state kind (``Store.TRANSCRIPT_KIND``) with exactly the field shapes
Codex produces, so ``Store.completed_final_texts``, ``Store.review_page``, the library replay and the web
readers cannot tell them apart. Every row additionally carries store-only ``backend`` (and ``native_id``,
``message_id``, ``parent_item_id`` when known); rows never enter model context.

Final-answer synthesis: backends without a native phase write assistant text with ``phase=None``. When a turn
terminates as ``completed``, the text rows of the LAST top-level assistant message (grouped by ``message_id``)
that has text and no tool call get ``phase="final_answer"``; every other top-level text row of that turn becomes
``commentary`` (so earlier would-be finals of one busy period, e.g. before a steer-triggered continuation, are
commentary). Subagent rows (``parent_item_id`` set) are never stamped. Interrupted or failed turns stamp nothing.
"""

from threading import RLock
import time

from ...core.contracts import new_id
from ..session.usage import RelayUsageTotal, account_usage

EVENT_SOURCE = "agent_backend"
TURN_STATUSES = ("completed", "interrupted", "failed")
TOOL_ROW_TYPES = ("dynamicToolCall", "commandExecution", "fileChange", "mcpToolCall")


class TranscriptWriter:
    """Writes neutral transcript rows, ``agent.*`` events and relay usage for one agent thread.

    store: the run ``Store`` (writer mode). backend: backend kind stored on every row ("claude_code",
    "opencode", ...). thread_id: native thread/session id stored on every row. The writer is thread-safe.
    """

    def __init__(self, store, *, backend, thread_id):
        self.store, self.backend, self.thread_id = store, backend, thread_id
        self.turn_id = None
        self.lock = RLock()
        self._started = {}
        self._usage = RelayUsageTotal(store.read_state("runtime", "model_usage_relay", None), thread_id)

    # ----- rows -----

    def _row(self, identifier, fields, *, turn_id=None, status="completed", native_id=None, message_id=None,
             parent_item_id=None, raw_ref=None):
        with self.lock:
            previous = self.store.transcript_item(identifier, None)
            row = dict(previous) if previous else dict(phase=None) if fields.get("type") == "agentMessage" else {}
            row.update({key: value for key, value in fields.items() if value is not None or key not in row})
            refs = list(row.get("raw_refs") or []) + ([raw_ref] if raw_ref else [])
            row.update(id=identifier, thread_id=self.thread_id, turn_id=turn_id or row.get("turn_id") or self.turn_id,
                       status=status, backend=self.backend, raw_refs=refs)
            for key, value in (("native_id", native_id), ("message_id", message_id), ("parent_item_id", parent_item_id)):
                if value is not None:
                    row[key] = value
            self.store.set_transcript_item(identifier, row)
            return row

    def user_message(self, text, *, item_id=None, native_id=None, turn_id=None, raw_ref=None, steer=False):
        """Operator/Auriga prompt as ``userMessage`` (content=[{type:"text", text}]). Returns the row id."""
        identifier = item_id or new_id("item")
        fields = dict(type="userMessage", content=[dict(type="text", text=text)])
        if steer:
            fields["steer"] = True
        self._row(identifier, fields, turn_id=turn_id, native_id=native_id, raw_ref=raw_ref)
        return identifier

    def assistant_text(self, text, *, message_id=None, item_id=None, native_id=None, parent_item_id=None,
                       turn_id=None, status="completed", raw_ref=None):
        """Assistant prose as ``agentMessage`` with ``phase=None`` (stamped at turn completion).

        message_id: native assistant message id; rows (text and tool calls) sharing it form one message for
        the final-answer rule. parent_item_id: set for subagent output (never final text). Re-calling with the
        same ``item_id`` replaces the text (streaming); an already stamped phase is kept. Returns the row id.
        """
        identifier = item_id or new_id("item")
        self._row(identifier, dict(type="agentMessage", text=text), turn_id=turn_id, status=status,
                  native_id=native_id, message_id=message_id, parent_item_id=parent_item_id, raw_ref=raw_ref)
        return identifier

    def reasoning(self, text, *, message_id=None, item_id=None, native_id=None, parent_item_id=None, turn_id=None,
                  raw_ref=None):
        """Reasoning summary as ``reasoning`` (summary=[text]); provider signatures must not be passed in."""
        identifier = item_id or new_id("item")
        self._row(identifier, dict(type="reasoning", summary=[text], content=[]), turn_id=turn_id,
                  native_id=native_id, message_id=message_id, parent_item_id=parent_item_id, raw_ref=raw_ref)
        return identifier

    def tool_started(self, call_id, tool, arguments, *, message_id=None, native_id=None, namespace=None,
                     parent_item_id=None, turn_id=None, raw_ref=None):
        """Registry (dynamic) tool call begins: ``dynamicToolCall`` row whose id is the host call id."""
        self._started[call_id] = time.monotonic()
        self._row(call_id, dict(type="dynamicToolCall", tool=tool, arguments=arguments, namespace=namespace,
                                contentItems=None, success=None), turn_id=turn_id, status="inProgress",
                  native_id=native_id, message_id=message_id, parent_item_id=parent_item_id, raw_ref=raw_ref)
        return call_id

    def tool_completed(self, call_id, *, success, content_items=None, tool=None, arguments=None, message_id=None,
                       native_id=None, turn_id=None, raw_ref=None):
        """Registry tool call finished. success: True/False, or None when the outcome is unknown (status stays
        "failed" only for False). content_items: Codex-style ``[{type:"inputText",text}|{type:"inputImage",...}]``."""
        fields = dict(type="dynamicToolCall", tool=tool, arguments=arguments, contentItems=content_items,
                      success=success)
        started = self._started.pop(call_id, None)
        if started is not None:
            fields["durationMs"] = int((time.monotonic() - started) * 1000)
        self._row(call_id, fields, turn_id=turn_id, status="failed" if success is False else "completed",
                  native_id=native_id, message_id=message_id, raw_ref=raw_ref)
        return call_id

    def command(self, item_id, command, *, status="completed", aggregated_output=None, exit_code=None, cwd=None,
                message_id=None, native_id=None, parent_item_id=None, turn_id=None, raw_ref=None):
        """Native shell command as ``commandExecution``. exit_code None = unknown/not finished. Upsert."""
        self._row(item_id, dict(type="commandExecution", command=command, aggregatedOutput=aggregated_output,
                                exitCode=exit_code, cwd=cwd), turn_id=turn_id, status=status, native_id=native_id,
                  message_id=message_id, parent_item_id=parent_item_id, raw_ref=raw_ref)
        return item_id

    def file_change(self, item_id, changes, *, status="completed", message_id=None, native_id=None,
                    parent_item_id=None, turn_id=None, raw_ref=None):
        """Native file edit as ``fileChange``; changes: list of ``{path, kind, diff}`` dicts. Upsert."""
        self._row(item_id, dict(type="fileChange", changes=changes), turn_id=turn_id, status=status,
                  native_id=native_id, message_id=message_id, parent_item_id=parent_item_id, raw_ref=raw_ref)
        return item_id

    def native_tool(self, item_id, tool, arguments=None, *, result=None, error=None, status="completed",
                    message_id=None, native_id=None, parent_item_id=None, turn_id=None, raw_ref=None):
        """Any other native tool (Read/Glob/Grep/...) as generic ``mcpToolCall`` (tool, arguments, result)."""
        fields = dict(type="mcpToolCall", tool=tool, arguments=arguments, result=result)
        if error is not None:
            fields["error"] = error if isinstance(error, dict) else dict(message=str(error))
        self._row(item_id, fields, turn_id=turn_id, status=status, native_id=native_id, message_id=message_id,
                  parent_item_id=parent_item_id, raw_ref=raw_ref)
        return item_id

    def compaction(self, *, item_id=None, native_id=None, turn_id=None, raw_ref=None):
        """Context compaction marker as ``contextCompaction``."""
        identifier = item_id or new_id("item")
        self._row(identifier, dict(type="contextCompaction"), turn_id=turn_id, native_id=native_id, raw_ref=raw_ref)
        return identifier

    def error(self, message, *, item_id=None, native_id=None, turn_id=None, raw_ref=None):
        """Backend/model error as ``error`` row (error.message). Never put provider secrets in ``message``."""
        identifier = item_id or new_id("item")
        self._row(identifier, dict(type="error", error=dict(message=message)), turn_id=turn_id, native_id=native_id,
                  raw_ref=raw_ref)
        return identifier

    # ----- events and turns -----

    def event(self, event_type, payload=None, *, turn_id=None, tool_call_id=None, item_id=None):
        """Emit one ``agent.*`` event (source ``agent_backend``) with thread/turn/call/item correlation."""
        if not isinstance(event_type, str) or not event_type.startswith("agent."):
            raise ValueError("agent_event_type_must_start_with_agent")
        correlation = {key: value for key, value in dict(thread_id=self.thread_id, turn_id=turn_id or self.turn_id,
                       tool_call_id=tool_call_id, item_id=item_id).items() if value}
        return self.store.event(event_type, dict(payload or {}), correlation, source=EVENT_SOURCE)

    def turn_started(self, turn_id=None):
        """Begin a turn (host-minted id when None); emits ``agent.turn_started``. Returns the turn id."""
        with self.lock:
            self.turn_id = turn_id or new_id("turn")
            self.event("agent.turn_started", {}, turn_id=self.turn_id)
            return self.turn_id

    def turn_completed(self, status, *, error=None, turn_id=None):
        """Terminal of a turn: status completed|interrupted|failed; error: message text or None.

        On ``completed`` stamps final-answer phases (see module doc); emits ``agent.turn_completed``."""
        if status not in TURN_STATUSES:
            raise ValueError("invalid_turn_status")
        with self.lock:
            turn = turn_id or self.turn_id
            if status == "completed":
                self.stamp_final_answer(turn)
            self.event("agent.turn_completed", dict(status=status, error=dict(message=error) if error else None),
                       turn_id=turn)
            return turn

    def steer(self, text, *, item_id=None, native_id=None, turn_id=None, raw_ref=None):
        """Operator text injected into the running turn: ``userMessage`` row (``steer: true``) plus an
        ``agent.turn_steered`` event. Pass a stable ``item_id`` when the backend later echoes the text."""
        identifier = self.user_message(text, item_id=item_id, native_id=native_id, turn_id=turn_id, raw_ref=raw_ref,
                                       steer=True)
        self.event("agent.turn_steered", {}, turn_id=turn_id, item_id=identifier)
        return identifier

    def permission_denied(self, tool, *, tool_use_id=None, reason=None, turn_id=None):
        """Record a native-tool permission denial (``agent.permission_denied``): tool name, optional native tool
        use id and reason text."""
        return self.event("agent.permission_denied", dict(tool=tool, tool_use_id=tool_use_id, reason=reason),
                          turn_id=turn_id, tool_call_id=tool_use_id)

    def stamp_final_answer(self, turn_id=None):
        """Apply the final-answer rule to one turn's top-level rows (idempotent). Returns stamped row ids."""
        turn = turn_id or self.turn_id
        with self.lock:
            rows = [row for row in self.store.states(self.store.TRANSCRIPT_KIND)
                    if row.get("thread_id") == self.thread_id and row.get("turn_id") == turn
                    and not row.get("parent_item_id")]
            messages = {}
            for row in rows:
                key = row.get("message_id") or row["id"]
                entry = messages.setdefault(key, dict(texts=[], tools=False))
                if row.get("type") == "agentMessage":
                    entry["texts"].append(row["id"])
                elif row.get("type") in TOOL_ROW_TYPES:
                    entry["tools"] = True
            texts = {row["id"]: row for row in rows if row.get("type") == "agentMessage"}
            candidates = [entry["texts"] for entry in messages.values() if not entry["tools"]
                          and any(isinstance(texts[i].get("text"), str) and texts[i]["text"].strip()
                                  for i in entry["texts"])]
            final = set(candidates[-1]) if candidates else set()
            stamped = []
            for identifier, row in texts.items():
                phase = "final_answer" if identifier in final else "commentary"
                if row.get("phase") != phase:
                    self.store.set_transcript_item(identifier, dict(row, phase=phase))
                    stamped.append(identifier)
            return stamped

    # ----- relay usage -----

    def add_usage(self, usage, *, model_context_window=None, turn_id=None):
        """Accumulate one delivered completion's normalized usage (``WireAdapter.normalize_usage`` output).

        Writes ``runtime/token_usage`` (Codex shape, ``raw_ref`` null, ``unknown_calls``) and
        ``runtime/model_usage_relay`` (same plus ``calls``). A total counter becomes None, and stays None, as
        soon as one call lacks it (unknown is not zero). Returns the token_usage record."""
        with self.lock:
            turn = turn_id or self.turn_id
            record = self._usage.add(usage, model_context_window=model_context_window, thread_id=self.thread_id, turn_id=turn)
            calls = record.pop("calls")
            self.store.set_state("runtime", "token_usage", record)
            self.store.set_state("runtime", "model_usage_relay", dict(record, calls=calls))
            if turn:
                self.store.set_state("token_usage", turn, record)
            return record

    def write_accounting(self, *, source, model=None, provider=None, price_card=None):
        """Derive ``runtime/token_accounting`` from the accumulated usage, e.g. source="relay.<wire>.usage.sum"."""
        usage = self.store.read_state("runtime", "token_usage", None)
        if not usage:
            return None
        accounting = account_usage(usage, model=model, provider=provider, price_card=price_card, source=source)
        self.store.set_state("runtime", "token_accounting", accounting)
        if usage.get("turn_id"):
            self.store.set_state("token_accounting", usage["turn_id"], accounting)
        return accounting
