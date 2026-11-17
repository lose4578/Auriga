"""Projects Claude Agent SDK messages onto the neutral transcript (design 5.1 item mapping, plan 07 section 2.3).

``ClaudeProjector.project(message)`` takes ONE message object of ``ClaudeSDKClient.receive_messages()`` and writes the
matching rows/events through a ``TranscriptWriter``. It never decides turn state: it returns a small ``Projected``
record (``kind`` + facts) that ``ClaudeCodeSession`` switches on (init data, session state, result classification,
compaction signals). The module imports no SDK code: messages and content blocks are recognised by class name
(``AssistantMessage``, ``TextBlock`` ...; subclasses such as ``TaskStartedMessage`` count as ``SystemMessage``), so the
real SDK dataclasses and any test double of the same shape work alike.

Mapping (every row carries ``backend`` and, when known, ``native_id``/``message_id``; subagent output carries
``parent_item_id`` and is never final text):

* assistant ``TextBlock``      -> ``agentMessage`` (``phase`` stamped by the writer at turn end);
* ``ThinkingBlock``            -> ``reasoning`` (the signature is not stored);
* ``Bash``                     -> ``commandExecution`` (row id = tool_use id; exit code stays null: the SDK reports none);
* ``Edit``/``Write``/``NotebookEdit`` -> ``fileChange``; every other native tool -> generic ``mcpToolCall`` row;
* registry tools ``mcp__auriga__*`` -> NO row of their own: the MCP transport writes the ``dynamicToolCall`` row under
  the same id (the tool_use id is the host call id). The projector only attaches ``message_id``/``parent_item_id``
  to that row, as soon as the row exists, so the final-answer rule sees the tool call as part of its message. A failed
  tool_result of a registry tool for which the transport never wrote a row (the CLI refused the call) becomes one
  failed generic row under the tool_use id;
* ``ToolResultBlock``          -> completes the native row (``failed`` when ``is_error``); image items are not stored;
* a string ``UserMessage`` equal to a pending prompt/steer (``expect_echo``) -> marks that row consumed (steer: also
  emits ``agent.turn_steered``); the CLI echoes them because the session passes ``--replay-user-messages``;
* ``SystemMessage`` ``compact_boundary``/``microcompact_boundary`` -> ``contextCompaction`` row;
  ``permission_denied`` and ``ResultMessage.permission_denials`` -> ``agent.permission_denied`` once per tool_use id;
* assistant messages with ``error`` set (the CLI's synthetic "API Error: ..." text) and failed ``ResultMessage`` ->
  ``error`` rows (never ``agentMessage``, so they can not become the final answer);
* ``init``/``session_state_changed``/``status``/``api_retry``/rate-limit/task/other system frames -> no row (the session
  logs them raw).

Usage: ``AssistantMessage.usage`` repeats on every block message of one API response, so it is kept once per
``message_id`` (latest wins) in ``message_usage``. It is an observer-side cross-check only; the accounting source is
the relay (``TranscriptWriter.add_usage``), so nothing here writes usage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .options import TOOL_NAME_PREFIX

__all__ = ["Projected", "ClaudeProjector", "classify_result", "ABORT_REASONS"]

ABORT_REASONS = frozenset({"aborted_tools", "aborted_streaming"})   # ResultMessage.terminal_reason of an interrupt
_USAGE_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
_FILE_TOOLS = {"Edit": "update", "Write": "write", "NotebookEdit": "update"}


@dataclass
class Projected:
    """What ``project`` found: ``kind`` plus facts for the session.

    kinds (``data`` keys): ``init`` (tools, model, session_id, mcp_servers, permission_mode, cli_version);
    ``state`` (state: running|idle|requires_action); ``compacting`` (); ``compaction`` (subtype, trigger, pre_tokens,
    post_tokens); ``permission_denied`` (tool, tool_use_id); ``result`` (status, error, terminal_reason, subtype,
    is_error, api_error_status, denials); ``assistant`` (message_id, parent_tool_use_id, item_ids, tool_use_ids,
    has_error); ``tool_result`` (tool_use_ids, unmatched); ``steer_echo``/``prompt_echo`` (item_id); ``user`` (); ``other``
    (subtype, type).
    """
    kind: str
    data: dict = field(default_factory=dict)


def classify_result(result, *, interrupt_requested=False):
    """``(status, error)`` of one ``ResultMessage`` per the plan table; the last result before the idle state wins.

    ``interrupted`` when ``interrupt_requested`` or ``terminal_reason`` is ``aborted_tools``/``aborted_streaming``
    (error None); ``completed`` when ``is_error`` is false (error None); otherwise ``failed`` with
    ``"claude_code_turn_failed:<terminal_reason or subtype>[:<api_error_status>]"`` (no provider text).
    """
    reason = getattr(result, "terminal_reason", None)
    if interrupt_requested or reason in ABORT_REASONS:
        return "interrupted", None
    if not getattr(result, "is_error", False):
        return "completed", None
    status = getattr(result, "api_error_status", None)
    return "failed", "claude_code_turn_failed:" + str(reason or getattr(result, "subtype", None) or "unknown") + (
        f":{status}" if status is not None else "")


def _names(obj):
    return {cls.__name__ for cls in type(obj).__mro__}


def _text_of(content):
    """Text of a tool_result ``content`` (str | list of content dicts | None); image items are skipped."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content if isinstance(content, list) else [content]:
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def _plain_user_text(content):
    """The text of a string user message, or of a list made only of text blocks; None for anything else."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all("TextBlock" in _names(block) for block in content):
        return "".join(block.text for block in content)
    return None


def _edit_diff(arguments):
    old, new = arguments.get("old_string"), arguments.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return None
    return "\n".join([*("-" + line for line in old.split("\n")), *("+" + line for line in new.split("\n"))])


class ClaudeProjector:
    """Message-to-transcript projector of one Claude Code session (one ``TranscriptWriter``). Single consumer: the
    session's reader calls ``project`` sequentially; the writer itself is thread-safe.

    ``transcript``: the session's ``TranscriptWriter``. ``prefix``: model-visible registry tool prefix
    (default ``mcp__auriga__``). Attributes: ``init`` (data of the last init message or None), ``message_usage``
    (``{message_id: usage dict}``), ``tools`` (``{tool_use_id: record}`` of every tool call seen).
    """

    def __init__(self, transcript, *, prefix=TOOL_NAME_PREFIX):
        self.transcript, self.prefix = transcript, prefix
        self.init = None
        self.message_usage = {}
        self.tools = {}
        self._pending_echo = []
        self._denied = set()
        self._assistant_error = False

    # ----- host-side expectations -----

    def expect_echo(self, text, item_id, *, steer=False):
        """Register the text of a prompt (``steer=False``) or steer the session has just sent, with the transcript row
        id it wrote. The CLI echoes it as a ``UserMessage`` when consumed; ``project`` then matches it (FIFO among
        equal texts), marks the row consumed and, for a steer, emits ``agent.turn_steered``."""
        self._pending_echo.append((text, item_id, bool(steer)))

    def forget_echo(self, item_id):
        """Withdraw the pending echo expectation of transcript row ``item_id`` (an input that will not be echoed for
        that row any more), so a later input with the same text gets its own echo. Returns True when one was removed."""
        for index, (_, pending_id, _) in enumerate(self._pending_echo):
            if pending_id == item_id:
                del self._pending_echo[index]
                return True
        return False

    def native_usage_total(self):
        """Sum of the four token counters over ``message_usage`` (each message id counted once); a cross-check
        against the relay accounting, not an accounting source."""
        total = {key: 0 for key in _USAGE_FIELDS}
        for usage in self.message_usage.values():
            for key in _USAGE_FIELDS:
                value = usage.get(key)
                total[key] += value if type(value) is int and value >= 0 else 0
        return total

    def abandon_open_items(self, status="failed"):
        """Set every native row this projector wrote that is still ``inProgress`` (no tool_result arrived: crash,
        disconnect) to ``status`` (``failed`` by default); returns the row ids changed. Registry-tool rows belong to
        the MCP transport and are never touched."""
        changed = []
        with self.transcript.lock:
            for call_id, record in self.tools.items():
                if record["registry"]:
                    continue
                row = self.transcript.store.transcript_item(call_id, None)
                if row and row.get("status") == "inProgress":
                    self.transcript.store.set_transcript_item(call_id, dict(row, status=status))
                    changed.append(call_id)
        return changed

    # ----- dispatch -----

    def project(self, message, *, raw_ref=None):
        """Write the rows/events for one SDK message and return its ``Projected`` record (never raises for an
        unknown message type: it yields ``Projected("other", ...)``)."""
        names = _names(message)
        if "AssistantMessage" in names:
            return self._assistant(message, raw_ref)
        if "UserMessage" in names:
            return self._user(message, raw_ref)
        if "ResultMessage" in names:
            return self._result(message, raw_ref)
        if "SystemMessage" in names:
            return self._system(message, raw_ref)
        return Projected("other", dict(type=type(message).__name__))

    # ----- system -----

    def _system(self, message, raw_ref):
        subtype, data = message.subtype, message.data if isinstance(message.data, dict) else {}
        if subtype == "init":
            self.init = dict(data)
            return Projected("init", dict(tools=data.get("tools"), model=data.get("model"),
                                          session_id=data.get("session_id"), mcp_servers=data.get("mcp_servers"),
                                          permission_mode=data.get("permissionMode"),
                                          cli_version=data.get("claude_code_version")))
        if subtype == "session_state_changed":
            return Projected("state", dict(state=data.get("state")))
        if subtype == "status" and data.get("status") == "compacting":
            return Projected("compacting")
        if subtype in ("compact_boundary", "microcompact_boundary"):
            metadata = data.get("compact_metadata") if isinstance(data.get("compact_metadata"), dict) else {}
            self.transcript.compaction(native_id=data.get("uuid"), raw_ref=raw_ref)
            return Projected("compaction", dict(subtype=subtype, trigger=metadata.get("trigger"),
                                                pre_tokens=metadata.get("pre_tokens"),
                                                post_tokens=metadata.get("post_tokens")))
        if subtype == "permission_denied":
            self._deny(data.get("tool_name"), data.get("tool_use_id"), data.get("decision_reason_type"))
            return Projected("permission_denied", dict(tool=data.get("tool_name"), tool_use_id=data.get("tool_use_id")))
        return Projected("other", dict(type="system", subtype=subtype))

    def _deny(self, tool, tool_use_id, reason):
        key = tool_use_id or (tool, len(self._denied))
        if key in self._denied:
            return False
        self._denied.add(key)
        self.transcript.permission_denied(tool or "unknown", tool_use_id=tool_use_id, reason=reason)
        return True

    # ----- result -----

    def _result(self, message, raw_ref):
        status, error = classify_result(message)
        recorded = []
        for denial in message.permission_denials or []:
            if isinstance(denial, dict) and self._deny(denial.get("tool_name"), denial.get("tool_use_id"), None):
                recorded.append(denial.get("tool_use_id"))
        if (message.is_error and message.terminal_reason not in ABORT_REASONS and not self._assistant_error):
            self.transcript.error(error or "claude_code_turn_failed",
                                  item_id=f"result:{message.uuid}" if message.uuid else None,
                                  native_id=message.uuid, raw_ref=raw_ref)
        self._assistant_error = False
        return Projected("result", dict(status=status, error=error, terminal_reason=message.terminal_reason,
                                        subtype=message.subtype, is_error=bool(message.is_error),
                                        api_error_status=message.api_error_status, denials=recorded))

    # ----- assistant -----

    def _annotate(self, row_id, **fields):
        """Add store-only linkage fields to an existing row; False when the row does not exist yet."""
        with self.transcript.lock:
            row = self.transcript.store.transcript_item(row_id, None)
            if not row:
                return False
            update = {key: value for key, value in fields.items() if value is not None and row.get(key) != value}
            if update:
                self.transcript.store.set_transcript_item(row_id, dict(row, **update))
            return True

    def _assistant(self, message, raw_ref):
        parent, message_id, uuid = message.parent_tool_use_id, message.message_id, message.uuid
        if isinstance(message.usage, dict) and message_id:
            self.message_usage[message_id] = dict(message.usage)
        item_ids, tool_ids = [], []
        error_texts = []
        for index, block in enumerate(message.content or []):
            kind = _names(block)
            item_id = f"{uuid}:{index}" if uuid else None
            if "TextBlock" in kind:
                if not block.text:
                    continue
                if message.error:
                    error_texts.append(block.text)
                    continue
                item_ids.append(self.transcript.assistant_text(
                    block.text, message_id=message_id, item_id=item_id, native_id=uuid, parent_item_id=parent,
                    raw_ref=raw_ref))
            elif "ThinkingBlock" in kind:
                if block.thinking:
                    item_ids.append(self.transcript.reasoning(
                        block.thinking, message_id=message_id, item_id=item_id, native_id=uuid,
                        parent_item_id=parent, raw_ref=raw_ref))
            elif "ToolUseBlock" in kind:
                tool_ids.append(block.id)
                self._tool_use(block, message_id, parent, raw_ref)
        if message.error:
            self._assistant_error = True
            self.transcript.error(" ".join(error_texts) or f"claude_code_assistant_error:{message.error}",
                                  item_id=f"{uuid}:error" if uuid else None, native_id=uuid, raw_ref=raw_ref)
        return Projected("assistant", dict(message_id=message_id, parent_tool_use_id=parent, item_ids=item_ids,
                                           tool_use_ids=tool_ids, has_error=bool(message.error)))

    def _tool_use(self, block, message_id, parent, raw_ref):
        name, arguments = block.name, block.input if isinstance(block.input, dict) else {}
        registry = name.startswith(self.prefix)
        record = dict(name=name[len(self.prefix):] if registry else name, arguments=arguments, registry=registry,
                      message_id=message_id, parent=parent)
        self.tools[block.id] = record
        if registry:
            self._annotate(block.id, message_id=message_id, native_id=block.id, parent_item_id=parent)
            return
        common = dict(status="inProgress", message_id=message_id, native_id=block.id, parent_item_id=parent,
                      raw_ref=raw_ref)
        if name == "Bash":
            self.transcript.command(block.id, arguments.get("command"), **common)
        elif name in _FILE_TOOLS:
            self.transcript.file_change(block.id, self._changes(name, arguments), **common)
        else:
            self.transcript.native_tool(block.id, name, arguments, **common)

    @staticmethod
    def _changes(name, arguments):
        path = arguments.get("file_path") or arguments.get("notebook_path")
        return [dict(path=path, kind=_FILE_TOOLS[name], diff=_edit_diff(arguments) if name == "Edit" else None)]

    # ----- user -----

    def _user(self, message, raw_ref):
        content, parent = message.content, message.parent_tool_use_id
        if isinstance(content, list):
            results = [block for block in content if "ToolResultBlock" in _names(block)]
            if results:
                ids, unmatched = [], []
                for block in results:
                    (ids if self._tool_result(block, parent, raw_ref) else unmatched).append(block.tool_use_id)
                return Projected("tool_result", dict(tool_use_ids=ids, unmatched=unmatched))
        text = _plain_user_text(content)
        if text is not None and parent is None:
            for index, (pending, item_id, steer) in enumerate(self._pending_echo):
                if pending == text or pending.strip() == text.strip():
                    del self._pending_echo[index]
                    if steer:
                        self.transcript.steer(text, item_id=item_id, native_id=message.uuid, raw_ref=raw_ref)
                    else:
                        self.transcript.user_message(text, item_id=item_id, native_id=message.uuid, raw_ref=raw_ref)
                    return Projected("steer_echo" if steer else "prompt_echo", dict(item_id=item_id))
        return Projected("user")

    def _tool_result(self, block, parent, raw_ref):
        """Complete the row of ``block.tool_use_id``; False when this projector never saw the tool_use."""
        record = self.tools.get(block.tool_use_id)
        if record is None:
            return False
        failed = bool(block.is_error)
        text = _text_of(block.content)
        status = "failed" if failed else "completed"
        row_id = block.tool_use_id
        common = dict(status=status, message_id=record["message_id"], native_id=row_id,
                      parent_item_id=record["parent"], raw_ref=raw_ref)
        name = record["name"]
        if record["registry"]:
            if not self._annotate(row_id, message_id=record["message_id"], native_id=row_id,
                                  parent_item_id=record["parent"]) and failed:
                self.transcript.native_tool(row_id, self.prefix + name, record["arguments"], error=text, **common)
        elif name == "Bash":
            self.transcript.command(row_id, record["arguments"].get("command"), aggregated_output=text, **common)
        elif name in _FILE_TOOLS:
            self.transcript.file_change(row_id, self._changes(name, record["arguments"]), **common)
        else:
            self.transcript.native_tool(row_id, name, record["arguments"], result=text,
                                        error=text if failed else None, **common)
        return True
