"""Projects OpenCode 1.19.5 messages and parts onto the neutral transcript (design 04 section 5.1, plan 07 sections
3.4 and 3.7, spike 06 S7/S8/S9).

Two inputs, one result:

* ``sync(messages)`` takes a full ``GET /session/{id}/message`` snapshot and upserts every row it implies. It is the
  authoritative path: ``/event`` has no replay, so after any gap (and before every turn verdict) the session
  re-projects the whole snapshot. Re-projection is idempotent: row ids are native part/message ids, unchanged rows
  are not rewritten, and nothing is appended.
* ``project_event(frame)`` applies one ``/event`` frame (``message.updated``, ``message.part.updated``,
  ``message.part.delta``) for live display. It is advisory; the next ``sync`` overwrites whatever it wrote.

The projector never decides turn state (``verdict.turn_verdict`` does) and never writes usage (the relay is the
accounting source). Like ``claude_code.projector`` it returns a ``Projected(kind, data)`` record the session switches
on. Mapping (every row carries ``backend``; ``native_id`` is the OpenCode/provider id; ``message_id`` is the assistant
message id, so the final-answer rule sees the tool calls of the same message):

* assistant ``text`` part        -> ``agentMessage`` (row id = part id; ``inProgress`` until the part or its message
  finished; the writer stamps ``phase`` at turn end);
* ``reasoning`` part             -> ``reasoning`` (text only; provider metadata/signatures are not stored);
* tool ``bash``                  -> ``commandExecution`` (command, output, ``metadata.exit`` as exit code);
* tool ``edit``/``write``/``apply_patch`` -> ``fileChange``; every other native tool (``read``, ``glob``, ``grep``,
  ...) -> generic ``mcpToolCall`` (tool, arguments, result, error). Row id = part id, ``native_id`` = ``callID``;
* registry tools ``auriga_*``    -> NO row of their own: the MCP transport wrote the ``dynamicToolCall`` row under the
  host call id (OpenCode sends no call id over MCP). The projector attaches ``native_id`` (the part's ``callID``) and
  ``message_id`` to the first unmatched transport row of the current turn with the same tool name and canonical
  arguments (best effort, never required for correctness). A registry part that ended in ``error`` while no transport
  row of that tool is unmatched (OpenCode refused the call before MCP) becomes one failed generic row under the part
  id, named as the model saw it;
* a native tool part still ``pending``/``running`` in a message that is not the last assistant message of the
  snapshot (the step died with the server, S8) -> its row becomes ``failed`` with ``interrupted: true`` (final; never
  reopened); registry rows are left as the transport wrote them;
* user ``text`` parts            -> the host writes the ``userMessage`` row (``run_turn``/``steer``) BEFORE
  ``prompt_async``; the projector matches a native user message to the first unmatched, non-withdrawn host row of the
  current turn with the same text (FIFO) and attaches ``native_id`` = message id. Unmatched native user text (the
  synthetic "Continue ..." message after compaction, anything else) becomes its own ``userMessage`` row (id = message
  id) with ``source: "native"`` and ``synthetic``;
* ``compaction`` part            -> ``contextCompaction`` (``auto``, ``overflow``); the compaction summary (assistant
  ``summary: true``) is written with ``parent_item_id`` = that compaction row, so it is never final text;
* assistant ``info.error`` other than ``MessageAbortedError`` -> ``error`` row ``<message id>:error`` with
  ``opencode_turn_failed:<name>[:<code>]`` (no provider text); ``retry`` part -> ``error`` row
  ``opencode_retry:<attempt>:<name>[:<code>]``;
* ``step-start``/``step-finish``/``patch``/``snapshot``/``file``/``agent``/``subtask`` parts -> no row.

Turn attribution: a row gets the writer's current turn when it is first written; existing rows keep theirs. Call
``sync`` once in ``start()`` before the first turn, so a resumed session adopts the native history without moving it
into the new turn.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ....core.contracts import canonical
from .verdict import ABORTED_ERROR, error_code

__all__ = ["Projected", "OpenCodeProjector", "TOOL_NAME_PREFIX", "INTERRUPTED_TOOL_ERROR"]

TOOL_NAME_PREFIX = "auriga_"       # model-visible registry tool prefix (MCP server key "auriga" + "_")
INTERRUPTED_TOOL_ERROR = "opencode_tool_interrupted"
_FILE_TOOLS = ("edit", "write", "apply_patch")
_STREAMED = ("text", "reasoning")
_PATCH_FILE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$", re.MULTILINE)
_PATCH_KIND = {"Add": "add", "Update": "update", "Delete": "delete"}


@dataclass
class Projected:
    """What ``sync``/``project_event`` found: ``kind`` plus facts for the session.

    kinds (``data`` keys): ``snapshot`` (messages, last_user_message_id, pending_submissions, compactions,
    unmatched_registry, interrupted_tools); ``message`` (message_id, role, finish, error); ``part`` (part_id, type,
    message_id); ``compaction`` (part_id, message_id, auto, overflow, new); ``delta`` (part_id, applied);
    ``deferred`` (part_id: its message info is unknown, the next ``sync`` writes it); ``compacted`` ();
    ``ignored`` (another session); ``other`` (type).
    """
    kind: str
    data: dict = field(default_factory=dict)


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _same_text(a: str, b: str) -> bool:
    return a == b or a.strip() == b.strip()


def _canonical(arguments):
    try:
        return canonical(dict(arguments) if isinstance(arguments, Mapping) else {})
    except (TypeError, ValueError):
        return None


def _row_text(row) -> str:
    return "".join(_text(item.get("text")) for item in row.get("content") or () if isinstance(item, Mapping))


class _Rows:
    """This thread's ``userMessage``/``dynamicToolCall`` rows (the only ones the projector looks up) in store order,
    read once on first use (one query; ``contentItems`` - tool images - are not decoded) and kept current by
    the projector's own annotations."""

    KINDS = ("userMessage", "dynamicToolCall")

    def __init__(self, store, thread_id):
        self.store, self.thread_id, self._rows = store, thread_id, None

    @property
    def rows(self):
        if self._rows is None:
            with self.store.lock:
                bodies = self.store.db.execute(
                    "SELECT json_remove(body, '$.contentItems') FROM state WHERE kind=? "
                    "AND json_extract(body,'$.thread_id')=? AND json_extract(body,'$.type') IN (?,?) ORDER BY rowid",
                    (self.store.TRANSCRIPT_KIND, self.thread_id, *self.KINDS)).fetchall()
            self._rows = [json.loads(body) for body, in bodies]
        return self._rows

    def find(self, kind, native_id):
        return next((row for row in self.rows if row.get("type") == kind and row.get("native_id") == native_id), None)

    def put(self, row):
        if row is None or self._rows is None or row.get("type") not in self.KINDS:
            return
        for index, existing in enumerate(self._rows):
            if existing["id"] == row["id"]:
                self._rows[index] = row
                return
        self._rows.append(row)


class OpenCodeProjector:
    """Message/part-to-transcript projector of one OpenCode session (one ``TranscriptWriter`` whose ``thread_id`` is the
    OpenCode session id). Single consumer: the session calls ``sync``/``project_event`` from one thread at a time; the
    writer itself is thread-safe.

    transcript: the session's ``TranscriptWriter``. prefix: model-visible registry tool prefix (default ``auriga_``).
    """

    def __init__(self, transcript, *, prefix=TOOL_NAME_PREFIX):
        self.transcript, self.prefix = transcript, prefix
        self.store = transcript.store
        self._infos = {}           # message id -> latest info seen
        self._user_parts = {}      # user message id -> {part id: part} (event path)
        self._compactions = {}     # compaction trigger message id -> compaction part id
        self._streams = {}         # part id -> streamed part (event path: text/reasoning accumulated from deltas)
        self._registry = {}        # callID -> transport row id (or the failed generic row id)
        self._users = {}           # native user message id -> userMessage row id
        self._native_rows = set()  # native tool row ids written by this projector

    # ----- host-side helpers -----

    def withdraw_submission(self, item_id):
        """Mark a host ``userMessage`` row whose ``prompt_async`` was definitely rejected (4xx): it is never matched to
        a native message and is not a pending submission. Returns False when the row does not exist."""
        return self._annotate(item_id, withdrawn=True)

    def submissions(self, turn_id=None):
        """``(last_user_message_id, pending)`` of ``turn_id`` (default: the writer's current turn): the native id of the
        latest host prompt/steer row of that turn (None while it is unmatched) and how many of its host rows are still
        unmatched. ``last_user_message_id`` is the anchor ``verdict.turn_verdict`` needs. The session should re-read
        it under the lock its ``steer()`` takes before finishing a turn, so a steer written after the snapshot keeps the
        turn running."""
        return self._submissions(_Rows(self.store, self.transcript.thread_id), turn_id)

    def _submissions(self, rows, turn_id=None):
        turn = turn_id or self.transcript.turn_id
        hosts = [row for row in rows.rows if row.get("type") == "userMessage" and row.get("turn_id") == turn
                 and row.get("source") != "native" and not row.get("withdrawn")]
        pending = sum(1 for row in hosts if not row.get("native_id"))
        return (hosts[-1].get("native_id") if hosts else None), pending

    def abandon_open_items(self, status="failed"):
        """Set every native tool row this projector wrote that is still ``inProgress`` to ``status`` (``failed`` by
        default) with ``interrupted: true``; returns the row ids changed. Registry rows are never touched."""
        changed = []
        with self.transcript.lock:
            for row_id in sorted(self._native_rows):
                row = self.store.transcript_item(row_id, None)
                if row and row.get("status") == "inProgress":
                    self.store.set_transcript_item(row_id, dict(row, status=status, interrupted=True,
                                                                error=dict(message=INTERRUPTED_TOOL_ERROR)))
                    changed.append(row_id)
        return changed

    # ----- snapshot -----

    def sync(self, messages: Sequence) -> Projected:
        """Idempotently upsert all rows implied by one ``GET /session/{id}/message`` snapshot (creation order).

        Returns ``Projected("snapshot", ...)``: ``last_user_message_id``/``pending_submissions`` (``submissions()``),
        ``compactions`` (compaction part ids written for the first time), ``unmatched_registry`` (callIDs of registry
        tool parts with no transport row yet), ``interrupted_tools`` (native rows marked interrupted in this sync)."""
        thread = self.transcript.thread_id
        messages = [m for m in messages or () if isinstance(m, Mapping) and isinstance(m.get("info"), Mapping)
                    and m["info"].get("sessionID", thread) == thread]
        rows = _Rows(self.store, thread)
        facts = dict(compactions=[], unmatched_registry=[], interrupted_tools=[])
        last_assistant = max((i for i, m in enumerate(messages) if m["info"].get("role") == "assistant"), default=-1)
        for message in messages:
            for part in message.get("parts") or ():
                if isinstance(part, Mapping) and part.get("type") == "compaction":
                    self._compactions[message["info"].get("id")] = part.get("id")
        for index, message in enumerate(messages):
            info = message["info"]
            self._infos[info.get("id")] = info
            parts = [part for part in message.get("parts") or () if isinstance(part, Mapping)]
            if info.get("role") == "user":
                self._user_message(info, parts, rows)
            elif info.get("role") == "assistant":
                self._assistant_error(info)
            for part in parts:
                self._part(part, info, rows, facts, superseded=index < last_assistant)
        last, pending = self.submissions()               # fresh read: a steer row may have been written meanwhile
        return Projected("snapshot", dict(messages=len(messages), last_user_message_id=last,
                                          pending_submissions=pending, **facts))

    # ----- events -----

    def project_event(self, frame) -> Projected:
        """Apply one ``/event`` frame (advisory; ``sync`` stays authoritative). Never raises for unknown frames."""
        kind = frame.get("type") if isinstance(frame, Mapping) else None
        props = _mapping(frame.get("properties")) if isinstance(frame, Mapping) else {}
        session = props.get("sessionID") or _mapping(props.get("info")).get("sessionID") or _mapping(
            props.get("part")).get("sessionID")
        if session and session != self.transcript.thread_id:
            return Projected("ignored", dict(type=kind))
        if kind == "message.updated" and isinstance(props.get("info"), Mapping):
            info = props["info"]
            self._infos[info.get("id")] = info
            if info.get("role") == "assistant":
                self._assistant_error(info)
            return Projected("message", dict(message_id=info.get("id"), role=info.get("role"),
                                             finish=info.get("finish"), error=_mapping(info.get("error")).get("name")))
        if kind == "message.part.updated" and isinstance(props.get("part"), Mapping):
            return self._event_part(dict(props["part"]))
        if kind == "message.part.delta":
            return self._event_delta(props)
        if kind == "session.compacted":
            return Projected("compacted")
        return Projected("other", dict(type=kind))

    def _event_part(self, part):
        part_id, message_id = part.get("id"), part.get("messageID")
        info = self._infos.get(message_id)
        if info is None or not part_id:
            return Projected("deferred", dict(part_id=part_id))
        rows = _Rows(self.store, self.transcript.thread_id)
        facts = dict(compactions=[], unmatched_registry=[], interrupted_tools=[])
        if part.get("type") == "compaction":
            self._compactions[message_id] = part_id
        if info.get("role") == "user":
            self._user_parts.setdefault(message_id, {})[part_id] = part
            self._user_message(info, list(self._user_parts[message_id].values()), rows)
        if part.get("type") in _STREAMED:                # deltas extend it until the part reports its end
            if _mapping(part.get("time")).get("end"):
                self._streams.pop(part_id, None)
            else:
                self._streams[part_id] = part
        self._part(part, info, rows, facts, superseded=False)
        if part.get("type") == "compaction":
            return Projected("compaction", dict(part_id=part_id, message_id=message_id, auto=part.get("auto"),
                                                overflow=part.get("overflow"), new=bool(facts["compactions"])))
        return Projected("part", dict(part_id=part_id, type=part.get("type"), message_id=message_id))

    def _event_delta(self, props):
        part_id, delta = props.get("partID"), props.get("delta")
        part = self._streams.get(part_id)
        info = self._infos.get(props.get("messageID"))
        if props.get("field") != "text" or not isinstance(delta, str) or part is None or info is None:
            return Projected("delta", dict(part_id=part_id, applied=False))
        part = dict(part, text=_text(part.get("text")) + delta)
        self._streams[part_id] = part
        self._part(part, info, _Rows(self.store, self.transcript.thread_id),
                   dict(compactions=[], unmatched_registry=[], interrupted_tools=[]), superseded=False)
        return Projected("delta", dict(part_id=part_id, applied=True))

    # ----- rows -----

    def _annotate(self, row_id, rows=None, **fields):
        """Add store-only fields to an existing row (``None`` values are skipped); False when the row does not exist."""
        with self.transcript.lock:
            row = self.store.transcript_item(row_id, None)
            if not row:
                return False
            update = {key: value for key, value in fields.items() if value is not None and row.get(key) != value}
            if update:
                row = dict(row, **update)
                self.store.set_transcript_item(row_id, row)
        if rows is not None:
            rows.put(row)
        return True

    def _assistant_error(self, info):
        error = _mapping(info.get("error"))
        if error and error.get("name") != ABORTED_ERROR and info.get("id"):
            self.transcript.error(error_code(error), item_id=f"{info['id']}:error", native_id=info["id"])

    @staticmethod
    def _finished(part, info):
        return bool(_mapping(part.get("time")).get("end") or _mapping(info.get("time")).get("completed")
                    or info.get("finish") or info.get("error"))

    def _part(self, part, info, rows, facts, *, superseded):
        kind, part_id, role = part.get("type"), part.get("id"), info.get("role")
        if not part_id:
            return
        if kind in _STREAMED and role == "assistant":
            text = _text(part.get("text"))
            if not text.strip() and self.store.transcript_item(part_id, None) is None:
                return                                  # empty streamed part: no row until it has text
            parent = None
            if info.get("summary") is True:     # compaction summary: child of the compaction row, never final text
                parent = self._compactions.get(info.get("parentID")) or info.get("parentID")
            common = dict(message_id=info.get("id"), item_id=part_id, native_id=part_id, parent_item_id=parent)
            if kind == "text":
                self.transcript.assistant_text(text, status="completed" if self._finished(part, info) else
                                               "inProgress", **common)
            else:
                self.transcript.reasoning(text, **common)
        elif kind == "tool" and role == "assistant":
            if _text(part.get("tool")).startswith(self.prefix):
                self._registry_part(part, info, rows, facts)
            else:
                self._native_tool(part, info, facts, superseded=superseded)
        elif kind == "compaction":
            new = self.store.transcript_item(part_id, None) is None
            self.transcript.compaction(item_id=part_id, native_id=part_id)
            self._annotate(part_id, auto=part.get("auto"), overflow=part.get("overflow"))
            if new:
                facts["compactions"].append(part_id)
        elif kind == "retry":
            code = error_code(_mapping(part.get("error")))[len("opencode_turn_failed:"):]
            self.transcript.error(f"opencode_retry:{part.get('attempt')}:{code}", item_id=part_id, native_id=part_id)

    # ----- tools -----

    def _native_tool(self, part, info, facts, *, superseded):
        part_id, name = part["id"], _text(part.get("tool"))
        existing = self.store.transcript_item(part_id, None)
        if existing and existing.get("interrupted"):
            return                                       # an interrupted row is final
        state = _mapping(part.get("state"))
        native, arguments, metadata = state.get("status"), _mapping(state.get("input")), _mapping(state.get("metadata"))
        open_state = native in ("pending", "running")
        interrupted = open_state and superseded
        status = "failed" if interrupted or native == "error" else "inProgress" if open_state else "completed"
        common = dict(status=status, message_id=info.get("id"), native_id=part.get("callID") or None)
        output = state.get("output") if isinstance(state.get("output"), str) else None
        error = state.get("error") if isinstance(state.get("error"), str) else None
        if name == "bash":
            exit_code = metadata.get("exit")
            live = metadata.get("output") if isinstance(metadata.get("output"), str) else None
            self.transcript.command(part_id, arguments.get("command"), aggregated_output=output or live,
                                    exit_code=exit_code if type(exit_code) is int else None, **common)
        elif name in _FILE_TOOLS:
            self.transcript.file_change(part_id, self._changes(name, arguments, metadata), **common)
        else:
            self.transcript.native_tool(part_id, name, dict(arguments), result=output, error=error, **common)
        self._native_rows.add(part_id)
        if interrupted:
            self._annotate(part_id, interrupted=True, error=dict(message=INTERRUPTED_TOOL_ERROR))
            facts["interrupted_tools"].append(part_id)
        elif error is not None and name in ("bash", *_FILE_TOOLS):
            self._annotate(part_id, error=dict(message=error))

    @staticmethod
    def _changes(name, arguments, metadata):
        """``fileChange.changes`` (``[{path, kind, diff}]``) from the tool input and, once finished, its metadata."""
        if name == "apply_patch":
            files = metadata.get("files")
            if isinstance(files, list) and files:
                return [dict(path=f.get("filePath") or f.get("relativePath"), kind=f.get("type"), diff=f.get("patch"))
                        for f in files if isinstance(f, Mapping)]
            return [dict(path=path.strip(), kind=_PATCH_KIND[kind], diff=None)
                    for kind, path in _PATCH_FILE.findall(_text(arguments.get("patchText")))]
        diff = metadata.get("diff") if isinstance(metadata.get("diff"), str) else None
        return [dict(path=arguments.get("filePath"), kind="update" if name == "edit" else "write", diff=diff)]

    def _registry_part(self, part, info, rows, facts):
        call_id, part_id = part.get("callID") or None, part["id"]
        state = _mapping(part.get("state"))
        name = _text(part.get("tool"))[len(self.prefix):]
        if call_id in self._registry:
            return                                       # matched and annotated earlier (rows keep their linkage)
        if state.get("status") not in ("running", "completed", "error"):
            return                                       # pending: no input yet, nothing to match
        row_id = None
        if call_id:
            found = rows.find("dynamicToolCall", call_id)
            row_id = found["id"] if found else None
        if row_id is None and self.store.transcript_item(part_id, None) is not None:
            row_id = part_id                             # failed generic row written earlier
        if row_id is None:
            turn = self.transcript.turn_id
            scope = [row for row in rows.rows if row.get("type") == "dynamicToolCall" and not row.get("native_id")
                     and row.get("tool") == name and (turn is None or row.get("turn_id") == turn)]
            key = _canonical(state.get("input"))
            match = next((row for row in scope if key is not None and _canonical(row.get("arguments")) == key), None)
            if match is not None:
                row_id = match["id"]
            elif state.get("status") == "error" and not scope:
                self.transcript.native_tool(part_id, self.prefix + name, dict(_mapping(state.get("input"))),
                                            error=_text(state.get("error")) or None, status="failed",
                                            message_id=info.get("id"), native_id=call_id)
                rows.put(self.store.transcript_item(part_id))
                row_id = part_id
        if row_id is None:
            facts["unmatched_registry"].append(call_id)
            return
        if call_id:
            self._registry[call_id] = row_id
        self._annotate(row_id, rows, native_id=call_id, message_id=info.get("id"))

    # ----- users -----

    def _user_message(self, info, parts, rows):
        message_id = info.get("id")
        texts = [part for part in parts if part.get("type") == "text" and isinstance(part.get("text"), str)]
        if not message_id or not texts or message_id in self._users:
            return
        found = rows.find("userMessage", message_id)
        if found is not None:
            self._users[message_id] = found["id"]
            return
        real = [part["text"] for part in texts if not part.get("synthetic")]
        text = "".join(real) if real else "".join(part["text"] for part in texts)
        if real:
            turn = self.transcript.turn_id
            for row in rows.rows:
                if (row.get("type") == "userMessage" and not row.get("native_id") and row.get("source") != "native"
                        and not row.get("withdrawn") and (turn is None or row.get("turn_id") == turn)
                        and _same_text(_row_text(row), text)):
                    self._annotate(row["id"], rows, native_id=message_id)
                    self._users[message_id] = row["id"]
                    return
        self.transcript.user_message(text, item_id=message_id, native_id=message_id)
        self._annotate(message_id, rows, source="native", synthetic=not real)
        self._users[message_id] = message_id
