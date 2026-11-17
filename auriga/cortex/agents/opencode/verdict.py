"""Pure turn verdict for OpenCode 1.19.5 (design 04 section 2.2 OpenCode row, plan 07 section 3.4, spike S7.4/S8).

OpenCode has no turn id and its ``/event`` stream has no replay, so the end of an Auriga turn is decided from an
authoritative snapshot: ``GET /session/{id}/message`` (``messages``) plus this session's entry of
``GET /session/status`` (``status``). ``turn_verdict`` is a pure function of those two answers and two host facts; it
never performs I/O and never mutates its input. ``IncompleteTurnGuard`` (plan rule 7) turns a verdict that stays
non-terminal while the session is idle into ``failed: opencode_turn_incomplete``.

Rules (mirroring OpenCode's own loop exit in ``session/prompt.ts``: the loop stops once the latest assistant message
answers the latest user message with a finish other than ``tool-calls``/``unknown`` and carries no pending tool call):

1. ``status`` busy or retry (anything but idle) -> ``running`` (``busy``). OpenCode keeps a session busy for the whole
   loop, including steers picked up at a step boundary and the extra model call that answers a late steer.
2. The anchor ``last_user_message_id`` (latest prompt/steer the host submitted in this turn) must be in the snapshot,
   else ``running`` (``user_message_pending``): ``prompt_async`` answers 204 before the message exists.
3. ``last_user`` = the latest user message at or after the anchor (later ones are steers whose id the host has not
   learned yet, the compaction trigger message, or the synthetic "Continue ..." message written after compaction).
4. ``answer`` = the latest assistant message whose ``parentID`` is ``last_user``; none -> ``running``
   (``awaiting_answer``). Earlier assistant messages of the same busy period (e.g. a final text that a steer
   superseded) are commentary.
5. ``interrupt_requested`` (the host sent ``POST /abort``) -> ``interrupted`` once the session is idle, whatever the
   answer shows (an APIError caused by the relay cancelling the pending model call, or an answer that completed just
   before the abort landed). Same convention as ``claude_code.projector.classify_result``.
6. ``answer.error.name == "MessageAbortedError"`` or a tool part ``error == "Tool execution aborted"`` ->
   ``interrupted`` (``aborted``). Any other ``answer.error`` -> ``failed`` with
   ``"opencode_turn_failed:<error.name>[:<data.statusCode>]"`` (no provider text).
7. ``finish`` null/``tool-calls``/``unknown``, or a tool part that is not an interrupted orphan -> ``running``
   (``answer_unfinished``; still looping, or a step that died with the server - S8). A compaction summary
   (``info.summary == true``) is not an answer to the operator -> ``running`` (``compaction_summary``).
8. Otherwise ``completed``; ``final_text`` = the answer's non-empty, non-synthetic text parts joined by ``"\\n"``
   (None when there is none, e.g. ``finish: "length"`` with empty output), the same join as
   ``AgentSession.completed_final_answer``.

Captured ordering the caller must respect (S7 D/E, S6 stream error): after an abort or a provider failure OpenCode
emits ``session.error``, ``session.status idle`` and the FIRST ``session.idle`` BEFORE it writes the tool part error
and the message error; a second idle follows the write. A reconcile at the first idle therefore sees
``answer_unfinished`` (unless ``interrupt_requested``); the caller must reconcile again on the later idle (do not
drop it as a duplicate), and ``IncompleteTurnGuard`` only fires after ``min_gap_s``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = ["TurnVerdict", "turn_verdict", "status_type", "final_text_of", "error_code", "IncompleteTurnGuard",
           "ABORTED_ERROR", "TOOL_ABORTED_TEXT", "UNFINISHED_FINISH", "INCOMPLETE_ERROR", "INCOMPLETE_MIN_GAP_S"]

ABORTED_ERROR = "MessageAbortedError"          # assistant ``info.error.name`` after POST /abort
TOOL_ABORTED_TEXT = "Tool execution aborted"   # tool part ``state.error`` of a call cut by POST /abort
UNFINISHED_FINISH = frozenset({None, "tool-calls", "unknown"})   # OpenCode keeps looping on these
INCOMPLETE_ERROR = "opencode_turn_incomplete"
INCOMPLETE_MIN_GAP_S = 2.0     # plan rule 7: two idle observations at least this far apart (s)


@dataclass(frozen=True)
class TurnVerdict:
    """Outcome of one reconciliation.

    status: ``completed`` | ``interrupted`` | ``failed`` | ``running`` (not terminal yet).
    final_text: final answer text (``completed`` only; None when the answer has no text).
    error: stable error string for ``failed`` (no provider text), else None.
    message_id: id of the assistant message the verdict is based on (None when there is none).
    reason: short code why: running -> ``busy`` | ``user_message_pending`` | ``awaiting_answer`` |
        ``answer_unfinished`` | ``compaction_summary``; terminal -> ``answered`` | ``aborted`` |
        ``interrupt_requested`` | ``error`` | ``incomplete``.
    idle: True when the session status was idle (only idle non-terminal verdicts feed ``IncompleteTurnGuard``).
    """
    status: str
    final_text: str | None = None
    error: str | None = None
    message_id: str | None = None
    reason: str | None = None
    idle: bool = False

    @property
    def terminal(self) -> bool:
        return self.status != "running"


def _info(message) -> dict:
    info = message.get("info") if isinstance(message, Mapping) else None
    return info if isinstance(info, Mapping) else {}


def _parts(message) -> list:
    parts = message.get("parts") if isinstance(message, Mapping) else None
    return [part for part in parts if isinstance(part, Mapping)] if isinstance(parts, list) else []


def status_type(status, session_id: str | None = None) -> str:
    """Normalise the status argument to ``"idle"`` / ``"busy"`` / ``"retry"`` / other native type.

    Accepts this session's entry of ``GET /session/status`` (``{"type": "busy"}``), ``None`` (absent from the map =
    idle), a type string, or the whole map (``{}`` = all idle; looked up by ``session_id``)."""
    if status is None:
        return "idle"
    if isinstance(status, str):
        return status or "idle"
    if isinstance(status, Mapping):
        if "type" in status:
            return str(status.get("type") or "idle")
        entry = status.get(session_id) if session_id else None
        if isinstance(entry, Mapping):
            return str(entry.get("type") or "idle")
        return "idle"
    raise TypeError("opencode_status_must_be_mapping_or_str")


def final_text_of(message) -> str | None:
    """Joined non-empty text parts of one message (synthetic/ignored parts excluded); None when there is none."""
    texts = [part["text"] for part in _parts(message)
             if part.get("type") == "text" and isinstance(part.get("text"), str) and part["text"].strip()
             and not part.get("synthetic") and not part.get("ignored")]
    return "\n".join(texts) if texts else None


def error_code(error: Mapping) -> str:
    """``"opencode_turn_failed:<name>[:<data.statusCode>]"`` of an assistant ``info.error`` (never provider text)."""
    error = error if isinstance(error, Mapping) else {}
    data = error.get("data") if isinstance(error.get("data"), Mapping) else {}
    code = data.get("statusCode")
    suffix = f":{code}" if isinstance(code, int) and not isinstance(code, bool) else ""
    return "opencode_turn_failed:" + str(error.get("name") or "unknown") + suffix


def _pending_tool(part: Mapping) -> bool:
    """A tool part OpenCode would still act on (its ``hasToolCalls``): any tool part except an interrupted orphan."""
    if part.get("type") != "tool" or (isinstance(part.get("metadata"), Mapping) and part["metadata"].get(
            "providerExecuted")):
        return False
    state = part.get("state") if isinstance(part.get("state"), Mapping) else {}
    metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
    return not (state.get("status") == "error" and metadata.get("interrupted") is True)


def turn_verdict(messages: Sequence, status, *, last_user_message_id: str | None,
                 interrupt_requested: bool = False) -> TurnVerdict:
    """Decide the current turn from one snapshot (see the module rules).

    messages: ``GET /session/{id}/message`` answer (``[{"info": {...}, "parts": [...]}]``, creation order).
    status: this session's ``GET /session/status`` entry, None, a type string, or the whole map (``status_type``).
    last_user_message_id: OpenCode id of the latest user message the host submitted in this turn (prompt or steer),
        None while that submission is not visible yet (``OpenCodeProjector.sync`` reports it).
    interrupt_requested: True once the host requested an interrupt of this turn.
    """
    messages = [message for message in messages or () if _info(message)]
    session_id = next((_info(m).get("sessionID") for m in messages if _info(m).get("sessionID")), None)
    if status_type(status, session_id) != "idle":
        return TurnVerdict("running", reason="busy")
    ids = [_info(message).get("id") for message in messages]
    if last_user_message_id is None or last_user_message_id not in ids:
        return TurnVerdict("running", reason="user_message_pending", idle=True)
    anchor = ids.index(last_user_message_id)
    users = [message for message in messages[anchor:] if _info(message).get("role") == "user"]
    last_user = _info(users[-1]).get("id") if users else last_user_message_id
    answers = [message for message in messages[anchor:] if _info(message).get("role") == "assistant"
               and _info(message).get("parentID") == last_user]
    answer = answers[-1] if answers else None
    message_id = _info(answer).get("id") if answer else None
    if interrupt_requested:
        return TurnVerdict("interrupted", message_id=message_id, reason="interrupt_requested", idle=True)
    if answer is None:
        return TurnVerdict("running", reason="awaiting_answer", idle=True)
    info, parts = _info(answer), _parts(answer)
    error = info.get("error") if isinstance(info.get("error"), Mapping) else None
    aborted_tool = any(part.get("type") == "tool" and isinstance(part.get("state"), Mapping)
                       and part["state"].get("status") == "error" and part["state"].get("error") == TOOL_ABORTED_TEXT
                       for part in parts)
    if (error and error.get("name") == ABORTED_ERROR) or aborted_tool:
        return TurnVerdict("interrupted", message_id=message_id, reason="aborted", idle=True)
    if error:
        return TurnVerdict("failed", error=error_code(error), message_id=message_id, reason="error", idle=True)
    if info.get("finish") in UNFINISHED_FINISH or any(_pending_tool(part) for part in parts):
        return TurnVerdict("running", message_id=message_id, reason="answer_unfinished", idle=True)
    if info.get("summary") is True:
        return TurnVerdict("running", message_id=message_id, reason="compaction_summary", idle=True)
    return TurnVerdict("completed", final_text=final_text_of(answer), message_id=message_id, reason="answered",
                       idle=True)


class IncompleteTurnGuard:
    """Plan rule 7: a turn whose verdict stays ``running`` while the session is idle is failed after two idle
    observations at least ``min_gap_s`` apart (a prompt that never materialised, a step that died with the server).

    min_gap_s: minimum time (s) between the first and the confirming idle non-terminal observation; default 2.0.
    Observations while busy, and any terminal verdict, reset the guard. Use one guard per turn (or ``reset()``).
    ``observe(verdict, now, interrupt_requested=False)`` returns the verdict unchanged, or the guard verdict:
    ``failed`` / ``opencode_turn_incomplete`` (``interrupted`` when the host requested an interrupt).
    ``now`` is a monotonic time in seconds supplied by the caller (the guard never reads a clock).
    """

    def __init__(self, *, min_gap_s: float = INCOMPLETE_MIN_GAP_S):
        if not min_gap_s >= 0:
            raise ValueError("min_gap_s_must_be_non_negative")
        self.min_gap_s = float(min_gap_s)
        self.first_idle_at: float | None = None

    def reset(self) -> None:
        self.first_idle_at = None

    def observe(self, verdict: TurnVerdict, now: float, *, interrupt_requested: bool = False) -> TurnVerdict:
        if verdict.terminal or not verdict.idle:
            self.first_idle_at = None
            return verdict
        if self.first_idle_at is None:
            self.first_idle_at = now
            return verdict
        if now - self.first_idle_at < self.min_gap_s:
            return verdict
        if interrupt_requested:
            return TurnVerdict("interrupted", message_id=verdict.message_id, reason="interrupt_requested", idle=True)
        return TurnVerdict("failed", error=INCOMPLETE_ERROR, message_id=verdict.message_id, reason="incomplete",
                           idle=True)
