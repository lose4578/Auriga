"""Backend-neutral readers of native agent evidence for audit and report.

Typed run facts still come only from the Store; these readers extract what the
agent runtime itself proves it did (tool items, images returned to the model,
turn terminals, prompts submitted, host interrupts).

Codex archives (default, every archive without ``manifest["agent_backend"]``):
``codex-raw.jsonl`` JSON-RPC records ``{received_at, direction, message}``.

Agent-backend archives (``claude_code``, ``opencode``): ``agent-raw.jsonl`` records
``{received_at, direction, channel, message}`` written by ``Store.raw_agent``.
``direction`` is ``sent`` (host -> agent) or ``received`` (agent -> host).
Conventions the readers rely on:

* ``mcp`` channel, MCP JSON-RPC.  A ``tools/call`` request (``method``, ``id``,
  ``params``) is paired with the response carrying the same ``id`` and a ``result``.
  The tool item id is ``params["_meta"]["item_id"]`` when present, else ``str(id)``.
  Image content ``{"type": "image", "data": <base64>, "mimeType": ...}`` in
  ``result.content`` is hashed (sha256 of the decoded bytes).
* ``control`` channel.  ``{"command": "submit_prompt", "thread_id", "turn_id",
  "input": [{"type": "text", "text": ...}]}`` and ``{"command": "interrupt",
  "thread_id", "turn_id"}``.
* Store events ``agent.turn_started`` / ``agent.turn_completed`` (correlation
  ``thread_id`` and ``turn_id``; completed payload ``status``).
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path

CODEX_KINDS = ("codex",)
AGENT_KINDS = ("claude_code", "opencode")


def parse_timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


@dataclass
class NativeEvidence:
    """Everything is restricted to the main thread (``manifest["thread_id"]``) unless stated.

    tool_item_ids: ids proving a native tool call existed (any thread; Codex may include None
        for an ``item/started`` without id, kept verbatim from the historical audit).
    sent_image_sha256: sha256 of images actually returned to the agent in tool results.
    turn_terminals: ``{id, status}`` per terminal turn record, in log order.
    turn_started_at / turn_completed: per turn id, the native start time and
        ``(status, completed_at)`` (last record wins); times are aware datetimes or None.
    interrupt_requests: host-initiated interrupts ``{turn_id, sent_at}`` in log order.
    interrupted_completion_seq: turn id -> Store ``server_seq`` of the event recording that
        turn's interrupted completion (backs the terminal-interrupt ownership check).
    submitted_prompts: one list of text inputs per native prompt submission, in order.
    completed_turn_texts: assistant texts of the terminal turns (continuity marker); only
        filled when requested (``require_texts``).
    log_present: whether the raw log file exists.
    """
    thread_id: object = None
    log_present: bool = False
    tool_item_ids: set = field(default_factory=set)
    sent_image_sha256: set = field(default_factory=set)
    turn_terminals: list = field(default_factory=list)
    turn_started_at: dict = field(default_factory=dict)
    turn_completed: dict = field(default_factory=dict)
    interrupt_requests: list = field(default_factory=list)
    interrupted_completion_seq: dict = field(default_factory=dict)
    submitted_prompts: list = field(default_factory=list)
    completed_turn_texts: list = field(default_factory=list)


def _event_pages(store):
    cursor = 0
    while page := store.events(after=cursor):
        yield from page
        next_cursor = page[-1]["server_seq"]
        if next_cursor <= cursor:
            raise ValueError("nonadvancing_event_cursor")
        cursor = next_cursor


def codex_turn_timeline(raw_records, events, store, thread_id, evidence=None):
    """Fill turn timing, interrupt requests and interrupted-completion event seqs from Codex records.

    Lenient on purpose (skips malformed records), exactly the historical
    ``_owned_terminal_interrupts`` extraction.  ``events``/``store`` may be None to skip
    the Store-event part.
    """
    evidence = evidence if evidence is not None else NativeEvidence(thread_id=thread_id)
    for record in raw_records:
        if not isinstance(record, dict) or not isinstance(record.get("message"), dict):
            continue
        message = record["message"]
        params = message.get("params")
        if not isinstance(params, dict):
            continue
        received_at = parse_timestamp(record.get("received_at"))
        if record.get("direction") == "sent" and message.get("method") == "turn/interrupt":
            if params.get("threadId") == thread_id:
                evidence.interrupt_requests.append(dict(turn_id=params.get("turnId"), sent_at=received_at))
        elif record.get("direction") == "received" and message.get("method") in ("turn/started", "turn/completed"):
            turn = params.get("turn")
            if not isinstance(turn, dict) or not turn.get("id"):
                continue
            if params.get("threadId") != thread_id:
                continue
            if message["method"] == "turn/started":
                evidence.turn_started_at[turn["id"]] = received_at
            else:
                evidence.turn_completed[turn["id"]] = (turn.get("status"), received_at)
    if store is not None and events is not None:
        for event in events:
            if not isinstance(event, dict):
                continue
            correlation = event.get("correlation")
            if (event.get("type") != "codex.turn/completed" or event.get("source") != "codex_app_server"
                    or not isinstance(correlation, dict) or correlation.get("thread_id") != thread_id):
                continue
            payload = event.get("payload")
            reference = payload.get("raw_ref") if isinstance(payload, dict) else None
            if not reference:
                continue
            try:
                _, content = store.resolve_artifact(reference, allowed_scopes=("observer",))
                message = json.loads(content)
            except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            params = message.get("params") if isinstance(message, dict) else None
            turn = params.get("turn") if isinstance(params, dict) else None
            if (isinstance(params, dict) and params.get("threadId") == thread_id
                    and isinstance(turn, dict) and turn.get("status") == "interrupted" and turn.get("id")):
                evidence.interrupted_completion_seq[turn["id"]] = event.get("server_seq")
    return evidence


class CodexRawEvidence:
    """Parses ``codex-raw.jsonl`` (moved verbatim from audit.py and report.py).

    profile "audit": log required; envelope and params validated; full extraction.
    profile "report": log optional; direction and ``turn/start`` validated; only
    ``submitted_prompts`` and ``log_present`` are filled.
    """

    def __init__(self, root, manifest, store=None, *, profile="audit", events=None, require_texts=False):
        self.root, self.manifest, self.store = Path(root), manifest, store
        self.profile, self.events, self.require_texts = profile, events, require_texts

    def load(self):
        path = self.root / "codex-raw.jsonl"
        thread_id = self.manifest.get("thread_id")
        evidence = NativeEvidence(thread_id=thread_id, log_present=path.exists())
        if self.profile == "report":
            self._report(path, thread_id, evidence)
        else:
            self._audit(path, thread_id, evidence)
        return evidence

    def _report(self, path, thread_id, evidence):
        if not path.exists():
            return
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if (not isinstance(record, dict) or not isinstance(record.get("message"), dict)
                        or record.get("direction") not in ("sent", "received")):
                    raise ValueError("invalid_native_rpc_record")
                message = record["message"]
                if "params" in message and not isinstance(message["params"], dict):
                    raise ValueError("invalid_native_rpc_record")
                if record["direction"] != "sent" or message.get("method") != "turn/start":
                    continue
                params = message.get("params", {})
                inputs = params.get("input")
                if (not isinstance(params.get("threadId"), str) or not params["threadId"]
                        or not isinstance(inputs, list) or any(not isinstance(item, dict)
                            or not isinstance(item.get("type"), str)
                            or (item["type"] == "text" and not isinstance(item.get("text"), str))
                            for item in inputs)):
                    raise ValueError("invalid_native_turn_start")
                if thread_id and params["threadId"] == thread_id:
                    evidence.submitted_prompts.append([item["text"] for item in inputs if item["type"] == "text"])

    def _audit(self, path, thread_id, evidence):
        raw_records = [json.loads(line) for line in path.read_text().splitlines()]
        if any(not isinstance(record, dict) or not isinstance(record.get("message"), dict)
               or ("params" in record["message"] and not isinstance(record["message"]["params"], dict))
               for record in raw_records):
            raise ValueError("invalid_native_rpc_record")
        raw = [record["message"] for record in raw_records]
        evidence.tool_item_ids = {(m.get("params") or {}).get("item", {}).get("id") for m in raw if m.get("method") == "item/started"}
        for message in raw:
            for content in (message.get("result") or {}).get("contentItems", []):
                if content.get("type") == "inputImage":
                    evidence.sent_image_sha256.add(hashlib.sha256(base64.b64decode(content["imageUrl"].split(",", 1)[1])).hexdigest())
        turns = [m["params"]["turn"] for m in raw if m.get("method") == "turn/completed" and m["params"].get("threadId") == thread_id]
        evidence.turn_terminals = [dict(id=t["id"], status=t["status"]) for t in turns]
        if self.require_texts:
            evidence.completed_turn_texts = [item.get("text", "") for t in turns for item in t.get("items", [])
                                             if item.get("type") == "agentMessage"]
        codex_turn_timeline(raw_records, self.events if self.events is not None or self.store is None else list(_event_pages(self.store)),
                            self.store, thread_id, evidence)


class AgentRawEvidence:
    """Reads ``agent-raw.jsonl`` (mcp/control channels) plus ``agent.turn_*`` Store events."""

    CHANNELS = ("native", "mcp", "control")

    def __init__(self, root, manifest, store=None, *, profile="audit", events=None, require_texts=False):
        self.root, self.manifest, self.store = Path(root), manifest, store
        self.profile, self.events, self.require_texts = profile, events, require_texts

    def _records(self, path):
        if self.profile == "report" and not path.exists():
            return []
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for record in records:
            if (not isinstance(record, dict) or not isinstance(record.get("message"), dict)
                    or record.get("direction") not in ("sent", "received")
                    or record.get("channel") not in self.CHANNELS):
                raise ValueError("invalid_agent_raw_record")
        return records

    def load(self):
        path = self.root / "agent-raw.jsonl"
        thread_id = self.manifest.get("thread_id")
        evidence = NativeEvidence(thread_id=thread_id, log_present=path.exists())
        pending = {}
        for record in self._records(path):
            message, channel = record["message"], record["channel"]
            if channel == "mcp":
                if message.get("method") == "tools/call" and "id" in message:
                    params = message.get("params") if isinstance(message.get("params"), dict) else {}
                    meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
                    pending[message["id"]] = meta.get("item_id") or str(message["id"])
                elif "id" in message and isinstance(message.get("result"), dict) and message["id"] in pending:
                    evidence.tool_item_ids.add(pending[message["id"]])
                    content = message["result"].get("content")
                    for block in content if isinstance(content, list) else []:
                        if isinstance(block, dict) and block.get("type") == "image" and isinstance(block.get("data"), str):
                            evidence.sent_image_sha256.add(hashlib.sha256(base64.b64decode(block["data"])).hexdigest())
            elif channel == "control" and message.get("thread_id") == thread_id:
                command = message.get("command")
                if command == "submit_prompt":
                    inputs = message.get("input")
                    if not isinstance(inputs, list):
                        raise ValueError("invalid_agent_raw_record")
                    evidence.submitted_prompts.append([item["text"] for item in inputs
                        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)])
                elif command == "interrupt":
                    evidence.interrupt_requests.append(dict(turn_id=message.get("turn_id"),
                        sent_at=parse_timestamp(record.get("received_at"))))
        if self.store is not None:
            events = self.events if self.events is not None else list(_event_pages(self.store))
            for event in events:
                correlation = event.get("correlation") if isinstance(event, dict) else None
                if (not isinstance(correlation, dict) or correlation.get("thread_id") != thread_id
                        or not correlation.get("turn_id")
                        or event.get("type") not in ("agent.turn_started", "agent.turn_completed")):
                    continue
                turn_id, at = correlation["turn_id"], parse_timestamp(event.get("received_at"))
                if event["type"] == "agent.turn_started":
                    evidence.turn_started_at[turn_id] = at
                    continue
                payload = event.get("payload")
                status = payload.get("status") if isinstance(payload, dict) else None
                evidence.turn_terminals.append(dict(id=turn_id, status=status))
                evidence.turn_completed[turn_id] = (status, at)
                if status == "interrupted":
                    evidence.interrupted_completion_seq[turn_id] = event.get("server_seq")
            if self.require_texts:
                done = {t["id"] for t in evidence.turn_terminals}
                evidence.completed_turn_texts = [
                    row.get("text", "") for row in self.store.states(self.store.TRANSCRIPT_KIND)
                    if row.get("type") == "agentMessage" and row.get("turn_id") in done
                    and row.get("thread_id") == thread_id and isinstance(row.get("text"), str)]
        return evidence


def load_native_evidence(root, manifest, store=None, *, profile="audit", events=None, require_texts=False):
    """Pick the reader from ``manifest["agent_backend"]["kind"]`` (default "codex") and load.

    profile: "audit" (log required, full extraction) or "report" (log optional,
    prompt submissions only).  store/events: Store and its event list, needed for
    interrupt ownership; require_texts: also collect assistant texts of terminal turns.
    """
    if profile not in ("audit", "report"):
        raise ValueError("unsupported_native_evidence_profile")
    block = manifest.get("agent_backend")
    kind = block.get("kind", "codex") if isinstance(block, dict) else "codex"
    if kind in CODEX_KINDS:
        reader = CodexRawEvidence
    elif kind in AGENT_KINDS:
        reader = AgentRawEvidence
    else:
        raise ValueError("unsupported_agent_backend")
    return reader(root, manifest, store, profile=profile, events=events, require_texts=require_texts).load()
