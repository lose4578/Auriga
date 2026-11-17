"""Build ``continuity_fixture.json``, the ContinuityTracker acceptance data (design §4.3, P1-3a; subagent chains:
design Revision 4 R4.3, R4-4).

Sources are the generation requests captured by the two agent-backend spikes and by probe R4-0:

* S-B, Claude Code CLI 2.1.321 (report 05): every ``<evidence>/*/model-requests*.jsonl`` with at least
  two generation requests that carry ``messages``. The thread key is the ``X-Claude-Code-Session-Id`` header.
* S-C, opencode 1.19.5 (report 06): every evidence set named in ``<evidence>/continuity-eval.json``,
  read from ``bodies.jsonl``, else from the generation rows of ``requests.json``. The thread key is the
  ``x-opencode-session-id`` header.
* R4-0, Claude Code CLI 2.1.321 with subagents on (probe R4-0, ``scratch/r4-0``): every
  ``<evidence>/*/model-requests.jsonl`` of a finished set (``result.json`` present) with at least one
  generation request that carries the ``x-claude-code-agent-id`` header, i.e. one made by a ``Task``/``Agent``
  subagent. Main and subagent requests share the session id. These sequences carry ``"source": "r4-0"``
  so that the spike sequences keep their own expected counts.

Every request is kept, in capture order, with what the tracker and its replay need:

* ``thread``: the session id, mapped to ``t1``, ``t2``, ... per sequence.
* ``status``: the HTTP status the fake model answered, or null.
* ``delivered``: true when the completion reached the agent as a usable answer. That means HTTP 200,
  no client disconnect, and not in ``UNDELIVERED`` (aborted model calls documented by the spike
  reports).
* ``label``: ground truth from content markers, never from any tracker rule.
  ``helper`` = no tools declared (title or summary call; checked against its prompt text).
  ``compaction_request`` = Claude Code's own summary request, which carries tools and only appends to
  the previous history.
  ``context_lost`` = the first main request on a thread after a compaction (a summary request), or a
  main request that holds more ``[Old tool result content cleared]`` markers than the previous main
  request on its thread (a prune rewrite).
  ``continuous`` = everything else.
  ``subagent`` = a request with tools that carries ``x-claude-code-agent-id``: the CLI's own attribution
  of the request to a subagent (the tracker never reads that header). The other labels of an R4-0
  sequence are computed over its other requests only.
* OpenCode system text: the S-C spike sent the Auriga instructions as the ``prompt_async`` ``system``
  field. opencode appends that text to the system text of the latest user message only, so it vanished
  after auto-compaction (report 06 S9.1 item 6). The shipped backend sends the instructions through the
  config ``instructions`` file instead (report 06 S9 change 4, ``opencode/config.py``), which is part of
  the system text of every request. The builder therefore removes the per-message text (from
  ``PROMPT_SYSTEM_MARK`` to the end of the system text, always the last part) from every OpenCode request,
  so the system keys are the ones of the shipped channel: constant across a compaction. Without this
  step, the ContinuityTracker (design Revision 4, R4.3) would read the post-compaction requests as a
  subagent chain. Affected sequences carry ``system_note``.
* ``body``: the request body reduced to ``system`` / ``instructions`` / ``messages`` / ``input`` /
  ``tools``. Tools keep only ``type`` and ``name``. Conversation items and system values are stored once
  per sequence in ``values`` and referenced by index (a list field holds a list of indexes). Every string
  longer than ``LONG_TEXT`` characters, and every string with a local path or a credential-like
  marker, is replaced by ``<sha256:16 hex len=N>``, which removes media bytes, environment dumps and
  long tool output. The replacement is a deterministic function of the string, so equality between
  any two strings, and with it every continuity relation, is preserved.

The evidence directories are the spike scratch areas of reports 05 and 06. They are not checked in,
and the fixture is the durable copy. Usage, from the worktree root::

    mamba run -n Auriga python tests/fixtures/continuity/build_fixture.py \
        --claude /tmp/auriga-agent-harness/scratch/s-b/evidence \
        --opencode /tmp/auriga-agent-harness/scratch/s-c/evidence \
        --subagents /tmp/auriga-agent-harness/scratch/r4-0/evidence
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().with_name("continuity_fixture.json")
LONG_TEXT = 48
PRIVATE_MARKERS = ("/Users/", "/private/", "/tmp/", "/home/", "/var/", "API_KEY", "TOKEN", "SECRET", "PASSWORD")
CLAUDE_SUMMARY_MARK = "create a detailed summary"
CLAUDE_TITLE_MARK = "Write the title"
CLAUDE_AGENT_HEADER = "x-claude-code-agent-id"
OPENCODE_SUMMARY_MARK = "context summarization agent"
PRUNE_MARK = "[Old tool result content cleared]"
# HTTP 200 answers the agent never received as a usable completion (index = position among the
# sequence's generation requests).
UNDELIVERED = {
    ("claude_code/s3-gated-model-p_now/model-requests", 1):
        "report 05 S3: steer with priority now while the model call was gated -> terminal_reason aborted_streaming",
    ("opencode/s7-anthropic", 8): "report 06 S7.2 scenario D: abort during the model call (MessageAbortedError, no parts)",
    ("opencode/s7-responses", 8): "report 06 S7.2 scenario D: abort during the model call (MessageAbortedError, no parts)",
    ("opencode/s7-chat", 8): "report 06 S7.2 scenario D: abort during the model call (MessageAbortedError, no parts)",
}
WIRES = {"anthropic": "anthropic_messages", "responses": "openai_responses", "chat": "openai_chat"}
# Start of the spike's ``prompt_async`` ``system`` text inside an OpenCode system string (see the docstring).
PROMPT_SYSTEM_MARK = "\nAURIGA-INSTRUCTIONS"
PROMPT_SYSTEM_NOTE = ("prompt_async system text removed from the system text (shipped channel: config instructions "
                      "file, constant across compaction)")


def _strip_prompt_system(text):
    index = text.find(PROMPT_SYSTEM_MARK) if isinstance(text, str) else -1
    return (text, False) if index < 0 else (text[:index], True)


def _strip_content(content):
    if isinstance(content, str):
        return _strip_prompt_system(content)
    changed = False
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text, hit = _strip_prompt_system(part["text"])
                part, changed = dict(part, text=text), changed or hit
            out.append(part)
        content = out
    return content, changed


def opencode_shipped_system(body):
    """``(body, changed)``: ``body`` with the ``prompt_async`` system text removed from its system part (Anthropic
    ``system``; leading ``system``/``developer`` items of Responses ``input`` and Chat ``messages``)."""
    body, changed = dict(body), False
    if "system" in body:
        body["system"], changed = _strip_content(body["system"])
    for key in ("input", "messages"):
        items = body.get(key)
        if not isinstance(items, list):
            continue
        items = list(items)
        for k, item in enumerate(items):
            if not (isinstance(item, dict) and item.get("role") in ("system", "developer")):
                break
            content, hit = _strip_content(item.get("content"))
            if hit:
                items[k], changed = dict(item, content=content), True
        body[key] = items
    return body, changed


def scrub(node):
    if isinstance(node, dict):
        return {k: scrub(v) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(v) for v in node]
    if isinstance(node, str) and (len(node) > LONG_TEXT or any(m in node for m in PRIVATE_MARKERS)):
        return "<sha256:%s len=%d>" % (hashlib.sha256(node.encode("utf-8", "surrogatepass")).hexdigest()[:16], len(node))
    return node


def reduce_tools(tools):
    out = []
    for tool in tools or []:
        kept = {k: tool[k] for k in ("type", "name") if k in tool}
        if isinstance(tool.get("function"), dict):
            kept["function"] = {"name": tool["function"].get("name")}
        out.append(kept)
    return out


def wire_of(body):
    if "input" in body:
        return "openai_responses"
    if "system" in body or not (body.get("messages") and body["messages"][0].get("role") == "system"):
        return "anthropic_messages"
    return "openai_chat"


class Sequence:
    def __init__(self, sid, client, wire, thread_header):
        self.data = {"id": sid, "client": client, "wire": wire, "thread_header": thread_header,
                     "values": [], "requests": []}
        self._index, self._threads = {}, {}

    def ref(self, value):
        value = scrub(value)
        key = json.dumps(value, ensure_ascii=True)
        if key not in self._index:
            self._index[key] = len(self.data["values"])
            self.data["values"].append(value)
        return self._index[key]

    def add(self, headers, body, status, delivered, label, note=None):
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        thread = self._threads.setdefault(headers.get(self.data["thread_header"], ""), "t%d" % (len(self._threads) + 1))
        reduced = {}
        for key in ("system", "instructions"):
            if key in body:
                reduced[key] = self.ref(body[key])
        for key in ("messages", "input"):
            if isinstance(body.get(key), list):
                reduced[key] = [self.ref(item) for item in body[key]]
            elif key in body:
                reduced[key] = self.ref(body[key])
        if "tools" in body:
            reduced["tools"] = self.ref(reduce_tools(body["tools"]))
        row = {"thread": thread, "status": status, "delivered": delivered, "label": label, "body": reduced}
        if note:
            row["note"] = note
        self.data["requests"].append(row)


def _agent_id(row):
    return {k.lower(): v for k, v in (row.get("headers") or {}).items()}.get(CLAUDE_AGENT_HEADER)


def claude_sequences(root, *, source=None):
    """S-B sequences; with ``source="r4-0"`` the R4-0 subagent sequences (see the module docstring)."""
    for path in sorted(glob.glob(os.path.join(root, "*", "model-requests*.jsonl"))):
        rows = [json.loads(line) for line in open(path)]
        rows = [r for r in rows if r.get("is_generation") and isinstance(r.get("json"), dict) and r["json"].get("messages")]
        if len(rows) < 2:
            continue
        if source is not None and not (os.path.exists(os.path.join(os.path.dirname(path), "result.json"))
                                       and any(_agent_id(r) and r["json"].get("tools") for r in rows)):
            continue
        prefix = "claude_code/" if source is None else "claude_code/%s/" % source
        sid = prefix + "%s/%s" % (os.path.basename(os.path.dirname(path)), Path(path).stem)
        seq = Sequence(sid, "claude_code", "anthropic_messages", "x-claude-code-session-id")
        if source is not None:
            seq.data["source"] = source
        previous_summary, pruned = False, 0
        for k, r in enumerate(rows):
            body = r["json"]
            tail = json.dumps(body["messages"][-1:])
            if _agent_id(r) and body.get("tools"):
                label = "subagent"
            elif not body.get("tools"):
                assert CLAUDE_TITLE_MARK in tail, (sid, k, "tool-less request that is not a title call")
                label = "helper"
            elif CLAUDE_SUMMARY_MARK in tail:
                label = "compaction_request"
            else:
                marks = json.dumps(body["messages"]).count(json.dumps(PRUNE_MARK)[1:-1])
                label = "context_lost" if previous_summary or marks > pruned else "continuous"
                pruned = marks
            if label not in ("helper", "subagent"):
                previous_summary = label == "compaction_request"
            status = r.get("response_status")
            note = UNDELIVERED.get((sid, k))
            delivered = status == 200 and not r.get("client_disconnected") and note is None
            if r.get("client_disconnected"):
                note = "client disconnected before the answer was delivered"
            seq.add(r.get("headers"), body, status, delivered, label, note)
        yield seq.data


def opencode_sequences(root):
    reference = json.load(open(os.path.join(root, "continuity-eval.json")))
    for name in reference:
        base = os.path.join(root, name)
        listed = [r for r in json.load(open(os.path.join(base, "requests.json"))) if r.get("is_generation")]
        if os.path.exists(os.path.join(base, "bodies.jsonl")):
            rows = [json.loads(line) for line in open(os.path.join(base, "bodies.jsonl"))]
            assert len(listed) == len(rows), (name, "bodies.jsonl and requests.json disagree")
            statuses = [r.get("response_status") for r in listed]
        else:
            rows = listed
            statuses = [r.get("response_status") for r in rows]
        sid = "opencode/" + name
        wire = wire_of(rows[0]["body"])
        seq = Sequence(sid, "opencode", wire, "x-opencode-session-id")
        after_summary, pruned = set(), {}
        normalised = False
        for k, (r, status) in enumerate(zip(rows, statuses)):
            body, changed = opencode_shipped_system(r["body"])
            normalised = normalised or changed
            thread = {kk.lower(): v for kk, v in r["headers"].items()}.get("x-opencode-session-id", "")
            text = json.dumps(body)
            if not body.get("tools"):
                assert OPENCODE_SUMMARY_MARK in text, (sid, k, "tool-less request that is not a summary call")
                label = "helper"
                after_summary.add(thread)
            else:
                marks = text.count(json.dumps(PRUNE_MARK)[1:-1])
                lost = thread in after_summary or marks > pruned.get(thread, 0)
                label = "context_lost" if lost else "continuous"
                after_summary.discard(thread)
                pruned[thread] = marks
            note = UNDELIVERED.get((sid, k))
            delivered = status == 200 and not r.get("client_disconnected") and note is None
            rule_b = reference[name][k]["B"]
            if (label == "context_lost") != rule_b.startswith(("history_rewritten", "images_removed", "system_changed")):
                note = ((note + "; ") if note else "") + (
                    "report 06 rule B flagged %s; ground truth: no loss (the aborted user text is merged "
                    "into the next user message)" % rule_b)
            seq.add(r["headers"], body, status, delivered, label, note)
        if normalised:
            seq.data["system_note"] = PROMPT_SYSTEM_NOTE
        yield seq.data


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--claude", required=True, help="S-B evidence directory (Claude Code spikes, report 05)")
    parser.add_argument("--opencode", required=True, help="S-C evidence directory (opencode spikes, report 06)")
    parser.add_argument("--subagents", required=True, help="R4-0 evidence directory (Claude Code with subagents)")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    sequences = (list(claude_sequences(args.claude)) + list(opencode_sequences(args.opencode))
                 + list(claude_sequences(args.subagents, source="r4-0")))
    fixture = {
        "format": "auriga.continuity_fixture.v1",
        "about": ("Generation requests captured by the Claude Code (S-B, CLI 2.1.321) and opencode (S-C, 1.19.5) "
                  "spikes and by probe R4-0 (Claude Code with subagents) against loopback fake models, reduced for the ContinuityTracker replay; built by "
                  "build_fixture.py (see its docstring for the reduction, the labels, UNDELIVERED and the OpenCode "
                  "system-text normalisation)."),
        "sequences": sequences,
    }
    text = json.dumps(fixture, ensure_ascii=True, separators=(",", ":"))
    # One sequence per line keeps diffs reviewable.
    lines = ['{"format":%s,"about":%s,"sequences":[' % (json.dumps(fixture["format"]), json.dumps(fixture["about"]))]
    lines += [json.dumps(s, ensure_ascii=True, separators=(",", ":")) + ("," if i < len(sequences) - 1 else "")
              for i, s in enumerate(sequences)]
    lines.append("]}")
    out = "\n".join(lines) + "\n"
    assert json.loads(out) == json.loads(text)
    Path(args.out).write_text(out)
    print("wrote %s: %d sequences, %d requests, %d bytes" % (
        args.out, len(sequences), sum(len(s["requests"]) for s in sequences), len(out.encode())))


if __name__ == "__main__":
    main()
