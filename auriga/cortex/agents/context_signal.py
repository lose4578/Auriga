"""Relay-side context-continuity signal for agent backends (design §4.3, decision R3-2).

With ``context_delivery="delta"`` the harness leaves out of a tool result the images and metadata it
already delivered in the same image-context epoch, because the model still holds them in its
history. Claude Code and OpenCode manage that history themselves: auto-compaction replaces it with a
summary, and pruning or a client rewrite can drop old tool results. The relay sees every generation
request the agent sends, so it is the authoritative and synchronous place to notice that the model
no longer holds what it was given. Each new history is compared with the last one whose completion
reached the agent, and any loss is reported, so the caller can open a new epoch before the next tool
result is built. Native compaction signals, such as the Claude Code ``PreCompact`` hook or OpenCode's
``session.compacted``, are extra triggers handled by the session, never a substitute.

The rule is the same for every wire and client:

1. **Chains** are keyed by ``(thread key, system_key)`` (design Revision 4, R4.3). The thread key is the
   native session the client sent (the relay passes the wire's own key, ``wire.thread_key(headers)``, and
   only an empty key falls back to ``default_thread_id``, so an OpenCode child session keeps its own
   ``x-opencode-session-id``). ``system_key`` is the digest of the system text only
   (``conversation_key``). It never includes a conversation item, because compaction replaces the head
   of the history.
2. Every request gets one ``call_purpose`` (``CALL_PURPOSES``):
   ``"helper"``: the request declares no tools, such as an OpenCode compaction summary or title call, a
   Claude Code title call, or the relay evidence client.
   ``"main"``: a request with tools on the thread's **main chain**. The main chain of a main thread is
   the chain whose ``system_key`` equals that of the first committed request with tools on it. That
   includes every post-compaction request (the instructions live in the system text on every backend, so
   compaction does not change ``system_key``) and Claude Code's own compaction-summary request, which
   only appends to the previous history. The caller says which threads are main threads
   (``main_thread``). The relay marks the actor's thread as one.
   ``"subagent"``: any other request with tools. That is a request on a thread that is not a main thread
   (an OpenCode child session), or a request on a main thread whose system text differs from the main
   chain's (a Claude Code subagent of the ``Agent``/``Task`` tool, which shares the session id; probe
   R4-0) and whose history does not continue the main chain (rule 7).
   Only main requests are compared and committed. Helper and subagent requests never report a
   discontinuity and never become a baseline, so a subagent's own compaction or pruning is never
   reported, and subagent requests that interleave with the main chain leave its baseline unchanged.
3. **Fingerprints**: each conversation item becomes ``(head_digest, (block_digest, ...))``. The head is
   the item without its content (role, tool-call fields, ...). The blocks are the canonical content
   blocks: a plain string equals one text block, and ``input_text`` / ``output_text`` / ``text`` are the
   same text block type. Transport-volatile keys (``VOLATILE_KEYS``) are dropped at every depth,
   except inside tool-call arguments.
   System and developer text is excluded from the items and digested separately (``conversation_key``).
4. **Continuity** holds when the committed baseline is a prefix of the new history. The baseline's
   *last* item may gain appended blocks with the same head, because Claude Code merges steer text, tool
   results and post-failure prompts into the last user message, and OpenCode on the Anthropic wire
   merges user texts after an abort. Anything else is a discontinuity, reported in this order:
   ``history_shortened`` when the new history also has fewer items than the baseline (compaction and
   dropped turns), otherwise ``history_rewritten`` (an earlier item changed, for example a pruned tool
   result). If the history is continuous, ``images_removed`` is reported when the image count dropped.
   After that, ``system_changed`` is reported when the request's system key differs from the main
   chain's (rule 7). System keys are compared only when the caller passes them on both requests.
5. **Commit on delivery**: ``observe`` and ``admit`` never change state. The caller calls ``commit``
   only after that request's completion was delivered to the agent as usable. A failed, denied or
   aborted request is never a baseline: Claude Code re-sends a failed prompt merged into the next one.
6. The first main request on a thread, or the first one after ``reset``, is not compared. Its commit
   fixes the main chain's ``system_key``.
7. **The main chain follows its own system change**: a request with tools on a main thread whose system
   key differs from the main chain's, but whose history continues the committed main baseline (rule 4,
   baseline not empty), is the main conversation with a changed system text. Examples are a live model
   switch or a date in the system text. It is ``"main"`` with reason ``system_changed``, and its commit
   moves the main chain to the new key. A subagent starts its own conversation, so its history never
   continues the main baseline.

The tracker takes only wire-neutral inputs. The pure functions below compute them from Anthropic
Messages, OpenAI Responses and OpenAI Chat Completions request bodies. They are the implementations
that the wire adapters' ``conversation_key`` / ``history_fingerprints`` / ``is_helper_request`` will
delegate to, for example
``history_fingerprints(anthropic_messages_history(request))``. Call them on the request as the agent
sent it after the egress projection, and before relay image trim or reuse. A body the relay trimmed
itself would look like a rewrite.

Digests are internal comparison keys. They never enter model-visible context, events or archives.
"""
import hashlib
import json
import threading
from typing import NamedTuple

HISTORY_REWRITTEN = "history_rewritten"
HISTORY_SHORTENED = "history_shortened"
IMAGES_REMOVED = "images_removed"
SYSTEM_CHANGED = "system_changed"
REASONS = (HISTORY_SHORTENED, HISTORY_REWRITTEN, IMAGES_REMOVED, SYSTEM_CHANGED)
# call_purpose values (module docstring, rule 2). The relay records them as ``identity.call_purpose``. They are
# evidence only and never change counting: every delivered usable generation completion is charged.
MAIN = "main"
HELPER = "helper"
SUBAGENT = "subagent"
CALL_PURPOSES = (MAIN, HELPER, SUBAGENT)

# Keys dropped at every depth before digesting. These are transport facts, not context the model reads:
# ``cache_control`` (Anthropic prompt-cache markers the client moves to the newest blocks on every
# request), ``providerOptions`` (AI SDK provider metadata), and identifier and pairing fields that a
# client may regenerate when it re-serialises its history (``id``, ``call_id``, ``tool_use_id``,
# ``tool_call_id``). Dropping an identifier cannot hide a loss: the content the identifiers point at
# is still compared.
VOLATILE_KEYS = frozenset({"cache_control", "providerOptions", "id", "call_id", "tool_use_id", "tool_call_id"})
# Content-part types that carry plain text. They are canonicalised to the single type "text", so that
# a plain string content equals one text block on every wire.
TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
# Content-part types counted as images. Anthropic: "image", including those nested in "tool_result".
# Responses: "input_image". Chat: "image_url". Any source counts, inline data or reference.
IMAGE_PART_TYPES = frozenset({"image", "input_image", "image_url"})
# Leading items with these roles hold system or developer text on the Responses and Chat wires.
SYSTEM_ROLES = frozenset({"system", "developer"})
# Anthropic block types whose ``input`` holds tool arguments the model wrote. The arguments are digested
# verbatim (no volatile-key removal) and never searched for images.
TOOL_CALL_BLOCK_TYPES = frozenset({"tool_use", "server_tool_use"})


class HistoryView(NamedTuple):
    """One request body split into its system part and its conversation items (wire layout removed).

    system: tuple of the request's system/developer entries in order. An entry is a string, a list of
        content blocks or a message item, exactly as sent. The tuple is empty when the request has none.
    items: tuple of the conversation items (message / tool-call / tool-output dicts) in context order,
        system entries excluded. ``history_fingerprints`` and ``image_count`` read only these.
    """
    system: tuple
    items: tuple


# ------------------------------------------------------------------ per-wire splitters (pure)
def anthropic_messages_history(request):
    """Split an Anthropic Messages body (``POST /v1/messages``).

    system: the top-level ``system`` field (string or text-block list), when present.
    items: every entry of ``messages``. That includes the ``role: "system"`` messages Claude Code puts
    in the history (environment block, steer framing, post-compaction date), because they are part
    of the conversation and do not replace the top-level system prompt.
    """
    request = request if isinstance(request, dict) else {}
    system = request.get("system")
    messages = request.get("messages")
    return HistoryView(() if system is None else (system,),
                       tuple(messages) if isinstance(messages, list) else ())


def openai_responses_history(request):
    """Split an OpenAI Responses body (``POST /responses``).

    system: the ``instructions`` field, when present, followed by the leading run of ``input`` items
    whose role is ``system`` or ``developer``.
    items: the remaining ``input`` items. A string ``input`` is one user message.
    """
    request = request if isinstance(request, dict) else {}
    instructions = request.get("instructions")
    raw = request.get("input")
    items = [{"role": "user", "content": raw}] if isinstance(raw, str) else list(raw) if isinstance(raw, list) else []
    lead = _leading_system_count(items)
    head = () if instructions is None else (instructions,)
    return HistoryView(head + tuple(items[:lead]), tuple(items[lead:]))


def openai_chat_history(request):
    """Split an OpenAI Chat Completions body (``POST /v1/chat/completions``).

    system: the leading run of ``messages`` whose role is ``system`` or ``developer``.
    items: the remaining ``messages``.
    """
    request = request if isinstance(request, dict) else {}
    messages = request.get("messages")
    items = list(messages) if isinstance(messages, list) else []
    lead = _leading_system_count(items)
    return HistoryView(tuple(items[:lead]), tuple(items[lead:]))


def _leading_system_count(items):
    count = 0
    for item in items:
        if not (isinstance(item, dict) and item.get("role") in SYSTEM_ROLES):
            break
        count += 1
    return count


# ------------------------------------------------------------------ wire-neutral inputs (pure)
def is_helper_request(request):
    """True when the request body declares no tools (no ``tools`` key, or an empty list).

    That is the helper rule shared by all three wires: compaction-summary, title and side calls of the
    agent clients, and the relay evidence client. Helpers are attributed ``call_purpose="helper"`` and
    charged to the budget like any generation call, but the tracker never compares or commits them.
    """
    return not (isinstance(request, dict) and request.get("tools"))


def conversation_key(view):
    """Hex digest of the canonical system/developer text of ``view`` (``HistoryView``).

    Only the system part is digested. A plain string equals one text block, and volatile keys are
    dropped. Two requests have equal keys exactly when their system text is equal. Passed to the
    tracker as ``system_key``.
    """
    return _digest([_canonical_entry(entry) for entry in view.system])


def history_fingerprints(view):
    """Canonical per-item fingerprints of ``view.items``.

    Returns a list with one ``(head_digest, (block_digest, ...))`` tuple per conversation item, in order.
    head_digest: hex sha256 of the item without its content key and without ``VOLATILE_KEYS``.
    block_digest: hex sha256 of one canonical content block. The content key is ``content``, or
    ``output`` for Responses tool outputs. A string is one ``{"type": "text", "text": s}`` block, None
    or a missing content is no block, and an Anthropic ``tool_result``'s nested ``content`` is
    canonicalised the same way. Image bytes are part of the digest.
    """
    fingerprints = []
    for item in view.items:
        head, blocks = _canonical_item(item)
        fingerprints.append((_digest(head), tuple(_digest(block) for block in blocks)))
    return fingerprints


def image_count(view):
    """Number of image parts in ``view.items``: dicts whose ``type`` is in ``IMAGE_PART_TYPES``, at any
    depth (Anthropic ``tool_result`` content included), with any source (inline data or reference).
    Tool-call arguments are not searched."""
    return _count_images(list(view.items))


def _strip(node):
    if isinstance(node, dict):
        if node.get("type") in TOOL_CALL_BLOCK_TYPES and "input" in node:
            # Tool arguments are model output; keep them verbatim, even a key that is also a volatile name.
            return {k: (v if k == "input" else _strip(v)) for k, v in node.items() if k not in VOLATILE_KEYS}
        return {k: _strip(v) for k, v in node.items() if k not in VOLATILE_KEYS}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def _canonical_blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        content = [content]
    return [_canonical_block(block) for block in content]


def _canonical_block(block):
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if not isinstance(block, dict):
        return block
    out = _strip(block)
    if out.get("type") in TEXT_PART_TYPES:
        out["type"] = "text"
    elif out.get("type") == "tool_result" and "content" in out:
        out["content"] = _canonical_blocks(out["content"])
    return out


def _content_key(item):
    return "content" if "content" in item else "output" if "output" in item else None


def _canonical_item(item):
    if not isinstance(item, dict):
        return item, []
    key = _content_key(item)
    head = {k: _strip(v) for k, v in item.items() if k != key and k not in VOLATILE_KEYS}
    return head, _canonical_blocks(item[key]) if key else []


def _canonical_entry(entry):
    if isinstance(entry, dict) and "role" in entry:
        head, blocks = _canonical_item(entry)
        return {"head": head, "blocks": blocks}
    return _canonical_blocks(entry)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _count_images(node):
    if isinstance(node, dict):
        own = 1 if node.get("type") in IMAGE_PART_TYPES else 0
        skip = "input" if node.get("type") in TOOL_CALL_BLOCK_TYPES else None
        return own + sum(_count_images(v) for k, v in node.items() if k != skip)
    if isinstance(node, list):
        return sum(_count_images(v) for v in node)
    return 0


# ------------------------------------------------------------------ tracker
class _Baseline(NamedTuple):
    fingerprints: tuple
    image_count: int
    system_key: str | None


def _freeze(fingerprints):
    frozen = []
    for entry in fingerprints:
        ok = isinstance(entry, (tuple, list)) and len(entry) == 2
        head, blocks = entry if ok else (None, None)
        if not (ok and isinstance(head, str) and isinstance(blocks, (tuple, list))
                and all(isinstance(block, str) for block in blocks)):
            raise TypeError("fingerprints must be (head_digest: str, block_digests: sequence of str) pairs")
        frozen.append((head, tuple(blocks)))
    return tuple(frozen)


def _check_count(image_count):
    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count < 0:
        raise ValueError("image_count must be a non-negative int")
    return image_count


def _history_reason(base, new):
    n = len(base)
    if n == 0:
        return None
    if len(new) < n:
        return HISTORY_SHORTENED
    last_head, last_blocks = base[-1]
    if new[:n - 1] == base[:n - 1] and new[n - 1][0] == last_head and new[n - 1][1][:len(last_blocks)] == last_blocks:
        return None
    return HISTORY_REWRITTEN


class Admission(NamedTuple):
    """What ``ContinuityTracker.admit`` decided for one request.

    call_purpose: ``"main"``, ``"helper"`` or ``"subagent"`` (module docstring, rule 2).
    reason: discontinuity reason of a main request (one of ``REASONS``), else None. It is always None for
        helper and subagent requests.
    """
    call_purpose: str
    reason: str | None


def _admission(base, new, count, system_key):
    """Classify and compare one request with tools on a main thread against the thread's committed main
    baseline ``base`` (None: none yet). Pure; module docstring, rules 2, 4, 6 and 7."""
    if base is None:
        return Admission(MAIN, None)
    history = _history_reason(base.fingerprints, new)
    same_chain = system_key is None or base.system_key is None or system_key == base.system_key
    if not same_chain and (history is not None or not base.fingerprints):
        return Admission(SUBAGENT, None)
    if history is not None:
        return Admission(MAIN, history)
    if count < base.image_count:
        return Admission(MAIN, IMAGES_REMOVED)
    return Admission(MAIN, None if same_chain else SYSTEM_CHANGED)


class ContinuityTracker:
    """Per-thread continuity baseline of the main conversation chain (see the module docstring).

    Thread-safe: one instance is shared by all relay handler threads. ``admit`` and ``observe`` only read
    state. ``commit`` and ``reset`` replace one thread's baseline atomically. Concurrent requests on one
    thread are independent, because ``commit`` takes the delivered request's own inputs (last
    delivered wins), so no observation is held between ``admit`` and ``commit``. Only main chains have
    state: one baseline per main thread, whose ``system_key`` is the main chain's key.

    Inputs, shared by ``admit``, ``observe`` and ``commit``:
    thread: native thread key (str; ``""`` is a valid key) as resolved by the relay.
    helper: True for a request that declares no tools (``is_helper_request``). Helpers are never
        compared and never committed.
    fingerprints: ``history_fingerprints(view)`` of the request, a sequence of
        ``(head_digest, block_digests)`` pairs.
    image_count: ``image_count(view)`` of the request, a non-negative int (images in the history).
    system_key: ``conversation_key(view)`` of the request, or None (the default). None skips the
        system comparison: such a request is on the main chain of a main thread, and a baseline committed
        without a key accepts any key. ``system_changed`` and subagent chains on a main thread exist only
        when the committed baseline and the new request both carry a key.
    main_thread: True (the default) when ``thread`` is a thread whose main chain is tracked. The relay
        passes True for the actor's thread (``default_thread_id`` or a thread registered with source
        ``"actor"``), and for every thread when it has neither. A request with tools on any other thread
        is ``"subagent"``.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._baselines = {}

    def admit(self, thread, *, helper, fingerprints, image_count, system_key=None, main_thread=True):
        """Classify a request and compare a main request with its thread's committed baseline. Never
        changes state. Returns an ``Admission(call_purpose, reason)``.

        ``reason`` is None when the request is continuous, is a helper or subagent request, or the thread
        has no committed baseline. Otherwise it is one of ``"history_shortened"``,
        ``"history_rewritten"``, ``"images_removed"`` or ``"system_changed"`` (module docstring, rules 4
        and 7). If the client re-sends a discontinuous history that was never delivered, the same reason
        is reported again, because the baseline did not move.
        """
        new = _freeze(fingerprints)
        count = _check_count(image_count)
        if helper:
            return Admission(HELPER, None)
        if not main_thread:
            return Admission(SUBAGENT, None)
        with self._lock:
            base = self._baselines.get(thread)
        return _admission(base, new, count, system_key)

    def observe(self, thread, *, helper, fingerprints, image_count, system_key=None, main_thread=True):
        """``admit(...).reason``: the discontinuity reason of a main request, else None. Never changes state."""
        return self.admit(thread, helper=helper, fingerprints=fingerprints, image_count=image_count,
                          system_key=system_key, main_thread=main_thread).reason

    def commit(self, thread, *, helper, fingerprints, image_count, system_key=None, main_thread=True,
               call_purpose=None):
        """Make a main request's history the thread's baseline, and its ``system_key`` the main chain's key.
        Call only after its completion was delivered to the agent as usable. Arguments are the ones passed
        to ``admit``.

        call_purpose: the ``Admission.call_purpose`` of this request at its admission (the relay passes
            it, so the delivery commits what was classified), or None (the default) to classify it now
            with the same rule. A no-op unless the purpose is ``"main"``: helper and subagent requests
            never become a baseline.
        """
        new = _freeze(fingerprints)
        count = _check_count(image_count)
        if call_purpose is not None and call_purpose not in CALL_PURPOSES:
            raise ValueError("call_purpose must be one of %s" % (CALL_PURPOSES,))
        if helper or not main_thread or call_purpose in (HELPER, SUBAGENT):
            return
        with self._lock:
            if call_purpose is None and _admission(self._baselines.get(thread), new, count,
                                                   system_key).call_purpose != MAIN:
                return
            self._baselines[thread] = _Baseline(new, count, system_key)

    def reset(self, thread):
        """Forget the thread's baseline and with it the main chain's key. The next request with tools on
        the thread is not compared, and its commit fixes the main chain again."""
        with self._lock:
            self._baselines.pop(thread, None)
