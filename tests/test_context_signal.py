"""ContinuityTracker and its wire-neutral inputs (design §4.3, decision R3-2, work package P1-3a), with the subagent
chains and ``call_purpose`` of design Revision 4 R4.3 (work package R4-4)."""
from collections import Counter
from itertools import islice
import json
from pathlib import Path
import threading
import unittest
from unittest import mock

from auriga.cortex.agents import context_signal as cs
from auriga.cortex.agents.context_signal import ContinuityTracker

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "continuity" / "continuity_fixture.json"
SPLITTERS = {"anthropic_messages": cs.anthropic_messages_history, "openai_responses": cs.openai_responses_history,
             "openai_chat": cs.openai_chat_history}
LOSS_REASONS = (cs.HISTORY_SHORTENED, cs.HISTORY_REWRITTEN)


def inputs(view, *, helper=False, system=False):
    """Keyword inputs of observe/commit for a HistoryView."""
    kw = dict(helper=helper, fingerprints=cs.history_fingerprints(view), image_count=cs.image_count(view))
    if system:
        kw["system_key"] = cs.conversation_key(view)
    return kw


def anthropic(messages, system="SYS", tools=True):
    body = {"model": "m", "system": system, "messages": messages}
    if tools:
        body["tools"] = [{"name": "auriga_observe", "input_schema": {"type": "object"}}]
    return body


def anthropic_inputs(body, **kw):
    return inputs(cs.anthropic_messages_history(body), helper=cs.is_helper_request(body), **kw)


def fp(*items):
    """Hand-made fingerprints: each item is (head, n_blocks) or (head, (block, ...))."""
    out = []
    for head, blocks in items:
        blocks = tuple("b%d" % i for i in range(blocks)) if isinstance(blocks, int) else tuple(blocks)
        out.append((head, blocks))
    return out


# ---------------------------------------------------------------------------------- acceptance fixture
def fixture_body(sequence, request):
    values = sequence["values"]
    return {key: [values[i] for i in ref] if isinstance(ref, list) else values[ref]
            for key, ref in request["body"].items()}


def replay(fixture, *, commit_when):
    """Replay every sequence through a fresh tracker per relay process; score main requests that have a
    committed baseline on their thread. Claude Code's own compaction-summary request is not scored (it
    only appends to the history; nothing is lost yet) but is reported separately. Sequences with a
    ``source`` (R4-0) are scored in a cell of their own; their subagent requests count as ``subagent`` (no
    reason) or ``subagent_raised`` (a reason), never as main negatives."""
    cells, reasons, summary_requests, helper_mismatch = {}, Counter(), [], []
    for sequence in fixture["sequences"]:
        tracker, committed = ContinuityTracker(), set()
        cell = cells.setdefault(sequence.get("source", sequence["client"]), Counter(tp=0, fp=0, fn=0, tn=0))
        for k, request in enumerate(sequence["requests"]):
            body = fixture_body(sequence, request)
            kw = inputs(SPLITTERS[sequence["wire"]](body), helper=cs.is_helper_request(body), system=True)
            if kw["helper"] != (request["label"] == "helper"):
                helper_mismatch.append((sequence["id"], k))
            reason = tracker.observe(request["thread"], **kw)
            if request["label"] == "compaction_request":
                summary_requests.append(reason)
            elif request["label"] == "subagent":
                cell["subagent_raised" if reason else "subagent"] += 1
            elif not kw["helper"] and request["thread"] in committed:
                lost = request["label"] == "context_lost"
                cell["tp" if reason and lost else "fp" if reason else "fn" if lost else "tn"] += 1
                if reason:
                    reasons[(sequence["client"], reason)] += 1
            elif kw["helper"]:
                assert reason is None
            if commit_when(request) and not kw["helper"]:
                tracker.commit(request["thread"], **kw)
                committed.add(request["thread"])
    return {client: dict(cell) for client, cell in cells.items()}, dict(reasons), summary_requests, helper_mismatch


# ---------------------------------------------------------------------------------- synthetic subagent sequences
# The R4-0 capture (fixture sequences with "source": "r4-0") has real subagent requests, but only next to short main
# chains that never compact, and no OpenCode child session. The sequences below are therefore SYNTHETIC combinations:
# two captured sequences of one client, one replayed as the main chain and the other interleaved with it as a subagent
# chain. Spike rows used as a subagent get the shape a subagent request has (as R4-0 shows for Claude Code): a first
# user message of their own (the Task prompt; the scripted spikes all start with the same prompt, a real subagent
# never does) and, for Claude Code, whose subagents share the session id, a system text of their own. Those texts are
# stand-ins, not captured strings. Captured R4-0 subagent rows keep their own system text and prompt.
SYNTHETIC_TASK_PROMPT = {"role": "user", "content": [
    {"type": "text", "text": "Synthetic Task prompt: inspect the actor directory and report what you find."}]}
SYNTHETIC_SUBAGENT_SYSTEM = [
    {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
    {"type": "text", "text": "Synthetic subagent system text: you are a subagent; use the tools to finish the task."}]
MAIN_SYSTEM = object()  # sub_system sentinel: the subagent rows reuse the main chain's system text (OpenCode general)


def subagent_interleaving(main, sub, *, sub_thread=None, sub_system=SYNTHETIC_SUBAGENT_SYSTEM, task_prompt=True,
                          start=3, burst=2):
    """A fixture-format sequence (Anthropic Messages wire): every request of ``main`` in order, with every request of
    ``sub`` interleaved as a subagent chain, ``burst`` subagent requests after each main request from the
    ``start``-th on, the rest at the end. Each request gets ``chain``: ``"main"`` or ``"subagent"``.

    sub_thread: thread of the subagent rows; None (default) = the main chain's thread (a Claude Code subagent).
    sub_system: ``system`` value of the subagent rows that declare tools; ``MAIN_SYSTEM`` = the main chain's own;
        None = keep the rows' own (captured subagent requests).
    task_prompt: True (default) = the subagent rows' first user message (where it equals the one of ``sub``'s first
        request) becomes ``SYNTHETIC_TASK_PROMPT``; False = keep it (captured subagent requests).
    """
    assert main["wire"] == sub["wire"] == "anthropic_messages"
    values = list(main["values"])
    offset = len(values)
    values.extend(sub["values"])

    def add(value):
        values.append(value)
        return len(values) - 1
    task_ref = add(SYNTHETIC_TASK_PROMPT)
    first_ref = sub["requests"][0]["body"]["messages"][0] + offset
    if sub_system is MAIN_SYSTEM:
        system_ref = next(r["body"]["system"] for r in main["requests"] if "tools" in r["body"])
    else:
        system_ref = None if sub_system is None else add(sub_system)
    thread = sub_thread or main["requests"][0]["thread"]
    sub_rows = []
    for request in sub["requests"]:
        body = {key: [i + offset for i in ref] if isinstance(ref, list) else ref + offset
                for key, ref in request["body"].items()}
        if task_prompt and body.get("messages") and body["messages"][0] == first_ref:
            body["messages"][0] = task_ref
        if "tools" in body and system_ref is not None:
            body["system"] = system_ref
        sub_rows.append(dict(request, thread=thread, body=body, chain="subagent"))
    rows, pending = [], iter(sub_rows)
    for k, request in enumerate(main["requests"]):
        rows.append(dict(request, chain="main"))
        if k + 1 >= start:
            rows.extend(islice(pending, burst))
    rows.extend(pending)
    return dict(id="synthetic:%s+%s" % (main["id"], sub["id"]), client=main["client"], wire=main["wire"],
                values=values, requests=rows)


def replay_chains(sequence, *, main_threads=None, system=True):
    """Admit every request of ``sequence`` through one tracker and commit the delivered ones with their admitted
    purpose, as the relay does. main_threads: threads passed as ``main_thread=True`` (None = every thread).
    Returns one ``Admission`` per request."""
    tracker, admissions = ContinuityTracker(), []
    for request in sequence["requests"]:
        body = fixture_body(sequence, request)
        kw = inputs(SPLITTERS[sequence["wire"]](body), helper=cs.is_helper_request(body), system=system)
        main_thread = main_threads is None or request["thread"] in main_threads
        admission = tracker.admit(request["thread"], main_thread=main_thread, **kw)
        if request["delivered"]:
            tracker.commit(request["thread"], main_thread=main_thread, call_purpose=admission.call_purpose, **kw)
        admissions.append(admission)
    return admissions


class AcceptanceFixtureTest(unittest.TestCase):
    """The spike request sequences (Claude Code CLI 2.1.321, opencode 1.19.5) replayed through the tracker."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_fixture_is_compact_and_covers_both_spikes(self):
        self.assertLess(FIXTURE.stat().st_size, 1_000_000)
        self.assertEqual(self.fixture["format"], "auriga.continuity_fixture.v1")
        clients = Counter(s["client"] for s in self.fixture["sequences"] if "source" not in s)
        self.assertEqual(clients, {"claude_code": 70, "opencode": 15})
        probe = [s for s in self.fixture["sequences"] if s.get("source") == "r4-0"]
        self.assertEqual((len(probe), {s["client"] for s in probe}), (4, {"claude_code"}))
        self.assertEqual(Counter(r["label"] for s in probe for r in s["requests"]), {"continuous": 14, "subagent": 9})
        self.assertEqual({s["wire"] for s in self.fixture["sequences"] if s["client"] == "opencode"},
                         set(SPLITTERS))
        labels = Counter((s["client"], r["label"]) for s in self.fixture["sequences"] for r in s["requests"])
        self.assertEqual(labels[("claude_code", "compaction_request")], 2)
        self.assertEqual(labels[("claude_code", "context_lost")], 2)
        self.assertEqual(labels[("opencode", "context_lost")], 4)
        self.assertEqual(labels[("opencode", "helper")], 3)
        # Media bytes, long text and local paths are reduced to digests.
        text = FIXTURE.read_text()
        for marker in ("iVBORw0KGgo", "/Users/", "/private/", "auriga-local-relay"):
            self.assertNotIn(marker, text)

    def test_replay_commit_on_delivery(self):
        """Acceptance (§4.3): the baseline advances only on a delivered usable completion."""
        cells, reasons, summary_requests, helper_mismatch = replay(self.fixture, commit_when=lambda r: r["delivered"])
        self.assertEqual(cells["claude_code"], {"tp": 2, "fp": 0, "fn": 0, "tn": 167})
        self.assertEqual(cells["opencode"], {"tp": 4, "fp": 0, "fn": 0, "tn": 207})
        # Compactions shorten the history (2 Claude Code, 3 opencode); the opencode prune rewrites a tool result.
        self.assertEqual(reasons, {("claude_code", cs.HISTORY_SHORTENED): 2, ("opencode", cs.HISTORY_SHORTENED): 3,
                                   ("opencode", cs.HISTORY_REWRITTEN): 1})
        self.assertEqual(summary_requests, [None, None])
        self.assertEqual(helper_mismatch, [])
        # Probe R4-0 (real Task/Agent subagents, foreground and background, interrupted task trees): every
        # subagent request passes without a reason, and the main chain around them stays continuous.
        self.assertEqual(cells["r4-0"], {"tp": 0, "fp": 0, "fn": 0, "tn": 10, "subagent": 9})

    def test_replay_commit_on_every_http_200_matches_architect_replay(self):
        """Cross-check with scratch/unified_tracker_eval.py, which committed every HTTP 200. The three opencode
        aborted model calls then become baselines, so the request after each one is scored as well (opencode
        TN 210 = the 209 negatives of that replay + the abort-then-merge request that report 06 rule B had
        labelled history_rewritten, which the last-item block extension accepts as continuous)."""
        cells, _, _, _ = replay(self.fixture, commit_when=lambda r: r["status"] == 200)
        self.assertEqual(cells["claude_code"], {"tp": 2, "fp": 0, "fn": 0, "tn": 167})
        self.assertEqual(cells["opencode"], {"tp": 4, "fp": 0, "fn": 0, "tn": 210})

    def test_captured_requests_get_their_labelled_call_purpose(self):
        """Every captured request with tools is on its thread's main chain, including the post-compaction requests
        (the shipped instruction channels keep the system text across a compaction), except the R4-0 requests the
        CLI itself attributes to a subagent (``x-claude-code-agent-id``), which are ``subagent``."""
        purposes = Counter()
        expected = {"helper": cs.HELPER, "subagent": cs.SUBAGENT}
        for sequence in self.fixture["sequences"]:
            for request, admission in zip(sequence["requests"], replay_chains(sequence)):
                purposes[admission.call_purpose] += 1
                self.assertEqual(admission.call_purpose, expected.get(request["label"], cs.MAIN),
                                 (sequence["id"], request))
                if request["label"] == "subagent":
                    self.assertIsNone(admission.reason)
        self.assertEqual(purposes[cs.SUBAGENT], 9)
        self.assertEqual(set(purposes), set(cs.CALL_PURPOSES))
        # Negative control: the P1-3a rule (one chain per thread, no system keys) reads the subagent requests as
        # rewrites of the main chain.
        probe = [s for s in self.fixture["sequences"] if s.get("source") == "r4-0"]
        raised = [a.reason for s in probe for a in replay_chains(s, system=False) if a.reason]
        self.assertGreaterEqual(len(raised), 10)
        noted = [s["id"] for s in self.fixture["sequences"] if "system_note" in s]
        self.assertEqual(len(noted), 11)
        self.assertTrue(all(sid.startswith(("opencode/s9-", "opencode/s6-")) for sid in noted), noted)

    def test_fixture_rejects_the_rule_without_its_allowances(self):
        """The fixture is sensitive to both allowances of rule 3: without string == text block, Claude Code's
        re-serialisation floods false positives (report 05 measured 63); without the last-item block
        extension, the compaction-summary requests and the merges after aborted calls are flagged."""
        canonical_blocks = cs._canonical_blocks

        def no_string_canonicalisation(content):
            return [content] if isinstance(content, str) else canonical_blocks(content)

        def strict_prefix(base, new):
            return None if tuple(new[:len(base)]) == tuple(base) else cs.HISTORY_REWRITTEN

        with mock.patch.object(cs, "_canonical_blocks", no_string_canonicalisation):
            cells, _, _, _ = replay(self.fixture, commit_when=lambda r: r["delivered"])
        self.assertGreater(cells["claude_code"]["fp"], 50)
        with mock.patch.object(cs, "_history_reason", strict_prefix):
            _, _, summary_requests, _ = replay(self.fixture, commit_when=lambda r: r["delivered"])
            cells, _, _, _ = replay(self.fixture, commit_when=lambda r: r["status"] == 200)
        self.assertEqual(summary_requests, [cs.HISTORY_REWRITTEN] * 2)
        self.assertEqual((cells["claude_code"]["fp"], cells["opencode"]["fp"]), (1, 1))


# ---------------------------------------------------------------------------------- tracker unit cases
class TrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = ContinuityTracker()

    def commit(self, thread, fingerprints, images=0, **kw):
        self.tracker.commit(thread, helper=False, fingerprints=fingerprints, image_count=images, **kw)

    def observe(self, thread, fingerprints, images=0, **kw):
        return self.tracker.observe(thread, helper=kw.pop("helper", False), fingerprints=fingerprints,
                                    image_count=images, **kw)

    def test_first_request_on_a_thread_is_not_compared(self):
        self.assertIsNone(self.observe("t", fp(("x", 1))))

    def test_prefix_extension_is_continuous(self):
        self.commit("t", fp(("u", 1), ("a", 2)))
        self.assertIsNone(self.observe("t", fp(("u", 1), ("a", 2))))
        self.assertIsNone(self.observe("t", fp(("u", 1), ("a", 2), ("u2", 1), ("a2", 1))))

    def test_last_item_may_gain_blocks(self):
        self.commit("t", fp(("u", 1), ("a", 1), ("tool", ("r1",))))
        self.assertIsNone(self.observe("t", fp(("u", 1), ("a", 1), ("tool", ("r1", "steer")))))
        self.assertIsNone(self.observe("t", fp(("u", 1), ("a", 1), ("tool", ("r1", "steer")), ("a", 1))))
        # The last item's existing blocks must stay a prefix, and its head must not change.
        self.assertEqual(self.observe("t", fp(("u", 1), ("a", 1), ("tool", ("steer", "r1")))), cs.HISTORY_REWRITTEN)
        self.assertEqual(self.observe("t", fp(("u", 1), ("a", 1), ("other", ("r1", "x")))), cs.HISTORY_REWRITTEN)

    def test_only_the_last_item_may_gain_blocks(self):
        self.commit("t", fp(("u", ("q",)), ("a", 1)))
        self.assertEqual(self.observe("t", fp(("u", ("q", "extra")), ("a", 1))), cs.HISTORY_REWRITTEN)

    def test_shortened_history(self):
        self.commit("t", fp(("u", 1), ("a", 1), ("u", 1), ("a", 1)))
        self.assertEqual(self.observe("t", fp(("u", 1), ("a", 1))), cs.HISTORY_SHORTENED)
        self.assertEqual(self.observe("t", []), cs.HISTORY_SHORTENED)
        # Compaction: fewer items and a new head item.
        self.assertEqual(self.observe("t", fp(("summary", 1), ("a", 1))), cs.HISTORY_SHORTENED)
        # The last item losing blocks is a rewrite, not a shortening (same item count).
        self.commit("t2", fp(("u", ("q", "r"))))
        self.assertEqual(self.observe("t2", fp(("u", ("q",)))), cs.HISTORY_REWRITTEN)

    def test_rewritten_earlier_item(self):
        self.commit("t", fp(("u", 1), ("tool", ("result",)), ("a", 1)))
        pruned = fp(("u", 1), ("tool", ("[cleared]",)), ("a", 1), ("u", 1))
        self.assertEqual(self.observe("t", pruned), cs.HISTORY_REWRITTEN)

    def test_image_drop(self):
        self.commit("t", fp(("u", 1), ("tool", 2)), images=2)
        self.assertEqual(self.observe("t", fp(("u", 1), ("tool", 2), ("a", 1)), images=1), cs.IMAGES_REMOVED)
        self.assertIsNone(self.observe("t", fp(("u", 1), ("tool", 2), ("a", 1)), images=3))
        # A history loss takes precedence over the image count.
        self.assertEqual(self.observe("t", fp(("u", 1)), images=0), cs.HISTORY_SHORTENED)

    def test_system_key_is_compared_only_when_both_sides_carry_one(self):
        self.commit("t", fp(("u", 1)), system_key="s1")
        self.assertEqual(self.observe("t", fp(("u", 1)), system_key="s2"), cs.SYSTEM_CHANGED)
        self.assertIsNone(self.observe("t", fp(("u", 1)), system_key="s1"))
        self.assertIsNone(self.observe("t", fp(("u", 1))))
        self.commit("t", fp(("u", 1)))
        self.assertIsNone(self.observe("t", fp(("u", 1)), system_key="s2"))
        # History and image reasons come first.
        self.commit("t", fp(("u", 1), ("a", 1)), images=1, system_key="s1")
        self.assertEqual(self.observe("t", fp(("u", 1)), system_key="s1"), cs.HISTORY_SHORTENED)
        self.assertEqual(self.observe("t", fp(("u", 1), ("a", 1)), images=0, system_key="s2"), cs.IMAGES_REMOVED)
        # Revision 4 (R4.3): another system text whose history does not continue the main chain is a subagent
        # chain, never a discontinuity (P1-3a reported history_shortened here).
        self.assertIsNone(self.observe("t", fp(("u", 1)), system_key="s2"))
        self.assertEqual(self.tracker.admit("t", helper=False, fingerprints=fp(("u", 1)), image_count=0,
                                            system_key="s2"), cs.Admission(cs.SUBAGENT, None))

    def test_observe_never_commits(self):
        self.commit("t", fp(("u", 1), ("a", 1)))
        longer = fp(("u", 1), ("a", 1), ("u", 1), ("a", 1))
        self.assertIsNone(self.observe("t", longer))
        # The baseline did not move to `longer`, so this two-item history is still continuous.
        self.assertIsNone(self.observe("t", fp(("u", 1), ("a", 1))))
        # A discontinuous history that was not delivered is reported again on every re-send.
        self.assertEqual(self.observe("t", fp(("x", 1))), cs.HISTORY_SHORTENED)
        self.assertEqual(self.observe("t", fp(("x", 1))), cs.HISTORY_SHORTENED)

    def test_commit_on_delivery_only(self):
        """A failed attempt is never a baseline: Claude Code re-sends the failed prompt merged with the next one."""
        delivered = fp(("u", 1), ("a", 1), ("tool", ("r",)))
        self.commit("t", delivered)
        failed = fp(("u", 1), ("a", 1), ("tool", ("r", "prompt-1")))          # answered 5xx, not committed
        self.assertIsNone(self.observe("t", failed))
        merged = fp(("u", 1), ("a", 1), ("tool", ("r", "prompt-1", "prompt-2")))
        self.assertIsNone(self.observe("t", merged))
        self.commit("t", merged)
        self.assertEqual(self.observe("t", delivered), cs.HISTORY_REWRITTEN)

    def test_committing_a_discontinuous_history_makes_it_the_new_baseline(self):
        self.commit("t", fp(("u", 1), ("a", 1), ("u", 1)))
        compacted = fp(("summary", 1), ("tail", 1))
        self.assertEqual(self.observe("t", compacted), cs.HISTORY_SHORTENED)
        self.commit("t", compacted)
        self.assertIsNone(self.observe("t", compacted + fp(("a", 1))))

    def test_helper_is_neither_compared_nor_committed(self):
        self.commit("t", fp(("u", 1), ("a", 1)), images=1)
        self.assertIsNone(self.observe("t", fp(("summarize", 1)), helper=True))
        self.tracker.commit("t", helper=True, fingerprints=fp(("summarize", 1)), image_count=0)
        self.assertIsNone(self.observe("t", fp(("u", 1), ("a", 1), ("u", 1)), images=1))
        # A helper on a thread without a baseline does not create one.
        self.tracker.commit("fresh", helper=True, fingerprints=fp(("title", 1)), image_count=0)
        self.assertIsNone(self.observe("fresh", fp(("other", 1))))

    def test_separate_threads(self):
        self.commit("session-a", fp(("u", ("same",)), ("a", 1), ("u", 1)))
        self.commit("session-b", fp(("u", ("same",))))
        self.assertIsNone(self.observe("session-b", fp(("u", ("same",)), ("a", 1))))
        self.assertEqual(self.observe("session-a", fp(("u", ("same",)), ("a", 1))), cs.HISTORY_SHORTENED)
        self.assertIsNone(self.observe("", fp(("x", 1))))
        self.commit("", fp(("x", 1)))
        self.assertEqual(self.observe("", fp(("y", 1))), cs.HISTORY_REWRITTEN)
        self.assertIsNone(self.observe("session-b", fp(("u", ("same",)))))

    def test_reset(self):
        self.commit("t", fp(("u", 1), ("a", 1)))
        self.tracker.reset("t")
        self.assertIsNone(self.observe("t", fp(("other", 1))))
        self.tracker.reset("never-seen")

    def test_inputs_are_validated(self):
        for bad in ([("head",)], [("head", "blocks-as-one-string")], [("head", (1, 2))], [(None, ())], ["head"]):
            with self.assertRaises(TypeError):
                self.observe("t", bad)
            with self.assertRaises(TypeError):
                self.commit("t", bad)
        for bad in (-1, 1.5, True, None):
            with self.assertRaises(ValueError):
                self.observe("t", fp(("u", 1)), images=bad)
            with self.assertRaises(ValueError):
                self.commit("t", fp(("u", 1)), images=bad)

    def test_caller_mutation_after_commit_does_not_change_the_baseline(self):
        history = [("u", ["b0"])]
        self.commit("t", history)
        history[0][1].append("b1")
        history.append(("a", ["x"]))
        self.assertEqual(self.observe("t", []), cs.HISTORY_SHORTENED)
        self.assertIsNone(self.observe("t", fp(("u", ("b0",)))))

    def test_thread_safety(self):
        """Interleaved observe/commit from many threads, on private and on shared thread keys."""
        tracker, workers, steps = ContinuityTracker(), 16, 150
        barrier, errors = threading.Barrier(workers), []

        def run(worker):
            try:
                key, history = "thread-%d" % worker, []
                barrier.wait()
                for step in range(steps):
                    history.append(("item-%d" % step, ("block-%d-%d" % (worker, step),)))
                    reason = tracker.observe(key, helper=False, fingerprints=history, image_count=step)
                    if reason is not None:
                        errors.append((worker, step, reason))
                    tracker.commit(key, helper=False, fingerprints=history, image_count=step)
                    # Shared key: each commit is a complete baseline, so an observation always sees one of them.
                    shared = [("shared", ("w%d" % worker,))]
                    tracker.observe("shared", helper=False, fingerprints=shared, image_count=0)
                    tracker.commit("shared", helper=False, fingerprints=shared, image_count=0)
                if tracker.observe(key, helper=False, fingerprints=history[:-1], image_count=steps) != cs.HISTORY_SHORTENED:
                    errors.append((worker, "final", "shortened not detected"))
            except Exception as exc:  # pragma: no cover - reported below
                errors.append((worker, "exception", repr(exc)))

        threads = [threading.Thread(target=run, args=(w,)) for w in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(errors, [])
        final = tracker.observe("shared", helper=False, fingerprints=[("shared", ("w0",))], image_count=0)
        self.assertIn(final, (None, cs.HISTORY_REWRITTEN))


# ---------------------------------------------------------------------------------- chains and call_purpose (R4.3)
class ChainTest(unittest.TestCase):
    """Chains keyed by (thread, system_key); only the main chain of a main thread is compared and committed."""

    def setUp(self):
        self.tracker = ContinuityTracker()

    def admit(self, thread, fingerprints, images=0, **kw):
        return self.tracker.admit(thread, helper=kw.pop("helper", False), fingerprints=fingerprints,
                                  image_count=images, **kw)

    def commit(self, thread, fingerprints, images=0, **kw):
        self.tracker.commit(thread, helper=kw.pop("helper", False), fingerprints=fingerprints, image_count=images,
                            **kw)

    def test_call_purposes(self):
        self.assertEqual(cs.CALL_PURPOSES, ("main", "helper", "subagent"))
        self.assertEqual(self.admit("t", fp(("u", 1)), helper=True, system_key="h"), (cs.HELPER, None))
        self.assertEqual(self.admit("t", fp(("u", 1)), system_key="s"), (cs.MAIN, None))  # new thread
        self.assertEqual(self.admit("t", fp(("u", 1)), system_key="s", main_thread=False), (cs.SUBAGENT, None))
        self.assertEqual(self.admit("t", fp(("u", 1)), helper=True, main_thread=False), (cs.HELPER, None))

    def test_subagent_chain_on_the_main_thread_never_raises_and_never_commits(self):
        main = fp(("u", 1), ("a", 1), ("u", ("r",)))
        self.commit("t", main, images=1, system_key="main")
        sub = fp(("task", 1), ("a", 1), ("u", ("r1",)), ("a", 1))
        for history in (sub, sub + fp(("u", 1)), fp(("summary", 1)), fp(("task", 1), ("a", 1), ("u", ("cleared",)))):
            # The subagent's own growth, compaction and pruning: never a discontinuity, never a baseline.
            self.assertEqual(self.admit("t", history, system_key="sub"), (cs.SUBAGENT, None))
            self.commit("t", history, system_key="sub")
            self.commit("t", history, system_key="sub", call_purpose=cs.SUBAGENT)
        # The main chain's baseline did not move, so it still continues, and its compaction is still detected.
        self.assertEqual(self.admit("t", main + fp(("a", 1)), images=1, system_key="main"), (cs.MAIN, None))
        self.assertEqual(self.admit("t", fp(("summary", 1)), system_key="main"), (cs.MAIN, cs.HISTORY_SHORTENED))

    def test_main_chain_is_fixed_by_the_first_committed_request_with_tools(self):
        # A helper and an undelivered request fix nothing.
        self.commit("t", fp(("title", 1)), helper=True, system_key="title")
        self.assertEqual(self.admit("t", fp(("u", 1)), system_key="main"), (cs.MAIN, None))
        self.assertEqual(self.admit("t", fp(("x", 1)), system_key="other"), (cs.MAIN, None))
        self.commit("t", fp(("u", 1)), system_key="main")
        self.assertEqual(self.admit("t", fp(("x", 1)), system_key="other"), (cs.SUBAGENT, None))
        self.assertEqual(self.admit("t", fp(("x", 1)), system_key="main"), (cs.MAIN, cs.HISTORY_REWRITTEN))
        # Each thread has its own main chain.
        self.assertEqual(self.admit("t2", fp(("x", 1)), system_key="other"), (cs.MAIN, None))

    def test_a_thread_that_is_not_a_main_thread_is_a_subagent_even_with_the_main_system_text(self):
        """OpenCode child sessions: own session id, possibly the same system text (the general agent)."""
        self.commit("main", fp(("u", 1), ("a", 1)), system_key="s")
        self.commit("child", fp(("task", 1)), system_key="s", main_thread=False)
        self.commit("child", fp(("task", 1)), system_key="s", main_thread=False, call_purpose=cs.MAIN)
        self.assertEqual(self.admit("child", fp(("other", 1)), system_key="s", main_thread=False), (cs.SUBAGENT, None))
        self.assertEqual(self.admit("child", fp(("other", 1)), system_key="s"), (cs.MAIN, None))  # never committed

    def test_main_chain_follows_a_system_change_that_continues_its_history(self):
        """A live model switch or a date in the system text: the main conversation goes on under a new key."""
        self.commit("t", fp(("u", 1), ("a", 1)), system_key="s1")
        grown = fp(("u", 1), ("a", 1), ("u", 1))
        self.assertEqual(self.admit("t", grown, system_key="s2"), (cs.MAIN, cs.SYSTEM_CHANGED))
        self.commit("t", grown, system_key="s2", call_purpose=cs.MAIN)
        self.assertEqual(self.admit("t", grown + fp(("a", 1)), system_key="s2"), (cs.MAIN, None))
        # The old key is now another chain; a later compaction of the main chain is still detected.
        self.assertEqual(self.admit("t", fp(("task", 1)), system_key="s1"), (cs.SUBAGENT, None))
        self.assertEqual(self.admit("t", fp(("summary", 1)), system_key="s2"), (cs.MAIN, cs.HISTORY_SHORTENED))
        # An empty main baseline is no evidence of continuation.
        self.commit("e", [], system_key="s1")
        self.assertEqual(self.admit("e", fp(("x", 1)), system_key="s2"), (cs.SUBAGENT, None))

    def test_commit_takes_the_admitted_purpose(self):
        self.commit("t", fp(("u", 1)), system_key="main")
        sub = fp(("task", 1))
        # Classified now (no purpose given): a subagent request is not committed.
        self.commit("t", sub, system_key="sub")
        self.assertEqual(self.admit("t", fp(("u", 1), ("a", 1)), system_key="main"), (cs.MAIN, None))
        # The admitted purpose wins over the state at delivery time.
        self.commit("t", fp(("u", 1), ("a", 1)), system_key="main", call_purpose=cs.SUBAGENT)
        self.assertEqual(self.admit("t", fp(("u", 1)), system_key="main"), (cs.MAIN, None))
        self.commit("t", sub, system_key="sub", call_purpose=cs.MAIN)
        self.assertEqual(self.admit("t", sub + fp(("a", 1)), system_key="sub"), (cs.MAIN, None))
        self.commit("t", fp(("h", 1)), helper=True, call_purpose=cs.MAIN)  # a helper is never committed
        self.assertEqual(self.admit("t", sub + fp(("a", 1)), system_key="sub"), (cs.MAIN, None))
        for bad in ("primary", "", 1):
            with self.assertRaises(ValueError):
                self.commit("t", sub, call_purpose=bad)

    def test_reset_forgets_the_main_chain(self):
        self.commit("t", fp(("u", 1)), system_key="main")
        self.tracker.reset("t")
        self.assertEqual(self.admit("t", fp(("x", 1)), system_key="other"), (cs.MAIN, None))
        self.commit("t", fp(("x", 1)), system_key="other")
        self.assertEqual(self.admit("t", fp(("u", 1)), system_key="main"), (cs.SUBAGENT, None))


class SyntheticSubagentSequenceTest(unittest.TestCase):
    """Captured main chains with an interleaved subagent chain: synthetic from spike rows, or the captured R4-0
    subagent requests (see ``subagent_interleaving``)."""

    @classmethod
    def setUpClass(cls):
        cls.sequences = {s["id"]: s for s in json.loads(FIXTURE.read_text())["sequences"]}

    def check(self, sequence, admissions):
        """Subagent rows: never a reason. Main rows: helper/main as labelled, a reason exactly on context_lost rows."""
        lost, sub = [], Counter()
        for k, (request, admission) in enumerate(zip(sequence["requests"], admissions)):
            if request["chain"] == "subagent":
                sub[admission.call_purpose] += 1
                self.assertIsNone(admission.reason, (k, request))
                self.assertIn(admission.call_purpose, (cs.SUBAGENT, cs.HELPER), (k, request))
            else:
                self.assertEqual(admission.call_purpose, cs.HELPER if request["label"] == "helper" else cs.MAIN)
                self.assertEqual(admission.reason is not None, request["label"] == "context_lost", (k, request))
                if admission.reason:
                    lost.append(admission.reason)
        return lost, sub

    def test_claude_main_compaction_with_an_interleaved_task_subagent(self):
        sequence = subagent_interleaving(self.sequences["claude_code/s5-compact/model-requests"],
                                         self.sequences["claude_code/s5-normal/model-requests"])
        self.assertEqual(Counter(r["chain"] for r in sequence["requests"]), {"main": 26, "subagent": 24})
        lost, sub = self.check(sequence, replay_chains(sequence))
        self.assertEqual(lost, [cs.HISTORY_SHORTENED] * 2)
        self.assertEqual(sub, {cs.SUBAGENT: 24})

    def test_claude_main_compaction_with_captured_subagent_requests(self):
        """The R4-0 subagent requests (real system text and Task prompts, two subagents) interleaved with the
        captured S-B compaction: still exactly the two main compactions."""
        probe = self.sequences["claude_code/r4-0/cc-agent/model-requests"]
        sub = dict(probe, requests=[r for r in probe["requests"] if r["label"] == "subagent"])
        sequence = subagent_interleaving(self.sequences["claude_code/s5-compact/model-requests"], sub,
                                         sub_system=None, task_prompt=False, start=4, burst=1)
        lost, purposes = self.check(sequence, replay_chains(sequence))
        self.assertEqual(lost, [cs.HISTORY_SHORTENED] * 2)
        self.assertEqual(purposes, {cs.SUBAGENT: 5})

    def test_claude_subagent_compaction_never_raises(self):
        """The captured Claude Code compaction (two summaries, two shortened histories) replayed as the subagent."""
        sequence = subagent_interleaving(self.sequences["claude_code/s5-normal/model-requests"],
                                         self.sequences["claude_code/s5-compact/model-requests"], start=1, burst=1)
        lost, sub = self.check(sequence, replay_chains(sequence))
        self.assertEqual(lost, [])
        self.assertEqual(sub, {cs.SUBAGENT: 26})
        # Negative control: without system keys the interleaved chain is read as the main chain and floods reasons.
        reasons = [a.reason for a in replay_chains(sequence, system=False) if a.reason]
        self.assertGreater(len(reasons), 10)

    def test_opencode_child_session_with_the_main_system_text(self):
        """An OpenCode child session (own x-opencode-session-id, same system text as the main session: the general
        agent) runs the captured prune sequence while the main session compacts."""
        sequence = subagent_interleaving(self.sequences["opencode/s9-anthropic-compact"],
                                         self.sequences["opencode/s9-anthropic-prune"], sub_thread="t-child",
                                         sub_system=MAIN_SYSTEM)
        lost, sub = self.check(sequence, replay_chains(sequence, main_threads={"t1"}))
        self.assertEqual(lost, [cs.HISTORY_SHORTENED])
        self.assertEqual(sub, {cs.SUBAGENT: 15})
        # Negative control: had the child been tracked on the main thread (the relay's attribution maps an
        # unregistered session id to default_thread_id), the same system text would make it a main request.
        merged = dict(sequence, requests=[dict(r, thread="t1") for r in sequence["requests"]])
        reasons = [a.reason for a in replay_chains(merged, main_threads={"t1"}) if a.reason]
        self.assertGreater(len(reasons), 10)


# ---------------------------------------------------------------------------------- wire inputs
class AnthropicInputsTest(unittest.TestCase):
    def test_string_content_equals_one_text_block(self):
        a = anthropic([{"role": "user", "content": "hello"}])
        b = anthropic([{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}])
        self.assertEqual(anthropic_inputs(a), anthropic_inputs(b))
        c = anthropic([{"role": "user", "content": [{"type": "text", "text": "hello!"}]}])
        self.assertNotEqual(anthropic_inputs(a)["fingerprints"], anthropic_inputs(c)["fingerprints"])

    def test_claude_code_reserialisation_is_continuous(self):
        """Claude Code turns the previous last message (and the env system message) from a cached block list into a
        plain string, moves cache markers and adds the steer text to the last user message (reports 05 S3/S5)."""
        env = "# Environment\nworking directory <run>"
        first = anthropic([
            {"role": "user", "content": [{"type": "text", "text": "Run the probe.", "cache_control": {"type": "ephemeral"}}]},
            {"role": "system", "content": [{"type": "text", "text": env, "cache_control": {"type": "ephemeral"}}],
             "output_config": {"effort": "medium"}},
        ])
        second = anthropic([
            {"role": "user", "content": "Run the probe."},
            {"role": "system", "content": env, "output_config": {"effort": "medium"}},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "mcp__auriga__observe", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                {"type": "text", "text": "observation"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}],
                "cache_control": {"type": "ephemeral"}}]},
        ])
        third = json.loads(json.dumps(second))
        del third["messages"][3]["content"][0]["cache_control"]
        third["messages"][3]["content"].append({"type": "text", "text": "STEER", "cache_control": {"type": "ephemeral"}})
        tracker = ContinuityTracker()
        tracker.commit("s", **anthropic_inputs(first, system=True))
        self.assertIsNone(tracker.observe("s", **anthropic_inputs(second, system=True)))
        tracker.commit("s", **anthropic_inputs(second, system=True))
        self.assertIsNone(tracker.observe("s", **anthropic_inputs(third, system=True)))
        self.assertEqual(anthropic_inputs(third)["image_count"], 1)

    def test_split_keeps_role_system_messages_as_items(self):
        body = anthropic([{"role": "user", "content": "u"}, {"role": "system", "content": "env"}],
                         system=[{"type": "text", "text": "S"}])
        view = cs.anthropic_messages_history(body)
        self.assertEqual(view.system, ([{"type": "text", "text": "S"}],))
        self.assertEqual(len(view.items), 2)
        self.assertEqual(cs.conversation_key(view), cs.conversation_key(cs.anthropic_messages_history(anthropic([], system="S"))))
        self.assertNotEqual(cs.conversation_key(view), cs.conversation_key(cs.anthropic_messages_history(anthropic([], system="T"))))
        self.assertEqual(cs.anthropic_messages_history({}), cs.HistoryView((), ()))
        self.assertEqual(cs.anthropic_messages_history(None), cs.HistoryView((), ()))

    def test_volatile_keys_dropped_but_content_compared(self):
        def tool_round(call_id, result):
            return anthropic([
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "observe", "input": {"id": "obs-1"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result}]},
            ])
        base = anthropic_inputs(tool_round("toolu_A", "ok"))
        self.assertEqual(base, anthropic_inputs(tool_round("toolu_B", "ok")))
        self.assertNotEqual(base, anthropic_inputs(tool_round("toolu_A", "[Old tool result content cleared]")))
        # Nested tool_result content is canonical as well: a string equals one text block.
        self.assertEqual(base, anthropic_inputs(tool_round("toolu_A", [{"type": "text", "text": "ok"}])))
        # Tool arguments are model output and digested verbatim, including a key named like a volatile field.
        moved = tool_round("toolu_A", "ok")
        moved["messages"][1]["content"][0]["input"] = {"id": "obs-2"}
        self.assertNotEqual(base["fingerprints"], anthropic_inputs(moved)["fingerprints"])

    def test_image_bytes_and_count(self):
        def with_image(data, n=1):
            return anthropic([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}] * n}]}])
        self.assertNotEqual(anthropic_inputs(with_image("AAAA"))["fingerprints"], anthropic_inputs(with_image("BBBB"))["fingerprints"])
        self.assertEqual(anthropic_inputs(with_image("AAAA", 3))["image_count"], 3)
        no_image_in_args = anthropic([{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "x", "input": {"type": "image"}}]}])
        self.assertEqual(anthropic_inputs(no_image_in_args)["image_count"], 0)

    def test_helper_request(self):
        self.assertTrue(cs.is_helper_request(anthropic([{"role": "user", "content": "title?"}], tools=False)))
        self.assertTrue(cs.is_helper_request({"messages": [], "tools": []}))
        self.assertTrue(cs.is_helper_request(None))
        self.assertFalse(cs.is_helper_request(anthropic([])))


class ResponsesInputsTest(unittest.TestCase):
    def body(self, items, instructions="INSTR", tools=True):
        body = {"model": "m", "instructions": instructions, "input": items}
        if tools:
            body["tools"] = [{"type": "function", "name": "auriga_observe"}]
        return body

    def view(self, body):
        return cs.openai_responses_history(body)

    def test_split_and_system_key(self):
        body = self.body([{"role": "developer", "content": "D"}, {"role": "system", "content": "S"},
                          {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}])
        view = self.view(body)
        self.assertEqual(len(view.system), 3)
        self.assertEqual(len(view.items), 1)
        changed = self.view(self.body([{"role": "developer", "content": "D2"}, {"role": "user", "content": "hi"}]))
        self.assertNotEqual(cs.conversation_key(view), cs.conversation_key(changed))
        self.assertEqual(cs.history_fingerprints(view), cs.history_fingerprints(changed))
        # A developer item after the first conversation item stays in the history.
        late = self.view(self.body([{"role": "user", "content": "hi"}, {"role": "developer", "content": "D"}]))
        self.assertEqual((len(late.system), len(late.items)), (1, 2))
        self.assertEqual(self.view({"input": "hi"}).items, ({"role": "user", "content": "hi"},))

    def test_string_equals_input_text_block_and_ids_dropped(self):
        a = self.view(self.body([{"type": "message", "role": "user", "content": "hi"},
                                 {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "observe", "arguments": "{}"},
                                 {"type": "function_call_output", "call_id": "call_1", "output": "done"}]))
        b = self.view(self.body([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                                 {"type": "function_call", "id": "fc_9", "call_id": "call_9", "name": "observe", "arguments": "{}"},
                                 {"type": "function_call_output", "call_id": "call_9", "output": [{"type": "input_text", "text": "done"}]}]))
        self.assertEqual(cs.history_fingerprints(a), cs.history_fingerprints(b))
        c = self.view(self.body([{"type": "message", "role": "user", "content": "hi"},
                                 {"type": "function_call", "call_id": "call_1", "name": "observe", "arguments": '{"x":1}'},
                                 {"type": "function_call_output", "call_id": "call_1", "output": "done"}]))
        self.assertNotEqual(cs.history_fingerprints(a)[1], cs.history_fingerprints(c)[1])

    def test_image_count(self):
        out = {"type": "function_call_output", "call_id": "c", "output": [
            {"type": "input_text", "text": "obs"}, {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}
        self.assertEqual(cs.image_count(self.view(self.body([out, out]))), 2)


class ChatInputsTest(unittest.TestCase):
    def body(self, messages, tools=True):
        body = {"model": "m", "messages": messages}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": "auriga_observe"}}]
        return body

    def test_split_string_equals_text_part_and_tool_ids_dropped(self):
        a = cs.openai_chat_history(self.body([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "observe", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
        ]))
        b = cs.openai_chat_history(self.body([
            {"role": "system", "content": [{"type": "text", "text": "S"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_7", "type": "function", "function": {"name": "observe", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_7", "content": [{"type": "text", "text": "done"}]},
            {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
        ]))
        self.assertEqual(len(a.system), 1)
        self.assertEqual(cs.conversation_key(a), cs.conversation_key(b))
        self.assertEqual(cs.history_fingerprints(a), cs.history_fingerprints(b))
        self.assertEqual(cs.image_count(a), 1)

    def test_abort_then_two_user_items_is_continuous(self):
        """opencode on Chat/Responses sends the post-abort prompt as a separate user item (report 06 S7.2)."""
        tracker = ContinuityTracker()
        first = self.body([{"role": "system", "content": "S"}, {"role": "user", "content": "USER-ONE"}])
        second = self.body([{"role": "system", "content": "S"}, {"role": "user", "content": "USER-ONE"},
                            {"role": "user", "content": "USER-AFTER-ABORT"}])
        view = cs.openai_chat_history
        tracker.commit("s", **inputs(view(first), system=True))
        self.assertIsNone(tracker.observe("s", **inputs(view(second), system=True)))


if __name__ == "__main__":
    unittest.main()
