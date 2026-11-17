"""Camera supervisor of a running WALL-OSS delegation (gate ``pickplace_v9``).

Every ``wall_oss.monitor_every_chunks`` executed chunks (4 = 64 policy steps by default) the delegation loop asks one
stateless vision-language model call whether the policy should continue. The call sees the overall task instruction,
the sub-instruction the actor delegated (and the actor's subtask note), the budget used and left, the head camera at the
start of the call and at up to ``SEQUENCE_HISTORY`` earlier checks, and the three current cameras, and answers one
verdict:

* ``normal``: progress is as expected over the whole sequence and the rest fits in the budget left;
* ``done``: the sub-instruction is completed; the loop stops the call (``stop_reason`` ``monitor_done``);
* ``takeover``: progress is not as expected (the same attempt repeated, no object getting closer to its end state, an
  object lost while nothing else progresses, or too slow for the budget left). After
  ``wall_oss.monitor_takeover_consecutive`` (default 1) takeover verdicts in a row the loop stops the call
  (``monitor_takeover``) and control returns to the actor.

It never ends a call on its own failure: a call that errors or answers outside the schema counts as ``normal`` and is
logged. It is separate from the actor's model thread (no shared context), so each check costs one small request.

The key is read from ``wall_oss.monitor_key_file`` (JSON with OPENAI_API_KEY / api_key, or KEY=VALUE lines) and is
never logged. Requests go to ``wall_oss.monitor_base_url`` + ``/chat/completions`` (OpenAI compatible).
"""
import base64
import json
import time
import urllib.request

VERDICTS = ("normal", "done", "takeover")
CAMERAS = (("cam_head", "head camera now"), ("cam_left_wrist", "left wrist camera now"),
           ("cam_right_wrist", "right wrist camera now"))
SYSTEM = ("You supervise a bimanual robot. An agent delegated one sub-instruction to a low-level visuomotor policy, "
          "which is now executing it. Every few policy steps you look at the cameras and decide whether to let the "
          "policy continue. Answer with JSON only.")


# Sequence question. A question that only compares the head camera with the one at the previous check keeps no memory:
# a policy retrying the same failing attempt looks like progress at every check, and a pause between two objects of a
# good call looks like no progress. This question sees the head camera at the call start and at up to SEQUENCE_HISTORY
# earlier checks, the current cameras and the budget used, and asks whether progress is as expected. Offline replay on
# 50 recorded calls (checks every 64 steps, one takeover verdict stops): 0/24 good calls stopped, 17/21 late-failing
# calls stopped at median step 256 instead of 480; a dropped object while others progress is not a takeover.
SEQUENCE_HISTORY = 4
SEQUENCE_TEXT = """Overall task: {task}
Sub-instruction given to the policy: {instruction}{note}
Budget: the policy has used {steps} of the {budget} policy steps of this delegation ({pct}% used, {left} left; 25 steps per second).
Images, oldest first: the head camera at the start of this delegation (step 0){history}; then the current {cameras}.

Judge the progress over the whole sequence, not only the last interval, against what the sub-instruction needs and the budget left:
- done: the sub-instruction is completed now; the policy should be stopped.
- normal: progress is as expected. Across the sequence, objects of the sub-instruction have reached their end state or are clearly and steadily getting there, and at this rate the rest fits in the budget left. Approaching, grasping, regrasping, short pauses and moving on to the next object are normal when the sequence shows completed or advancing objects. One object dropped or misplaced is also normal for now while the policy keeps completing the other objects (the agent fixes it afterwards); mention it in completed.
- takeover: progress is not as expected. For example: the same attempt is repeated without being completed (an object brought to the same place again and again); across the sequence no object of the sub-instruction has reached its end state or come clearly closer to it; an object was dropped, knocked over or moved away from its goal and the policy is neither recovering it nor completing other objects; the policy works on something else; or at the current rate the budget left will clearly not be enough.
Reply with JSON only: {{"verdict": "normal" | "done" | "takeover", "confidence": 0.0-1.0, "completed": "<what of the sub-instruction is already done>", "reason": "<one short sentence>"}}"""


def sequence_prompt(context):
    budget = max(1, int(context.get("budget") or 1))
    steps = int(context.get("steps") or 0)
    names = {"cam_head": "head camera (step %d)" % steps, "cam_left_wrist": "left wrist camera",
             "cam_right_wrist": "right wrist camera"}
    cameras = [names[camera] for camera, _ in CAMERAS if camera in context.get("cameras", ())]
    note = f"\nAgent's note for this sub-instruction: {context['note']}" if context.get("note") else ""
    return SEQUENCE_TEXT.format(task=context.get("task_instruction") or "unknown", instruction=context["instruction"],
                                note=note, steps=steps, budget=budget, pct=round(100 * steps / budget),
                                left=max(0, budget - steps),
                                history="".join(f", at step {s}" for s in context.get("history_steps", ())[1:]),
                                cameras=", ".join(cameras) or "cameras")


def _mime(data):
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def read_key(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    try:
        value = json.loads(text)
    except ValueError:
        value = dict(line.strip().split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("#"))
    for name in ("OPENAI_API_KEY", "api_key", "API_KEY"):
        if isinstance(value, dict) and isinstance(value.get(name), str) and value[name].strip():
            return value[name].strip()
    raise ValueError("monitor_key_file_has_no_key")


def parse_verdict(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("monitor_answer_not_json")
    answer = json.loads(text[start:end + 1])
    verdict = str(answer.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"monitor_verdict_invalid:{verdict[:40]}")
    confidence = answer.get("confidence")
    result = dict(verdict=verdict, confidence=confidence if isinstance(confidence, (int, float)) else None,
                  reason=str(answer.get("reason", ""))[:500])
    if answer.get("completed") is not None:
        result["completed"] = str(answer.get("completed"))[:500]
    return result


class GptProgressMonitor:
    """One OpenAI-compatible chat/completions request per check; never raises from ``check``."""

    def __init__(self, base_url, key, model, timeout_s=120.0, retries=2):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._key, self.model, self.timeout_s, self.retries = key, model, timeout_s, retries

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.monitor_base_url, read_key(settings.monitor_key_file), settings.monitor_model,
                   settings.monitor_timeout_s)

    def check(self, context, images):
        """context: task_instruction, instruction, note, steps, budget, history_steps; images: {name: bytes} with the
        current cameras and 'history' (list of head images, oldest first, the first at the call start)."""
        text = sequence_prompt(dict(context, cameras=tuple(name for name in images if name != "history")))
        ordered = list(images.get("history") or []) + [images.get(camera) for camera, _ in CAMERAS]
        content = [dict(type="text", text=text)]
        for data in ordered:
            if data:
                content.append(dict(type="image_url", image_url=dict(
                    url=f"data:{_mime(data)};base64,{base64.b64encode(data).decode()}")))
        body = json.dumps(dict(model=self.model, messages=[dict(role="system", content=SYSTEM),
                                                           dict(role="user", content=content)])).encode()
        started, error = time.monotonic(), None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(self.url, data=body, headers={
                    "Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                    "User-Agent": "auriga-monitor/1"})
                raw = urllib.request.urlopen(request, timeout=self.timeout_s).read().decode("utf-8", "replace")
                if "Concurrency limit exceeded" in raw:
                    raise RuntimeError("monitor_concurrency_rejected")
                response = json.loads(raw)
                result = parse_verdict(response["choices"][0]["message"]["content"])
                return dict(result, usage=response.get("usage"), latency_s=round(time.monotonic() - started, 2),
                            attempts=attempt + 1, model=self.model)
            except Exception as exc:  # network, provider or parse error: retry, then report as an error
                error = f"{type(exc).__name__}: {exc}"[:300]
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        return dict(verdict=None, error=error, latency_s=round(time.monotonic() - started, 2),
                    attempts=self.retries + 1, model=self.model)
