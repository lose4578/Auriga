"""Capture the Responses request the pinned Codex app-server sends for a turn with ``outputSchema`` (P1-7).

Design §9.3: the relay evidence client's Responses builder copies ``text.format`` (and ``strict``) from what the
pinned Codex evidence path asks the provider for, so both evidence paths request the same structured output. This
script drives the real pinned Codex app-server (``codex_cli_bin.bundled_codex_path()``) through the unchanged
``CodexEvidenceAgent`` against ``tests/support/fake_model_servers.FakeResponsesServer`` (loopback only, no model, no
network, sandbox off) and writes ``codex_output_schema_request.json`` next to this file: the evidence request's
top-level key list and every field the builder mirrors (``text``, ``reasoning``, ``store``, ``stream``, ``tools``,
``tool_choice``, ``parallel_tool_calls``, ``include``, the ``input`` item roles/types, and the content parts of the
evidence message with image bytes replaced; Codex's own context items, e.g. its environment block with the host
date and timezone, keep only their part types). Usage, from the worktree root::

    PYTHONPATH=tests mamba run -n Auriga python tests/fixtures/evidence/capture_codex_output_schema.py
"""
import base64
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from codex_cli_bin import bundled_codex_path  # noqa: E402

from auriga.core.config import Config  # noqa: E402
from auriga.core.storage.store import Store  # noqa: E402
from auriga.cortex.context.task_planning import TaskPlanning  # noqa: E402
from auriga.cortex.session.codex import CodexSession, CODEX_CLI_VERSION  # noqa: E402
from auriga.cortex.session.codex_evidence_agent import CodexEvidenceAgent  # noqa: E402
from support.fake_model_servers import FakeResponsesServer, text_step  # noqa: E402

OUT = Path(__file__).resolve().with_name("codex_output_schema_request.json")
# A schema with the constructs the Auriga evidence schemas use (enum, nested object, array, required, closed objects).
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["assessment", "notes"],
          "properties": {"assessment": {"type": "string", "enum": ["supported", "contradicted", "uncertain"]},
                         "notes": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                              "required": ["text"],
                                                              "properties": {"text": {"type": "string"}}}}}}
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg==")
IMAGE_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


def shape(part):
    """A content part with inline image bytes replaced by a marker (the bytes are the input, not the shape)."""
    out = dict(part)
    if isinstance(out.get("image_url"), str) and out["image_url"].startswith("data:"):
        out["image_url"] = "<data-url>" if out["image_url"] == IMAGE_URL else "<other-data-url>"
    return out


def capture():
    answer = json.dumps({"assessment": "uncertain", "notes": []})
    with FakeResponsesServer(default=text_step(answer)) as server, tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        (source / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nbase_url="'
                                            + server.base_url + '/v1"\n')
        store = Store(root / "run", dict(run_id="capture", episode_id="e", env_instance_id="i", env_generation=1))
        config = Config.from_dict(dict(model=dict(codex_home=str(source)), features=dict(sandbox_enabled=False)))

        class CaptureSession(CodexSession):
            def _prepare(self, *args):
                provider = super()._prepare(*args)
                path = self.codex_home / "config.toml"
                path.write_text(path.read_text().replace("memories = true", "memories = false"))
                return provider

        session = CaptureSession(store.root, store, lambda _: {}, config=config, task_planning=TaskPlanning(store))
        try:
            session.start([], "capture actor instructions")
            result = CodexEvidenceAgent(session).review_evidence(
                input_items=[dict(type="text", text="evidence text"), dict(type="image", url=IMAGE_URL)],
                output_schema=SCHEMA, request_id="capture")
            helper = [r for r in server.generation_requests() if r.header("thread-id") == result["thread_id"]]
            assert len(helper) == 1, len(helper)
            body = helper[0].json
        finally:
            bridge = session.client._proc if session.client else None
            session.close()
            if bridge:
                for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                    if stream and not stream.closed:
                        stream.close()
            store.close()
    items = []
    inputs = body.get("input", [])
    for index, item in enumerate(inputs):
        entry = {k: item.get(k) for k in ("type", "role") if k in item}
        if index == len(inputs) - 1:  # the evidence message; Codex's own context items keep only their shape
            entry["content"] = [shape(part) for part in item.get("content", [])]
        else:
            entry["content_part_types"] = sorted({p.get("type") for p in item.get("content", []) if isinstance(p, dict)})
        items.append(entry)
    return {
        "about": ("Responses request sent by the pinned Codex app-server (openai-codex " + CODEX_CLI_VERSION
                  + ", binary " + bundled_codex_path().name + ") for one CodexEvidenceAgent turn with outputSchema "
                  "against FakeResponsesServer; captured by capture_codex_output_schema.py."),
        "codex_cli_version": CODEX_CLI_VERSION,
        "output_schema": SCHEMA,
        "top_level_keys": sorted(body),
        "model": body.get("model"),
        "instructions_is_base_instructions": body.get("instructions") is not None and body.get("instructions").startswith(
            "You are an independent visual evidence reviewer."),
        "text": body.get("text"),
        "reasoning": body.get("reasoning"),
        **{key: body.get(key) for key in ("store", "stream", "tools", "tool_choice", "parallel_tool_calls", "include")},
        "input_items": items,
    }


if __name__ == "__main__":
    captured = capture()
    OUT.write_text(json.dumps(captured, indent=1, sort_keys=True) + "\n")
    print(json.dumps(captured, indent=1, sort_keys=True))
