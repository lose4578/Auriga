"""Capture the first actor generation request each agent backend sends upstream (P4-3 prompt-parity review).

The capture runs the REAL Workbench startup of a benchmark episode (robot information, ``actor_instructions``,
runtime capabilities, the task prompt with public rules and task notes, the registry actor catalogue) with a
process-free calibrated ARX X5 fixture robot (``test_robot_calibration_information.CalibratedRobot``), through the real
backend session (``factory.create_session``), the real relay and the real pinned agent binary, against a fake loopback
model server from ``fake_model_servers`` (never a real model, never the network). ``features.sandbox_enabled`` is off:
bubblewrap is Linux-only, and with the sandbox on only the ``SANDBOX_INSTRUCTIONS`` block of the instructions changes
(identically for every backend; Workbench chooses it).

Backends (``BACKENDS``): ``codex`` (app-server 0.160.0 bundled with ``openai-codex``), ``claude_code`` (CLI 2.1.321
bundled with ``claude-agent-sdk`` 0.2.180), ``opencode_anthropic`` / ``opencode_responses`` (``opencode serve`` 1.19.5
named by ``AURIGA_OPENCODE_EXECUTABLE``, upstream wire ``anthropic_messages`` / ``openai_responses``).

    capture(backend) -> Capture       # Workbench inputs + the recorded first actor request
    analyze(capture) -> dict          # where the Auriga text landed, byte identity, tool catalogue comparison
    snapshot(capture) -> dict         # redacted request layout: Auriga parts by reference, native text in full,
                                      # run paths -> <run>, ids -> <id>, media -> <data-url>; no hashes

Command line, from the worktree root (writes ``<backend>.json`` = ``{"analysis", "snapshot"}`` per backend)::

    AURIGA_OPENCODE_EXECUTABLE=/abs/opencode PYTHONPATH=tests mamba run -n Auriga python -m support.prompt_capture \
        [--backend codex --backend claude_code ...] --out /tmp/prompt-parity

Used by ``tests/test_agent_prompt_parity.py`` and ``docs/development/reports/agent-backend-prompt-parity.md``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .fake_model_servers import FakeAnthropicServer, FakeResponsesServer, text_step

__all__ = ["BACKENDS", "Capture", "analyze", "capture", "main", "published_tools", "snapshot", "text_blocks"]

UPSTREAM_KEY_ENV = "AURIGA_PROMPT_CAPTURE_UPSTREAM_KEY"
OPENCODE_EXECUTABLE_ENV = "AURIGA_OPENCODE_EXECUTABLE"
TASK = "Stack the three bowls together."
CAPTURE_TIMEOUT_S = 120

# backend -> (session kind, upstream wire, model name). Codex uses the default model of ``model.name``; the other
# model ids only need the family prefix OpenCode keys its native base prompt and edit tools on (claude* / gpt-*).
BACKENDS = {
    "codex": ("codex", "openai_responses", "gpt-6-astra"),
    "claude_code": ("claude_code", "anthropic_messages", "claude-parity-model"),
    "opencode_anthropic": ("opencode", "anthropic_messages", "claude-parity-model"),
    "opencode_responses": ("opencode", "openai_responses", "gpt-parity-model"),
}
TOOL_PREFIX = {"codex": "", "claude_code": "mcp__auriga__", "opencode": "auriga_"}


@dataclass
class Capture:
    """One backend's Workbench inputs and its first actor generation request.

    ``instructions`` / ``prompt`` / ``tools``: exactly what Workbench passed to ``session.start`` and the first
    ``session.run_turn`` (str, str, registry catalogue list). ``request``: the recorded ``RecordedRequest``.
    ``naming_note``: the backend's tool-name-prefix sentence ("" for Codex). ``run_root``: the (deleted) run
    directory, kept only for redaction. ``seen``: ``(method, path, role)`` of every request the fake model got.
    """
    backend: str
    kind: str
    wire: str
    model: str
    instructions: str
    prompt: str
    tools: list
    request: object
    naming_note: str
    tool_prefix: str
    run_root: str
    thread_id: str | None
    seen: list = field(default_factory=list)
    binary: dict = field(default_factory=dict)

    @property
    def body(self):
        return self.request.json


def _fixture_robot(root):
    from test_robot_calibration_information import CalibratedRobot

    class ParityRobot(CalibratedRobot):
        """Calibrated ARX X5 fixture with the public task text and the motion-review hook the Harness requires."""

        def start(self):
            return dict(super().start(), robot_model_id="arx-x5")

        def preview_motion(self, *args, **kwargs):  # never called: the fake model makes no tool call
            raise NotImplementedError("prompt capture fixture")

        def observe(self):
            raw = super().observe()
            raw["instruction"] = TASK
            return raw

    return ParityRobot(root)


def _config(backend, server, root, opencode_executable):
    from auriga.core.config import Config
    kind, wire, model = BACKENDS[backend]
    features = {"self_briefing": False, "terminal": False, "sandbox_enabled": False,
                "grasp_memory_dir": str(root / "grasp-memory")}
    if kind == "codex":
        profile = root / "codex-profile"
        profile.mkdir()
        (profile / "config.toml").write_text('model_provider = "fixture"\n[model_providers.fixture]\n'
                                             f'base_url = "{server.base_url}/v1"\n')
        return Config.from_dict({"model": {"codex_home": str(profile), "name": model, "effort": "high"},
                                 "features": features})
    upstream = {"wire_api": wire, "base_url": server.base_url, "api_key_env": UPSTREAM_KEY_ENV}
    if wire == "anthropic_messages":
        upstream["auth_scheme"] = "x-api-key"
    if kind == "claude_code":
        upstream.pop("wire_api")
        agent = {"backend": "claude_code", "claude_code": {"upstream": upstream}}
    else:
        if not opencode_executable:
            raise RuntimeError(f"{OPENCODE_EXECUTABLE_ENV} must name the pinned opencode binary")
        agent = {"backend": "opencode", "opencode": {"executable": opencode_executable, "upstream": upstream}}
    return Config.from_dict({"agent": agent, "model": {"name": model, "effort": "high"}, "features": features})


def _naming_note(kind):
    if kind == "claude_code":
        from auriga.cortex.agents.claude_code.options import naming_note
        return naming_note()
    if kind == "opencode":
        from auriga.cortex.agents.opencode.session import naming_note
        return naming_note()
    return ""


def _versions(manifest):
    """Pinned runtime versions recorded in the run manifest (versions only; no paths, no digests)."""
    block = manifest.get("agent_backend") or {}
    found = dict(kind=block.get("kind"))
    for key in ("cli", "sdk", "binary", "server"):
        value = block.get(key)
        if isinstance(value, dict) and value.get("version"):
            found[key + "_version"] = value["version"]
    if manifest.get("codex_cli"):
        found["cli_version"] = manifest["codex_cli"]
    if manifest.get("codex_sdk"):
        found["sdk_version"] = manifest["codex_sdk"]
    return found


def _is_actor_request(kind, request, thread_id):
    """The actor loop's generation requests: Codex = the actor thread, not a native helper (memory consolidation);
    Claude Code = tools declared, not a subagent; OpenCode = tools declared and the actor session."""
    if not request.is_generation or not isinstance(request.json, dict):
        return False
    if kind == "codex":
        return request.header("thread-id") == thread_id and request.header("x-openai-subagent") is None
    if not request.json.get("tools"):
        return False
    if kind == "claude_code":
        return request.header("x-claude-code-agent-id") is None
    return request.header("x-opencode-session-id") == thread_id


def _role(kind, request, thread_id):
    if _is_actor_request(kind, request, thread_id):
        return "actor"
    if kind == "codex" and request.header("x-openai-subagent"):
        return "native_helper:" + request.header("x-openai-subagent")
    return "generation" if request.is_generation else "auxiliary"


def capture(backend, *, opencode_executable=None, timeout_s=CAPTURE_TIMEOUT_S):
    """Run one Workbench startup on ``backend`` (a ``BACKENDS`` key) and return its ``Capture``.

    ``opencode_executable``: absolute path of the pinned opencode binary (default ``$AURIGA_OPENCODE_EXECUTABLE``).
    Raises ``TimeoutError`` when no actor request arrives within ``timeout_s`` seconds. The run directory and every
    agent process are removed before returning.
    """
    from unittest import mock
    from auriga.apps.gateway.workbench import Workbench
    kind, wire, model = BACKENDS[backend]
    opencode_executable = opencode_executable or os.environ.get(OPENCODE_EXECUTABLE_ENV)
    server_class = FakeResponsesServer if wire == "openai_responses" else FakeAnthropicServer
    server = server_class([], default=text_step("done"), gate_timeout_s=30).start()
    inputs = {}
    try:
        with tempfile.TemporaryDirectory(prefix="auriga-prompt-capture-") as directory, \
                mock.patch.dict(os.environ, {UPSTREAM_KEY_ENV: "host-secret-prompt-capture"}):
            root = Path(directory).resolve()
            config = _config(backend, server, root, opencode_executable)
            workbench = Workbench(root / "run", robot=_fixture_robot(root / "robot"), visual_review_model=None,
                                  config=config)
            session = workbench.session
            start, run_turn = session.start, session.run_turn

            def recording_start(tools, instructions, *args, **kwargs):
                inputs.update(tools=tools, instructions=instructions)
                return start(tools, instructions, *args, **kwargs)

            def recording_turn(prompt, *args, **kwargs):
                inputs.setdefault("prompt", prompt)
                return run_turn(prompt, *args, **kwargs)
            session.start, session.run_turn = recording_start, recording_turn

            async def run():
                try:
                    await workbench.start("unused", benchmark=True)
                    deadline = time.monotonic() + timeout_s
                    while time.monotonic() < deadline:
                        found = [r for r in server.requests if _is_actor_request(kind, r, session.thread_id)]
                        if found:
                            return found[0]
                        await asyncio.sleep(0.1)
                    raise TimeoutError(f"{backend}: no actor generation request within {timeout_s} s")
                finally:
                    await workbench.close()
            request = asyncio.run(run())
            thread_id = session.thread_id
            manifest = root / "run" / "manifest.json"
            provenance = _versions(json.loads(manifest.read_text())) if manifest.exists() else {}
            seen = [(r.method, r.path, _role(kind, r, thread_id)) for r in server.requests]
            return Capture(backend=backend, kind=kind, wire=wire, model=model, instructions=inputs["instructions"],
                           prompt=inputs["prompt"], tools=inputs["tools"], request=request,
                           naming_note=_naming_note(kind), tool_prefix=TOOL_PREFIX[kind], run_root=str(root / "run"),
                           thread_id=thread_id, seen=seen, binary=provenance)
    finally:
        server.close()


# ---- reading request bodies ----------------------------------------------------------------------------------
def text_blocks(capture):
    """Ordered model-visible text blocks of the request: ``[(location, role, text)]``.

    Responses: top-level ``instructions`` then every ``input`` message part; Anthropic: every ``system`` block then
    every ``messages`` text block. Tool definitions are not text blocks (see ``published_tools``)."""
    body, blocks = capture.body, []
    if capture.wire == "openai_responses":
        if isinstance(body.get("instructions"), str) and body["instructions"]:
            blocks.append(("instructions", "instructions", body["instructions"]))
        for index, item in enumerate(body.get("input") or []):
            if item.get("type", "message") != "message":
                continue
            content = item.get("content")
            if isinstance(content, str):
                blocks.append((f"input[{index}]", item.get("role"), content))
                continue
            for part_index, part in enumerate(content or []):
                if isinstance(part.get("text"), str):
                    blocks.append((f"input[{index}].content[{part_index}]", item.get("role"), part["text"]))
        return blocks
    system = body.get("system")
    if isinstance(system, str):
        blocks.append(("system", "system", system))
    for index, block in enumerate(system if isinstance(system, list) else []):
        blocks.append((f"system[{index}]", "system", block.get("text", "")))
    for index, message in enumerate(body.get("messages") or []):
        content = message.get("content")
        if isinstance(content, str):
            blocks.append((f"messages[{index}]", message.get("role"), content))
            continue
        for part_index, part in enumerate(content or []):
            if part.get("type") == "text":
                blocks.append((f"messages[{index}].content[{part_index}]", message.get("role"), part["text"]))
    return blocks


def published_tools(capture):
    """Every tool the request publishes, in order: ``[{"name", "namespace", "type", "description", "schema"}]``.

    Codex 0.160.0 sends ``tools: []`` and publishes its catalogue in an ``additional_tools`` input item (namespaces of
    functions, plus ``tool_search``); the other clients use the top-level ``tools`` list."""
    body, out = capture.body, []

    def add(tool, namespace=None):
        if tool.get("type") == "namespace":
            for inner in tool.get("tools") or []:
                add(inner, tool.get("name"))
            return
        out.append(dict(name=tool.get("name") or tool.get("type"), namespace=namespace, type=tool.get("type"),
                        description=tool.get("description"),
                        schema=tool.get("input_schema", tool.get("parameters"))))
    for tool in body.get("tools") or []:
        add(tool)
    for item in body.get("input") or [] if capture.wire == "openai_responses" else []:
        if item.get("type") == "additional_tools":
            for tool in item.get("tools") or []:
                add(tool)
    return out


def _schema_changes(registry, published, removed, added):
    """Count schema keywords dropped from / added to ``registry`` in ``published`` (recursively, by key)."""
    if isinstance(registry, dict) and isinstance(published, dict):
        for key, value in registry.items():
            if key not in published:
                removed[key] += 1
            else:
                _schema_changes(value, published[key], removed, added)
        for key in published:
            if key not in registry:
                added[key] += 1
    elif isinstance(registry, list) and isinstance(published, list):
        for left, right in zip(registry, published):
            _schema_changes(left, right, removed, added)


def _top_level_combinators(schema):
    from auriga.cortex.agents.mcp_server import TOP_LEVEL_COMBINATORS
    return sorted(key for key in TOP_LEVEL_COMBINATORS if isinstance(schema, dict) and key in schema)


def analyze(capture):
    """Facts of one capture (JSON-serialisable; no hashes, no run paths).

    ``instructions``: ``location`` and ``role`` of the block holding the Workbench instructions, ``occurrences`` in all
    text blocks, ``delivered_identical`` (the block contains ``instructions + "\\n\\n" + naming note`` for MCP
    backends, the instructions alone for Codex), ``native_before_chars`` / ``native_after_chars`` of that block,
    ``header`` (text directly in front of the instructions in the same block). ``first_user_turn``: location and
    byte identity with the Workbench prompt. ``tools``: per registry actor tool its published name, presence,
    description identity / truncation and schema keyword changes; ``native_tools``: names of the other tools."""
    blocks = text_blocks(capture)
    expected = capture.instructions + ("\n\n" + capture.naming_note if capture.naming_note else "")
    occurrences = sum(text.count(capture.instructions) for _, _, text in blocks)
    holder = next(((location, role, text) for location, role, text in blocks if capture.instructions in text), None)
    instructions = dict(chars=len(capture.instructions), occurrences=occurrences, location=None, role=None,
                        delivered_identical=False, naming_note=capture.naming_note or None)
    if holder is not None:
        location, role, text = holder
        start = text.index(capture.instructions)
        before, after = text[:start], text[start + len(capture.instructions):]
        instructions.update(location=location, role=role, delivered_identical=expected in text,
                            native_before_chars=len(before), native_after_chars=len(after),
                            header=before.rstrip("\n").rsplit("\n", 1)[-1] if before.strip() else "")
    turn = next(((location, role) for location, role, text in blocks if text == capture.prompt), None)
    from auriga.cortex.agents.mcp_server import portable_input_schema
    published = published_tools(capture)
    by_name = {tool["name"]: tool for tool in published}
    registry_names = {capture.tool_prefix + tool["name"] for tool in capture.tools}
    tools = []
    for tool in capture.tools:
        name = capture.tool_prefix + tool["name"]
        entry = dict(name=tool["name"], published_name=name, deferred=bool(tool.get("deferLoading", False)),
                     present=name in by_name, registry_description_chars=len(tool["description"]),
                     top_level_combinators=_top_level_combinators(tool.get("inputSchema")))
        if name in by_name:
            item = by_name[name]
            removed, added = Counter(), Counter()
            _schema_changes(tool["inputSchema"], item["schema"], removed, added)
            description = item["description"] or ""
            entry.update(namespace=item["namespace"], description_identical=description == tool["description"],
                         published_description_chars=len(description),
                         truncated=description.endswith("[truncated]") and description != tool["description"],
                         schema_identical=item["schema"] == tool["inputSchema"],
                         # MCP backends publish the portable projection (top-level combinators removed)
                         schema_portable_identical=item["schema"] == portable_input_schema(tool["inputSchema"]),
                         schema_removed=dict(sorted(removed.items())), schema_added=dict(sorted(added.items())))
        tools.append(entry)
    native = [dict(name=tool["name"], namespace=tool["namespace"], type=tool["type"]) for tool in published
              if tool["name"] not in registry_names]
    return dict(backend=capture.backend, wire=capture.wire, model=capture.model, binary=capture.binary,
                requests_seen=[list(item) for item in capture.seen], instructions=instructions,
                first_user_turn=dict(location=turn[0] if turn else None, role=turn[1] if turn else None,
                                     identical=turn is not None, chars=len(capture.prompt)),
                text_blocks=[dict(location=location, role=role, chars=len(text)) for location, role, text in blocks],
                tools=tools, native_tools=native, top_level_keys=sorted(capture.body))


# ---- redacted snapshot -------------------------------------------------------------------------------------
_DATA_URL = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
_ID_KEYS = {"prompt_cache_key", "client_metadata", "metadata", "session_id", "user_id"}


def _redactor(capture):
    """Text -> text with the run directory as ``<run>`` (also its macOS ``/private`` twin and the dash-sanitised form
    Claude Code uses for its per-project memory directory), the capture's temporary root as ``<tmp>``, the actor
    thread id as ``<thread>`` and inline media as ``<data-url>``."""
    def variants(path):
        forms = {path, path.removeprefix("/private")}
        forms.add("/private" + path if path.startswith("/var/") else path)
        return forms | {re.sub(r"[^A-Za-z0-9]", "-", form) for form in forms}
    replacements = [(value, "<run>") for value in variants(capture.run_root)]
    replacements += [(value, "<tmp>") for value in variants(str(Path(capture.run_root).parent))]
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    def redact(text):
        text = _DATA_URL.sub("<data-url>", text)
        for value, marker in replacements:
            text = text.replace(value, marker)
        if capture.thread_id:
            text = text.replace(capture.thread_id, "<thread>")
        return text
    return redact


def snapshot(capture):
    """Redacted layout of the first actor request.

    ``blocks``: ordered text blocks; the Workbench instructions, the naming note and the first user turn are replaced
    by ``<AURIGA_INSTRUCTIONS>``, ``<AURIGA_NAMING_NOTE>`` and ``<AURIGA_FIRST_USER_TURN>`` (they are byte-identical
    across backends, see ``analyze``); all other (native) text is kept in full with run paths -> ``<run>`` and media
    -> ``<data-url>``. ``fields``: the remaining top-level request fields (ids -> ``<id>``). ``tools``: published name,
    kind (``auriga``/``native``), namespace and description length. ``headers``: header names."""
    redact = _redactor(capture)
    registry_names = {capture.tool_prefix + tool["name"] for tool in capture.tools}
    blocks = []
    for location, role, text in text_blocks(capture):
        content = text
        if text == capture.prompt:
            content = "<AURIGA_FIRST_USER_TURN>"
        else:
            if capture.naming_note:
                content = content.replace(capture.naming_note, "<AURIGA_NAMING_NOTE>")
            content = content.replace(capture.instructions, "<AURIGA_INSTRUCTIONS>")
        blocks.append(dict(location=location, role=role, chars=len(text), text=redact(content)))

    def scrub(key, value):
        if key in _ID_KEYS:
            return "<id>"
        if isinstance(value, str):
            return redact(value)
        return value
    content_keys = {"system", "messages", "input", "tools", "instructions"}
    fields = {key: scrub(key, value) for key, value in sorted(capture.body.items()) if key not in content_keys}
    tools = [dict(name=tool["name"], kind="auriga" if tool["name"] in registry_names else "native",
                  namespace=tool["namespace"], type=tool["type"], description_chars=len(tool["description"] or ""))
             for tool in published_tools(capture)]
    return dict(backend=capture.backend, wire=capture.wire, model=capture.model, path=capture.request.path,
                headers=sorted(capture.request.headers), fields=fields, blocks=blocks, tools=tools)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", action="append", choices=sorted(BACKENDS),
                        help="backend to capture (repeatable; default: every backend whose runtime is installed)")
    parser.add_argument("--out", required=True, type=Path, help="directory for <backend>.json")
    parser.add_argument("--opencode", default=os.environ.get(OPENCODE_EXECUTABLE_ENV),
                        help=f"pinned opencode binary (default ${OPENCODE_EXECUTABLE_ENV})")
    arguments = parser.parse_args(argv)
    selected = arguments.backend or [name for name in BACKENDS
                                     if BACKENDS[name][0] != "opencode" or arguments.opencode]
    arguments.out.mkdir(parents=True, exist_ok=True)
    for backend in selected:
        result = capture(backend, opencode_executable=arguments.opencode)
        record = dict(analysis=analyze(result), snapshot=snapshot(result))
        path = arguments.out / f"{backend}.json"
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
        facts = record["analysis"]["instructions"]
        missing = [tool["name"] for tool in record["analysis"]["tools"] if not tool["present"]]
        truncated = [tool["name"] for tool in record["analysis"]["tools"] if tool.get("truncated")]
        print(f"{backend}: instructions at {facts['location']} identical={facts['delivered_identical']} "
              f"occurrences={facts['occurrences']}; missing tools={missing}; truncated={truncated} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
