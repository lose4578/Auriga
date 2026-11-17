"""Build ``opencode_wire_bodies.json``: real opencode 1.19.5 generation requests for the relay wire tests (P3-4).

Source: the S6 spike evidence of report 06 (``s6-{responses,anthropic}-basic``: one session per wire with text,
MCP image results, an isError result, two parallel calls, a native ``bash`` call and a second user turn, captured by a
loopback fake model). Kept per wire: the client request headers of the first generation request (``Host`` and
``Content-Length`` dropped) and the first and last generation bodies, verbatim except for this sanitisation:

* the native ``bash`` result (an ``env | sort`` dump of the isolated spike run) becomes ``ENV_PLACEHOLDER``;
* the spike run directory prefix (under ``/private/tmp`` or ``/tmp``) becomes ``/run``.

The bodies keep the OpenCode-specific fields the egress walker must leave alone (``prompt_cache_key``, ``include``,
``store``, ``max_output_tokens`` / ``max_tokens``, ``tool_choice``, ``eager_input_streaming``, ``cache_control``) and
the 4x3 PNG the spike served. Usage, from the worktree root::

    mamba run -n Auriga python tests/fixtures/opencode/build_opencode_bodies.py \
        --evidence /tmp/auriga-agent-harness/scratch/s-c/evidence
"""
import argparse
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().with_name("opencode_wire_bodies.json")
ENV_PLACEHOLDER = "env output removed from the fixture"
RUN_PATH = re.compile(r"(?:/private)?/tmp/auriga-agent-harness/scratch/s-c/runs/[^/\s\"\\]+")


def sanitise(node, env_call_ids):
    if isinstance(node, dict):
        if node.get("type") == "function_call_output" and node.get("call_id") in env_call_ids:
            return dict(node, output=ENV_PLACEHOLDER)
        if node.get("type") == "tool_result" and node.get("tool_use_id") in env_call_ids:
            return dict(node, content=ENV_PLACEHOLDER)
        return {k: sanitise(v, env_call_ids) for k, v in node.items()}
    if isinstance(node, list):
        return [sanitise(v, env_call_ids) for v in node]
    if isinstance(node, str):
        return RUN_PATH.sub("/run", node)
    return node


def env_calls(body):
    found = set()
    for item in body.get("input") or body.get("messages") or []:
        if item.get("type") == "function_call" and item.get("name") == "bash":
            found.add(item["call_id"])
        for block in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "bash":
                found.add(block["id"])
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    cases = {}
    for wire, name in (("openai_responses", "s6-responses-basic"), ("anthropic_messages", "s6-anthropic-basic")):
        base = Path(args.evidence) / name
        requests = [r for r in json.loads((base / "requests.json").read_text()) if r.get("is_generation")]
        headers = {k: v for k, v in requests[0]["headers"].items() if k.lower() not in ("host", "content-length")}
        bodies = {}
        for which in ("first", "last"):
            body = json.loads((base / f"{which}_generation_body.json").read_text())
            bodies[which] = sanitise(body, env_calls(body))
        cases[wire] = dict(path=requests[0]["path"], headers=headers, first_body=bodies["first"],
                           last_body=bodies["last"])
    text = json.dumps(dict(about=__doc__.split("\n\n")[0] + " See build_opencode_bodies.py.", cases=cases),
                      indent=1, ensure_ascii=True) + "\n"
    assert "auriga-agent-harness" not in text and "ALL_PROXY" not in text and "PASSWORD" not in text
    OUT.write_text(text)
    print("wrote", OUT, len(text.encode()), "bytes")


if __name__ == "__main__":
    main()
