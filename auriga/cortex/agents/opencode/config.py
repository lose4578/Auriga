"""OpenCode launch builder (plan 07 section 3.3 step 1, design 04 sections 6, 7.2, 8): config body, private dirs,
instructions file, ``rg`` exposure, environment, version and route-list checks, provenance.

Nothing here starts a process except the one ``<executable> --version`` probe, and nothing contacts a model.
``build_launch`` returns an ``OpenCodeLaunch`` (argv, environment, working directory, config) for the session to spawn.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from auriga.cortex.agents.isolation import (
    clean_environment, file_sha256_cached, private_dirs, require_exact_version,
)

__all__ = [
    "OPENCODE_VERSION", "RELAY_PROVIDER_ID", "RELAY_PLACEHOLDER_KEY", "WIRE_NPM_PACKAGES", "MANAGED_CONFIG_PATHS",
    "OpenCodeConfigError", "OpenCodeLaunch", "build_config", "build_launch", "check_managed_config",
    "resolve_ripgrep", "probe_cli_version", "check_health_version", "load_expected_routes", "route_list_digest",
    "check_route_list", "provenance_block", "read_effective_tool_ids", "INSTRUCTIONS_FILE_NAME",
    "SANDBOX_EXECUTABLE", "SANDBOX_RIPGREP", "SANDBOX_HARNESS_SOURCE", "sandbox_spec",
    "EFFORT_CHOICES", "EFFORT_WIRE_FIELDS", "effort_variants",
]

OPENCODE_VERSION = "1.19.5"                     # exact pin of opencode-ai (platform binary)
RELAY_PROVIDER_ID = "relay"                     # neutral provider alias; the model id is the configured model name
RELAY_PLACEHOLDER_KEY = "auriga-local-relay"    # never a real credential; the relay holds the upstream key
INSTRUCTIONS_FILE_NAME = "AURIGA.md"
SANDBOX_EXECUTABLE = "/opt/auriga-bin/opencode"  # the pinned binary inside the bubblewrap namespace (read-only mount)
SANDBOX_RIPGREP = "/opt/auriga-bin/rg"           # ``rg`` mounted by AgentSandbox (first on the in-sandbox PATH)
SANDBOX_HARNESS_SOURCE = "/harness"              # read-only snapshot of the published Harness source
WIRE_NPM_PACKAGES = {                           # provider SDK bundled in the opencode binary, per upstream wire
    "anthropic_messages": "@ai-sdk/anthropic",
    "openai_responses": "@ai-sdk/openai",
    "openai_chat": "@ai-sdk/openai-compatible",
}
MANAGED_CONFIG_PATHS = (                        # a managed config overrides OPENCODE_CONFIG_CONTENT (spike S8)
    "/etc/opencode",
    "/Library/Application Support/opencode",
    "/Library/Managed Preferences/ai.opencode.managed.plist",
)
_FLAGS = {
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1", "OPENCODE_PURE": "1", "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1", "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_LSP_DOWNLOAD": "1", "OPENCODE_DISABLE_EMBEDDED_WEB_UI": "1",
}
# Native tools removed from the agent (default-allow policy: everything else, including task / todowrite / skill / lsp,
# stays).  webfetch / websearch / codesearch reach the internet (outside information); question blocks on a human.
_DENIED_TOOLS = {"webfetch": "online_access", "websearch": "online_access", "codesearch": "online_access",
                 "question": "waits_for_human"}
_ROUTES_FILE = Path(__file__).with_name("opencode_routes.json")

# Reasoning effort (``model.effort``) per upstream wire: the accepted values, each declared as a model ``variant`` of
# the generated provider and selected per prompt (``prompt_async`` ``variant``; OpenCode passes the user message's
# variant on to its subagents and to compaction). Identity mapping, no per-model table: a tier the model rejects
# fails at the provider, as on the other harnesses. ``EFFORT_WIRE_FIELDS`` names the request-body field the variant
# sets (provenance only).
EFFORT_CHOICES = {
    "anthropic_messages": ("low", "medium", "high", "xhigh", "max"),
    "openai_responses": ("none", "minimal", "low", "medium", "high", "xhigh"),
}
EFFORT_WIRE_FIELDS = {"anthropic_messages": "output_config.effort", "openai_responses": "reasoning.effort"}


class OpenCodeConfigError(RuntimeError):
    """Start-time refusal; ``str(error)`` is a stable code (``opencode_managed_config_present``, ...)."""


@dataclass(frozen=True)
class OpenCodeLaunch:
    """Everything needed to spawn ``opencode serve`` (all paths absolute)."""
    argv: tuple
    env: dict
    cwd: Path
    config: dict
    instructions_path: Path
    executable: str
    sha256: str
    rg_path: str
    dirs: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------- config body
def build_config(*, model: str, wire_api: str, relay_base_url: str, context_limit_tokens: int,
                 output_limit_tokens: int, instructions_path: str | os.PathLike, mcp_url: str | None = None,
                 mcp_headers: Mapping[str, str] | None = None, subagents: bool = True,
                 variants: Mapping[str, Mapping] | None = None, mcp_timeout_ms: int = 600000) -> dict:
    """The whole ``OPENCODE_CONFIG_CONTENT`` document.

    Parameters
    ----------
    model:
        Model id exactly as configured (never aliased: it selects OpenCode's native base prompt and edit tools).
    wire_api:
        Upstream wire (``anthropic_messages`` | ``openai_responses`` | ``openai_chat``); selects the bundled provider SDK.
    relay_base_url:
        Root URL of the local accounting relay (``http://127.0.0.1:<port>``); ``/v1`` is appended.
    context_limit_tokens, output_limit_tokens:
        Model ``limit.context`` / ``limit.output`` (tokens); the context limit drives native auto-compaction.
    instructions_path:
        Absolute path of the read-only instructions file (config ``instructions``; survives compaction and steers).
    mcp_url, mcp_headers:
        Streamable-HTTP MCP endpoint and headers (bearer) of the host tool server; ``mcp_timeout_ms`` (ms) is the
        per-call timeout (default 600000, large because tool calls can be long).  Omit ``mcp_url`` for no MCP block.
    subagents:
        True (default) leaves the native ``task`` (subagent) tool allowed; False denies it.
    variants:
        Optional model ``variants`` map (effort name -> provider options) for the prompt ``variant``; default none.
    """
    if wire_api not in WIRE_NPM_PACKAGES:
        raise OpenCodeConfigError(f"opencode_unsupported_wire:{wire_api}")
    if not model:
        raise OpenCodeConfigError("opencode_model_required")
    permission = {"*": "allow", **{name: "deny" for name in _DENIED_TOOLS},
                  "external_directory": "allow", "doom_loop": "allow"}
    if not subagents:
        permission["task"] = "deny"
    model_entry = {
        "name": model, "tool_call": True, "attachment": True, "reasoning": False, "temperature": True,
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "limit": {"context": int(context_limit_tokens), "output": int(output_limit_tokens)},
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
    }
    if variants:
        model_entry["variants"] = {name: dict(value) for name, value in variants.items()}
    # ``lsp`` / ``formatter`` are deliberately absent: in 1.19.5 an absent key means off (OpenCode default), and
    # switching them on lets agent-written files (package.json ...) trigger package installs.
    config = {
        "share": "disabled", "autoupdate": False, "snapshot": False,
        "enabled_providers": [RELAY_PROVIDER_ID], "model": f"{RELAY_PROVIDER_ID}/{model}",
        "small_model": f"{RELAY_PROVIDER_ID}/{model}",
        "agent": {"title": {"disable": True}},
        "compaction": {"prune": False},
        "tool_output": {"max_lines": 100000, "max_bytes": 16000000},
        "instructions": [os.fspath(instructions_path)],
        "permission": permission,
        "provider": {RELAY_PROVIDER_ID: {
            "npm": WIRE_NPM_PACKAGES[wire_api], "name": RELAY_PROVIDER_ID,
            "options": {"baseURL": relay_base_url.rstrip("/") + "/v1", "apiKey": RELAY_PLACEHOLDER_KEY,
                        "chunkTimeout": False, "headerTimeout": False},
            "models": {model: model_entry}}},
    }
    if mcp_url:
        config["mcp"] = {"auriga": {"type": "remote", "url": mcp_url, "headers": dict(mcp_headers or {}),
                                    "oauth": False, "timeout": int(mcp_timeout_ms)}}
    return config


def effort_variants(wire_api: str) -> dict:
    """``{effort name: provider options}`` for every ``EFFORT_CHOICES[wire_api]`` name (the model ``variants`` of
    the generated config; the bundled AI SDK turns the options into the request body).

    anthropic_messages (``@ai-sdk/anthropic``): ``{"thinking": {"type": "adaptive", "display": "summarized"},
    "effort": <name>}`` -> ``thinking.type = adaptive`` and ``output_config.effort = <name>``.
    openai_responses (``@ai-sdk/openai``): ``{"reasoningEffort": <name>, "reasoningSummary": "auto", "include":
    ["reasoning.encrypted_content"]}`` -> ``reasoning.effort``, ``reasoning.summary`` and the encrypted reasoning
    items that let a stateless request carry the model's earlier reasoning.
    Another wire raises ``OpenCodeConfigError("opencode_effort_unsupported_wire:<wire>")``."""
    if wire_api == "anthropic_messages":
        return {name: {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": name}
                for name in EFFORT_CHOICES[wire_api]}
    if wire_api == "openai_responses":
        return {name: {"reasoningEffort": name, "reasoningSummary": "auto",
                       "include": ["reasoning.encrypted_content"]}
                for name in EFFORT_CHOICES[wire_api]}
    raise OpenCodeConfigError(f"opencode_effort_unsupported_wire:{wire_api}")


# ------------------------------------------------------------------------------------------------ checks
def check_managed_config(paths: Sequence[str] | None = None, *, user: str | None = None) -> None:
    """Single existence check: a system-managed OpenCode config overrides the generated tool permissions.

    Raises ``OpenCodeConfigError("opencode_managed_config_present")`` when any of ``MANAGED_CONFIG_PATHS`` (or the
    per-user macOS managed-preferences plist for ``user``) exists.  ``paths`` replaces the default list (tests).
    """
    candidates = list(MANAGED_CONFIG_PATHS if paths is None else paths)
    if paths is None:
        name = user or os.environ.get("USER") or os.environ.get("LOGNAME")
        if name:
            candidates.append(f"/Library/Managed Preferences/{name}/ai.opencode.managed.plist")
    for candidate in candidates:
        if os.path.lexists(candidate):
            raise OpenCodeConfigError("opencode_managed_config_present")


def resolve_ripgrep(explicit: str | None = None, search_path: str | None = None) -> str:
    """Absolute path of the host ``rg`` (exposed through the private bin dir; OpenCode would otherwise download it).
    ``explicit`` wins; else ``shutil.which("rg", path=search_path)``.  Missing -> ``opencode_ripgrep_missing``."""
    found = explicit if explicit else shutil.which("rg", path=search_path)
    if not found or not os.path.isabs(found) or not os.access(found, os.X_OK):
        raise OpenCodeConfigError("opencode_ripgrep_missing")
    return found


def probe_cli_version(executable: str, *, timeout_s: float = 15.0) -> str:
    """Run ``<executable> --version`` in a minimal environment; return the stripped version token.

    Without ``HOME``/``XDG_*`` set, OpenCode resolves the invoking user's real passwd home and creates
    ``~/.cache/opencode/bin`` and ``~/.local/share/opencode/repos`` there.  The probe instead runs with a fresh
    temporary directory as ``HOME`` and every ``XDG_*_HOME``, so nothing outside it is read or written (mirrors
    ``claude_code_runtime_problem``'s probe sandbox); the directory and anything written to it are discarded after
    the call.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="auriga-opencode-probe-") as home:
            env = {"PATH": os.defpath, "LC_ALL": "C", "HOME": home,
                   "XDG_CONFIG_HOME": os.path.join(home, "config"), "XDG_DATA_HOME": os.path.join(home, "data"),
                   "XDG_CACHE_HOME": os.path.join(home, "cache"), "XDG_STATE_HOME": os.path.join(home, "state")}
            completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=timeout_s,
                                       env=env, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as error:
        raise OpenCodeConfigError(f"opencode_version_probe_failed:{type(error).__name__}") from error
    return completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""


def check_health_version(health: object) -> str:
    """``GET /global/health`` must report ``healthy`` and the pinned version (``opencode_version_mismatch``)."""
    version = health.get("version") if isinstance(health, dict) else None
    require_exact_version("opencode", version, OPENCODE_VERSION)
    if not health.get("healthy"):
        raise OpenCodeConfigError("opencode_unhealthy")
    return version


def load_expected_routes() -> list[str]:
    """The checked-in sorted ``"METHOD /path"`` operation list of opencode 1.19.5 (188 operations)."""
    return list(json.loads(_ROUTES_FILE.read_text(encoding="utf-8"))["routes"])


def route_list_digest(routes: Sequence[str]) -> str:
    """Canonical sha256 (sorted, unique, compact JSON array) recorded as ``agent_backend.route_list_sha256``."""
    canonical = json.dumps(sorted(set(routes)), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def check_route_list(actual: Sequence[str], expected: Sequence[str] | None = None) -> str:
    """Compare ``GET /doc`` operations with the checked-in list; return the digest or raise
    ``OpenCodeConfigError("opencode_route_list_mismatch")`` (``.added`` / ``.removed`` attached)."""
    wanted = set(load_expected_routes() if expected is None else expected)
    got = set(actual)
    if got != wanted:
        error = OpenCodeConfigError("opencode_route_list_mismatch")
        error.added, error.removed = sorted(got - wanted), sorted(wanted - got)
        raise error
    return route_list_digest(got)


_EDIT_TOOLS = ("edit", "write", "apply_patch", "patch", "multiedit")   # all governed by the ``edit`` permission


def read_effective_tool_ids(client: object, agent: str = "build", *, model: str,
                            provider: str = RELAY_PROVIDER_ID) -> list[str]:
    """Sorted native tool ids ``agent`` (default ``build``, the one the harness prompts) can actually call.

    ``GET /experimental/tool?provider=<provider>&model=<model>`` (``client.tool_list``) lists the tools OpenCode
    offers that provider/model (its registry applies the model-id rule: ``apply_patch`` for non-``gpt-`` ids,
    ``edit``/``write`` for ``gpt-`` ids).  The internal ``invalid`` repair tool is in that list but is excluded from
    the tools given to the model (OpenCode ``session/llm.ts``), so it is dropped; then the ids the agent's merged
    permission ruleset (``GET /agent``) denies with pattern ``*`` are removed, as OpenCode itself does (the last
    matching rule decides; ``edit`` covers write/apply_patch).  The host's MCP tools are not in the list.  The
    whole-registry ``/experimental/tool/ids`` is no longer read (it lists tools never offered).  Feed it to
    ``provenance_block(enabled_tools=...)``.  A malformed reply raises ``OpenCodeConfigError("opencode_tool_ids_invalid")``."""
    items = client.tool_list(provider, model)
    ids = [item.get("id") if isinstance(item, dict) else None for item in items] if isinstance(items, list) else None
    agents = client._request("GET", "/agent")
    if ids is None or not all(isinstance(item, str) for item in ids) or not isinstance(agents, list):
        raise OpenCodeConfigError("opencode_tool_ids_invalid")
    rules = next((item.get("permission") for item in agents if isinstance(item, dict) and item.get("name") == agent),
                 None)
    if not isinstance(rules, list):
        raise OpenCodeConfigError("opencode_tool_ids_invalid")

    def denied(tool_id: str) -> bool:
        permission = "edit" if tool_id in _EDIT_TOOLS else tool_id
        action = None
        for rule in rules:
            name = rule.get("permission", "")
            if rule.get("pattern") == "*" and (name == "*" or name == permission):
                action = rule.get("action")
        return action == "deny"
    return sorted(item for item in ids if item != "invalid" and not denied(item))


# ----------------------------------------------------------------------------------------------- launch
def build_launch(run_dir: str | os.PathLike, *, executable: str, model: str, wire_api: str, relay_base_url: str,
                 instructions_text: str, port: int, password: str, context_limit_tokens: int,
                 output_limit_tokens: int, mcp_url: str | None = None, mcp_headers: Mapping[str, str] | None = None,
                 subagents: bool = True, variants: Mapping[str, Mapping] | None = None,
                 ripgrep: str | None = None, base_env: Mapping[str, str] | None = None,
                 check_version: bool = True, managed_config_paths: Sequence[str] | None = None,
                 sandbox: bool = False, instance_id: str | None = None) -> OpenCodeLaunch:
    """Prepare the private tree below ``run_dir`` and return the launch description.

    Order: managed-config check -> executable + exact ``--version`` + sha256 -> ``rg`` -> private dirs
    (``<run>/private/opencode/{xdg/{config/opencode,data,cache,state},cfgdir,instructions,bin}``, ``private/tmp``) ->
    read-only instructions file (0444) -> config -> both config dirs chmod 0555 (prevents OpenCode's startup
    package install) -> clean environment.  ``port`` is the free loopback port chosen by the host and ``password``
    the per-run ``OPENCODE_SERVER_PASSWORD``.  ``check_version=False`` skips the ``--version`` probe (tests only).

    ``sandbox=True`` prepares the launch for ``AgentSandbox`` (design R4.4): ``argv`` starts with
    ``SANDBOX_EXECUTABLE`` (the namespace path of the pinned binary, mounted from ``executable``), no private ``rg``
    link is made (the sandbox mounts the host ``rg`` at ``SANDBOX_RIPGREP`` and puts it on ``PATH``; ``rg_path``
    records that path), and ``env`` holds only the OpenCode variables: the sandbox sets ``PATH``, ``HOME`` (the
    actor directory) and ``TMPDIR`` (``/tmp`` tmpfs) itself.  ``sandbox_spec`` turns the launch into the mounts.
    ``instance_id`` (the session's ``runtime/agent.instance_id``, default None) sets ``isolation.AGENT_MARKER_ENV`` so
    ``isolation.terminate_agent_processes`` finds the server's detached children at teardown.
    """
    check_managed_config(managed_config_paths)
    if not os.path.isabs(executable) or not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise OpenCodeConfigError("opencode_executable_invalid")
    if check_version:
        require_exact_version("opencode", probe_cli_version(executable), OPENCODE_VERSION)
    digest = file_sha256_cached(executable)
    rg = resolve_ripgrep(ripgrep)
    run = Path(run_dir).resolve(strict=True)
    actor = run / "actor"
    actor.mkdir(exist_ok=True)
    dirs = private_dirs(run, ["private/tmp", "private/opencode/xdg/config/opencode", "private/opencode/xdg/data",
                              "private/opencode/xdg/cache", "private/opencode/xdg/state", "private/opencode/cfgdir",
                              "private/opencode/instructions", "private/opencode/bin"])
    bin_dir = dirs["private/opencode/bin"]
    if not sandbox:
        link = bin_dir / "rg"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(rg, link)
    instructions = dirs["private/opencode/instructions"] / INSTRUCTIONS_FILE_NAME
    if instructions.exists():
        os.chmod(instructions, 0o600)
    instructions.write_text(instructions_text, encoding="utf-8")
    os.chmod(instructions, 0o444)
    config = build_config(model=model, wire_api=wire_api, relay_base_url=relay_base_url,
                          context_limit_tokens=context_limit_tokens, output_limit_tokens=output_limit_tokens,
                          instructions_path=instructions, mcp_url=mcp_url, mcp_headers=mcp_headers,
                          subagents=subagents, variants=variants)
    config_home = dirs["private/opencode/xdg/config/opencode"].parent
    for name in ("private/opencode/xdg/config/opencode", "private/opencode/cfgdir"):
        # OpenCode's Config.ensureGitignore writes <configdir>/.gitignore on first use when it is missing, and
        # treats the write as fatal unless the error is exactly PermissionDenied (EACCES, from chmod 0555 below).
        # A sandboxed read-only bind of these directories instead fails with EROFS, which OpenCode does not
        # tolerate and dies on. Pre-creating an (empty) .gitignore here means OpenCode finds it already present
        # (it checks existence first) and never attempts the write, under either read-only mode.
        gitignore = dirs[name] / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("", encoding="utf-8")
        os.chmod(dirs[name], 0o555)
    extra = {
        "XDG_CONFIG_HOME": str(config_home), "XDG_DATA_HOME": str(dirs["private/opencode/xdg/data"]),
        "XDG_CACHE_HOME": str(dirs["private/opencode/xdg/cache"]),
        "XDG_STATE_HOME": str(dirs["private/opencode/xdg/state"]),
        "OPENCODE_CONFIG_CONTENT": json.dumps(config, ensure_ascii=False),
        "OPENCODE_CONFIG_DIR": str(dirs["private/opencode/cfgdir"]),
        "OPENCODE_SERVER_PASSWORD": password, **_FLAGS,
    }
    env = clean_environment(base_env or {}, path_dirs=[] if sandbox else [bin_dir], home=actor,
                            tmpdir=dirs["private/tmp"], extra=extra, instance_id=instance_id)
    if sandbox:
        for name in ("PATH", "HOME", "TMPDIR"):     # owned by the sandbox (reserved or its own defaults)
            env.pop(name, None)
    argv = (SANDBOX_EXECUTABLE if sandbox else executable, "serve", "--hostname", "127.0.0.1", "--port", str(int(port)))
    return OpenCodeLaunch(argv=argv, env=env, cwd=actor, config=config, instructions_path=instructions,
                          executable=os.path.realpath(executable), sha256=digest,
                          rg_path=SANDBOX_RIPGREP if sandbox else rg, dirs=dirs)


def sandbox_spec(launch: OpenCodeLaunch, *, harness_source: str | os.PathLike) -> dict:
    """Keyword arguments of ``AgentSandbox`` for a ``build_launch(..., sandbox=True)`` launch (design R4.4).

    * ``state_dirs``: the private XDG ``data``, ``cache`` and ``state`` directories, read-write at their host paths
      (OpenCode's database, logs and tool output live there).
    * ``executables``: the pinned binary, read-only, at ``SANDBOX_EXECUTABLE``.
    * ``read_paths``: the config home and ``OPENCODE_CONFIG_DIR`` (read-only: no startup package install), the
      instructions file (read-only) and ``harness_source`` (the ``actor-source`` snapshot) at ``/harness``.
    * ``env``: ``launch.env`` (XDG roots, ``OPENCODE_CONFIG_CONTENT``, server password, flags).

    The caller adds ``ports`` (relay and MCP, outbound) and ``inbound_ports=[server port]`` (the host reaches the
    server through ``AgentSandbox.inbound_socket_path``).  The checkout, the user's home and the RoboDojo /
    simulator tree are never mounted.
    """
    state = [launch.dirs[f"private/opencode/xdg/{name}"] for name in ("data", "cache", "state")]
    read_only = [launch.dirs["private/opencode/xdg/config/opencode"].parent, launch.dirs["private/opencode/cfgdir"],
                 launch.instructions_path]
    return {
        "state_dirs": [(path, path) for path in state],
        "executables": [(launch.executable, SANDBOX_EXECUTABLE)],
        "read_paths": [(path, path) for path in read_only] + [(os.fspath(harness_source), SANDBOX_HARNESS_SOURCE)],
        "env": dict(launch.env),
    }


def provenance_block(launch: OpenCodeLaunch, *, route_list_sha256: str | None, wire_api: str, model: str,
                     effort: str | None = None, instructions_sha256: str | None = None,
                     tool_name_prefix: str = "auriga_", subagents: bool = True,
                     enabled_tools: Sequence[str] | None = None) -> dict:
    """``manifest.agent_backend`` fragment for OpenCode (observer-only; never enters model context).  Secrets
    (password, MCP bearer) are not recorded; the model id is the configured name (``native_model_id``).

    ``native_tools.disabled`` lists ``{name, reason}`` (``online_access`` | ``waits_for_human`` |
    ``subagents_disabled``); ``enabled_tools`` is the agent's effective tool-id list read back from the running
    server (``read_effective_tool_ids``), ``None`` when not read."""
    disabled = [{"name": name, "reason": reason} for name, reason in _DENIED_TOOLS.items()]
    if not subagents:
        disabled.append({"name": "task", "reason": "subagents_disabled"})
    return {
        "kind": "opencode", "sdk": None,
        "cli": {"version": OPENCODE_VERSION, "executable": launch.executable, "sha256": launch.sha256},
        "wire_api": wire_api, "native_model_id": model, "provider_alias": RELAY_PROVIDER_ID,
        "route_list_sha256": route_list_sha256, "native_effort": {"variant": effort} if effort else None,
        "tool_transport": {"kind": "mcp_streamable_http", "tool_name_prefix": tool_name_prefix},
        "prompt_channel": {"instructions_channel": "config_instructions_file", "native_base_prompt": "preserved",
                           "instructions_sha256": instructions_sha256},
        "native_tools": {"enabled": sorted(enabled_tools) if enabled_tools is not None else None,
                         "disabled": disabled},
        "native_settings": {"flags": dict(_FLAGS), "config_dirs_read_only": True, "rg": launch.rg_path,
                            "compaction_prune": False, "title_agent": "disabled",
                            "provider_timeouts": {"chunkTimeout": False, "headerTimeout": False}},
    }
