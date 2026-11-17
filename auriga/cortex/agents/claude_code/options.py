"""Options, clean environment, launcher and provenance of the Claude Code agent backend (design rev 3, sections 6, 7.2, 7.3, 8).

Pure builders: nothing here starts a process (except the one ``<cli> --version`` probe in ``resolve_runtime``), talks
to a model or imports ``claude_agent_sdk`` at module import time (the SDK is an optional extra). ``ClaudeCodeSession``
calls ``build_launch`` once per ``start()`` and hands ``launch.options`` to ``ClaudeSDKClient``.

What the launch consists of
---------------------------
* ``launch.env``       the COMPLETE environment of the CLI (never derived from ``os.environ``);
* ``launch.launcher``  ``env -i`` launcher script passed to the SDK as ``cli_path`` (the SDK merges ``options.env``
                       over the parent environment, so only a launcher can start the CLI from an empty one). It
                       re-exports the three variables the SDK injects at spawn time (``REEXPORT_VARS``);
* ``launch.options``   ``ClaudeAgentOptions``: the CLI's DEFAULT native tool set minus the denylist, permission mode
                       ``bypassPermissions``, no setting sources, strict MCP config, MCP server entry of the host endpoint, instructions via ``--append-system-prompt-file``,
                       host-chosen session id or ``resume``;
* ``launch.runtime``   the verified SDK/CLI versions, real executable path and sha256 (provenance records the real
                       CLI, never the launcher).

Sandbox mode (``features.sandbox_enabled``, design R4.4): ``build_launch(sandbox=SandboxPlan(...))`` builds a generic
``AgentSandbox`` and the launcher becomes ``exec <env -i bwrap ... python3 -c supervisor> <real CLI> "$@"`` (see
``write_sandbox_launcher``). The SDK still spawns the launcher on the host and talks to it over stdio, which the
bubblewrap supervisor passes straight through to the CLI. Inside the namespace the CLI sees: its own binary (read-only),
the run-private config / tmp / run directories (read-write, same paths as outside), the actor directory (read-write, cwd
and HOME), the instructions file (read-only), the published Harness source at ``/harness`` (read-only) and the system
runtime; the relay and MCP endpoint answer at the SAME ``127.0.0.1:<port>`` as on the host (in-namespace forwarders). The
checkout, the user's home and the RoboDojo / simulator sources are not mounted.

Every variable and option below was confirmed against the real CLI 2.1.321 in spikes S1-S5 (see
``docs/development/`` agent backend notes); the value tables are the single place to change them.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shlex
import stat
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ....core.config import ClaudeCodeConfig
from ...session.agent_sandbox import ACTOR_ISOLATION, AgentSandbox
from ..factory import NATIVE_EFFORT_CHOICES
from ..isolation import (AGENT_MARKER_ENV, clean_environment, file_sha256_cached, private_dirs, require_exact_version,
                         write_env_launcher)

__all__ = [
    "SDK_PACKAGE", "SDK_VERSION", "CLI_VERSION", "BACKEND_KIND", "MCP_SERVER_NAME", "TOOL_NAME_PREFIX",
    "DISALLOWED_TOOLS", "SUBAGENT_TOOL_NAMES", "REEXPORT_VARS", "PLACEHOLDER_API_KEY", "EFFORT_CHOICES",
    "STREAM_IDLE_TIMEOUT_MS", "LIVE_EFFORT", "MAX_BUFFER_BYTES", "ClaudeRuntime", "ClaudeLaunch", "effort_choices", "native_effort",
    "naming_note", "compose_instructions", "resolve_runtime", "build_launch", "native_tools_block", "provenance_block",
    "SandboxPlan", "SANDBOX_OWNED_ENV", "HARNESS_SOURCE_MOUNT", "sandbox_ipc_dir", "prepare_sandbox_ipc",
    "write_sandbox_launcher", "AURIGA_OWNED_ENV", "SETTINGS_CREDENTIAL_ENV", "read_claude_settings_upstream",
    "read_claude_settings_env",
]

SDK_PACKAGE = "claude-agent-sdk"
SDK_VERSION = "0.2.180"            # exact pin of the SDK (optional extra auriga-harness[claude])
CLI_VERSION = "2.1.321"            # exact pin of the CLI bundled with that SDK; also required of an explicit cli_path
BACKEND_KIND = "claude_code"
MCP_SERVER_NAME = "auriga"
TOOL_NAME_PREFIX = f"mcp__{MCP_SERVER_NAME}__"   # model-visible name prefix of every registry tool (design 3.5)

# Native tool policy (design R4.1, user decision 2026-11-12): the agent keeps the CLI's DEFAULT tool set (no ``tools=``
# option, no allowlist; the set the pinned CLI really exposes is read back from the init message, see
# ``native_tools_block``). Only online access and tools that wait for a human are removed; scheduling tools
# (CronCreate/CronDelete/CronList/ScheduleWakeup), Monitor, background Bash, memory, skills and slash commands stay.
# (name, reason): every entry is recorded in the manifest. WebFetch is also off by CLAUDE_CODE_DISABLE_WEB_FETCH; the
# last three are absent from the 2.1.321 SDK default set and listed so the reason is on record (probe R4-0 s.1).
DISALLOWED_TOOLS = (("WebSearch", "online_access"), ("WebFetch", "online_access"),
                    ("AskUserQuestion", "waits_for_human"), ("EnterPlanMode", "waits_for_human"),
                    ("ExitPlanMode", "waits_for_human"))
# The subagent tool is ``Agent`` in model requests and ``Task`` in the init message (2.1.321); denying either removes
# both, both are listed for robustness. Denied only when ``agent.claude_code.subagents`` is false.
SUBAGENT_TOOL_NAMES = ("Agent", "Task")
# ``env -i`` drops these three variables, which the SDK sets at spawn time. CLAUDE_CODE_SDK_READS_SESSION_STATE changes
# how the SDK detects the end of a run, so the launcher must pass them through (spike S1 ``pure-env-i``).
REEXPORT_VARS = ("CLAUDE_CODE_ENTRYPOINT", "CLAUDE_AGENT_SDK_VERSION", "CLAUDE_CODE_SDK_READS_SESSION_STATE")
PLACEHOLDER_API_KEY = "auriga-local-relay"   # the agent only ever sees this local placeholder, never an upstream key
EFFORT_CHOICES = NATIVE_EFFORT_CHOICES[BACKEND_KIND]   # SDK ``effort`` levels: low, medium, high, xhigh, max
# CLI stream idle timeout (ms). The binary clamps the value to [300000, 1800000]; 1800000 is the largest value, so a
# relay that holds a generation for many minutes (buffered upstream, keepalive pings only) is not abandoned by the CLI
# (spike S5). The relay's upstream generation timeout must not exceed it.
STREAM_IDLE_TIMEOUT_MS = 1_800_000
# The SDK has no live effort control: ``effort`` takes effect only from session start (probe T0). The session reports
# ``AgentCapabilities.live_effort=False`` and the console says "applies after restart".
LIVE_EFFORT = False
# SDK stdout line buffer (bytes). The 1 MiB default kills the SDK reader on a 3 MB image tool result (spike S2).
MAX_BUFFER_BYTES = 32 * 1024 * 1024

_DISABLE_FLAGS = {
    # outside traffic and housekeeping
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    # web tools and tool search
    "CLAUDE_CODE_DISABLE_WEB_FETCH": "1", "ENABLE_TOOL_SEARCH": "false",
}
_RELAY_FLAGS = {
    "CLAUDE_CODE_GZIP_REQUEST_BODIES": "0",                # redundant for a loopback relay, kept as belt and braces
    "CLAUDE_CODE_MAX_RETRIES": "0",                        # the relay owns retries; one attempt per request
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",      # no second, non-streaming request after a stream failure
}
# MCP limits (design 3.4): without them the CLI rewrites large text results, times tool calls out after 60 s or
# backgrounds long calls.
_MCP_FLAGS = {
    "MAX_MCP_OUTPUT_TOKENS": "2000000",                    # avoids an extra count_tokens request on big text results
    "MCP_TOOL_TIMEOUT": "3600000",                         # ms; also bounds non-SSE replies (default 60 s)
    "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT": "0",              # no idle abort; the MCP server sends progress keepalives
    "CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS": "0",             # never move a long MCP call to the background
    "CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH": "1000000",   # characters; default 4096 truncates the generated field guides
}
_SIGNALLING_FLAGS = {"CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS": "1"}   # needed to see session_state_changed frames
# Information boundary (AGENTS.md "Context carries only necessary information", "Restricted Information Sources"): a
# deliberate change of one native default. When any directory above the actor directory holds a ".git" entry (a run
# below a checkout's runs/, an empty /tmp/.git on a shared host) the CLI puts a gitStatus block into the first user
# message: the host repository's branch, git user name, untracked host paths and recent commit subjects. This variable
# is the CLI's only switch for it (2.1.321 and 2.1.325, measured with tests/support/prompt_capture.py); the same switch
# also drops the generic "# Git" commit/PR advice (no host data) from the native Bash tool description, and nothing else
# of the request changes. It wins over the ``includeGitInstructions`` setting and, being Auriga-owned, cannot be
# re-enabled from a provider settings file.
_BOUNDARY_FLAGS = {"CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1"}
# Native memory (CLAUDE_CODE_DISABLE_AUTO_MEMORY) and CLAUDE.md discovery stay at the CLI defaults (user decision
# 2026-11-12); ``model.native_memories=false`` sets only the former. Background tasks, Monitor and cron are enabled: an
# unsolicited native turn is adopted by the session (design R4.2).
_NO_AUTO_MEMORY = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
_MODEL_VARS = ("ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
               "ANTHROPIC_DEFAULT_OPUS_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")   # one model for everything
# Removes the unrelated git-attribution reminder from the first user message.
_SETTINGS_JSON = json.dumps({"includeCoAuthoredBy": False}, separators=(",", ":"))
# Values that are paths or ports of this run: left out of the recorded ``native_settings`` (the manifest keeps the
# switches, not the run layout).
_RUN_SPECIFIC_ENV = frozenset({"PATH", "HOME", "TMPDIR", "CLAUDE_CODE_TMPDIR", "XDG_RUNTIME_DIR", "CLAUDE_CONFIG_DIR",
                               "ANTHROPIC_BASE_URL", AGENT_MARKER_ENV})
# ``agent.claude_code.upstream.source="claude_settings"``: the settings file's ``env`` entries the host relay consumes
# (endpoint and credential; never forwarded to the CLI).
SETTINGS_CREDENTIAL_ENV = frozenset({"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"})
# Every variable ``build_launch`` / ``clean_environment`` / the SDK set in the CLI environment (plus ANTHROPIC_MODEL,
# which would compete with the model Auriga passes as ``--model``): a settings-file ``env`` entry with one of these
# names is refused (``claude_settings_env_reserved``), never silently dropped or allowed to override Auriga's value.
AURIGA_OWNED_ENV = frozenset({
    "CLAUDE_CONFIG_DIR", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "CLAUDE_STREAM_IDLE_TIMEOUT_MS",
    "CLAUDE_CODE_TMPDIR", "XDG_RUNTIME_DIR", "IS_SANDBOX", "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "ANTHROPIC_MODEL",
    "PATH", "HOME", "TMPDIR", "LANG", AGENT_MARKER_ENV, *_DISABLE_FLAGS, *_RELAY_FLAGS, *_MCP_FLAGS,
    *_SIGNALLING_FLAGS, *_BOUNDARY_FLAGS, *_NO_AUTO_MEMORY, *_MODEL_VARS, *REEXPORT_VARS})


# ------------------------------------------------------------------------------------------------ effort

def effort_choices():
    """``model.effort`` values this backend accepts: ``("low", "medium", "high", "xhigh", "max")`` (SDK levels)."""
    return tuple(EFFORT_CHOICES)


def native_effort(effort):
    """The SDK ``effort`` for ``model.effort`` (identity for the five Claude levels).

    Raises ``ValueError("agent_effort_unsupported:claude_code:<value>")`` for anything else (``none``, ``minimal``,
    ``ultra``, ...); the launch gate in ``factory.backend_requirements`` refuses the same values earlier.
    """
    if effort not in EFFORT_CHOICES:
        raise ValueError(f"agent_effort_unsupported:{BACKEND_KIND}:{effort}")
    return effort


# ---------------------------------------------------------------------------------------------- instructions

def naming_note():
    """The one-sentence tool-name-prefix note appended to the instructions (design 3.5).

    The actor instructions and tool descriptions use bare registry names (``robot_observe``); Claude Code lists them
    as ``mcp__auriga__robot_observe``.
    """
    return (f"Tool names in these instructions are written without the prefix '{TOOL_NAME_PREFIX}' that your tool "
            "list shows; they refer to the same tools.")


def compose_instructions(instructions):
    """Text of the ``--append-system-prompt-file`` file: the already presented ``instructions`` (str) followed by the
    naming note, separated by a blank line. An empty ``instructions`` yields the note alone."""
    text = (instructions or "").rstrip()
    return (text + "\n\n" if text else "") + naming_note() + "\n"


# ----------------------------------------------------------------------------------------- pins and runtime

@dataclass(frozen=True)
class ClaudeRuntime:
    """Verified runtime of one launch: SDK version, CLI version, REAL executable path (not the launcher), sha256."""
    sdk_version: str
    cli_version: str
    executable: str
    sha256: str


def _installed_sdk_version():
    try:
        return importlib.metadata.version(SDK_PACKAGE)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("claude_code_sdk_missing: install the optional extra auriga-harness[claude] "
                           f"({SDK_PACKAGE}=={SDK_VERSION})") from error


def _bundled_cli():
    import claude_agent_sdk
    from claude_agent_sdk import _cli_version
    return Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude", _cli_version.__cli_version__


def _run_version_probe(executable, env):
    """``<cli> --version`` with the clean environment; returns stdout. One short process, no network."""
    try:
        completed = subprocess.run([os.fspath(executable), "--version"], env=dict(env), capture_output=True,
                                   text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"claude_code_cli_version_probe_failed:{type(error).__name__}") from error
    if completed.returncode != 0:
        raise RuntimeError(f"claude_code_cli_version_probe_failed:exit_{completed.returncode}")
    return completed.stdout


def resolve_runtime(cli_path=None, env=None, *, probe: Callable | None = None):
    """Verify the pins and locate the CLI. Raises ``VersionMismatch`` (``claude_code_version_mismatch``) on any
    difference and ``ValueError`` for a missing CLI; never falls back to a binary on ``PATH``.

    ``cli_path``: absolute path of the CLI (``agent.claude_code.cli_path``) or None for the CLI bundled with the SDK.
    ``env``: environment of the one ``--version`` probe (default: the empty environment plus ``PATH``).
    ``probe``: ``(executable, env) -> stdout`` replacement for tests.

    Checks, in order: installed SDK distribution version == ``SDK_VERSION``; the SDK's own ``__cli_version__`` ==
    ``CLI_VERSION``; ``<executable> --version`` == ``CLI_VERSION`` (an explicit ``cli_path`` of another version is
    refused). Returns ``ClaudeRuntime`` with the resolved real path and its cached sha256.
    """
    sdk_version = require_exact_version(BACKEND_KIND, _installed_sdk_version(), SDK_VERSION)
    bundled, bundled_version = _bundled_cli()
    require_exact_version(BACKEND_KIND, bundled_version, CLI_VERSION)
    executable = Path(cli_path) if cli_path else bundled
    if not executable.is_absolute():
        raise ValueError(f"claude_code_cli_path_must_be_absolute:{executable}")
    executable = executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"claude_code_cli_not_found:{executable}")
    output = (probe or _run_version_probe)(executable, env if env is not None else {"PATH": os.defpath})
    token = (output or "").strip().split(" ")[0] if (output or "").strip() else None
    cli_version = require_exact_version(BACKEND_KIND, token, CLI_VERSION)
    return ClaudeRuntime(sdk_version=sdk_version, cli_version=cli_version, executable=os.fspath(executable),
                         sha256=file_sha256_cached(executable))


# ---------------------------------------------------------------------------------------------------- sandbox

# Sandbox path of the published Harness source (read-only). The actor instructions name it when the sandbox is on
# (``actor_instructions(source_root=None)``; Workbench decides for every backend).
HARNESS_SOURCE_MOUNT = Path("/harness")
# Variables the sandbox sets itself inside the namespace (PATH adds the host ``rg`` directory, HOME is the actor
# directory, TMPDIR is the sandbox tmpfs, LANG is fixed). They are not handed over from the clean CLI environment;
# CLAUDE_CODE_TMPDIR / XDG_RUNTIME_DIR (run-private directories, mounted read-write at their own paths) are.
SANDBOX_OWNED_ENV = frozenset({"PATH", "HOME", "TMPDIR", "LANG"})
_SOCKET_PATH_LIMIT = 107   # sockaddr_un.sun_path on Linux, including the terminating NUL


@dataclass(frozen=True)
class SandboxPlan:
    """What the session decides for a sandboxed launch (``build_launch(sandbox=...)``).

    ``ipc``: run-owned IPC directory (``prepare_sandbox_ipc``), mounted read-write at its own path; holds the
        owner socket and the unix sockets of the in-namespace relays. ``ports``: host loopback TCP ports the CLI may
        reach, normally the model relay and the MCP endpoint; reachable inside at the same ``127.0.0.1:<port>``.
        ``read_paths``: ``[(host_path, sandbox_path)]`` read-only mounts, normally the actor-source snapshot at
        ``HARNESS_SOURCE_MOUNT``. ``packages``: True = the sandbox's PyPI index proxy, pip bootstrap and ``rg``
        (the only network the agent has, as for Codex).
    """
    ipc: Path
    ports: tuple
    read_paths: tuple = ()
    packages: bool = True


def sandbox_ipc_dir(root):
    """Run-owned IPC directory of the sandbox of the run at ``root``: ``$XDG_RUNTIME_DIR/auriga/<sha256(root)[:24]>``
    (``<tempdir>/auriga-ipc-<uid>/...`` without it), the same layout Codex uses. Unix socket paths below it must fit
    the 107-byte ``sockaddr_un`` limit independently of the archive path, so it is kept short and outside the run.
    ``ValueError("claude_code_ipc_path_too_long: ...")`` when even this is too long."""
    import hashlib
    import tempfile
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = (Path(runtime) / "auriga") if runtime else Path(tempfile.gettempdir()) / f"auriga-ipc-{os.getuid()}"
    if runtime and not os.path.isabs(runtime):
        raise ValueError("claude_code_runtime_directory_must_be_absolute")
    ipc = base / hashlib.sha256(os.fsencode(Path(root).resolve())).hexdigest()[:24]
    if len(os.fsencode(ipc / "sandbox-owner.sock")) >= _SOCKET_PATH_LIMIT:
        raise ValueError("claude_code_ipc_path_too_long: set XDG_RUNTIME_DIR to a shorter private runtime directory")
    return ipc


def prepare_sandbox_ipc(root):
    """Create (0700, owned by this user, never a symlink) and return ``sandbox_ipc_dir(root)``. A runtime directory
    given by ``XDG_RUNTIME_DIR`` must already be private (``unsafe_claude_code_runtime_directory`` otherwise)."""
    ipc = sandbox_ipc_dir(root)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        info = Path(runtime).lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("unsafe_claude_code_runtime_directory")
    for path in (ipc.parent, ipc):
        path.mkdir(mode=0o700, exist_ok=True)
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("unsafe_claude_code_ipc_directory")
        path.chmod(0o700)
    return ipc


def write_sandbox_launcher(path, command_prefix, executable, *, reexport=()):
    """Write the POSIX ``sh`` launcher of a sandboxed CLI and return its path (mode 0700, replaced; symlink refused).

    ``command_prefix``: ``AgentSandbox.prepare()[0]``, ``env -i bwrap <options> -- <in-namespace supervisor...>``;
        it must contain exactly one bare ``--`` (the bwrap option terminator). ``executable``: absolute path of the real
        CLI AS SEEN INSIDE THE NAMESPACE (the sandbox mounts it at its own host path). ``reexport``: names of
        variables the SDK sets at spawn time (``REEXPORT_VARS``).

    The script ends in ``exec <prefix up to --> [--setenv NAME "$NAME" ...] -- <supervisor> <executable> "$@"``: the
    SDK's arguments reach the CLI unchanged, stdin/stdout/stderr pass through bubblewrap and the supervisor, and each
    re-exported variable is copied from the launcher's own environment (only when set there) into the namespace with
    ``--setenv``, because ``env -i`` and ``--clearenv`` would drop it. Nothing else of the launcher's environment
    (the gateway's, merged in by the SDK) enters the namespace.
    """
    command = [os.fspath(item) for item in command_prefix]
    executable = os.fspath(executable)
    if not os.path.isabs(executable) or "\0" in executable or any("\0" in item for item in command):
        raise ValueError("launcher_executable_must_be_absolute")
    if command.count("--") != 1:
        raise ValueError("sandbox_prefix_separator_ambiguous")
    names = list(reexport)
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid_reexport_name:{name!r}")
    split = command.index("--")
    head, tail = command[:split], command[split:]
    lines = ["#!/bin/sh", "# generated by auriga.cortex.agents.claude_code.options; do not edit",
             "set -- " + " ".join(shlex.quote(item) for item in (*tail, executable)) + ' "$@"']
    for name in names:
        lines.append(f'if [ "${{{name}+set}}" = set ]; then set -- --setenv {name} "${name}" "$@"; fi')
    lines.append("exec " + " ".join(shlex.quote(item) for item in head) + ' "$@"')
    target = Path(path)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o700)
    except OSError as error:
        raise ValueError(f"launcher_path_unwritable:{target}:{error.strerror}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o700)
        handle.write("\n".join(lines) + "\n")
    return target


# ---------------------------------------------------------------------------------------------------- launch

@dataclass(frozen=True)
class ClaudeLaunch:
    """Everything ``ClaudeCodeSession.start`` needs; built by ``build_launch``.

    ``options``: ``ClaudeAgentOptions`` for ``ClaudeSDKClient``. ``env``: complete CLI environment. ``launcher``: the
    ``env -i`` script (``options.cli_path``). ``runtime``: ``ClaudeRuntime`` of the real CLI. ``thread_id``: the
    session id (host-chosen on a fresh start, the resumed one otherwise); ``resumed``: whether ``options.resume`` is
    used. ``config_dir``/``tmp_dir``/``actor_dir``/``instructions_file``: per-run paths. ``instructions_text`` /
    ``instructions_sha256``: the file contents and their digest. ``native_effort``: effective SDK effort.
    ``disallowed_tools``: ``({"name", "reason"}, ...)`` denied native tools (the enabled set is read back from the
    init message, ``native_tools_block``). ``native_settings``: exact switches applied
    (manifest ``agent_backend.native_settings``; no paths, ports or secrets). ``isolation``: ``ACTOR_ISOLATION``
    when the CLI runs in the bubblewrap sandbox, else ``"disabled"``; ``sandbox``: the prepared ``AgentSandbox`` (the
    session closes it after the CLI tree is gone) or None.
    """
    options: object
    env: dict
    launcher: Path
    runtime: ClaudeRuntime
    thread_id: str
    resumed: bool
    config_dir: Path
    tmp_dir: Path
    actor_dir: Path
    instructions_file: Path
    instructions_text: str
    instructions_sha256: str
    native_effort: str
    disallowed_tools: tuple
    native_settings: dict = field(default_factory=dict)
    isolation: str = "disabled"
    sandbox: object = None


def _write_private_text(path, text):
    import hashlib
    if os.path.islink(path):
        raise ValueError(f"private_file_symlink:{path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _running_as_root():
    return os.getuid() == 0


def build_launch(run_dir, *, model, effort, thread_id, instructions, relay_base_url, mcp_url, mcp_headers,
                 resume=False, config: ClaudeCodeConfig | None = None, native_memories=True, stderr=None, hooks=None,
                 relay_timeout_s=None, runtime: ClaudeRuntime | None = None, probe=None,
                 sandbox: SandboxPlan | None = None, instance_id: str | None = None,
                 settings_env: Mapping[str, str] | None = None):
    """Build the launch of one Claude Code session (see the module doc). Creates ``private/claude-config`` and
    ``private/tmp`` (0700), ``actor/``, the instructions file and the launcher below ``run_dir`` (must exist).

    Parameters
    ----------
    run_dir: the run directory (``<run>``); ``<run>/actor`` becomes cwd and ``HOME`` of the agent.
    model: model id sent to the relay (``model.name``); also set for the small/fast, haiku/sonnet/opus and subagent
        model variables so one model serves every call; no ``fallback_model``.
    effort: ``model.effort``; one of ``effort_choices()``, else ``ValueError``.
    thread_id: session id. Fresh start: a uuid4 string chosen by the host, passed as ``session_id``. Resume
        (``resume=True``): the previous id, passed as ``resume`` (never both: reusing a ``session_id`` fails with
        "already in use"). Must be a UUID.
    instructions: actor instructions after presentation (scrubbing); the naming note is appended and the result is
        passed as ``--append-system-prompt-file`` after the preserved ``claude_code`` preset prompt.
    relay_base_url: loopback relay URL (``ANTHROPIC_BASE_URL``). The credential is the fixed placeholder
        ``PLACEHOLDER_API_KEY``.
    mcp_url, mcp_headers: URL and headers of the host MCP endpoint (``AurigaMcpServer.url`` / ``.headers``), registered
        as server ``auriga`` (tools appear as ``mcp__auriga__<name>``).
    config: ``ClaudeCodeConfig`` (``cli_path``, ``subagents``, ``auto_compact_window_tokens``); default: defaults.
    native_memories: ``model.native_memories``; False sets ``CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`` (the only memory switch;
        CLAUDE.md discovery is never touched). Default True = the CLI default (memory under the run-private config dir).
    stderr: ``(line: str) -> None`` for CLI stderr (session appends it to ``private/agent.log``).
    hooks: ``ClaudeAgentOptions.hooks`` mapping (the session registers ``PreCompact``).
    relay_timeout_s: the relay's upstream generation timeout in seconds, or None (no relay timeout). Must not exceed
        ``STREAM_IDLE_TIMEOUT_MS / 1000`` or the CLI would give up before the relay does (``ValueError``).
    runtime, probe: a pre-resolved ``ClaudeRuntime`` / version-probe replacement (tests); default: ``resolve_runtime``.
    instance_id: the session's ``runtime/agent.instance_id``; set as ``AGENT_MARKER_ENV`` in the CLI environment so
        every process the agent starts can be found at teardown (``isolation.terminate_agent_processes``). Default
        None: no marker. Not recorded in ``native_settings`` (run-specific).
    settings_env: provider variables forwarded from the settings file of ``upstream.source="claude_settings"``
        (``read_claude_settings_env``), e.g. ``{"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}``; added to the CLI
        environment unchanged (so also to the launcher and the sandbox namespace). A name in ``AURIGA_OWNED_ENV`` or
        ``SETTINGS_CREDENTIAL_ENV`` raises ``ValueError``. ``native_settings`` records only their names
        (``settings_env``), never the values. Default None: nothing forwarded.
    sandbox: ``SandboxPlan`` to run the CLI in the bubblewrap ``AgentSandbox`` (``features.sandbox_enabled``), else
        None. The sandbox is prepared here (host-side relays start) and returned as ``launch.sandbox``; the caller
        closes it. ``IS_SANDBOX`` keeps its single rule: set only when this process is uid 0 (the sandbox keeps the host
        uid, so the CLI's bypass-permissions root check is hit exactly when the gateway runs as root, sandbox or not).
    """
    from claude_agent_sdk import ClaudeAgentOptions
    config = config or ClaudeCodeConfig()
    if not isinstance(model, str) or not model:
        raise ValueError("invalid_model")
    sdk_effort = native_effort(effort)
    try:
        session_uuid = str(uuid.UUID(str(thread_id)))
    except ValueError as error:
        raise ValueError("claude_code_session_id_must_be_uuid") from error
    if relay_timeout_s is not None and relay_timeout_s * 1000 > STREAM_IDLE_TIMEOUT_MS:
        raise ValueError("relay_timeout_exceeds_claude_stream_idle_timeout")
    root = Path(run_dir)
    dirs = private_dirs(root, ["private", "private/claude-config", "private/tmp", "private/run"])
    config_dir, tmp_dir = dirs["private/claude-config"], dirs["private/tmp"]
    run_private = dirs["private/run"]
    actor_dir = root.resolve() / "actor"
    actor_dir.mkdir(exist_ok=True)

    env_extra = {"CLAUDE_CONFIG_DIR": os.fspath(config_dir), "ANTHROPIC_BASE_URL": relay_base_url,
                 "ANTHROPIC_API_KEY": PLACEHOLDER_API_KEY, "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(STREAM_IDLE_TIMEOUT_MS),
                 **_DISABLE_FLAGS, **_RELAY_FLAGS, **_MCP_FLAGS, **_SIGNALLING_FLAGS, **_BOUNDARY_FLAGS,
                 # task outputs and the messaging socket stay inside the run, never in the shared host /tmp
                 "CLAUDE_CODE_TMPDIR": os.fspath(tmp_dir), "XDG_RUNTIME_DIR": os.fspath(run_private),
                 **{name: model for name in _MODEL_VARS}}
    if not native_memories:
        env_extra.update(_NO_AUTO_MEMORY)
    # The CLI refuses bypassPermissions as uid 0 unless it is told it is in a deliberate sandbox (probe R4-0 s.2); the
    # Auriga bwrap sandbox keeps the host uid, so this is needed only when the gateway itself runs as root.
    is_sandbox = _running_as_root()
    if is_sandbox:
        env_extra["IS_SANDBOX"] = "1"
    if config.auto_compact_window_tokens is not None:
        env_extra["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(config.auto_compact_window_tokens)
    forwarded = dict(settings_env or {})
    reserved = sorted(set(forwarded) & (AURIGA_OWNED_ENV | SETTINGS_CREDENTIAL_ENV | set(env_extra)))
    if reserved:
        raise ValueError(f"claude_settings_env_reserved:{','.join(reserved)}")
    env_extra.update(forwarded)
    env = clean_environment({}, home=actor_dir, tmpdir=tmp_dir, extra=env_extra, instance_id=instance_id)

    runtime = runtime or resolve_runtime(config.cli_path, env, probe=probe)
    text = compose_instructions(instructions)
    instructions_file = dirs["private"] / "instructions.txt"
    digest = _write_private_text(instructions_file, text)
    prepared_sandbox = None
    if sandbox is None:
        launcher = write_env_launcher(dirs["private"] / "claude-launcher.sh", runtime.executable, env,
                                      reexport=REEXPORT_VARS)
    else:
        # Everything the CLI reads or writes must exist inside the namespace at the path the SDK passes: the real
        # binary (read-only), the run-private directories named by the environment (read-write), the instructions file
        # (read-only; named by --append-system-prompt-file) and the sandbox plan's read-only paths (/harness).
        # The environment is the same clean CLI environment; the variables the sandbox owns are left to it.
        prepared_sandbox = AgentSandbox(
            dirs["private"] / "sandbox", sandbox.ipc, actor_dir,
            state_dirs=[(config_dir, config_dir), (tmp_dir, tmp_dir), (run_private, run_private)],
            executables=[(runtime.executable, runtime.executable)],
            env={key: value for key, value in env.items() if key not in SANDBOX_OWNED_ENV},
            ports=sandbox.ports, packages=sandbox.packages,
            read_paths=[(instructions_file, instructions_file), *sandbox.read_paths])
        try:
            command_prefix, _overrides = prepared_sandbox.prepare()
            launcher = write_sandbox_launcher(dirs["private"] / "claude-launcher.sh", command_prefix,
                                              runtime.executable, reexport=REEXPORT_VARS)
        except BaseException:
            prepared_sandbox.close()
            raise

    isolation = ACTOR_ISOLATION if prepared_sandbox is not None else "disabled"
    disallowed = [dict(name=name, reason=reason) for name, reason in DISALLOWED_TOOLS]
    if not config.subagents:
        disallowed += [dict(name=name, reason="subagents_disabled") for name in SUBAGENT_TOOL_NAMES]
    extra_args = {"append-system-prompt-file": os.fspath(instructions_file),
                  "replay-user-messages": None}   # steer/prompt text is echoed when consumed (transcript rows)
    keyword = dict(resume=thread_id) if resume else dict(session_id=session_uuid)
    try:
        options = ClaudeAgentOptions(
            disallowed_tools=[item["name"] for item in disallowed],   # no tools=/allowed_tools: the CLI default set
            system_prompt={"type": "preset", "preset": "claude_code"}, permission_mode="bypassPermissions",
            setting_sources=[], strict_mcp_config=True,
            mcp_servers={MCP_SERVER_NAME: {"type": "http", "url": mcp_url, "headers": dict(mcp_headers)}},
            model=model, effort=sdk_effort, cwd=actor_dir, cli_path=os.fspath(launcher), settings=_SETTINGS_JSON,
            extra_args=extra_args, max_buffer_size=MAX_BUFFER_BYTES, stderr=stderr, hooks=hooks, **keyword)

        native_settings = dict(
            env={key: value for key, value in env.items() if key not in _RUN_SPECIFIC_ENV and key not in forwarded},
            # names only: a forwarded value may be a provider header or another secret
            settings_env=sorted(forwarded),
            reexported_env=list(REEXPORT_VARS),
            options=dict(permission_mode="bypassPermissions", setting_sources=[], strict_mcp_config=True, settings=_SETTINGS_JSON,
                         system_prompt="preset:claude_code", extra_args=sorted(extra_args), max_buffer_size=MAX_BUFFER_BYTES,
                         tools="cli_default", disallowed_tools=[dict(item) for item in disallowed], fallback_model=None,
                         can_use_tool=False),
            is_sandbox_env=is_sandbox, native_memory=bool(native_memories),
            stream_idle_timeout_ms=STREAM_IDLE_TIMEOUT_MS,
            auto_compact_window_tokens=config.auto_compact_window_tokens)
        if prepared_sandbox is not None:
            # Names only: no host paths or ports (run layout) enter the manifest.
            native_settings["sandbox"] = dict(
                isolation=isolation, harness_source=os.fspath(HARNESS_SOURCE_MOUNT), packages=sandbox.packages,
                mounts_read_write=["actor", "claude-config", "tmp", "run", "ipc"],
                mounts_read_only=["claude_cli", "instructions_file", *(os.fspath(dst) for _, dst in sandbox.read_paths)],
                sandbox_env=sorted(key for key in env if key not in SANDBOX_OWNED_ENV),
                namespace_env_owned_by_sandbox=sorted(SANDBOX_OWNED_ENV))
        return ClaudeLaunch(
            options=options, env=env, launcher=launcher, runtime=runtime, thread_id=thread_id if resume else session_uuid,
            resumed=bool(resume), config_dir=config_dir, tmp_dir=tmp_dir, actor_dir=actor_dir,
            instructions_file=instructions_file, instructions_text=text, instructions_sha256=digest,
            native_effort=sdk_effort, disallowed_tools=tuple(disallowed),
            native_settings=native_settings, isolation=isolation, sandbox=prepared_sandbox)
    except BaseException:
        if prepared_sandbox is not None:
            prepared_sandbox.close()
        raise


# ------------------------------------------------------------------------------------------------ provenance

def native_tools_block(launch: ClaudeLaunch, init_tools=None):
    """Manifest ``native_tools``: ``{"enabled": [...], "disabled": [{"name", "reason"}, ...]}``.

    ``init_tools``: ``data["tools"]`` of the CLI's init message (``SystemMessage`` subtype ``init``). ``enabled`` is that
    list without the ``mcp__`` tools, in the CLI's order; the host never writes the list by hand, because the default
    set belongs to the pinned CLI. Without an init message ``enabled`` is ``None`` (unknown). ``disabled`` is what the
    host removed (``launch.disallowed_tools``) with the reason of each.
    """
    enabled = None if init_tools is None else [str(name) for name in init_tools if not str(name).startswith("mcp__")]
    return dict(enabled=enabled, disabled=[dict(item) for item in launch.disallowed_tools])


def _load_claude_settings_env(path):
    """``(resolved path, the "env" object)`` of a Claude Code settings file; ``ValueError`` naming the file."""
    location = Path(path).expanduser()
    try:
        data = json.loads(location.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"claude_settings_unreadable:{location}: {type(error).__name__}") from None
    env = data.get("env") if isinstance(data, dict) else None
    if not isinstance(env, dict):
        raise ValueError(f"claude_settings_missing_env:{location}: expected an \"env\" object")
    return location, env


def read_claude_settings_upstream(path):
    """``(base_url, headers)`` from a Claude Code settings file in the official format.

    ``path``: settings.json with ``{"env": {"ANTHROPIC_BASE_URL": ..., "ANTHROPIC_AUTH_TOKEN" | "ANTHROPIC_API_KEY": ...}}``.
    ``ANTHROPIC_AUTH_TOKEN`` is sent as ``Authorization: Bearer``, ``ANTHROPIC_API_KEY`` as ``x-api-key`` (the token
    wins when both are set, as in Claude Code). ``ANTHROPIC_BASE_URL`` defaults to the Anthropic API. Raises
    ``ValueError`` naming the file and the missing field, never a key value.
    """
    location, env = _load_claude_settings_env(path)
    base_url = env.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    token, api_key = env.get("ANTHROPIC_AUTH_TOKEN"), env.get("ANTHROPIC_API_KEY")
    if isinstance(token, str) and token.strip():
        headers = {"Authorization": "Bearer " + token.strip()}
    elif isinstance(api_key, str) and api_key.strip():
        headers = {"x-api-key": api_key.strip()}
    else:
        raise ValueError(f"claude_settings_missing_key:{location}: set env.ANTHROPIC_AUTH_TOKEN or env.ANTHROPIC_API_KEY")
    if not isinstance(base_url, str):
        raise ValueError(f"claude_settings_invalid_base_url:{location}")
    return base_url.strip().rstrip("/"), headers


def read_claude_settings_env(path):
    """The provider variables of a Claude Code settings file that are forwarded to the CLI environment.

    Every entry of the file's ``env`` object except ``SETTINGS_CREDENTIAL_ENV`` (``ANTHROPIC_BASE_URL``,
    ``ANTHROPIC_AUTH_TOKEN``, ``ANTHROPIC_API_KEY``: consumed by the host relay, never given to the CLI), as Claude
    Code itself applies ``settings.json`` ``env``. This keeps provider quirks in the provider's own file, for example
    ``{"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}`` (the CLI then sends no ``thinking.display``) for a gateway that
    answers ``display: "updates"`` with HTTP 400 ``PROVIDER_THINKING_DISPLAY_UNSUPPORTED``, or
    ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` for a model with a small output limit. Returns ``{name: value}`` (possibly
    empty). Raises ``ValueError`` naming the file and the variable, never a value: ``claude_settings_env_reserved``
    for a name Auriga sets itself (``AURIGA_OWNED_ENV``), ``claude_settings_env_invalid_name`` for a name that is not
    a shell variable name, ``claude_settings_env_not_string`` for a value that is not a string (or contains NUL).
    """
    location, env = _load_claude_settings_env(path)
    forwarded = {}
    for name, value in env.items():
        if name in SETTINGS_CREDENTIAL_ENV:
            continue
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"claude_settings_env_invalid_name:{location}: {name!r}")
        if name in AURIGA_OWNED_ENV:
            raise ValueError(f"claude_settings_env_reserved:{location}: {name} is set by Auriga for the Claude Code "
                             "CLI; remove it from env")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError(f"claude_settings_env_not_string:{location}: {name} must be a string")
        forwarded[name] = value
    return forwarded


def provenance_block(launch: ClaudeLaunch, *, model, counting, wire_api="anthropic_messages", upstream_provider=None,
                     isolation="disabled", context_delivery="full", init_tools=None, init_model=None,
                     capabilities=None, evidence_client="relay", decision_reminder_channel=None):
    """The manifest ``agent_backend`` block (design 7.3) for Claude Code; JSON-serialisable, observer data only.

    ``launch``: from ``build_launch``. ``model``: configured model name. ``counting``: the model-call counting
    semantics string. ``wire_api``/``upstream_provider``: relay wire and provider label. ``isolation``:
    ``"disabled"`` until the Auriga sandbox supports this backend. ``context_delivery``: ``"full"``/``"delta"``.
    ``init_tools``/``init_model``: ``data["tools"]``/``data["model"]`` of the CLI's init message when known; the
    recorded ``native_tools.enabled`` is then the tool list the CLI really exposes (without ``mcp__`` names, see
    ``native_tools_block``), otherwise ``None`` (unknown). ``capabilities``: ``AgentCapabilities.to_dict()`` or None.
    ``decision_reminder_channel``: the channel string the session uses (None = omit).
    """
    runtime = launch.runtime
    block = dict(
        kind=BACKEND_KIND,
        sdk=dict(name=SDK_PACKAGE, version=runtime.sdk_version),
        cli=dict(version=runtime.cli_version, executable=runtime.executable, sha256=runtime.sha256),
        wire_api=wire_api, upstream_provider=upstream_provider, model_configured=model, effort=launch.native_effort,
        native_effort=dict(effort=launch.native_effort, live=LIVE_EFFORT, applies="session_start"),
        isolation=isolation, counting=counting,
        tool_transport=dict(kind="mcp_streamable_http", tool_name_prefix=TOOL_NAME_PREFIX, deferred_tools="eager",
                            client_parallel_hint="mcp_read_only_hint"),
        prompt_channel=dict(instructions_channel="append_system_prompt_file", native_base_prompt="preserved",
                            instructions_sha256=launch.instructions_sha256, tool_name_prefix=TOOL_NAME_PREFIX,
                            naming_note=naming_note()),
        native_tools=native_tools_block(launch, init_tools),
        native_settings=launch.native_settings, context_delivery=context_delivery, evidence_client=evidence_client,
        session_resumed=launch.resumed)
    if init_model is not None:
        block["native_model_id"] = init_model
    if capabilities is not None:
        block["capabilities"] = dict(capabilities)
    if decision_reminder_channel is not None:
        block["decision_reminder_channel"] = decision_reminder_channel
    return block
