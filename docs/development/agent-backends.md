# Agent Backends

Auriga drives the actor through a swappable agent runtime. `agent.backend` (config) / `--agent-backend` (CLI, one
run) selects one of three backends: `codex` (default), `claude_code`, `opencode`. All three implement the same
backend-neutral contract, so `Workbench`, `Harness`, the evaluation queue, the experiment library, and the reports
treat them uniformly; only `agent.*` sub-settings and the recorded native differences below vary. See the
[README](../../README.md#choosing-the-agent-runtime) for the install recipes and a quick summary, and
[Configuration](configuration.md#agent-backend-agent) for every `agent.*` field.

## Auriga backport scope

This backport targets Auriga `main` at `0fd2337`. It carries the shared agent runtime from
Capella's `feat/so101-real-robot` history (integration `c2657d4`, plus Claude settings/env and host Git-context
fixes), retaining the original optional OpenCode 1.19.5 adapter and the default Codex backend. It does not
bring in SO-101 hardware, calibration, remote GPU services, the newer provider-profile console or plugin mode.
Choose a backend and provider through configuration/CLI; the original Auriga web workbench displays its
transcript and lifecycle. Session `set_model`/`set_effort` APIs do not imply a new console settings endpoint.
Upstream Linux and provider results below describe the source implementation; backport validation is recorded
separately in [the migration report](reports/auriga-claude-backport.md).

## Overview

| | Codex | Claude Code | OpenCode |
|---|---|---|---|
| Native runtime | `openai-codex` app-server | `claude-agent-sdk==0.2.180` (bundles CLI `2.1.321`) | `opencode-ai@1.19.5` platform binary |
| Tool transport | native `dynamicTools` (in-process) | host-owned MCP over loopback HTTP | host-owned MCP over loopback HTTP |
| Model wire(s) | OpenAI Responses | Anthropic Messages | Anthropic Messages or OpenAI Responses |
| Optional extra | built in | `mamba run -n Auriga python -m pip install -e '.[claude]' -c requirements-claude.lock.txt` | `npm i --save-exact opencode-ai@1.19.5`, absolute `executable` path |
| Native terminal attach | yes | no | no |
| Native evidence agent | yes (`CodexEvidenceAgent`) | no (relay client) | no (relay client) |
| Native memory / cross-turn | yes (Codex home) | yes (run-private `CLAUDE_CONFIG_DIR`) | no |

## Architecture

### `AgentSession` contract and capabilities

Every backend subclasses `AgentSession` ([`auriga/cortex/agents/base.py`](../../auriga/cortex/agents/base.py)): a
blocking, thread-safe, one-native-thread/one-active-turn object with the lifecycle `start()` / `run_turn()` /
`interrupt()` / `close()`, plain-string `state` (`not_started`, `idle`, `running`, `completed`, `interrupted`,
`failed`, `disconnected`, `closed`, `replay`), and completion observed through `turn_done`, `state`, and
`turn_result`. `run_turn()` returns at admission, never at completion. `cortex/agents/factory.py` (`create_session`,
`backend_requirements`, `agent_backend`) picks the subclass named by `agent.backend` and imports only that backend's
SDK, so a replay viewer of a Claude Code or OpenCode archive opens with no native runtime installed, and
`backend_requirements` refuses a launch configuration a backend cannot honour (missing runtime, incompatible
upstream wire, unsupported `model.effort`, …) with one actionable sentence before any run directory, robot, or
native process exists.

Optional features are declared as `AgentCapabilities` flags (`steer`, `native_requests`, `terminal_attach`, `fork`,
`deferred_tools`, `native_evidence_agent`, `sandbox`, `native_memories`, `live_effort`, `native_wakeup`,
`compaction_signal`, …) instead of callers probing the backend name; see the dataclass docstring for every flag's
meaning. `CodexSession` ([`auriga/cortex/session/codex.py`](../../auriga/cortex/session/codex.py)) declares the
richest set (terminal attach, fork, deferred tools, native evidence agent, live effort, sandbox, native memories).
`ClaudeCodeSession` ([`auriga/cortex/agents/claude_code/session.py`](../../auriga/cortex/agents/claude_code/session.py))
declares `steer`, `native_wakeup` (R4.2 below), `compaction_signal="native+relay"`, `context_delivery="full"`.
`OpenCodeSession` ([`auriga/cortex/agents/opencode/session.py`](../../auriga/cortex/agents/opencode/session.py))
declares `steer`, `live_effort`, `compaction_signal="native+relay"`, `context_delivery="full"`.

**Self-initiated native turns (Claude Code).** The CLI can start a turn by itself while the host turn is terminal
or idle (a background shell finishing, a scheduled wake-up; `CronCreate`/`CronDelete`/`CronList`/`ScheduleWakeup`
stay enabled). The session adopts it as an Auriga turn (`agent.turn_started {origin: "native_wakeup"}`, a new turn
id, `turn_done` cleared, ending at the next native idle); a `run_turn()` issued while such a turn is running joins
it instead of raising `turn_already_active`. Codex and OpenCode never start a turn unprompted.

### MCP tool transport

Claude Code and OpenCode both connect to one host-owned loopback MCP endpoint,
[`AurigaMcpServer`](../../auriga/cortex/agents/mcp_server.py) (streamable HTTP, the minimal in-repo JSON-RPC subset
both clients use — no `mcp` package dependency). `tools/list` publishes the full registry catalogue with the
session's presentation scrub, projecting each `inputSchema` to a portable form (top-level
`anyOf`/`oneOf`/`allOf`/`if`/`then`/`else` removed; nested ones untouched) because Claude Code silently drops an MCP
tool whose schema has one and the Anthropic API rejects it. `tools/call` is the only path from an MCP call to
`Harness.handle_tool`, through [`ToolDispatcher`](../../auriga/cortex/agents/dispatch.py) (arrival-ordered
concurrent dispatch, keyed off the registry's own `ToolSpec.parallel`: a parallel-safe call may overlap other
parallel-safe calls, but execution order always equals arrival order for ordered calls — no name list in the
transport). [`McpSessionMixin`/`McpToolBridge`](../../auriga/cortex/agents/mcp_session.py) hold the turn helpers,
relay wiring, and shutdown order both MCP backends share. See [MCP tool catalogue](#mcp-tool-catalogue-claude-code-and-opencode) below for the recorded delivery differences from the Codex native path.

### Relay wire adapters and client profiles

One relay, [`ModelRequestProxy`](../../auriga/cortex/session/model_proxy.py), carries every generic policy (the
180-call budget, retries, the recovery gate, cancellation, evidence attribution, activity clocks, the image-trim/reuse
state machine, egress scrub). Format-specific behaviour lives in a `WireAdapter`
([`cortex/session/wire/`](../../auriga/cortex/session/wire/)): `openai_responses.py` (Codex and OpenCode),
`anthropic_messages.py` (Claude Code and OpenCode); `openai_chat` has no relay adapter yet (OpenCode refuses to start
on it). A `ClientProfile` ([`wire/base.py`](../../auriga/cortex/session/wire/base.py)) selects per-client delivery
behaviour: `early_stream` (Codex: stream opens immediately; OpenCode/Claude Code on `client_loop_guard`: deferred
delivery, no byte sent until the attempt is validated) and `client_loop_guard` (see
[Relay client-loop guards](#relay-client-loop-guards-claude-code-and-opencode) below). The Codex profile's bytes are
pinned by golden fixtures (`test_wire_golden.py`) and never touched by the guards.

### Continuity / compaction detection

With `context_delivery="delta"` (Codex) the Harness may leave out of a tool result what it already delivered in the
current image-context epoch. Claude Code and OpenCode manage their own history (auto-compaction, pruning, client
rewrites can drop old context), so both run `context_delivery="full"` and the relay runs a synchronous
[`ContinuityTracker`](../../auriga/cortex/agents/context_signal.py) on every generation request: chains are keyed by
`(thread_key, system_key)`, each request gets a `call_purpose` (`main`/`helper`/`subagent`), and a new history is
compared against the last one delivered to the agent. A discontinuity (`history_shortened`, `history_rewritten`,
`images_removed`, `system_changed`) calls the session's `on_context_reset(reason)`, which schedules the Harness's
decision-context reminder on the next actor tool response — the same mechanism Codex drives from its own native
compaction event. Native hints (Claude Code's `PreCompact` hook / `compact_boundary`; OpenCode's
`session.compacted`) are extra triggers, never a substitute.

### Transcript, usage, and evidence

[`TranscriptWriter`](../../auriga/cortex/agents/transcript.py) writes every backend's rows into the existing
`codex_item` state kind with Codex's own field shapes (`backend`, `native_id`, `message_id`, `parent_item_id` stored
alongside), so `Store.completed_final_texts`, library replay, and the web readers cannot tell backends apart;
final-answer stamping (last top-level assistant message of a `completed` turn with text and no tool call) is
computed the same way for every backend. [`RelayEvidenceClient`](../../auriga/cortex/agents/evidence.py)
(`agent.evidence.client="relay"`, the default for every backend except Codex) answers one evidence packet with one
fresh, streamed, tool-free request through the run's own relay — no agent process, home directory, history, or
memory — offering the same constructor keywords and `review_evidence()` call as `CodexEvidenceAgent` so
`VisualReviewer` and `self_briefing.generate` use either unchanged.

## Configuration per backend

**Claude Code key in the official Claude format.** Instead of an environment variable, point the upstream at a
project-local Claude Code settings file (git-ignored `configs/claude.local.json`):

```json
// configs/claude.local.json  (same "env" block as ~/.claude/settings.json)
{"env": {"ANTHROPIC_BASE_URL": "https://relay.example", "ANTHROPIC_AUTH_TOKEN": "sk-..."}}
// configs/auriga.local.json
{"agent": {"claude_code": {"upstream": {"source": "claude_settings", "settings": "claude.local.json"}}}}
```

The gateway reads the endpoint and key from that file (`ANTHROPIC_AUTH_TOKEN` → `Authorization: Bearer`,
`ANTHROPIC_API_KEY` → `x-api-key`); only the file path is archived with the run, never the key, and the agent only
sees a local placeholder key (same model as `model.codex_home` for Codex). `ANTHROPIC_BASE_URL` is the root the relay
appends `/v1/messages` to, so it has no trailing `/v1` (the Claude Code convention).

Every other entry of the file's `env` object is forwarded to the CLI environment unchanged, as Claude Code itself
applies the `env` block of `settings.json` (`read_claude_settings_env` in
[`claude_code/options.py`](../../auriga/cortex/agents/claude_code/options.py)). Provider quirks therefore stay in the
provider's own file:

- `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_API_KEY` are consumed by the host relay and never
  forwarded.
- A variable Auriga sets itself in the CLI environment (`AURIGA_OWNED_ENV`: `CLAUDE_CONFIG_DIR`, the model variables
  and `ANTHROPIC_MODEL`, the relay/MCP/telemetry/signalling switches, `CLAUDE_STREAM_IDLE_TIMEOUT_MS`,
  `CLAUDE_CODE_TMPDIR`, `XDG_RUNTIME_DIR`, `CLAUDE_CODE_AUTO_COMPACT_WINDOW`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY`,
  `IS_SANDBOX`, `PATH`/`HOME`/`TMPDIR`/`LANG`, the process marker and the SDK spawn variables) is refused at launch
  with `claude_settings_env_reserved:<file>: <NAME> ...`; it is never dropped silently or allowed to override Auriga's
  value. Values must be strings (`claude_settings_env_not_string`).
- The forwarded variables also reach the `env -i` launcher and, with `features.sandbox_enabled`, the sandbox namespace.
- Provenance records only their names (`manifest.agent_backend.native_settings.settings_env`), never the values (a
  value may be a header secret); none of it enters model context.

Example: a multi-model gateway (DeepSeek, Kimi, GLM, Qwen, ... behind one Anthropic Messages endpoint) that rejects
Claude Code's `thinking: {"type": "adaptive", "display": "updates"}` with HTTP 400
`PROVIDER_THINKING_DISPLAY_UNSUPPORTED` on every request (the turn fails on its first model call). The CLI's own switch
`CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0` makes it send no `display` field:

```json
// configs/claude.local.json
{"env": {"ANTHROPIC_BASE_URL": "https://gateway.example/gateway",
         "ANTHROPIC_AUTH_TOKEN": "sk-...",
         "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}}
```

A model whose provider output limit is below the CLI's own `max_tokens` can be given `CLAUDE_CODE_MAX_OUTPUT_TOKENS`
in the same block; Auriga never clamps it on its own.

See [Configuration — Agent Backend](configuration.md#agent-backend-agent) for the full `agent.*` field reference
(`agent.backend`, `agent.claude_code.*`, `agent.opencode.*`, `agent.evidence.*`, upstream `source`/`wire_api`/
`base_url`/`api_key_env`/`auth_scheme`/`provider`). In short:

- **Claude Code**: `agent.claude_code.upstream` must be an explicit `anthropic_messages` endpoint; `cli_path` is
  optional (defaults to the SDK-bundled CLI); `subagents` (default `true`), `context_delivery`, and
  `auto_compact_window_tokens` are config-file only. `model.effort` must be `low`, `medium`, `high`, `xhigh`, or
  `max`.
- **OpenCode**: `agent.opencode.executable` (absolute platform binary path) and `agent.opencode.upstream` (any
  relay-served wire, or `source: "codex_home"`) are required; `context_limit_tokens` (default `200000`) and
  `output_limit_tokens` (default `32000`) size the model's context/generation window. `model.effort` choices depend
  on the selected wire (`anthropic_messages`: `low`..`max`; `openai_responses`: `none`..`xhigh`).
- **Evidence**: `agent.evidence.client` (`native` default, `relay`) and `agent.evidence.upstream` select the client
  for `features.visual_review_model` / `features.self_briefing`; see
  [Configuration](configuration.md#evidence-helpers-agentevidence).

## Native capability policy

Embedded agents keep their own native tools and abilities (shell, file edit, own code, subagents, todo, skills, ...);
the harness does not restrict an agent to Auriga's tools and prefers each harness's default behaviour. Only two kinds of
tool are removed: tools that reach the internet (outside information would break benchmark validity) and tools that
block forever waiting for a human in a headless run. The removals are recorded in `manifest.agent_backend.native_tools`
as `{"enabled": [...], "disabled": [{"name", "reason"}]}` with reasons `online_access`, `waits_for_human` or
`subagents_disabled`; this is observer-only data and never enters model context.

### Claude Code (2.1.321, `auriga/cortex/agents/claude_code/options.py`)

- No `tools=` allowlist option exists: the agent gets the CLI's **default** native tool set minus a denylist
  (`DISALLOWED_TOOLS`): `WebSearch`, `WebFetch` (`online_access`; `CLAUDE_CODE_DISABLE_WEB_FETCH=1` also set), and
  `AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode` (`waits_for_human`, plan exit needs a human approval). The
  subagent tool is `Agent` on the wire and `Task` in the CLI's own init message (2.1.321); both names are denied
  together only when `agent.claude_code.subagents=false` (`subagents_disabled`; default `true`).
- `permission_mode="bypassPermissions"` (parity with Codex `approval_policy="never"`): nothing prompts, so no
  hand-maintained allowlist is needed to avoid `dontAsk` denying every non-read-only call of an unlisted tool. The
  CLI refuses to run as uid 0 under bypass unless it detects a deliberate sandbox (`IS_SANDBOX=1`, bubblewrap,
  docker); the launcher sets `IS_SANDBOX=1` only when the gateway itself runs as uid 0 or inside the Auriga
  sandbox.
- Kept at the harness default (design rule: prefer the harness default; remove only online access and
  human-waiting tools): Bash (including background runs), Read/Edit/Write/NotebookEdit/Glob/Grep, `Task`
  (subagents), `TaskStop`, `Monitor`, `Workflow`, `SendMessage`/`ListAgents`, `EnterWorktree`/`ExitWorktree`,
  `ReportFindings`, `Skill` (bundled skills and slash commands only), and the scheduling tools
  `CronCreate`/`CronDelete`/`CronList`/`ScheduleWakeup` (kept enabled — a 2026-11-12 user decision against an
  earlier "no wall-clock scheduling" exception; see [Residuals](#residuals-claude-code)). Native memory
  (`model.native_memories`) is left at the harness default (on): it writes to and reads from the run-private
  `CLAUDE_CONFIG_DIR`, never a path outside the run, so it never crosses runs the way Codex's own home-directory
  memory does not either; `setting_sources=[]` keeps host-global and nested `CLAUDE.md`/`AGENTS.md` discovery off.
- `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH=1000000` and `ENABLE_TOOL_SEARCH=false` (every registry tool published
  eagerly, matching Claude Code's and OpenCode's `deferred_tools=False`) are set so nothing the model needs is cut.

### Residuals (Claude Code)

- Root caveat: the exemption `IS_SANDBOX=1`/bubblewrap/docker is a CLI-side string check, not a verified sandbox;
  Docker itself is not detected by the CLI.
- A native subagent (`Task`/`Agent`) runs in the same CLI process tree and its MCP calls carry
  `x-claude-code-agent-id` / `claudecode/toolUseId`, but the registry sees them through the same
  `Harness.handle_tool` barrier as the main actor.
- CLAUDE.md/AGENTS.md auto-discovery stays off everywhere (a validity rule: discovery walks up into the
  developer's archive root and the user's home, pulling host-dependent, non-reproducible files into context); native
  memory under `CLAUDE_CONFIG_DIR` is unaffected.
- Scheduling tools have no wall-clock-aware episode semantics: a timer that fires after the episode ended is
  dropped (the session is closed, or the Harness is no longer accepting); a turn held open by pending scheduled work
  still ends through the ordinary no-progress / inactivity stop.

### OpenCode (1.19.5, `auriga/cortex/agents/opencode/config.py`)

- Permission map: `{"*": "allow", "webfetch": "deny", "websearch": "deny", "codesearch": "deny", "question": "deny",
  "external_directory": "allow", "doom_loop": "allow"}`, plus `"task": "deny"` only when `agent.opencode.subagents` is
  false (default true).
  - `webfetch`, `websearch`, `codesearch`: `online_access`.
  - `question`: `waits_for_human`.
  - `task`: `subagents_disabled` (only when the option is turned off).
- Kept: bash, read, edit/write/apply_patch, glob, grep, `task` (subagents; child sessions are the harness's concern),
  `todowrite`, `skill` with OpenCode's bundled skills, and the `lsp` tool where OpenCode offers it.
- The `lsp` and `formatter` config keys are left out: in 1.19.5 an absent key means off, which is the OpenCode default.
  Switching them on makes agent-written files (for example `package.json`) trigger package installs that
  `OPENCODE_DISABLE_LSP_DOWNLOAD` (kept at `1`) does not block.
- External skill directories (`~/.claude`, `~/.agents`) stay off (`OPENCODE_DISABLE_EXTERNAL_SKILLS`,
  `OPENCODE_DISABLE_CLAUDE_CODE`): they are the user's files, not part of the run.
- `read_effective_tool_ids(client)` reads `/experimental/tool/ids` (the whole registry) and removes the ids the agent's
  permission ruleset (`/agent`) denies; the session records the result as `native_tools.enabled`.

- Reasoning effort: `config.effort_variants(wire)` declares one model variant per tier of `EFFORT_CHOICES[wire]`
  (`anthropic_messages`: `low`..`max` -> `output_config.effort` + adaptive thinking; `openai_responses`:
  `none`..`xhigh` -> `reasoning.effort` + `summary: auto` + encrypted reasoning items); the session selects
  `model.effort` per prompt as `variant`. `reasoning` stays `false` and the model has no `options`, so OpenCode's
  model-id variant heuristics never apply. Measured on 1.19.5 (`tests/test_agent_opencode_effort_smoke.py`): every
  tier reaches main, subagent and compaction requests; the bundled `@ai-sdk/anthropic` 3.0.111 accepts `xhigh`; the
  thinking block (with signature) / reasoning item (with encrypted content) of a tool-calling step is replayed in
  the next request. Residual differences: OpenCode sends `max_tokens = output_limit_tokens` (default 32000; Claude
  Code up to 128000) and adds `text.verbosity: "low"` for `gpt-5.` model ids.

### Residuals (OpenCode)

- The permission map is enforced by OpenCode; a system-managed OpenCode config would override it, so start-up refuses
  when one exists (`opencode_managed_config_present`).
- `websearch`/`codesearch` only exist in the registry for some providers; they are denied regardless.
- A native subagent runs in a child session of the same server; its calls are subject to the same permission map.
- Shell commands run by the agent are not network-isolated by this policy; only the dedicated online tools are removed.

## Relay client-loop guards (Claude Code and OpenCode)

The model relay (`auriga/cortex/session/model_proxy.py`, `ModelRequestProxy`) runs two transport guards on client
profiles with `client_loop_guard` (`claude_code`, `opencode`; never `codex`, whose relay bytes are pinned by
`tests/test_wire_golden.py`). Both refuse the request without forwarding or charging it, block the transport with
`manual_retry_allowed=True`, and answer with the profile's single-attempt denial (`auriga_client_retry_runaway`).

- **Runaway guard** (`RUNAWAY_MAX_IDENTICAL_ADMISSIONS = 8` within `RUNAWAY_WINDOW_S = 2.0` s): byte-identical
  consecutive bodies on one thread, as when a stream ends without a terminal. `error_type="ClientRetryRunaway"`, event
  `model_client_retry_runaway`.
- **Consume guard** (`CONSUME_GUARD_MAX_IDLE_COMPLETIONS = 3`): a main-chain request whose history adds only the
  model's own earlier output (`WireAdapter.history_model_output_flags`) to the last delivered main request is *idle*,
  because the client did not act on the completion it was given. After 3 delivered idle completions in a row, the next
  idle request is refused: `error_type="ClientDidNotConsumeCompletion"`, evidence
  `classification="client_did_not_consume_completion"`, event `model_client_did_not_consume_completion`. Observed with
  opencode 1.19.5 when provider usage carries `input_tokens: null`: every answer was charged, OpenCode could not parse
  it and re-requested (180 calls in about 4 s on the Responses wire). With the guard the loop costs 4 charged
  completions on either wire. Any tool result, user/steer message, text merged into the last client message, or
  continuity discontinuity resets the count. Helper and subagent requests are not compared. Measured false positives:
  none in the 420 delivered main completions of the captured Claude Code and OpenCode sessions
  (`tests/fixtures/continuity/`) (`tests/test_model_proxy_consume_guard.py`).
- Residuals: a client that adds its own note to every re-request counts as adding input and is not detected. A loop
  inside a subagent chain is not compared. Both are still bounded by the budget. Provider usage is delivered unchanged
  (no normalisation of invalid counters).

## Sandbox (`features.sandbox_enabled`, all backends)

The sandbox exists for RoboDojo benchmark runs: the model must not read the RoboDojo / simulator source (reward
hacking). Every backend runs its native agent in the same bubblewrap namespace (`AgentSandbox`,
[`agent_sandbox.py`](../../auriga/cortex/session/agent_sandbox.py); Codex through its subclass `CodexSandbox`), and
`factory.backend_requirements` accepts `features.sandbox_enabled=true` for `codex`, `claude_code` and `opencode`.

- Inside: the agent's own binary (read-only), its run-private state directories (read-write), the actor directory
  (cwd and `HOME`), the published Harness source snapshot at `/harness` (read-only; written to
  `<run>/private/actor-source`), the system runtime, `rg` and the PyPI proxy. The relay (and the MCP endpoint for
  Claude Code and OpenCode) is reached at its unchanged `127.0.0.1` port through run-owned Unix sockets; the host
  reaches OpenCode's HTTP server only through `<ipc>/inbound/<port>.sock` (no host TCP listener).
- Never mounted: the checkout, the user's home, RoboDojo and simulator assets, the run's observer data. The network
  namespace has only `lo`.
- Recorded: `runtime/agent.isolation` and `manifest.agent_backend.isolation` =
  `bubblewrap_namespaces_package_egress_v1` (`disabled` without the sandbox); a resume across modes is refused.
- Host requirement: Linux with `bwrap` and system `python3`; unprivileged user namespaces are preferred, otherwise the
  rootful `sudo -n bwrap` + `setpriv --clear-groups` fallback is used (also on hosts where the gateway user has
  supplementary groups). The launch gate cannot probe this, so a host without bubblewrap fails at session start with
  `codex_sandbox_bubblewrap_not_found_on_path_or_os_defpath` (Codex) or
  `agent_sandbox_bubblewrap_not_found_on_path_or_os_defpath` (Claude Code, OpenCode) before the native agent
  starts; nothing ever falls back to unsandboxed execution.
- Verification: conformance item 28 (`AgentBackendConformance.test_sandbox_namespace` for Claude Code and both
  OpenCode wires; `tests/test_codex_sandbox_conformance.py` for Codex) on a Linux host, see
  [Validation](validation.md#host-capabilities).
- Residuals: the sandbox userland is the explicit `SHELL_COMMANDS` list of
  [`system_runtime.py`](../../auriga/cortex/session/system_runtime.py) plus `python3` (for example `perl`, `id` and
  `setsid` are absent). Inside the namespace, teardown of detached children relies on the pid namespace ending with
  its init; outside it, on the process marker scan.

## MCP tool catalogue (Claude Code and OpenCode)

`tools/list` (`mcp_server.mcp_tool_catalog`) publishes the registry actor catalogue with the session's presentation
scrub. Recorded harness differences from the Codex path:

- **Portable input schema.** Top-level `anyOf`/`oneOf`/`allOf`/`if`/`then`/`else` are removed from every `inputSchema`
  (`portable_input_schema`; nested ones are untouched). Claude Code 2.1.321 silently drops an MCP tool whose schema
  has one, and Anthropic's API rejects it (OpenCode on `anthropic_messages` forwards it unchanged). Only `task_plan`
  has one today (operation-dependent required fields). The rules stay in the tool description (the generated
  `input: ... Object rules: ...` clause of the field guide, asserted for every registry tool by
  `tests/test_tool_registry_prose_bounds.py`) and `Harness.handle_tool` validates against the full registry schema.
- **Description cap.** Claude Code truncates MCP descriptions to 4096 characters; the launch environment sets
  `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH=1000000` (recorded in `native_settings.env`) so every generated field guide
  arrives complete.

## Process teardown (Claude Code and OpenCode)

Both sessions stop their native process through one shared path, `isolation.stop_agent_tree`
(`auriga/cortex/agents/isolation.py`). The launch environment carries `AURIGA_AGENT_INSTANCE=<runtime/agent.instance_id>`,
which every native tool child inherits. At close the session kills the process tree it observed and every process that
carries this marker, then scans again; `agent_processes.confirmed` (and with it `runtime/agent.clean_shutdown`) is true
only when nothing targeted survived and the final scan found no marked process. This also catches children a shell
call detached (`cmd &`, `setsid`) after they were re-parented to pid 1, and it applies after an agent crash. A resumed
session first stops what the previous instance of the run left running (`agent.previous_process_stopped`; the resume
is refused when that cannot be verified).

- Linux reads `/proc/<pid>/environ`. On macOS with SIP the kernel withholds the environment of Apple platform
  binaries (`/bin/sh`, `/usr/bin/perl`, `/bin/sleep`, ...) from other processes. For those the scan uses the working
  directory instead: such a process is selected when it is orphaned, or is a child of a marked process, and its cwd
  lies inside the actor directory. An operator's shell that merely `cd`s into the run is never selected.
- Residuals (documented, not mitigated): agent code can drop the marker (`env -u`, `env -i`). On macOS, a platform
  binary started that way and running outside the actor directory is not found. The marker value is visible to the
  agent's shell; it is an opaque session id, not a credential.

## Comparison protocol for harness experiments

A benchmark comparison across backends is a harness-identity comparison, not a model comparison, unless every other
condition is held fixed (AGENTS.md "Establish the Comparison Protocol First"). Hold fixed across arms:

- **Model** (`model.name`) and **upstream provider** (`agent.<backend>.upstream.provider`): the same served model on
  the same provider, verified by `conditions.agent_cli` / the recorded SDK/CLI versions, not assumed from config.
- **Reasoning effort tier** (`model.effort`): pick a tier every backend in the comparison natively accepts
  (`factory.NATIVE_EFFORT_CHOICES`, OpenCode's `config.EFFORT_CHOICES[wire]`) — Codex accepts every tier, Claude
  Code accepts `low`/`medium`/`high`/`xhigh`/`max`, OpenCode's choices depend on the selected wire.
- **Evidence client** (`agent.evidence.client`): set `"relay"` on every arm when `features.visual_review_model` or
  `features.self_briefing` is on (the fairness rule, [Configuration](configuration.md#agent-backend-agent)) — a
  Codex arm left on `"native"` would compare `CodexEvidenceAgent` against `RelayEvidenceClient` on the other arms,
  not the agent runtime.
- **Subagents setting** (`agent.claude_code.subagents` / `agent.opencode.subagents`): record as an explicit
  condition; it also forces `context_delivery="full"` and changes `McpSessionMixin` attribution, so flipping it
  changes more than "whether delegation is available".
- **Sandbox mode** (`features.sandbox_enabled`): hold fixed across arms; it is frozen for the run's lifetime and a
  resume across modes is rejected.
- Task, variant, seed, layout, and budgets, as for any comparison (AGENTS.md "Experiments & Evaluation").

What is a condition of the backend itself, not an experimental intervention to control away — each is a recorded,
by-design difference, not a bug:

- **Tool-catalogue delivery.** Codex defers 7 of 17 registry tools behind native `tool_search` (eager on Claude Code
  and OpenCode, whose own tool search stays off); Codex drops schema keywords (bounds, examples, conditionals,
  nested property descriptions of `robot_action` / `geometry_query.request`) that Claude Code and OpenCode on
  Anthropic Messages keep, while OpenCode on OpenAI Responses drops a similar (but not identical) subset. The
  dropped bounds/units/defaults stay in the generated field-guide prose on every backend (AGENTS.md's
  per-parameter documentation requirement), so the contract is never silently incomplete.
- **Generation limits.** OpenCode caps `max_tokens`/`max_output_tokens` at `agent.opencode.output_limit_tokens`
  (default 32000); Claude Code sends up to 128000 for a recognised Claude model id (32000 otherwise); Codex has no
  separate cap.
- **Verbosity / reasoning fields.** Codex sends `text.verbosity: "high"` and `reasoning.context: "all_turns"`;
  OpenCode adds `text.verbosity: "low"` for `gpt-5.` model ids on the Responses wire; Claude Code has no verbosity
  field.
- **Native base prompts and tool catalogues.** Each backend's own base instructions, environment block, memory
  section, skills list, and native tool set are part of the condition under test, not rewritten by Auriga (see
  [`agent-backend-prompt-parity.md`](reports/agent-backend-prompt-parity.md) section 4 for the measured sizes).
- **Subagent attribution.** A Claude Code or OpenCode subagent's MCP registry calls are attributed to the main host
  turn (MCP carries no caller id); child-session usage is charged to the main thread with no per-subagent
  breakdown. Codex's own multi-agent mode has its own accounting.
- **No Chat Completions wire yet.** `openai_chat` has no relay adapter in this checkout; OpenCode refuses to start
  on it.
- **Steer timing.** A Claude Code steer joins the active turn through `query()` mid-stream; an OpenCode steer joins
  only at the next step boundary (`prompt_async` while busy always answers `204` immediately, but the text is
  delivered at the boundary).
- **Self-initiated turns.** Claude Code's native wake-ups (background shells, scheduled wake-ups) are adopted as
  Auriga turns (`native_wakeup=True`); Codex and OpenCode never start a turn unprompted.

## Known limitations and residuals

- The sandbox (all three backends, see [Sandbox](#sandbox-featuressandbox_enabled-all-backends)) needs Linux with
  bubblewrap; it was verified on a Linux host on the rootful `sudo -n bwrap` path, the unprivileged user-namespace path
  was verified for Claude Code and OpenCode only before the final fixes.
- `features.lean_projection` (Responses-shaped history projection) is Codex-only
  (`lean_projection_requires_codex_backend`).
- Claude Code's own MCP image re-encoding (images over 512,000 bytes or 2000 px per side) is a recorded per-harness
  difference, not mitigated by new host logic (image transport otherwise uses the same trim/reuse logic, byte for
  byte, on every backend).
- `apps/evaluation/checkpoint.py` and `turn_probe.py` refuse non-Codex archives (native thread fork is Codex-only;
  `AgentCapabilities.fork`).
- A resumed non-Codex run is gated against its archived configuration, not the invocation's; a resume after an
  unclean exit invalidates the model context epoch on every backend.

## Reports

- [Agent Backend Prompt Parity](reports/agent-backend-prompt-parity.md) — captured-request comparison of the first
  actor generation request per backend: instruction channel placement, tool-catalogue delivery, native additions,
  and the `actor_instructions.py` wording review.
- [Agent Backend Spikes](reports/agent-backend-spikes.md) — summary of the pre-implementation spikes against the
  real Claude Code CLI and OpenCode binary, and the Linux sandbox verification.
