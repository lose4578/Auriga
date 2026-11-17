# Agent Backend Spikes (05, 06, 08, 09, 10)

Summary of the pre-implementation and verification spikes behind the swappable agent runtime (Codex / Claude Code /
OpenCode). Each spike drove the real pinned binary against loopback fakes only (no real model, no real credential,
no network); full evidence and reproduction scripts are scratch-only and not part of this repository. This page
keeps only what was measured and the design effect it had — see [Agent Backends](../agent-backends.md) for where
each effect landed in the shipped contract.

## Claude Code against a loopback fake (CLI 2.1.321, `claude-agent-sdk==0.2.180`)

What was measured: every endpoint and header the CLI sends (`POST /v1/messages?beta=true`, once
`count_tokens` when an MCP result is large and `MAX_MCP_OUTPUT_TOKENS` is at its default), the native tool list and
permission denial behaviour under `bypassPermissions`, HTTP MCP tool-use concurrency and result shape, `query()`
steering into a running turn, kill-mid-tool/resume behaviour, and native auto-compaction plus retry behaviour on
relay-style failures.

Design effect:
- Confirmed the SDK/CLI version pin (`0.2.180` bundles CLI `2.1.321`, not `2.1.322` as an earlier note assumed) —
  recorded as the pin in `claude_code/options.py` and the install recipe in the README.
- Settled the native default tool list (`Bash`, `Read`, `Edit`, `Write`, `NotebookEdit`; no `Glob`/`Grep` in
  2.1.321) and that `dontAsk`-style allowlisting was unworkable, leading to the `bypassPermissions` +
  denylist-only policy ([Native capability policy](../agent-backends.md#claude-code-21321-aurigacortexagentsclaude_codeoptionspy)).
- Found the MCP description truncation (4096 chars) and the top-level-`allOf` schema drop, both fixed by
  `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH=1000000` and `portable_input_schema` (see the prompt-parity report).
- Measured the stream-idle clamp (`[300000, 1800000]` ms) that bounds the relay's own upstream generation timeout.

## OpenCode against a loopback fake (1.19.5, both relay wires)

What was measured: request shape and instruction placement on `anthropic_messages` and `openai_responses`,
tool/schema transformations (bounds and defaults stripped on Responses, `additionalProperties:false` forced on MCP
schemas), where MCP tool-result images land, retry behaviour on relay denials and provider errors (OpenCode retries
on bare status-code substrings and words like "overloaded" even inside an HTTP 400 body), steering and abort
semantics, `/event` reconnect and terminal reconciliation via `GET /session/{id}/message`, and process/group
teardown of detached bash-tool children.

Design effect:
- The retry-pattern finding drove the relay's OpenCode denial messages (`OPENCODE_DENIAL_MESSAGES`) to avoid every
  string OpenCode's own retry regex matches, and the client-retry runaway guard
  ([Relay client-loop guards](../agent-backends.md#relay-client-loop-guards-claude-code-and-opencode)).
- Confirmed `isError:true` must never be set on an MCP result carrying images (OpenCode drops them), applied in
  `encode_mcp_result`.
- Confirmed instructions must go through a config file, not the `prompt_async` system field (lost after
  auto-compaction and after a steer without it) — the `Instructions:` placement recorded in the prompt-parity
  report.
- Measured that bounds/defaults missing from Responses-wire schemas must stay in the tool description prose,
  enforced by `tests/test_tool_registry_prose_bounds.py`.
- Confirmed OpenCode never starts a turn unprompted (`native_wakeup=False`).

## T0 — Claude Code turn-end semantics

What was measured: 28 captured cases (interrupts, API errors, mid-stream truncation, steer-then-completion,
mid-session model/effort changes), finding that every native `idle` SystemMessage arrives 0.000–0.002 s after the
turn's last `ResultMessage`, with zero cases of an idle arriving before its result or a host turn ending without
one. Background shells (`run_in_background=true`) were found to start an unsolicited native turn 4 s after the
host turn went idle.

Design effect: turn end is classified by the last `ResultMessage` since the turn began, confirmed at the next
native `session_state_changed: idle` (not a 5 s quiet-timer fallback, which would misfire during a background
shell or an in-flight subagent task); `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` removes the one measured source of
unsolicited native turns. The quiet-timer guard stays as a defensive fallback only, never armed while a native task
is in flight, and is pinned by unit tests replaying the captured sequences.

## R4-0 — Revision 4 assumption probe

What was measured: bypass-mode permission enforcement under the SDK and launcher (confirmed, no extra flag
needed), subagent header/system-text/MCP attribution for both Claude Code (`x-claude-code-agent-id`,
`claudecode/toolUseId`) and OpenCode (own `x-opencode-session-id`, identical system text), interrupt behaviour
during a running subagent, native memory read-back inside a private `CLAUDE_CONFIG_DIR`, OpenCode's `lsp`/
`formatter` config defaults (an absent key means off in 1.19.5), and `task` subagent abort semantics (OpenCode's
main session makes one more model call after an abort with a running `task` in flight).

Design effect: the self-initiated-activity adoption rule (R4.2), the subagent attribution rule used by
`context_signal.py`'s `call_purpose` classification, and the finding that an OpenCode interrupt must block the
relay for the main session and every known subagent session *before* aborting (the main agent's one extra model
call after abort must not be forwarded or charged).

## Linux sandbox verification (R4-8)

What was measured: real bubblewrap-namespace runs of `AgentSandbox` for Claude Code and OpenCode on a Linux host,
on both the unprivileged-user-namespace path and the rootful `sudo -n bwrap` fallback, plus the full host-capability
and conformance test suites in strict mode.

Design effect: Claude Code's sandbox passed outright. OpenCode needed two fixes before it could start under the
sandbox — a startup health-poll timeout bug (the first `/global/health` request during `opencode serve`'s startup
window never answers, consuming the whole 30 s budget) and an `EROFS` crash from OpenCode trying to write a
`.gitignore` into its read-only-mounted config directory — both fixed in the shipped code; a third defect (a
version probe writing into the real gateway user's home instead of a temporary one) was also fixed. With those
fixes, both non-Codex backends pass their real-namespace sandbox test and the full conformance suite on Linux. The
sandbox section of [Agent Backends](../agent-backends.md) and [Configuration](../configuration.md) record the
resulting support matrix.
