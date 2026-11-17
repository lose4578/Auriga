# Agent Backend Prompt Parity (P4-3)

Captured-request review of the first actor generation request that each agent backend sends upstream for the same
benchmark episode start. It answers three questions: where the Auriga instructions land and whether they arrive
byte-identical, what each harness adds natively, and whether any wording in
[`actor_instructions.py`](../../../auriga/cortex/context/actor_instructions.py) is wrong for a backend other than
Codex.

- Source: branch `agent-harness/p4-3` at integration tip `6e0438f` (no product code changed by this review).
- Host: macOS 26 arm64 (Darwin 25.6.0), `Auriga` Miniforge environment, 2026-11-12. `features.sandbox_enabled=false`
  (bubblewrap is Linux-only, so the sandbox variant was checked by code reading, see section 6).
- Pinned binaries: Codex app-server `codex-cli 0.160.0` (bundled with `openai-codex 0.160.0`); Claude Code CLI
  `2.1.321` (bundled with `claude-agent-sdk 0.2.180`); `opencode 1.19.5` (`opencode-darwin-arm64`).
- No real model and no network: every upstream is a loopback fake from
  [`tests/support/fake_model_servers.py`](../../../tests/support/fake_model_servers.py).

## 1. Method

[`tests/support/prompt_capture.py`](../../../tests/support/prompt_capture.py) runs the real `Workbench.start(benchmark=True)`
with a process-free calibrated ARX X5 fixture robot (public task "Stack the three bowls together.", public rules and
default task notes), the real `Harness` and tool registry, the real backend session from `factory.create_session`,
the real relay and the pinned binary. It records exactly what Workbench passes to `session.start(tools,
instructions)` and to the first `session.run_turn(prompt)`, and the first request of the actor loop that reaches the
fake model (Codex: the actor thread, not a native helper; Claude Code: tools declared, no subagent header; OpenCode:
tools declared, actor session header). `analyze()` compares them; `snapshot()` writes the redacted layout (Auriga
text by reference, native text in full, run paths as `<run>`, ids as `<id>`, media as `<data-url>`, no hashes).

Four upstream configurations were captured: `codex` (Responses), `claude_code` (Anthropic Messages),
`opencode_anthropic` and `opencode_responses` (OpenCode on each relay wire). The non-Codex model ids are
`claude-parity-model` / `gpt-parity-model`: only their family prefix matters (OpenCode chooses its native base prompt
and edit tools from it); Codex uses its default `gpt-6-astra`.

Inputs common to all four (byte-identical by construction, one Workbench code path): instructions 40,599 characters
(robot physical information with the inline URDF, then `Harness instructions`, the observation contract, handoff
policy, 3D perception text and runtime capabilities; the length includes the run's source path, which the
sandbox-off variant names); first user turn 5,630 characters; 17 registry actor tools (198,751 characters of
name + description + schema JSON).

## 2. Result summary

| Check | Codex | Claude Code | OpenCode (anthropic_messages) | OpenCode (openai_responses) |
|---|---|---|---|---|
| Instructions channel | developer message `input[2]`, part 0 | tail of `system[1]` after the `claude_code` preset | `system[0]` after `Instructions:` | developer message `input[0]` after `Instructions:` |
| Instructions byte-identical, delivered once | yes | yes | yes | yes |
| Added by Auriga besides the instructions | nothing | the naming note | the naming note | the naming note |
| First user turn byte-identical | yes (`input[6]`) | yes (`messages[0]`) | yes (`messages[0]`) | yes (`input[1]`) |
| Registry tools in the first request | 10 eager (+7 behind `tool_search`) | **16 (`task_plan` dropped)** | 17 | 17 |
| Tool descriptions unchanged | yes | **3 truncated at 4096 chars** | yes | yes |
| Tool schemas | Codex subset (bounds, examples, conditionals and the nested descriptions of `robot_action` / `geometry_query.request` dropped) | identical | identical (incl. `task_plan` top-level `allOf`) | OpenAI-compatible subset (bounds, examples, conditionals dropped; `const` -> `enum`) |
| Native tools published | 18 | 19 | 9 | 8 |
| Request body (JSON chars) | 199,960 | 252,156 | 277,447 | 251,664 |

**Wording changes to `actor_instructions.py`: none.** No sentence is factually wrong for Claude Code or OpenCode
(section 6). The parity problems found are in tool-catalogue delivery, not in the prompt text (section 7).

## 3. Captured request snapshots

Placeholders: `<AURIGA_INSTRUCTIONS>` (the Workbench instructions, byte-identical), `<AURIGA_NAMING_NOTE>` (the one
sentence of design section 3.5), `<AURIGA_FIRST_USER_TURN>` (the Workbench prompt). Character counts are of the
delivered text. Request headers are listed by name only.

### 3.1 Codex (`POST /v1/responses`)

```text
fields: model="gpt-6-astra", tool_choice="auto", parallel_tool_calls=false,
        reasoning={effort:"high", context:"all_turns"}, text={verbosity:"high"}, store=false, stream=true,
        include=["reasoning.encrypted_content"], prompt_cache_key=<id>, client_metadata=<id>
        (no top-level `instructions`, no top-level `tools`)
input[0] additional_tools (developer): namespace functions [exec_command, write_stdin, request_user_input,
         request_user_input_async, apply_patch (custom), view_image, get_goal, create_goal, update_goal,
         + 10 eager Auriga tools], namespace clock [curr_time, sleep], namespace collaboration [followup_task,
         interrupt_agent, list_agents, send_message, spawn_agent, wait_agent], tool_search (client execution)
input[1] developer  21,420  native base instructions: "You are Codex, an agent based on GPT-6. ..." (When to ask
                            the user for permission, Autonomy and persistence, Personality, Working with the user,
                            Rules for getting work done, Using skills, Apps (Connectors), Plugins)
input[2] developer  part 0  40,599  <AURIGA_INSTRUCTIONS>
                    part 1   2,048  <skills_instructions> (bundled system skills; skill root <run>/private/codex-home/skills/.system)
                    part 2     362  <permissions instructions> "sandbox_mode is danger-full-access ... Network access is enabled. Approval policy is currently never."
                    part 3     920  <collaboration_mode> Default (request_user_input availability)
input[3] developer   2,429  <multi_agent_role> (spawn_agent / followup_task / send_message)
input[4] developer     271  <multi_agent_mode> (no proactive delegation)
input[5] user          520  <environment_context> cwd <run>/actor, shell zsh, current_date, timezone, filesystem
input[6] user        5,630  <AURIGA_FIRST_USER_TURN>
headers: accept, content-type, originator, session-id, thread-id, user-agent, x-client-request-id,
         x-codex-beta-features, x-codex-turn-metadata, x-codex-window-id, x-openai-internal-codex-responses-lite
```

At startup Codex also sends a native helper request (`x-openai-subagent: memory_consolidation`, because
`model.native_memories` defaults to true); it is not an actor request.

### 3.2 Claude Code (`POST /v1/messages?beta=true`)

```text
fields: model="claude-parity-model", max_tokens=32000, thinking={type:"adaptive", display:"updates"},
        context_management={edits:[{type:"clear_thinking_20251015", keep:"all"}]}, output_config={effort:"high"},
        stream=true, metadata=<id>
system[0]          94  "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK."
system[1]      46,445  native preset (5,707 chars incl. separators: role, # Harness, # Session-specific guidance, # Memory,
                       # Environment, # Context management) + "\n\n" + <AURIGA_INSTRUCTIONS> + "\n\n" +
                       <AURIGA_NAMING_NOTE> + "\n"
messages[0] user  5,630  <AURIGA_FIRST_USER_TURN> (plain string)
messages[1] system 8,605  "# Environment": working directory <run>/actor, git no, platform, shell, OS version,
                          untrusted-download guidance, "You are powered by the model claude-parity-model.",
                          Agent tool agent types, bundled skills list, today's date
tools: Agent, Bash, CronCreate, CronDelete, CronList, Edit, EnterWorktree, ExitWorktree, ListAgents, Monitor,
       NotebookEdit, Read, ReportFindings, ScheduleWakeup, SendMessage, Skill, TaskStop, Workflow, Write,
       + 16 mcp__auriga__* tools (no task_plan)
headers: anthropic-beta (claude-code-20250219, interleaved-thinking-2025-05-14, context-management-2025-06-27,
         mid-conversation-tool-changes-2026-07-01, thinking-token-count-2026-05-13, effort-2025-11-24,
         thinking-display-updates-2026-08-18), anthropic-version, x-app, X-Claude-Code-Session-Id, X-Stainless-*
```

The `# Memory` section names the run-private memory directory
(`<run>/private/claude-config/projects/<run>-actor/memory/`); it is left in place by user decision (it is what makes
native memory usable). `max_tokens` depends on the model id: 32000 for the unrecognised `claude-parity-model`,
128000 for `claude-opus-5-5` and `claude-sonnet-5-5` (same capture with those ids; everything else in this section,
including the missing `task_plan`, was unchanged).

### 3.3 OpenCode on `anthropic_messages` (`POST /v1/messages`)

```text
fields: model="claude-parity-model", max_tokens=32000, thinking={type:"adaptive", display:"summarized"},
        output_config={effort:"high"}, tool_choice={type:"auto"}, stream=true
system[0]      50,079  native anthropic prompt (8,645 chars up to and including "Instructions:\n":
                       "You are OpenCode, the best coding agent on the planet.",
                       Tone and style, Professional objectivity, Task Management, Doing tasks, Tool usage policy,
                       Code References, "You are powered by the model named claude-parity-model. The exact model ID
                       is relay/claude-parity-model", <env> working directory <run>/actor, workspace root /, git no,
                       platform, date)
                       + "Instructions:\n" + <AURIGA_INSTRUCTIONS> + "\n\n" + <AURIGA_NAMING_NOTE> + "\n"
                       + "\nSkills provide ... <available_skills> customize-opencode"
messages[0] user  [text 5,630]  <AURIGA_FIRST_USER_TURN>
tools: auriga_* x17 (every tool has eager_input_streaming=true), bash, edit, glob, grep, read, skill, task,
       todowrite, write
headers: anthropic-beta, anthropic-version, x-opencode-session-id, x-session-affinity, x-session-id
```

`Instructions:` is the relay's neutralisation of OpenCode's `Instructions from: <private path>` header
(`wire/base.py` `neutralize_instruction_header`).

### 3.4 OpenCode on `openai_responses` (`POST /responses`)

```text
fields: model="gpt-parity-model", max_output_tokens=32000, reasoning={effort:"high", summary:"auto"},
        include=["reasoning.encrypted_content"], store=false, tool_choice="auto", stream=true, prompt_cache_key=<id>
input[0] developer 51,135  native gpt prompt (9,701 chars up to and including "Instructions:\n": "You are
                           OpenCode, You and the user share the same
                           workspace ...", Editing Approach, Autonomy and persistence, Editing constraints, Special
                           user requests, Frontend tasks, Working with the user, Formatting rules, Response channels,
                           model line, <env>) + "Instructions:\n" + <AURIGA_INSTRUCTIONS> + "\n\n" +
                           <AURIGA_NAMING_NOTE> + "\n" + skills block
input[1] user  [input_text 5,630]  <AURIGA_FIRST_USER_TURN>
tools: apply_patch, auriga_* x17, bash, glob, grep, read, skill, task, todowrite
headers: Authorization, x-opencode-session-id, x-session-affinity, x-session-id
```

## 4. Native additions per harness (part of the condition under test)

| | Codex | Claude Code | OpenCode |
|---|---|---|---|
| Base prompt | Codex base instructions, 21,420 chars | `claude_code` preset, 94 + 5,707 chars | provider prompt chosen by model id, with its `<env>` block: anthropic 8,645 / gpt 9,701 chars |
| Environment block | `<environment_context>` user message (cwd, shell, date, timezone, filesystem) | `# Environment` system message after the first user turn (cwd, git, platform, shell, OS version, model id, agent types, skills, date) | `<env>` inside the system text (cwd, workspace root, git, platform, date) plus the model id |
| Memory | none on a fresh run (the memory pipeline runs as a separate helper request) | `# Memory` section with the run-private memory directory | none |
| Skills | `<skills_instructions>` (bundled system skills) | bundled skills list | one built-in skill (`customize-opencode`) |
| Subagents | `<multi_agent_role>` / `<multi_agent_mode>`, collaboration namespace | Agent tool with agent types | `task` tool |
| Other | permissions block ("Network access is enabled" with the sandbox off), collaboration mode | Harness bullets (permission mode, hooks, parallel tool calls) | WebFetch advice for OpenCode docs although `webfetch` is denied |
| Native tools | 18 (incl. `request_user_input*`, answered empty by the host, `codex.py:506`) | 19 (WebSearch, WebFetch, AskUserQuestion, Enter/ExitPlanMode removed; tool search off) | 9 / 8 (webfetch, websearch, codesearch, question denied) |
| Native text total | 27,970 chars | 14,409 chars | 9,350 / 10,406 chars |
| Native tool definitions | 15,633 chars | 55,831 chars | 19,558 / 17,361 chars |

Uncontrollable native context (base prompt, environment block, own tool catalogue, model-dependent request fields) is
part of the condition under test (design section 6) and is not rewritten.

## 5. Auriga tool catalogue

All three clients keep the bare registry descriptions; the prefix comes only from the client (`""`,
`mcp__auriga__`, `auriga_`). Per tool (registry description length in characters):

| Tool | Chars | Codex | Claude Code | OpenCode anthropic | OpenCode responses |
|---|---:|---|---|---|---|
| task_plan | 6868 | ok; schema subset | **missing** | ok | ok; schema subset |
| robot_observe | 1859 | ok; schema subset | ok | ok | ok; schema subset |
| robot_capabilities | 734 | deferred (tool_search) | ok | ok | ok; schema subset |
| robot_information | 1566 | deferred (tool_search) | ok | ok | ok; schema subset |
| robot_contact_overlay | 1690 | ok; schema subset | ok | ok | ok; schema subset |
| robot_action_status | 930 | deferred (tool_search) | ok | ok | ok; schema subset |
| robot_wait | 1668 | ok; schema subset | ok | ok | ok; schema subset |
| history_list_actions | 2619 | ok; schema subset | ok | ok | ok; schema subset |
| history_view_observation | 1780 | ok; schema subset | ok | ok | ok; schema subset |
| history_read_trace | 1938 | deferred (tool_search) | ok | ok | ok; schema subset |
| history_annotate_action | 799 | deferred (tool_search) | ok | ok | ok; schema subset |
| geometry_query | 23852 | ok; schema subset | **truncated to 4109** | ok | ok; schema subset |
| program_run | 5741 | ok; schema subset | **truncated to 4109** | ok | ok; schema subset |
| program_status | 2864 | deferred (tool_search) | ok | ok | ok; schema subset |
| vision_compare | 2457 | deferred (tool_search) | ok | ok | ok; schema subset |
| robot_action | 35790 | ok; schema subset | **truncated to 4109** | ok | ok; schema subset |
| operation_memory | 1716 | ok; schema subset | ok | ok | ok; schema subset |

"ok" = description byte-identical to the registry; "schema subset" = the client dropped schema keywords:

- Codex 0.160.0: `minimum`/`maximum` (38 each), `minLength` 26, `maxItems` 23, `maxLength` 16, `examples` 10,
  `pattern` 9, `exclusiveMinimum` 9, `if`/`then`/`else`, `default`, `uniqueItems`, `prefixItems`, a nested `oneOf`
  (`geometry_query.request`) and 115 nested property descriptions (all of `robot_action` and
  `geometry_query.request`); `task_plan` keeps an empty `allOf: [{}]`.
- OpenCode on Responses: `minLength` 92, `maxItems` 65, `minItems` 59, `minimum` 47, `maximum` 44, `examples` 17,
  `const` 17 (rewritten as one-value `enum`), `exclusiveMinimum` 16, `maxLength` 16, `pattern` 9, `default` 5,
  `dependentRequired` 4, `exclusiveMaximum` 4, `if`/`then`/`else`, `uniqueItems`, `minProperties`, `prefixItems`;
  `task_plan` keeps `allOf: [{}]`. This is the R3 behaviour recorded in the design.
- Claude Code and OpenCode on Anthropic Messages: schemas byte-identical to the registry.

On both Responses paths the dropped bounds, units and defaults remain in the generated field guide of the
description, as AGENTS.md requires. On Claude Code that compensation fails for the three truncated tools (section 7.2).

## 6. Wording review of `actor_instructions.py`

Every sentence of the delivered instructions was checked against the captured requests and the session code. The
instructions text never names a harness or a tool prefix (pinned by `test_instructions_name_no_backend`).

| Text | Claim | Codex | Claude Code | OpenCode | Verdict |
|---|---|---|---|---|---|
| "Call the supplied tools directly. Their results contain text and independent images" | tool results carry separate image items | dynamic-tool image items | MCP image blocks in `tool_result` (conformance items 4, 5; CLI re-encodes images > 512,000 bytes / > 2000 px, recorded difference) | image blocks in `tool_result` / `function_call_output` (host never sets `isError` on image results) | correct |
| "No JavaScript wrapper, image extraction or base64 printing is needed." | nothing to unwrap | code mode is off (`code_mode_host=false`) | no such wrapper | no such wrapper | correct; Codex-motivated but true everywhere, so left as is |
| "Use native shell and file tools for computation, files and memory inside your sandbox." | native shell and file tools exist | `exec_command`, `apply_patch`, `view_image` | Bash, Read, Write, Edit, NotebookEdit | bash, read, write/edit or apply_patch, glob, grep | correct |
| "Read the public Harness implementation under /harness." (sandbox on) / "under `<source_root>`" (sandbox off) | the snapshot is readable there | sandbox mounts it at `/harness` (`codex.py:740`) | `/harness` (`claude_code/session.py:398`) | `/harness` (`opencode/config.py:366`) | correct; Workbench chooses the variant for every backend (`workbench.py:431`); the sandbox-off path is the run's `private/actor-source` |
| "Install Python packages with python3 -m pip; packages and working files stay in your sandbox." | pip works inside the sandbox | `packages=True` (`codex.py:739`) | `packages=True` (`claude_code/session.py:397`) | `packages=True` (`opencode/session.py:416`) | correct |
| "Web search and general network access are unavailable." / "Code runs without OS sandbox isolation. Web search is disabled." | no web search; network limited only in the sandbox | `web_search = "disabled"` | WebSearch/WebFetch removed, tool search off | webfetch/websearch/codesearch denied | correct |
| "Python client.call returns success, kind, data and display contentItems ..." | `program_run` client shape | Harness program client, backend-independent | same | same | correct |
| "The Harness preserves the current saved plan, attention and ledger when context is compacted." | the plan survives compaction | reminder in the compaction request | reminder in the next tool result after `on_context_reset` (`claude_code/session.py:788`) | same (`opencode/session.py:847`) | correct |
| `LEAN_PROJECTION_*` | Responses-shaped history projection | used | refused at start (`lean_projection_requires_codex_backend`) | refused | not delivered off Codex |

No sentence names deferred tool search, tool namespaces or other Codex-only mechanics. The one protocol-looking term,
`contentItems`, is the field of the Harness program client's reply (`auriga/runtime/programs/auriga_client.py`
`call`), which is the same on every backend. Three observations were recorded instead of changed:

1. The sandbox-off sentence "Use native shell and file tools for computation, files and memory in your working
   directory." is a placement preference. Native memory lives outside the working directory on Claude Code (its
   `# Memory` section names the run-private memory directory) and on Codex (`CODEX_HOME`), and the native section is
   the more specific one. It is not factually wrong and it is pinned by
   `tests/test_robot_information_integration.py`; a neutral rewrite ("... for computation and files in your working
   directory; native memory stays where your harness keeps it") would change the Codex prompt too and is left to a
   deliberate prompt revision.
2. "No JavaScript wrapper ..." is redundant on all three backends as configured (see the table); removing it would
   also change the Codex prompt.
3. Outside `actor_instructions.py`: the ARX X5 robot briefing
   (`auriga/robot/model/robot_information_data/arx_x5/briefing.txt`) says "Astra's normal motion tools request
   Cartesian end-effector/contact poses". "Astra" names the Codex actor model of the original setup; no harness tells
   the model it is Astra, and under Claude Code or OpenCode it is not. The meaning is still recoverable. The same name
   appears in helper prompts (`visual_review.py`) and in the `robot_policy_step` description. This is model naming,
   not backend mechanics, and is reported as an open item.

## 7. Defects found (tool-catalogue delivery, not wording)

### 7.1 `task_plan` schema: top-level `allOf`

`task_plan` publishes a top-level `allOf` with `if`/`then`/`else` operation rules
([`definitions.py:552-563`](../../../auriga/runtime/tools/definitions.py)). Anthropic tool input schemas do not
accept a top-level `anyOf`/`oneOf`/`allOf`:

- Claude Code 2.1.321 silently drops the tool. Its MCP loader normalises such schemas only behind the feature flag
  `tengu_mcp_normalize_root_combinators`; otherwise the verdict is `normalize_gated` ("its input schema uses
  top-level allOf, which the Anthropic API does not accept"). The tool is in the MCP `tools/list` answer but absent
  from the CLI init tool list and from every model request. Every physical action requires a submitted plan
  (`task_plan_required`), so a Claude Code actor cannot act. Measured: a copy of `task_plan` without `allOf` is
  published; any top-level `anyOf`/`oneOf`/`allOf` is dropped.
- OpenCode on `anthropic_messages` forwards the schema unchanged, so a real Anthropic upstream is expected to reject
  every actor request. This is based on the CLI's own rule; it was not verified against the provider (no real model
  calls in this package).
- On Codex and OpenCode-Responses the conditional already never reaches the model (`allOf: [{}]`).

The rules are enforced at runtime by `TaskPlanning` (`task_plan_submit_requires_plan`,
`task_plan_submit_requires_ledger`, `task_plan_update_requires_plan_attention_or_ledger`,
`task_plan_read_accepts_only_operation`), are stated in the `operation` field description, and the generated
`input: ... Object rules: ... all of (...)` clause of the field guide in the tool description spells out the conditional.

**Fixed in 000920a.** Registry contract and Codex path are unchanged. The MCP transport
(`mcp_server.portable_input_schema`, applied in `mcp_tool_catalog`) removes top-level
`anyOf`/`oneOf`/`allOf`/`if`/`then`/`else` from every published `inputSchema` (nested ones stay); `Harness.handle_tool`
still validates against the full registry schema. `tests/test_tool_registry_prose_bounds.py` asserts that every
top-level combinator's rule is stated in the description, so future tools stay portable. Recorded harness difference
in `docs/development/agent-backends.md`. Pinned by
`RegistrySchemaPortability.test_no_mcp_published_tool_schema_has_a_top_level_combinator` and
`ClaudeCodePromptParity.test_every_registry_tool_is_published` (no longer `expectedFailure`).

### 7.2 Claude Code MCP description cap (4096 characters)

`CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` defaults to 4096 in 2.1.321. Longer MCP descriptions are cut and end in
`… [truncated]` (4109 characters). `geometry_query` keeps its prose but loses about 19.7 k of its 22 k field guide.
`program_run` loses the last 1.6 k of its field guide. `robot_action` loses the end of its prose (examples and
continuation rules, prose ends at 4995) and its whole 31 k field guide. The input schemas themselves are intact (with
bounds and per-property descriptions), but the "Object rules" and kind-specific guidance exist only in the prose.
**Fixed in 000920a:** `"CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH": "1000000"` in `_MCP_FLAGS`
(`claude_code/options.py`, recorded in `native_settings.env`). Measured: every published description on Claude Code
and OpenCode is byte-identical to the registry, all 17 registry tools reach the model. Pinned by
`ClaudeCodePromptParity.test_tool_descriptions_unchanged` (no longer `expectedFailure`) and
`test_agent_claude_options` (`test_mcp_description_cap_is_lifted_and_recorded`).

## 8. Differences that remain by design

- Instructions channel per backend (design section 6) and the one naming-note sentence (design section 3.5); the
  instructions are otherwise byte-identical, and so is the first user turn.
- Codex defers 7 registry tools behind native `tool_search`; Claude Code and OpenCode publish all eagerly
  (`deferred_tools="eager"`, design section 3.8). Claude Code's own tool search is off (`ENABLE_TOOL_SEARCH=false`).
- Schema delivery per client and wire (section 5); bounds stay in the description field guide.
- Native base prompts, environment blocks, memory section, skills, subagent and scheduling tools (section 4), per
  the capability policy (agents keep all native abilities except online access and human-waiting tools; Codex's
  `request_user_input` is answered empty instead of removed).
- Harness-chosen request fields: output limit (Claude Code 128000 for current Claude ids, 32000 for an unrecognised
  id; OpenCode `output_limit_tokens`, default 32000), thinking display (Claude Code `updates`, OpenCode
  `summarized`), Claude Code `context_management` (`clear_thinking_20251015`, keep all), Codex
  `text.verbosity: "high"`, `reasoning.context: "all_turns"` and `parallel_tool_calls: false`.
- Claude Code places its environment block as a `system` message after the first user turn; Codex as a user message
  before it; OpenCode inside the system text.

## 9. Limits

- Sandbox-on captures were not possible on macOS. With the sandbox on, only the `SANDBOX_INSTRUCTIONS` block differs,
  Workbench selects it for every backend, and all three sandboxes mount the source at `/harness` with the package
  proxy (section 6 references). Codex's native permissions block will then differ from the one captured here.
- One capture per configuration, the first actor request only; later requests (tool results, compaction, steer) are
  covered by the conformance suite (`tests/support/agent_conformance.py`, item 24 for instruction persistence).
- Optional tools that are off by default (`vision_review`, depth tools, `robot_policy_step`) were not captured.
- Statements about the real Anthropic API in section 7.1 rest on the pinned CLI's rule, not on a provider call.

## 10. Reproduce

```bash
# full redacted snapshots (analysis + snapshot per backend) into a fresh directory
AURIGA_OPENCODE_EXECUTABLE=/abs/path/opencode PYTHONPATH=tests mamba run -n Auriga python -m support.prompt_capture --out /tmp/prompt-parity
# the pinned checks (OpenCode classes skip without AURIGA_OPENCODE_EXECUTABLE)
PYTHONPATH=tests mamba run -n Auriga python -m unittest tests.test_agent_prompt_parity -v
```

Run both from the worktree root, so that the worktree's `auriga` package is imported. Result on this host with
all three runtimes: 31 tests, OK, no expected failures (sections 7.1 and 7.2 fixed).
