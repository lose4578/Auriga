# Validation and Release Guide

Validation must demonstrate that changes behave correctly along their actual call paths and clearly state the scope of collected evidence. For minor documentation edits, checking links and content is sufficient; changes affecting permissions, physical execution, state transitions, archiving, or scoring require regression tests covering failure and edge paths. The CI workflow is defined in [`ci.yml`](../../.github/workflows/ci.yml).

## Environment and Standard Checks

Run all commands from the repository root. Development uses the Miniforge `Auriga` environment by default—never create a project `.venv` or upgrade an environment in place while live experiments are using it. Auriga requires Python >= 3.11 (CI runs in a clean environment with Python 3.12 and Node 22); dependency versions are governed by `environment.yml`, `requirements.lock.txt`, and `pyproject.toml`, with the Codex SDK/CLI currently pinned to `0.160.0`. Before running checks, verify the resolved SDK/CLI versions from the `Auriga` interpreter rather than relying on whatever `codex` binary is on `PATH` (see [Environment Setup](running.md#environment-setup)). `ffmpeg` is required for video-related checks.

```sh
# Skip environment creation if Auriga already exists; verify dependencies per the Running Guide first.
mamba env create -f environment.yml
npm ci --prefix web
mamba run -n Auriga python -m unittest discover -s tests
npm run check --prefix web
npm run build --prefix web
mamba run -n Auriga python scripts/build_package.py --output dist
mamba run -n Auriga python scripts/check_package.py dist/auriga_harness-0.2.0-py3-none-any.whl
git diff --check
```

When validating subsequent releases, pass the exact path to the newly built wheel so `dist/*.whl` does not inadvertently pick up older wheels. `check_package.py` installs the wheel into a temporary directory via `pip --target` using the current interpreter and verifies it outside the repository checkout under Python's `-I` isolated mode without creating a virtual environment. The CPU test suite requires neither a live model nor a GPU simulator; a few asset-mapping tests are skipped when an external RoboDojo checkout is absent. To exercise those asset tests, set `AURIGA_ROBODOJO_ROOT` and `AURIGA_ROBODOJO_PYTHON` to your local installation paths and re-run, recording pass and skip counts separately. Native app-server tests run against a local loopback model stub; if SDK/CLI versions do not match, switch to the proper environment rather than skipping failures.

## Targeted Regressions by Scope

Native Codex sandbox regressions with isolation enabled (`features.sandbox_enabled=true`, disabled by default) require Linux, a discoverable `bwrap` binary, and system `python3`; unprivileged user namespaces are preferred, falling back to `sudo -n` and `setpriv` when user namespaces are disabled on the host. These tests verify isolation boundaries using real files, subprocesses, and a loopback Responses stub while covering tool installation path variations and IPC directory configurations. Actual PyPI package downloads and authenticated upstream model calls are recorded separately and are never triggered implicitly by the CPU test suite. Regressions for `features.sandbox_enabled=false` must verify that the primary Codex session, terminal, evidence agent, and `program_run` start cleanly while sandbox probing/loading is forbidden, that private home directories, placeholder credentials, broker review, and resource limits remain enforced, and that cross-mode session resumption is rejected. Because host file and loopback network access are expected when the sandbox is disabled, test them against temporary files and local stubs—never report unsandboxed test passes as proof of OS isolation.

The table below lists targeted test modules under `tests/` (for example, `PYTHONPATH=tests mamba run -n Auriga python -m unittest test_tool_registry test_typed_actor_output`). When changes span multiple public boundaries or prepare a release, run the full CPU test suite after targeted tests pass.

| Change Scope | Primary Test Modules | Behavior to Verify |
| --- | --- | --- |
| `ToolSpec` / handlers / output | `test_tool_registry`, `test_typed_actor_output`, `test_program_review` | Both actor and program entry points, zero side effects on invalid inputs, role permissions, typed facts, and no action replay on post-execution presentation errors. |
| Actions / worker / stopping | `test_unified_actions`, `test_native_wait`, `test_runtime_lifecycle` | Budgets, idempotency, and checkpoints; `partial`/`stopped`/`unknown` handling; terminal capture failure preserves receipts and invalidates stale observations. |
| Scene evidence | `test_scene`, `test_scene_contracts`, `test_scene_service` | Per-entity provenance and clocks, dirty/freshness invalidation, and service/job lifecycle. |
| Store / scoring / archives / reports | `test_contract_store`, `test_read_models`, `test_scoring_policy`, `test_experiments`, `test_evaluation_reports` | Unchanged file bytes/hashes/mtimes on sealed WAL archives, corrupted archive isolation, fixed case denominator, complete references and scopes, and preserved scoring facts on import. |
| Evaluation queue and completion hook | `test_benchmark_queue`, `test_experiment_hook` | Exact `argv` recognition across current and legacy entry points, PID/start-ticks/output ownership checks, and OOM retry boundaries. |
| Web UI / read-only API | `npm run check --prefix web`, corresponding Python API tests, browser fixtures | Pagination, terminal state refresh, read-only replay, input injection guards, and mobile layout. |
| Config / assets / entry points / wheel | `test_config`, full wheel build and check | Out-of-checkout imports and CLI console scripts, default config bundling, static web assets, and provenance metadata. |
| Native Codex tools and permissions | `test_codex_sandbox`, `test_agent_sandbox`, `test_agent_claude_sandbox` and `test_agent_opencode_sandbox` (plan and launch-gate tests run everywhere; the real-namespace tests need `bubblewrap_namespaces`), `test_agent_source_snapshot`, `test_actor_source`, `test_codex_native_preferences`, `test_actor_isolation`, `test_program_isolation`, `test_codex_image_transport`, `test_codex_evidence_agent` | Real native shell/memory operations, host file/network denial when sandboxed, zero sandbox dependency with preserved broker/review/budgets when unsandboxed (default), resume mode consistency, source snapshot publishing and placeholder auth, and dynamic image transport. |
| Sandbox boundary, all backends (conformance item 28) | `test_agent_claude_code_conformance`, `test_agent_opencode_conformance` (`test_sandbox_namespace`), `test_codex_sandbox_conformance` | With `features.sandbox_enabled=true` the agent's own native shell runs a probe inside the namespace: `/harness` holds exactly the published source snapshot (read-only); the checkout, regular files of the user's home and a planted fake RoboDojo root next to the run directory are unreadable; the only interface is `lo`, the forwarded relay/MCP ports connect, a host-only loopback port and public addresses (`1.1.1.1:443`, `8.8.8.8:53`) fail within 5 s; a registry tool call round-trips; close leaves no bwrap, supervisor, agent or detached child process (tracked by pid and start time) and removes the run IPC directory; `runtime/agent.isolation` and provenance record `bubblewrap_namespaces_package_egress_v1`. OpenCode additionally reaches its server only through the inbound unix socket. Needs Linux with bubblewrap and the pinned runtimes (`requires_claude_cli`, `requires_opencode` with `AURIGA_OPENCODE_EXECUTABLE`, `requires_codex_binary`). |
| Relay evidence client | `test_relay_evidence_client`, `test_relay_bindings_evidence` | Per-wire request bodies through the relay against loopback fake servers, attribution, shared budget across the actor and `evidence` bindings, cancellation, recovery gate, egress scrub and alias restore, failure strings identical to `CodexEvidenceAgent`, the Codex native/relay switch, manifest `evidence_client`, a non-Codex self-briefing start, and the queue's evidence-path warning. Provider acceptance of the structured-output request is not covered (needs a user-authorised live probe). |
| Agent backend conformance bindings | `test_agent_claude_code_conformance`, `test_agent_opencode_conformance` (both relay wires), [`tests/support/agent_conformance.py`](../../tests/support/agent_conformance.py) | The shared conformance mixin's numbered items (tool catalogue delivery, image/MCP contract, continuity, subagent attribution, native-tool provenance, turn adoption, teardown) bound against each backend's real session and a loopback fake model; backend-specific `expectedFailure`s are recorded with the defect they track, not silently skipped. |
| Agent backend prompt parity | `test_agent_prompt_parity`, [`tests/support/prompt_capture.py`](../../tests/support/prompt_capture.py) | Byte-identical Auriga instructions and first user turn across Codex, Claude Code, and OpenCode (both relay wires); every registry tool reaches the model; no wording in `actor_instructions.py` is backend-specific. See [`agent-backend-prompt-parity.md`](reports/agent-backend-prompt-parity.md). |
| Model relay client-loop guards | `test_model_proxy_consume_guard`, `test_wire_golden`, `test_wire_openai_responses`, `test_wire_anthropic_messages`, `test_wire_opencode` | Byte-identical Codex relay bytes (golden fixtures, never touched by a `client_loop_guard` profile), the runaway and consume guards refusing without forwarding or charging a request, and the client-visible denial strings OpenCode does not retry. |
| Agent host capabilities (non-sandbox) | `test_agent_config`, `test_agent_dispatch`, `test_agent_isolation`, `test_agent_transcript`, `test_agent_mcp_server`, `test_agent_mcp_session`, `test_context_signal`, `test_agent_workbench`, `test_agent_harness_callers`, `test_agent_session_contract` | Backend-neutral `AgentSession`/`AgentCapabilities` contract, factory launch gating (`backend_requirements`), MCP tool bridge dispatch ordering, transcript row shapes and `agent.*` events, relay context-continuity signal, and process teardown markers; these run on every host (no kernel capability required). |

Use `find tests -name 'test_*.py'` (or `rg --files tests`) to check test module names before running, and add new regression tests to the existing test module owned by the affected subsystem. Fault-injection tests must assert persisted state facts and side-effect counts rather than merely matching error message substrings. If a historical minimal reproducer asserts buggy legacy behavior, keep it as baseline documentation rather than including it in the green regression suite.

### Host Capabilities

Sandbox and OS-isolation tests exercise real kernel features, so they run only on hosts that provide them and are skipped elsewhere instead of failing. Each gated test is decorated with `require(<probe>, "<what the test exercises>")` from [`tests/support/host_capabilities.py`](../../tests/support/host_capabilities.py). A probe checks the actual capability with the production lookup or a minimal kernel call, never the platform name alone, and is cached per test process. Probes that change process state (namespaces, seccomp filters) run in a short-lived child process.

| Probe | What it checks | Gated tests (examples) |
| --- | --- | --- |
| `bubblewrap_namespaces` | `bwrap` and system `python3` are found, and a namespace sandbox starts: first the production unprivileged user-namespace probe, then the `sudo -n` + `setpriv` fallback (5 s per attempt) | `test_codex_sandbox`, sandbox-on cases in `test_codex_native_preferences`, `test_codex_provider`, `test_codex_evidence_agent`, `test_actor_isolation`, the real-namespace tests of `test_agent_sandbox`, `test_agent_claude_sandbox`, `test_agent_opencode_sandbox`, and conformance item 28 (`test_sandbox_namespace`, `test_codex_sandbox_conformance`) |
| `program_isolation` | Landlock ABI >= 4, working seccomp filters, and `ldd`/`ldconfig` resolving `libseccomp.so.2`, as needed by bounded programs with the sandbox on (`program_services` fixtures default to it) | `test_program_isolation`, sandbox-on cases in `test_program_review`, `test_programs`, `test_actor_isolation`, `test_activity_sources`, `test_benchmark_terminal_policy`, `test_program_visual_feedback` |
| `landlock` | Linux on x86_64/aarch64 and a Landlock ABI >= 1 from the version query (nothing is restricted) | `test_isolation` |
| `seccomp` | A child process loads the production socket-denial filter and `socket()` is denied (20 s timeout) | Component of `program_isolation` and `native_replay_sandbox` |
| `native_replay_sandbox` | Landlock ABI >= 3, working seccomp filters, and `find_library("seccomp")` | Capturing cases in `test_native_terminal_replay` |
| `linux_runtime_tools` | `ldd` and `ldconfig` on the production search paths, and `ldconfig` resolves `libc.so.6` | `test_system_runtime` |
| `linux_proc` | `/proc/<pid>/stat` start ticks and `/proc/self/cmdline` are readable | PID-identity cases in `test_benchmark_queue` and `test_experiment_hook` |

Running item 28 on a benchmark host (strict mode, pinned runtimes, no model and no network beyond loopback):

```sh
AURIGA_REQUIRE_SANDBOX_TESTS=1 AURIGA_OPENCODE_EXECUTABLE=<opencode-linux-x64>/bin/opencode PYTHONPATH=tests \
  mamba run -n Auriga python -m unittest \
  test_agent_claude_code_conformance.ClaudeCodeConformance.test_sandbox_namespace \
  test_agent_opencode_conformance.OpenCodeAnthropicConformance.test_sandbox_namespace \
  test_agent_opencode_conformance.OpenCodeResponsesConformance.test_sandbox_namespace \
  test_codex_sandbox_conformance
```

Keep `TMPDIR` short (the default `/tmp`): Unix socket paths below temporary directories must fit in 107 bytes. The checkout, the user's home and the fake RoboDojo root must not be mount sources: run from a checkout outside the temporary directory, as usual.

On macOS every probe reports the capability as missing. A Linux host with bubblewrap, unprivileged user namespaces (or passwordless `sudo`), Landlock ABI >= 4, and libseccomp is expected to pass every probe. Run `PYTHONPATH=tests mamba run -n Auriga python -m support.host_capabilities` to print each probe's result for the current host. A skipped test appears in unittest output as `skipped 'requires host capability <probe>: <reason> [probe: <why it is missing>]'`, and skips are counted separately from passes and failures.

On hosts that are supposed to support the sandbox, such as Linux benchmark hosts and CI, set `AURIGA_REQUIRE_SANDBOX_TESTS=1`. With this variable set, a missing capability fails the test with `MissingHostCapability` instead of skipping it: a gated method fails, and a gated class errors in `setUpClass`. This prevents a capable host from silently skipping the sandbox suite, whether the host lost a capability or the production code a probe reuses has regressed. Gate only tests that cannot run without the capability. Do not gate failures caused by temporary-path canonicalization (`/var` versus `/private/var` on macOS), Unix-socket path length, timing, or missing optional tools; fix those or record them separately.

## Browser and Native Evidence

The synthetic-only mode of `web/check-task-review.mjs` injects synthetic data in a headless browser without launching a model or robot. Build the web assets first, then run with an installed Playwright setup (setting `PLAYWRIGHT_MODULE` to your local module path if needed):

```sh
AURIGA_BROWSER=firefox \
AURIGA_URL=http://127.0.0.1:18787 \
AURIGA_PREVIEW_ASSETS="$PWD/web/dist" \
AURIGA_SYNTHETIC_ONLY=1 \
AURIGA_SCREENSHOT_DIR=/tmp/auriga-browser-check \
node web/check-task-review.mjs
```

This check validates synthetic Web UI states, interactive controls, and read-only boundaries, but does not prove live task replay synchronization or GPU execution. Changes to physical motion semantics additionally require running on explicitly authorized hardware with frozen source code, configuration, provider, model, task, seed, and layout, recording the complete action requests, execution receipts, measured trajectories, independent native evaluator scores, original/replay media, and audit results. Passing software tests, launching a queue, model self-reports, or producing a video file are never substitutes for verified task success, and prior experiments do not automatically serve as evidence for modified source code.

`audit_run.py --profile native` normally requires a completed Codex turn to satisfy session checks; if `Harness` interrupts an active turn after a native terminal state has already been recorded, the audit accepts it only when the archive simultaneously proves that the terminal state came from the private native evaluator, the turn had started on the same thread, and the owner's interrupt request occurred after the native terminal timestamp. Such turns are counted under `native_terminal_interrupted_turns` without incrementing `native_completed_turns` and do not imply task success; a standalone `interrupted` turn never satisfies the native audit. `--require-continuity` still requires at least two completed turns.

Evaluation reports preserve `null` for missing, invalid, or unknown runs against a fixed case denominator, detailing how many runs were scorable alongside their failure categories. Integrity audits, raw RGB byte verification, reconstructed image availability, and task evaluation scores are reported as separate dimensions. After importing legacy archives, verify both key native summary fields and byte-level immutability of the source directory.

In `benchmark_report.py`, per-run `recovery_guidance` distinguishes the manifest's `configured_version`, the count of `benchmark.actor_turn_continuation` attempts on the primary thread, and the count of matching `sent turn/start` requests in `codex-raw.jsonl`. After restoring short references via the saved alias map, the reporter performs an exact match between the full event prompt and the `text` input items of native requests: each request is matched at most once, and secondary threads, `received` log entries, or partial prompt substrings are excluded. `versions` breaks down attempt and sent counts by the recovery version actually recorded in each event rather than backfilling from the configured version, and separates standard continuations from continuations carrying recovery context. Reports never copy raw prompt bodies or hashes.

`native_sent_recovery_count` only proves that the log recorded a request submitting that continuation prompt to the native runtime—it does not prove that the upstream accepted the request, that the model followed the guidance, or that task performance improved. `unmatched_continuation_attempt_count` indicates recorded continuation attempts for which no corresponding native send entry was found. For older archives lacking raw logs or primary thread identity, sent counts remain `null`; when both the primary thread identity and complete readable logs exist with zero matching events, the count is `0`. Malformed JSON-RPC log entries remain isolated in the report denominator under existing corrupted-archive rules rather than being coerced to zero triggers (see `test_evaluation_reports`).

## Release Checklist

Before cutting a release, save any uncommitted work and branch baselines, confirming the independent changes on both target `main` and the branch being merged. Use a standard merge commit to preserve ancestry rather than `reset` or `force push`, resolving conflicts according to module ownership and test coverage. Existing release tags are immutable annotated tags and must never be moved.

On the final merged tree, bump the package version in `pyproject.toml`, update the release notes, and run the complete software, web, and wheel verification suite. Verify the wheel metadata version, the commits pointed to by the previous and new tags, that both the prior `main` HEAD and merged branch HEAD are ancestors of the new `main`, that features unique to `main` are preserved, and that the working tree is clean. Create the new annotated tag only on the verified commit, push normally, and inspect remote refs to confirm. Synchronizing a Git tag and package version does not publish to PyPI; external deployments and official benchmark runs are separate deliverables.
