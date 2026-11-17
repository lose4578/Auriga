# Claude Code backend backport to Auriga

## Source and scope

Target: `lose4578/Auriga`, `main` at `0fd233705295c44136c392b8e7251393277ba580`.
Source: the agent-runtime history reachable from Capella's `feat/so101-real-robot`
(`221ad808e344848a26465184a835291d82f8aabf`), whose initial backend integration is `c2657d4`.
The initial diff is `c4ff16a..c2657d4`; SO-101 console files and their documentation are excluded.
Additional Claude fixes: `91c5dde` (official settings file), `00eb73d` (provider environment),
`79ffb5f` (disable the CLI's host Git context).

The shared AgentSession, MCP transport, relay wire adapters, context reset handling, evidence client,
usage/archive readers and web labels are carried together with the source's optional OpenCode 1.19.5 adapter.
Codex remains the default. Default config snapshots omit the new agent section; old archives resolve to Codex.
The old gateway's existing HTTP control route is adapted for definite rejection versus unknown admission;
no SO-101 console/settings endpoint, provider-profile handoff or external-agent plugin is added.
The original robot/motion/perception implementation and Auriga's virtual spatial field remain unchanged.

This is a feature patch, not a new robot-performance result or package release.

## Local environment

Validation uses `mamba run -n Auriga python` on macOS arm64, Python 3.12.
Installed SDKs: `openai-codex==0.160.0`, `openai-codex-cli-bin==0.160.0`,
`claude-agent-sdk==0.2.180`; the SDK-bundled Claude CLI reports `2.1.321`.
Codex executable: `/opt/homebrew/Caskroom/miniforge/base/envs/Auriga/lib/python3.12/site-packages/codex_cli_bin/bin/codex`.
No dependency environment was replaced. All model traffic in these checks goes to loopback test servers.
Linux bubblewrap isolation, the optional OpenCode binary and paid providers are not available in this validation.

## Validation

Checked on 2026-11-17 in an isolated worktree based on the target revision above. The tested changes are the
feature commit containing this report; the source workspace stayed clean at `ffd0fb8314cc20ffc0a54ec1f192733c8c3417e0`.

| Check | Result |
| --- | --- |
| Final affected backend/relay suite, 51 modules | 1,025 tests: 940 passed, 85 skipped; no failures, errors, timeouts or crashes |
| Claude native CLI conformance (included above) | 30 tests: 28 passed, 2 skipped |
| Claude native CLI smoke (included above) | 2 passed: tools, interruption, resume/crash and close against loopback model stubs |
| Web unit checks | 37 passed |
| Web production build | Passed |
| Synthetic browser check | Passed at widths 320, 736, 1024 and 1440; no browser errors |
| Claude browser presentation | Backend label and transcript visible; unsupported native TUI entry hidden |
| Wheel build and out-of-checkout installation check | Passed, including Claude session/wire imports and backend configuration |
| Isolated benchmark manifest generation | 6 passed in 127.945 s; no benchmark launched |
| Python compilation, Markdown relative links, `git diff --check` | Passed |

The 85 targeted skips cover missing OpenCode runtime, Linux namespace capability and an opt-in slow test.
No expected failures were used in that final suite. Native CLI tests exercise the pinned local runtime with
fake model servers; they do not prove provider acceptance or robot task performance.

### Full CPU suite and same-host baseline

The complete CPU suite is **not green on this macOS host**. Both checkouts were tested with the same interpreter,
four module workers and a 180-second per-module timeout:

| Checkout / run | Modules | Completed tests | Skipped | Failures | Errors | Timeouts | Crashes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Auriga `main`, `0fd2337` | 203 | 2,015 | 15 | 120 | 92 | 1 | 0 |
| Initial integrated backport, before final test-scope cleanup | 257 | 3,046 | 163 | 94 | 52 | 2 | 0 |

The full runs took 264.6 s and 358.9 s respectively. Counts exclude unfinished timeout modules. These are the
original full-run results, not totals synthesized from later targeted reruns.

After normalizing temporary directory names in subtest identities, the only new completed-test failures were
three tests for the source branch's console settings endpoint. That endpoint is outside this backport, so its
tests were excluded; the remaining gateway/control tests passed in the final affected suite. Existing failures
include `/var` versus `/private/var` assertions, Linux `pidfd`/`prctl`/`/proc` assumptions and bounded-program
subprocess setup. Some source tests now explicitly skip missing Linux capabilities; the reduced failure count
must not be interpreted as fixes to those platform limitations.

`test_benchmark_queue` exceeded 180 s in both checkouts at the same live-parallelism control test.
The additional backport timeout, `test_benchmark_group`, passed all 6 tests in 127.945 s when rerun alone
without the concurrent module workload (baseline under the parallel runner: 164.9 s). No unresolved new
completed-test failure was observed after the final targeted corrections; the full-suite platform failures
and queue timeout remain unresolved.
Linux CPU/isolation validation remains outstanding; no remote host or running experiment was used for it.

### Reproduction and local evidence

The per-module runner was a temporary copy of Capella's
`221ad808e344848a26465184a835291d82f8aabf:scripts/run_tests_per_module.py`, with its `ROOT` changed to `Path.cwd()`
so the same runner could test both worktrees. It only runs unit-test modules, not benchmark queues.

```sh
# Run in each checkout, with a different report filename for the baseline.
mamba run -n Auriga python /tmp/auriga-backport-test-runner.py \
  --jobs 4 --timeout 180 --report /tmp/auriga-backport-cpu.json

# Final affected suite; glob patterns are interpreted by the runner.
mamba run -n Auriga python /tmp/auriga-backport-test-runner.py \
  'test_agent_*' 'test_claude_settings_*' 'test_wire_*' \
  test_context_signal test_codex_parallel_dispatch test_model_proxy_anthropic \
  test_model_proxy_consume_guard test_model_proxy_continuity test_relay_bindings_evidence \
  test_relay_evidence_client test_relay_review_followups test_native_evidence \
  test_mixed_backend_benchmark test_harness_context_switches test_tool_parallel_contract \
  test_tool_registry_prose_bounds test_support_fake_claude_sdk test_support_fixtures \
  --jobs 4 --timeout 180 --report /tmp/auriga-backport-final-targeted.json

PYTHONPATH=tests mamba run -n Auriga python -m unittest -v test_benchmark_group
npm --prefix web ci
npm --prefix web run check
npm --prefix web run build
mamba run -n Auriga python scripts/build_package.py --output dist
mamba run -n Auriga python scripts/check_package.py dist/auriga_harness-0.2.0-py3-none-any.whl
mamba run -n Auriga python -m compileall -q auriga tests scripts
git diff --check HEAD
```

Local evidence: `/tmp/auriga-baseline-cpu.json`, `/tmp/auriga-backport-cpu.json`,
`/tmp/auriga-backport-final-targeted.json`, `/tmp/auriga-backport-group-final.log`,
`/tmp/auriga-backport-web-check.log`, `/tmp/auriga-backport-browser.log`,
`/tmp/auriga-backport-web-browser/claude-backend.json`, `/tmp/auriga-backport-wheel-check.log`.
The synthetic browser used `web/check-browser.mjs`, a local fixture server at port 18778,
`AURIGA_SYNTHETIC=1`, Playwright Core and installed Google Chrome. The temporary fixture server was stopped.
These machine-local logs are not shipped as product assets or exposed to the experimental actor.
