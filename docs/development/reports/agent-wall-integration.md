# Claude Code and WALL-OSS integration

## Revisions and scope

Integration on 2026-11-17 targets `lose4578/Auriga/main` at `0fd2337`, preserving both PR histories:

- [PR #5](https://github.com/lose4578/Auriga/pull/5): Claude Code agent backend, head `87e8b1b`.
- [PR #6](https://github.com/lose4578/Auriga/pull/6): WALL-OSS whole-subtask delegation, head `0e753c5`.
- Local merge commits: `825b0ad` and `13212cb`; integration fixes: `aebe57b`.

Three textual conflicts were resolved by preserving both sets of Harness imports, both the backend accounting
and policy-delegation report fields, and both configuration sections. Codex remains the default agent;
WALL-OSS remains disabled unless explicitly configured. No package version or release tag is changed.

## Integration fixes

Review of the combined execution path found three defects in the delegation path. Regression tests reproduced
all three before the fixes (two errors and one failure):

1. Replaying a supervised delegation recalculated its fixed budget from the now-smaller remaining episode budget
   and rejected identical caller intent. Replays now retain the original admitted budget and still reject changed intent.
2. Failure while rendering the final observation discarded the typed delegation response after physical execution.
   The handler now preserves the receipt and returns the existing post-handler delivery-error contract; replay does not move again.
3. Supervisor task text and notes bypassed the common actor scrubber. Policy instructions and supervisor text now
   use the shared `ActorPresentation` boundary, retaining original observer records separately.

The MCP catalogue test also verifies the delegation tool is available to the actor with `readOnlyHint=false`;
program access remains prohibited. The historical parallel-classification fixture remains unchanged, with the
new physical tool checked separately. No gate thresholds, policy motion semantics or task-specific instructions were changed.

## Validation

Environment: macOS arm64, `/opt/homebrew/Caskroom/miniforge/base/envs/Auriga/bin/python` (Python 3.12),
`openai-codex==0.160.0`, `openai-codex-cli-bin==0.160.0`, `claude-agent-sdk==0.2.180`.
The dependency environment was not modified. GitHub Actions is disabled on the repository, so no CI pass is claimed.

- Focused delegation, native EE fixture and dispatch-contract checks: **44 passed**.
- Web checks: **37 passed**; production build passed.
- Wheel build and out-of-checkout installation check passed, including the packaged XPolicyLab worker and Claude imports.
- Python compilation, changed documentation links and diff whitespace checks passed.

The full CPU run completed in 322.5 seconds: **260 modules, 3,088 tests, 171 skipped, 94 failures,
49 errors, one timeout and no crashes**. The suite is not green on this macOS host. After normalizing temporary
paths in subtest identities, every failure/error identity also occurs in the original `0fd2337` main baseline
(2,015 tests, 15 skipped, 120 failures, 92 errors and one timeout). The remaining timeout is
`test_benchmark_queue` at 180 seconds, also present in that baseline. `test_benchmark_group` passed all 6 tests
in 161.6 seconds. The baseline comparison establishes no newly observed failure identities, not Linux correctness.

The **55 affected agent/relay/delegation/contract modules** in that full run completed 1,075 tests:
**982 passed and 93 skipped**, with no failures or errors. This includes native Claude CLI conformance and
loopback smoke checks; 8 of the skips are the unavailable XPolicyLab client dependencies. The other skips are
optional OpenCode runtime, Linux capabilities and the opt-in slow test. Results cover the production tree at
`aebe57b`; only this report and its index entry were added afterward.

Commands (all from the integration checkout):

```sh
PYTHONPATH=tests mamba run -n Auriga python -m unittest \
  test_policy_delegation test_ee_chunk_action test_tool_parallel_contract
mamba run -n Auriga python /tmp/auriga-backport-test-runner.py \
  --jobs 4 --timeout 180 --report /tmp/auriga-agent-wall-cpu.json
npm --prefix web ci
npm --prefix web run check
npm --prefix web run build
mamba run -n Auriga python scripts/build_package.py --output dist
mamba run -n Auriga python scripts/check_package.py dist/auriga_harness-0.2.0-py3-none-any.whl
mamba run -n Auriga python -m compileall -q auriga tests scripts
git diff --check
```

The temporary per-module test runner and baseline method are described in the
[Claude backport validation report](auriga-claude-backport.md). Logs are local under `/tmp/auriga-agent-wall-*`
and `/tmp/auriga-merge-regressions-*`; no raw test archives or credentials are shipped.

The XPolicyLab protocol tests require `msgpack`, `msgpack_numpy` and `websockets` in the client environment and
are skipped on this host. Linux isolation and real model/policy/simulator execution were not tested in this
integration. The source PR's benchmark results are not evidence for the combined Claude Code + WALL-OSS configuration.
