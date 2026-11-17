# Experiment Library and Persistent Web Service

The Experiment Library web interface is served at `http://127.0.0.1:8780`. It runs independently of individual experiment processes using Python, `aiohttp`, and the built web assets from the `Auriga` mamba environment, without loading models or Isaac Sim and without allocating GPUs. The implementation lives in `auriga/apps/library/experiments.py`, `auriga/apps/library/service.py`, and `auriga/apps/evaluation/recording.py`.

## Storage Layout and Archiving Rules

The default experiment root is `~/exp/auriga`. New experiments do not require manually specifying an output directory; entry points allocate a unique run directory under the layout below:

```text
~/exp/auriga/
  runs/YYYY/MM/DD/category/task/run-id/
    experiment.json       # Category, task, title, tags, creation time, and identity
    runtime.json          # Atomically updated heartbeat, stage, terminal state, and live endpoint
    summary.json          # Final outcome; task score and interface validation recorded separately
    manifest.json         # Resolved configuration and software versions (when available)
    events.sqlite         # Present for runs that support protocol-level replay
    blobs/                # SHA-256 content-addressed RGB images, traces, media, and evaluations
    terminal.bin          # Present only when a native Codex terminal was recorded
    ...                   # Native reports, media, and producer-specific artifacts
  service/
    installation.json     # Installation receipt written by install_library_service.py --install
    experiment-groups.json  # Optional experiment group registry (format described below)
```

Directory dates use the allocation date in UTC+8 (`Asia/Shanghai`); JSON timestamps are stored in UTC, while web UI date filters and displays use UTC+8. When importing an existing archive, the directory date reflects the import date, `date_source` records the provenance of the legacy `created_at` field, and `imported_at` records the import timestamp. Because some legacy archives populated `created_at` from report file modification times (`mtime`), `created_at` must not be treated as the actual execution start time; actual start and finish times use the dedicated time projection described below. The original record's identity is stored in `evidence_run_id`, while web routes use a deterministic directory-derived `exp-...` ID so identically named runs from different sources never collide.

The library also discovers registered `experiment.json` files placed in custom directory layouts under the root, such as `benchmark/<group>/<batch>/runs/<case>/`. Directory scanning skips symbolic links and private credential directories, and stops recursing inside a directory once a registered run is found so archived evidence copies are never listed as duplicate experiments. Plain report directories without an `experiment.json` file are not indexed automatically.

Recommended categories are `main` (primary experiments), `reference` (API / golden references), `acceptance` (joint acceptance tests), `validation` (robot validation matrices), `geometry` (real recorded geometry probes), and `synthetic` (synthetic fixtures). Examples of task names include `stack_bowls` and `local-scene`. Titles and tags accept arbitrary descriptive text, while path components (`category`, `task`, and `run-id`) use ASCII letters, digits, dots, underscores, and hyphens. Explicit `--run-dir` / `--output` paths remain supported; legacy records stored outside the experiment root can be brought into the library via the `import` command.

## Web Interface and Live Updates

The home page opens in **Experiment Comparison** view by default. The left sidebar groups runs into collapsible research directions, distinguishing method iterations from baselines within each direction. Selecting a method version or clicking "Compare Entire Direction" selects all versions in that direction alongside their respective baselines (with no limit on the number of columns). Group names summarize what changed (for example, "Re-observe After Gripper Action - v1"), while preparation runs, fault recoveries, and backfills for the same condition are folded into the execution details. The overview header displays the group mean score, successes over protocol denominator, scored coverage, paired scene score deltas, and iteration lineage. From the sidebar you can search methods, branches, or groups, filter by protocol, and select custom cross-direction comparisons; switching the active comparison reference in the UI never mutates the recorded baseline relationships on disk.

The video matrix pairs runs by their full scene identity (`task`, `variant`, `eval seed`, and `layout`), rendering one scene per row and one experiment condition per column across all scenes. Each cell shows the evaluation score, start/end timestamps, and camera video. Row and column headers remain sticky during horizontal and vertical scrolling, and off-screen media elements load lazily on demand. Each row provides a "Play Row" control that synchronizes playback across that row only—supporting unified camera switching (`cam_head`, `cam_left_wrist`, `cam_right_wrist`), pause, playback speed, and scrubbing by video seconds. Playback aligns videos from the start of each recording without guaranteeing synchronization by action stage or physical simulation tick. Unscored runs, missing camera views, missing videos, and unmerged multiple attempts are badged explicitly, and recovery history lists each attempt's timestamps with a direct link to its full run record. Selection state is persisted in the URL query string (`scope=direction` keeps the full direction selected, `scope=custom` keeps custom columns, and legacy compare links preserve custom selection semantics). Refreshing metadata does not recreate active video players whose source URL has not changed.

### Run Timestamps and Provenance

The catalog, run detail endpoints, and individual attempt records all return `started_at`, `finished_at`, `registered_at`, and `imported_at` along with their corresponding `*_source` fields. Timestamps are normalized to UTC ISO-8601 strings and rendered in `UTC+8` (`YYYY-MM-DD HH:mm:ss`) in the web UI regardless of the browser's local timezone. `started_at` prefers the first `run.started` event matching the archive's run identity, falling back to explicit start fields in the raw `summary.json`, `runtime.json`, or report files; `finished_at` only accepts explicit completion timestamps from those raw files. Stop events, final log lines, heartbeats, registration times, manifest preparation times, and import times are never substituted for actual start or finish timestamps. Imported records read execution timestamps strictly from `evidence/`, while the outer `runtime.json` describes only the import lifecycle.

Missing timestamps, timezone-naive strings, numeric timestamps with ambiguous units, and invalid finish times earlier than `started_at` remain `null` and populate `time_warnings`. Group and execution-batch time spans are computed from the earliest known start and latest known finish across all unique attempts, accompanied by `time_coverage={run_count, started_count, finished_count}` so partial timestamp coverage is never mistaken for total wall-clock duration. `prepared_at` in `manifest.json` is retained separately, and `created_at` is no longer used as the start time in the UI. Sorting by "Run Started" and date-range filtering in the run list operate on `started_at`, placing unknown timestamps last and excluding runs with unknown start times when a date filter is active.

Timestamp normalization operates purely on read-only projections without rewriting original archives. Archived SQLite databases are opened via `open_read_store`, and the timestamp cache is keyed by file identity and WAL state so repeated scans of unchanged runs do not reopen SQLite files. Initial cold-cache warmup time and cached scan duration are tracked separately, and page requests never trigger ad-hoc filesystem scans.

The **All Runs** tab provides filtering by category, status, task, tag, title / run ID, and date range, along with sorting and pagination. Runs that do not belong to a benchmark comparison group remain accessible here. Selecting a run displays its task evaluation score, reports, media, and raw metadata. Runs backed by a protocol `events.sqlite` database can be opened in the full workbench view to inspect observations, actions, trajectories, events, scene versions, and the native Codex terminal; runs containing only static reports or videos display their available evidence directly without synthesizing a terminal.

### Experiment Groups and Comparison Policy

The read-only `GET /v1/experiment-groups` endpoint is powered by `experiment_comparisons.py`, which scans benchmark manifests under the experiment root and combines explicit case tags, `experiment_panels` / `case_roles`, protocol identities, and archived run identities to construct comparison groups. Group snapshots refresh in the background alongside the library catalog without scanning or modifying archives during HTTP requests. The API projects only evaluation scores, source provenance, and permitted experiment condition metadata, omitting full configurations, credentials, and internal error payloads. Group conditions include `agent_backend` (`auriga_config.agent.backend`, or the per-case `agent_backend` override; `codex` for archives without an `agent` section), so harness comparisons are visibly separated from model or prompt comparisons. Groups also carry `evidence_client` (`codex_agent` or `relay`; derived from the final backend and `agent.evidence.client`, `codex_agent` for pre-backend Codex archives) and `native_effort` (when the runs recorded it). When the member runs' own manifests record `agent_backend` (`kind`, `evidence_client`, `native_effort`), the recorded value is the true provenance and replaces the configured reading; a configured/recorded conflict adds a `recorded_<field>_differs_from_configured` group warning and attempts that disagree make the condition `null` with `recorded_<field>_differs_between_attempts`. A harness comparison is therefore a `baseline` and a `candidate` group whose `conditions` differ only in `agent_backend` (and the dependent `evidence_client`/`native_effort`); a backend difference between execution groups merged under one group is a conflict, never silently merged. Per-run reports add `counting_basis` (Codex's and the other backends' counting strings are one basis), `served_models` and relay-measured `model_usage_relay` next to the native usage; see the field table in [Configuration](configuration.md#mixed-backend-experiment-fields-report-and-library).

- Group mean scores multiply the normalized official evaluation score by `100` and are always displayed alongside scored coverage; unscored cases are never zero-filled. Success counts use the protocol's scheduled case count as the denominator (or display `?` when the protocol denominator is unknown).
- Pairwise score deltas between groups or across consecutive iterations are computed strictly over the intersection of scenes where both groups have official evaluation scores. If the two groups differ in scene coverage, the UI surfaces an explicit warning; numerical deltas alone do not prove causal attribution to a single code change.
- The registry file can merge preparation runs and fault-recovery batches under the same condition back into the primary experiment group via `execution_groups`, preserving the primary group's fixed protocol and denominator. Evaluating the primary group first followed by listed `execution_groups` in declaration order, the merger selects the **first official evaluation score** (including `0`) for each scene—subsequent higher scores never overwrite an earlier valid score. Only unscored or missing scenes are backfilled from recovery batches, while every attempt remains inspectable below the video player. If a scene has no official score across any attempt, the cell displays the status and video of the latest unscored attempt while keeping the score `null`. Multiple attempts that have not been explicitly merged remain flagged as pending confirmation.
- An optional registry file may be placed at `<experiment-root>/service/experiment-groups.json`; when absent, only automatic manifest scanning is used. The packaged `auriga/apps/library/experiment_group_relations.json` shares the same schema (empty in release builds) and applies only when the referenced manifests exist on disk; entries in `<experiment-root>/service/experiment-groups.json` then override matching entries by exact root-relative `manifest` path and `condition`. Because group IDs are derived solely from `(manifest, condition)`, editing human-readable display names never breaks group URLs. The registry only describes experiment metadata, provenance, and lineage relationships—scores are always read from the original run archives rather than duplicated in the registry file.

## Registry File Format

The registry file is a JSON object in which the Experiment Library reads only the top-level `groups` array (any other top-level keys are ignored):

```json
{
  "version": 1,
  "groups": [
    {
      "manifest": "benchmark/my-direction/v2/queue-manifest.json",
      "condition": "candidate",
      "name_zh": "Re-observe After Gripper Action - v2",
      "purpose": "Evaluate whether re-observing after gripper actuation reduces empty grasps",
      "series": "gripper-observe",
      "series_name_zh": "Post-Gripper Observation",
      "iteration": 2,
      "role": "candidate",
      "experiment_nature": ["ablation"],
      "parent": {"manifest": "benchmark/my-direction/v1/queue-manifest.json", "condition": "candidate"},
      "baseline": {"manifest": "benchmark/my-direction/v2/queue-manifest.json", "condition": "baseline"},
      "recovery_of": null,
      "relation_status": {"parent": "confirmed", "baseline": "confirmed"},
      "baseline_reason": "Paired baseline in the same batch with identical task, seed, layout, and budgets",
      "source": {"branch": "feature/gripper-observe", "revision": "<commit>"},
      "evidence": ["benchmark/my-direction/v2/README.md"]
    }
  ]
}
```

Every entry requires `manifest` (a `.json` manifest path inside the experiment root recorded relative to the root; paths inside `private` or excluded directories are rejected) and `condition` (the experiment condition of cases in that manifest, typically matching each case's `experiment_condition`). Entries with invalid schemas or out-of-bounds paths are skipped and reported in the `errors` array of the groups API response while remaining entries load normally. Optional descriptive fields are projected directly onto the group: `name_zh` (display name key retained for schema compatibility), `label`, `purpose` (hypothesis being tested), `series` and `series_name_zh` (method series identifier and display title), `iteration` (integer method iteration, or `null` if unknown), `role` (recommended values: `baseline`, `candidate`, `recovery`, `benchmark`, `validation`, or `preparation`), `experiment_nature` (array of strings), `note`, `baseline_reason`, `source` (from which `branch` and `revision` are read), and `evidence` (array of strings displayed as lineage evidence). Relationship fields behave as follows:

| Field | Meaning | Constraints |
| --- | --- | --- |
| `parent` | Prior experiment condition from which this method iterated | `{manifest, condition}` or `null`; an explicit `null` clears relationships inferred by automatic scanning, and a missing target group surfaces a `parent_group_unavailable` warning. Must not be guessed from execution attempt numbers. |
| `baseline` | Reference group against which this group should be compared | Uses the same exact `{manifest, condition}` reference and may differ from `parent`. Historical baselines must document any source, model, or environment confounders in `baseline_reason`. |
| `recovery_of` | Original batch for which this entry is a recovery/backfill run | Preserves individual results for both the original batch and the recovery batch without incrementing the method `iteration`. |
| `execution_groups` | Preparation, recovery, and repeat batches folded into this group | Ordered array of `{manifest, condition}` references excluding the group itself; original entries remain addressable and legacy links redirect to the canonical group. |
| `relation_status` | Verification status of `parent` and `baseline` links | Object where `parent` and `baseline` each take `confirmed`, `unresolved`, or `not_applicable`; other values are ignored. |
| `baseline_reason` | Rationale for selecting this baseline | `confirmed` only indicates that the link is backed by archive evidence; it does not automatically prove a single-variable controlled ablation. |

The registry can explicitly reference JSON manifests inside the experiment root that were not discovered by automatic directory scanning. Several additional fields exist strictly for evidence-backed historical reconciliation:
- `condition_aliases` (array of strings): merges the listed condition names into this entry's `condition` for parallel execution shards of the same method.
- `protocol_cases`: explicitly binds a protocol when the original manifest lacked one or used inconsistent condition names; case identities must not duplicate or conflict with the manifest.
- `manifest_copies`: declares backup manifest paths so byte-identical manifests at valid paths are not listed as duplicate groups.
- `execution_run_dirs`: restricts which run directories were actually launched in this batch, preventing later runs that reused unlaunched directory paths from backfilling scores into an older batch while preserving the original protocol and `missing` rows without shrinking the denominator.

Registry entries never move video files, rename run directories, or fabricate historical Git revisions from the current worktree `HEAD`.

The groups API returns `protocol_denominator_known`, `scope` (`protocol`, `recovery_subset`, or `recorded_cases`), `protocol_case_count`, and per-scene `selection`, `selection_reason`, `run`, and `attempts`. For merged groups, `selection` is one of `first_scored`, `latest_unscored`, or `missing`; each attempt includes its `execution_group_id`, group-level `execution_groups` summarizes the provenance of the primary group and folded batches, and `id_aliases` records folded legacy group IDs. Requesting details for a folded alias ID returns the canonical group and updates the selection in the browser URL. If any referenced group is missing, assigned to multiple targets, involved in a cycle, or conflicts in protocol or known condition metadata, the merge is aborted for that cluster and reported in `errors`. Runs with conflicting, missing, or duplicate scene identities are excluded from automatic pairing and remain accessible under **All Runs**.

The web UI loads group details and media manifests on demand:
- `GET /v1/experiment-groups?view=summary` returns all group metadata while trimming each case to `case_key` and the minimal `run.summary` fields required to determine official scores, task success, and unscored status (`null` when no run is selected). The summary view preserves exact paired scene statistics and fixed denominators while omitting per-scene identities, attempt lists, and provenance details, setting `details_loaded=false`.
- Selecting a group fetches `GET /v1/experiment-groups/{id}`, which returns the full group payload with `details_loaded=true`. Both endpoints compute the group-level `revision` from the full group payload. The unfiltered `GET /v1/experiment-groups` (or `?view=full`) continues to return full group payloads, unknown `view` values return `400`, and unknown group IDs return `404`.
- `GET /v1/experiments/{id}/media` returns only `id`, `view_url`, `replay_available`, the video media list, and sanitized `camera_id` / filename / `url` descriptors from `reports[].content.videos` for camera switching, without building a full task review or exposing full reports or server filesystem paths. Complete experiment details continue to use `GET /v1/experiments/{id}`. Videos remain stored on local disk and are streamed on demand via the existing files URL with `HEAD` and `Range` support.

Run lists, group summaries/details, media manifests, and full experiment detail responses set `Cache-Control: private, no-cache` with weak ETags, returning an empty `304 Not Modified` when `If-None-Match` matches. Clock updates to `observed_at` alone do not invalidate cached ETags, whereas request paths, query parameters, and underlying payload changes do; authentication and path checks always execute before conditional `304` responses. JSON responses negotiate `gzip` / `deflate` via `Accept-Encoding` and include `Vary: Accept-Encoding` (video streams and SSE endpoints bypass JSON compression). See `tests/test_library_loading.py` for HTTP, scoring equivalence, camera alias, permission, and `Range` regression tests.

In the background, the server scans metadata, SQLite event databases/WALs, terminal recordings, and active gateway endpoints once per second, publishing an atomic catalog snapshot and broadcasting change notifications over SSE. List, detail, and run requests read the latest published snapshot without triggering full-directory rescans, so newly registered runs or terminal state transitions appear within at most one scan interval plus scan duration. While SSE is connected, the browser avoids polling group summaries; if the connection drops or a refresh fails, it retries every 5 seconds and refreshes immediately when the tab regains visibility. The comparison view polls lightweight media manifests every 5 seconds only for currently visible or actively playing cells to pick up newly exported videos while streaming video bytes strictly on demand. Unapplied filter drafts, input focus, selected detail panels, expanded reports, and active video players are preserved across background refreshes.

### Live Modes and Historical Replay

- **Controllable Live Run**: Active when the runtime heartbeat is fresh and the local gateway's `/v1/run` identity matches. The gateway enforces single-writer leases, unique `request_id` checks, native question/approval handling, and action admission. Every proxied request carries the target run ID so the gateway rejects mismatched requests if a port is later reused by another run; the proxy never auto-retries write operations.
- **Read-Only Live Stream**: Producers such as the API reference runner, robot validation matrix, and geometry probes write `recorded_stream` heartbeats. Robot producers passively archive received RGB frames, action requests, progress events, verified receipts, and evaluations—issuing each action once without adding observations, actions, or retries. Live geometry stages and results can be inspected in the library detail view.
- **Historical Replay**: Once an experiment finishes or stops, the same URL automatically transitions to read-only replay mode. If a heartbeat expires or the backend identity cannot be verified, the UI reports a disconnected state while keeping saved evidence readable—never inferring task completion from a missing heartbeat alone. Terminal streams and event logs maintain independent per-run cursors without leaking state across experiments.

Historical replay reads directly from SQLite and archived files without instantiating `Workbench` or launching a robot worker. When a native PTY recording (`terminal.bin`) is present, the terminal view streams the original PTY bytes with `replay.kind=original_pty`. If `terminal.bin` is missing or fails to load but a complete session archive exists, a standalone adapter copies the archive into a private cache, probes SQLite table/column compatibility, and renders the saved history through an isolated, network-disabled native Codex TUI process. This fallback is badged with `recovered_native_tui` and `original_recording=false`, recording the CLI version, schema probe results, source hashes, and an immutability check confirming that the historical thread was not modified; it is strictly a derived view rather than original PTY bytes or live task resumption. Unrecognized schemas or incomplete renders are never published as valid cache entries.

In replay mode, write endpoints are rejected and the terminal is read-only. Closed databases without a WAL file are opened in `immutable` read-only mode, while active WAL databases use standard read-only mode to follow newly appended events. Media endpoints support `HEAD`, `Range`, and `X-Content-SHA256` headers, verifying file SHA-256 digests in streaming chunks before serving the file response so large videos are never buffered entirely in memory. The digest cache holds up to 512 entries and invalidates automatically whenever device, inode, size, `mtime`, or `ctime` changes, while path and scope checks still run on every request. Browser playback depends on the archived video codec; legacy `mp4v` files that a browser cannot decode inline remain downloadable as raw files.

The workbench view prefers the incremental `/v1/records` projection: immutable observations are fetched by record cursor, while mutable items and requests are fetched by persistent revision cursor without generating duplicate revisions when content is unchanged. Older SQLite archives remain viewable read-only without migrating their original evidence on disk. The incremental view retains up to `2,000` items/observations/requests and `5,000` stage events in the active window; when the window boundary is reached, the UI exposes an "Earlier Records" control that paginates historical native items from the server and clearly marks out-of-window context. When connected to an older server that lacks pagination, the UI falls back to the full legacy item/observation/request payload. Historical item pages display up to `5,000` associated events; if this cap or the current stage event cap is reached, the UI displays an explicit notice and links to the full `/v1/events?after=0` stream (where `after` can be set to the last read `server_seq` to paginate forward). Loading earlier item pages does not backfill truncated events for an individual item that exceeded the per-item event cap. Underlying SQLite databases and media files remain complete; browsing historical pages pauses live stage updates, and returning to the latest view simply reloads the page without starting a new run.

### Scoring and Termination Reconciliation

In the UI, `task_success=false` with `score=0` is displayed as failed with score `0`, whereas runs lacking a valid evaluation display `Unknown`. Passing interface checks, completing physical motion, and succeeding at the task are tracked as three distinct outcomes, and completing a geometry probe does not represent robot task success.

Task outcome and score availability are recorded independently. Confirmed startup-no-action timeouts, action-progress timeouts, model request budget exhaustion, run wall-clock budget exhaustion, and repeated actor inability stops are all classified as **task failures** while preserving any valid native partial/process score. For example, a run that timed out due to lack of action progress with a native process score of `40` displays a normalized task score of `0.4`, task success `No`, and failure reason `action_progress_timeout`. Normalized failure scores divide the native process score by `100` (or `0` for binary tasks without partial process scores), and a locked valid native terminal outcome always takes precedence over a simultaneous harness budget limit. Infrastructure interruptions, unconfirmed actions/stops, or invalid episodes remain classified as `unknown` without borrowing partial snapshots to fabricate success or scores; if a confirmed task failure on a valid episode lacks a verifiable score, it is reported as failed with its failure reason and an unavailable score.

`failure_reason` stores a stable machine code, human-readable description, and authority source, while `termination_reason`, `termination_authority`, `native_episode`, and `native_evaluation_snapshot` preserve the stop state and native evaluator evidence. Historical budget-terminated runs that were excluded under older scoring policies are reconciled into valid failed scores only when cleanup is confirmed, run identity, action counts, evaluation artifact bytes/SHA-256, and the final evaluation event all match, and no subsequent actions or resets occurred; `scoring_reconciliation` records the legacy classification and provenance references without mutating the archived files on disk. The Experiment Library, run detail views, offline score/cost reports, and paired comparisons all share `auriga.apps.library.scoring`, while the evaluation queue writes reconciled summaries alongside the untouched raw summaries required for OOM recovery verification. The success-rate denominator remains the scheduled protocol case count—a partial process score of `0.4` never counts as a `40%` task success rate.

In new benchmark runs, `actor_stall_policy` stops the episode early if the model's `final_answer` explicitly declares an inability to continue across `5` consecutive completed turns without any native action progress in between. `Harness` inspects only completed `final_answer` items on the primary thread and episode using deterministic English and Chinese inability patterns without making extra model calls; the counter resets whenever native action progress occurs, the episode or thread changes, or a turn finishes without declaring inability, counting at most once per turn. On the 5th consecutive turn, `Harness` records `benchmark.actor_inability` evidence, stops the run with `actor_repeated_inability`, and executes the standard stop, native `final_check` evaluation, video export, and cleanup pipeline. The run is scored as a task failure carrying its current valid native process score while recording `termination_authority=harness` and the actual `native_episode.ended` state; unconfirmed stops or invalid evidence are never force-scored.

## Starting and Maintaining the Service

```sh
# Build the web frontend first
npm --prefix web ci
npm --prefix web run build

# Run in the foreground (CPU-only)
mamba run -n Auriga python -m auriga.apps.library

# Print the systemd user unit for inspection; pass --install to install and enable it
mamba run -n Auriga python scripts/install_library_service.py
mamba run -n Auriga python scripts/install_library_service.py --install

# Set user bus environment variables if your shell session does not populate them
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
systemctl --user status auriga-library.service
journalctl --user -u auriga-library.service -n 50
systemctl --user restart auriga-library.service
loginctl show-user "$USER" -p Linger
```

`install_library_service.py --install` writes `~/.config/systemd/user/auriga-library.service` (managing only this unit) with `Restart=on-failure` and `UMask=0077`, backing up any pre-existing unit file of the same name under `service/` before overwriting it. `--root`, `--port`, `--scan-interval`, `--public-origin`, `--task-source-root`, and settings from the unified configuration's `library` section are baked into the unit's `ExecStart` command; when `--task-source-root` is omitted, `backend.root` from the unified configuration is used. Enable systemd user lingering (`loginctl enable-linger "$USER"`) if the service should remain running after you log out. Because the service binds strictly to loopback and validates `Host` and `Origin` headers, you can access it from another workstation over SSH port forwarding (`ssh -L 8780:127.0.0.1:8780 <user>@<host>`) and open `http://127.0.0.1:8780` locally.

Exposing the library over a network requires deploying an authenticating reverse proxy in front of the loopback port. The proxy must rewrite the `Host` header to the loopback address (e.g., `127.0.0.1:8780`), as the service rejects non-loopback `Host` headers, and you must pass `--public-origin https://library.example.com` (or set `library.public_origin`) to permit browser requests from that exact HTTPS origin. `public_origin` rejects trailing slashes, paths, wildcards, and embedded credentials, and does not perform user authentication itself—authentication and access control are entirely the responsibility of the reverse proxy. Mount the application at the domain root so same-origin HTTPS, SSE, and WSS paths resolve cleanly; note that authenticated users will have live control access to active workbenches.

New runs register in the library automatically and bind an ephemeral free loopback port by default (always select an actually idle GPU index):

```sh
mamba run -n Auriga python -u -m auriga.apps.gateway \
  --gpu 4 --category main --title 'stack_bowls geometry assisted' \
  --tag geometry-assisted --max-actions 10 --wall-budget-s 900

mamba run -n Auriga python scripts/reference_episode.py \
  --gpu 4 --category reference --title 'Simple API reference' --tag baseline

mamba run -n Auriga python scripts/robot_validation.py \
  --gpu 4 --category validation --title 'Robot interface matrix'

mamba run -n Auriga python scripts/probes/geometry_validate.py \
  --raw-recording <robot-validation-run>/calibrated-observations.jsonl \
  --train-indices 6,7 --heldout-index 8 --camera cam_left_wrist \
  --roi 100,150,450,100 --title 'Real RGB held-out geometry'
```

Output directories, heartbeats, and final summaries from all of these entry points land under the same experiment root. The standalone scene service continues to use its own `--root` directory and API for development without depending on the Experiment Library, and scene service scratch directories are not indexed as experiment runs.

## Downloading Raw Trajectory Bundles

When viewing a completed run in the Experiment Library, the detail panel provides a **Download Raw Trajectory** link. The browser downloads a `.zip` bundle via `GET /v1/experiments/{catalog_id}/trajectory-bundle` with `HEAD`, `Range`, and `X-Content-SHA256` support. The service resolves the original evidence strictly by catalog ID and builds a verified trajectory bundle in a separate cache directory without invoking a model or GPU. Active runs, runs missing trajectory evidence, and imported legacy copies display why export is unavailable, and runs with incomplete trajectory files are rejected during export validation.

You can pass the downloaded `.zip` directly to the offline replay tool on any machine with matching simulation versions and task assets to render high-resolution videos. For bundle contents, CLI usage, and resimulation constraints, see [High-Definition Offline Action Replay](hd-replay.md). The bundle carries only the actuator commands, measured robot states, and environment identifiers needed for resimulation—omitting model transcripts, server configurations, and credentials.

## Importing Legacy Runs

```sh
mamba run -n Auriga python -m auriga.apps.library.experiments import \
  runs/<old-run> --category acceptance --task stack_bowls \
  --title 'Imported legacy run' --tag legacy
mamba run -n Auriga python -m auriga.apps.library.experiments list --category acceptance
```

The importer rejects source directories that still have an active heartbeat or are locked by a running gateway. It copies original files into `evidence/`, verifies SHA-256 digests between every source file and its copy, and writes `import-inventory.json` while excluding credentials, `private/codex-home`, symbolic links, and special files. Source files are never deleted or modified. If a SQLite WAL file needs to be checkpointed for read-only replay, `replay/events.sqlite` is generated in a temporary copy while preserving the untouched source bytes under `evidence/`. Failed runs are imported with their failure status intact and are never rewritten as successes.

Run evidence and large media files belong exclusively under the external experiment root and must never be committed to Git.

The experiment detail view extracts task scoring logic from the RoboDojo task source code. By default, the library service reads task definitions from `backend.root` in the unified configuration; if your task sources live elsewhere, pass `--task-source-root` (or configure `library.task_source_root`). If the task source directory is absent, the scoring rules section in the detail view is simply left empty.
