# Image Transport and Storage Pipeline

By default, new runs follow a two-stage pipeline: raw camera RGB arrays are archived as **JPEG 95** and transmitted to the model as **WebP 85**. Original pixel dimensions, camera ordering, action boundaries, and sampling timestamps are preserved in full; data volume is never reduced by downscaling images or dropping cameras. Historical experiment archives are never rewritten.

## Configuration

By default, the Harness loads `configs/images.json` from the repository:

```json
{
  "transport_format": "webp",
  "webp_quality": 85,
  "webp_method": 4,
  "jpeg_quality": 95,
  "storage_jpeg_quality": 95
}
```

- `transport_format` accepts `"webp"` or `"jpeg"`. For upstream endpoints that only accept JPEG, set this to `"jpeg"` (controlled by `jpeg_quality`).
- `webp_quality`, `jpeg_quality`, and `storage_jpeg_quality` must be integers in `1–100`; `webp_method` must be an integer in `0–6`. Unrecognized fields, boolean values, or out-of-range numbers cause startup to fail immediately rather than silently falling back to defaults on a typo.
- JPEG encoding uses Pillow with `subsampling=2` (4:2:0) and `optimize=True`, relying solely on existing Pillow, OpenCV, and FFmpeg dependencies.
- An alternate configuration file can be specified via `AURIGA_IMAGE_CONFIG=/absolute/path/images.json` (omitted fields inherit the defaults above). Relative paths are resolved against the caller's working directory, so absolute paths are recommended for experiments.
- The Harness persists the active settings to `runtime/image_settings` in the run store and records them in the gateway manifest; resuming a run reuses its archived settings. The robot worker inherits frozen settings from the parent process at startup rather than re-reading mutable files on every frame (passed internally via `AURIGA_IMAGE_SETTINGS`, which normal users do not need to set manually).

If `libwebp` is unavailable or WebP encoding fails at runtime, the affected image automatically falls back to JPEG 95 and records `fallback_reason` in the delivery metadata. **This is a local encoder fallback, not an automatic retry after a remote HTTP endpoint rejects WebP.** If a remote provider only supports JPEG, configure `"transport_format": "jpeg"` before starting the run; physical robot actions are never re-executed to change image formats.

## Native Codex Interface

In the pinned Codex SDK/CLI `0.160.0`, dynamic tools return images using the following item shape:

```ts
{ type: "inputImage", imageUrl: string }
```

`imageUrl` is populated with `data:image/webp;base64,...` (or `data:image/jpeg;base64,...` when configured for JPEG or on fallback). The embodied actor invokes native dynamic tools directly, and Codex's direct adapter forwards image items as standalone `input_image` entries in the upstream Responses payload—requiring no URL extraction or `image(...)` helper calls by the model. Base64 serves strictly as the transport container for typed image items; an `input_text` block containing the same data URI is not treated as visual input. The native shell remains available for numeric computation and file operations, while closed-loop programs that call robot tools run through `program_run`.

Per-run private configuration disables `code_mode` and `code_mode_host` and overrides the target model's private catalog metadata to `tool_mode="direct"`. Disabling the feature flags alone is insufficient because `0.160.0` ships with `tool_mode="code_mode_only"` hardcoded for target models in its built-in catalog, which takes precedence over feature switches. The requested model name and reasoning effort are preserved, and all other model capabilities inherit from the pinned catalog. This transport boundary must not be worked around by swapping models or appending prompt instructions. Native `shell` / `unified_exec` and file tools remain independently enabled; `direct` transport governs only how dynamic tool calls and images are serialized.

Program tools return broker-captured previews, historical views, and supplementary images as native `inputImage` items with their provenance tags and original byte references intact; printing a raw data URL to `stdout` inside a program does not produce a model-visible image. Setting `program_status(detail="full")` controls log and metadata verbosity, not image fidelity.

Model-facing projections strip cryptographic hashes, verification digests, and redundant provenance metadata across both top-level fields and nested JSON strings. Because `source_url` exists only for audit provenance and its content is already delivered inline, it is omitted from model view, whereas links needed to read or interpret tool outputs are preserved according to each tool's contract. Transport data URIs for image/audio items, action and observation IDs, and readable short artifact references are preserved. The local SQLite store retains complete call graphs and raw result references archived prior to presentation filtering, along with full program execution evidence and raw logs. `codex-raw.jsonl` logs the streamlined payloads actually sent to the model, while full unfiltered outputs remain in local `tool_result` records and their associated artifacts.

While user message inputs in Codex also support `localImage.path` and `image.fileId`, dynamic tool outputs do not support those variants, so local file paths cannot be returned directly from tool calls. Furthermore, because the upstream provider's `/v1/files` endpoint returns HTTP 404 and offers no alternative image blob store, images are compressed and transmitted inline rather than uploaded once and referenced by ID—requiring no external image hosting service or permission changes.

When the exact same image is transmitted multiple times, an in-process LRU cache reuses the encoded data URI keyed by the source bytes and full encoding configuration. The cache holds up to 16 images of at most 1 MiB and `640 × 480` pixels each (larger images are encoded on the fly without caching). Returned encoding metadata dictionaries are shallow-copied so that a temporary WebP fallback never blocks subsequent WebP retries if the encoder recovers. This cache eliminates redundant CPU encoding time, **but outbound requests still carry the full base64 image payload, so network byte volume and visual token consumption are unchanged**.

Historical protocol definitions were cross-checked via `codex app-server generate-ts --experimental`, and current behavior is verified against the generated schema, source code, and local loopback HTTP request tests for pinned SDK/CLI `0.160.0`.

## Storage and Historical Retrieval

The robot worker encodes camera frames directly to `.jpg` files and returns `rgb_path` and `image_encoding` (while readers remain backward-compatible with legacy `png_path` fields). When a worker JPEG has a verified hash and matches the configured storage quality, the Harness ingests it directly into the Store without re-encoding. Legacy formats or images from other sources are transcoded to the configured storage JPEG quality, while existing artifacts in historical runs are never modified.

New `RobotBackend` instances always pass frozen image settings to the worker. If an older already-running parent process spawns a new worker without `AURIGA_IMAGE_SETTINGS`, the worker retains its legacy PNG IPC format to avoid changing the image contract mid-run; freshly launched processes use the JPEG/WebP defaults. Diagnostic scripts that launch the worker manually must also pass `AURIGA_IMAGE_SETTINGS` to enable the new format.

Newly captured immutable image files located on the same filesystem as the run directory share storage with Store blobs via hard links, keeping both paths valid while consuming disk space only once. Across filesystems or when hard links are disallowed, two separate valid JPEG files are retained so that storage optimization never risks losing an observation. Published images remain accessible to `history_view_observation`, local visual comparison, Sol verification, pi0.5 RGB ingestion, and the web inspection UI.

Cropped images are stored as JPEG while preserving the uncropped parent observation and recording the ROI, scale factor, and pixel-center transform. Diagnostic geometry overlays are likewise stored as JPEG and encoded according to the transport configuration when sent to the model. Here, "original observation" refers to the full-frame camera capture prior to cropping—**new runs no longer store lossless camera PNGs**.

Video export continues to use FFmpeg/H.264, preserving all three camera streams along with frame indices, PTS timestamps, and simulation-step mappings. Unreleased intermediate frames captured solely for video recording are written as temporary JPEGs: only after all three camera MP4s pass `ffprobe` validation (checking codec, pixel format, and exact decoded frame count) and the frame index is committed to disk are those temporary JPEGs deleted. If video export fails, all input frames are retained; re-running export returns the verified manifest, and any cleanup errors retain the files and log the failure in the manifest.

Published observation images are retained on disk by default. To reclaim disk space after a run has finished and its processes have exited, operators can explicitly run offline video compaction.

## Optional Post-Run Video Compaction

```sh
mamba run -n Auriga python scripts/compact_observation_images.py /absolute/path/to/ended-run --apply
```

Passing `--apply` permanently deletes verified source RGB JPEGs after archiving them into video segments—an explicit, lossy storage compaction step. It requires `runtime.json` to have a completion timestamp, a terminal status (`completed`, `stopped`, or `failed`), no active actions, and no live process at the recorded PID, and it acquires the run's `gateway.lock`. Compaction never runs automatically in the background or while robot actions are executing.

Before compacting a run, ensure that any inspection server uses the archive-aware read APIs and that the web frontend has been built from current source; legacy servers do not recognize video-archive indices.

Because standard task replay videos deduplicate frames by simulation step and do not cover every static observation capture, compaction builds dedicated archive videos directly from published RGB observations: images are grouped by camera and resolution into segments of up to 16 distinct frames encoded with H.264 CRF 18 at 5 fps without downscaling (using `yuv420p` for even dimensions and `yuv444p` for odd dimensions). The 5 fps container rate is purely an archive packaging convention and **does not represent physical task time**, which remains governed by each observation's `sim_step`, clock domain, and `observation_id`. Byte-identical images are encoded only once, allowing multiple observation references to point to the same verified video frame.

Each frame index entry records the source image SHA-256, video artifact reference, intra-segment frame index, width, height, decoder version, and the SHA-256 of the reconstructed PNG. Only after all candidate video segments are encoded, decoded frame-by-frame, and verified for exact frame counts are the corresponding RGB Store blobs and identical raw capture files within the run directory deleted. If a video segment is not smaller than its source JPEGs, the original JPEGs are kept. Blobs shared with non-RGB or observer artifacts are never deleted, and raw model messages, action records, task replay videos, and scene databases remain untouched. Because the index is persisted before blob deletion, re-running the script reuses verified video segments, completes any pending file cleanup, and reports combined statistics across existing and newly compacted segments rather than reporting zero on idempotent re-runs.

Read APIs explicitly distinguish between original bytes and reconstructed images:

- `Store.resolve_artifact` and default `GET /v1/artifacts/{id}` strictly require original source bytes. Once an image is compacted into video, its original bytes are gone and HTTP requests return `404` rather than passing off a reconstructed frame as the original artifact.
- `Store.resolve_image` and `GET /v1/artifacts/{id}?view=image` explicitly permit reading reconstructed video frames (preferring the original file whenever it is still present). Reconstructed responses return `image/png` with the actual reconstructed SHA-256, accompanied by `X-Image-Reconstructed: video` and `X-Source-SHA256` headers.
- Historical observation views, ROI crops, visual comparisons, Sol reviews, and pi0.5 image loaders use the image-aware API and annotate returned metadata with `reconstruction` (while HTTP observation views add `image_reconstruction` without mutating the immutable observation record). The web UI labels compacted camera views as `"Video reconstructed (lossy)"`.
- Video files continue to be served directly with HTTP Range support, while archived individual frames are decoded sequentially by intra-segment frame index rather than using timestamp seeks that could snap to adjacent keyframes.

Because reconstructed frames undergo an additional lossy H.264 pass, they cannot recover the exact original JPEG bytes. Raw model response logs still retain the exact payloads sent during the run, and `audit_run.py` explicitly reports `"exact source-byte audit incomplete"` when source JPEGs have been compacted rather than claiming a full source-byte audit pass. Experiments requiring exact source-image bytes for bit-exact reproduction or external scene references should not run `--apply`. Reconstructed frame hashes are verified against the decoding results recorded at compaction time; if a future decoder upgrade produces different pixel values, the hash mismatch is explicitly rejected.

Each compacted run records segment details, deleted unique source-image counts, original image bytes, archive video bytes, and any cleanup errors in `image-archive.json` and `runtime/image_archive`. Reported storage savings cover only replaced RGB blobs, not the entire run directory (which also contains raw model transcripts and task replay videos).

## Context and Image Compression for Model Requests

The table below summarizes every mechanism that can reduce payload size between the Harness and the upstream model API, ordered by architectural layer. "Rewrites Sent History" indicates whether an outbound request ceases to be a strict prefix extension of the preceding request, which invalidates the upstream prompt cache from the point of modification onward.

| Layer | Mechanism | Config Switch (Default) | Rewrites Sent History | Implementation |
|---|---|---|---|---|
| Single-image encoding | Send as WebP 85, store as JPEG 95, no downscaling | `images.transport_format`, etc. (`"webp"`) | No | `auriga/core/storage/images.py` |
| Current tool reply | Streamline observation metadata | `features.actor_payload_profile` (`"lean_v1"`) | No | `harness.py`, `actor_payload.py` |
| Current tool reply | Omit camera views unchanged since last model delivery | `images.feedback_policy` (`"all"`, i.e., disabled) | No | `view_selection.py` |
| Single request | Replace byte-identical duplicate images within the same request with a reference | `images.reuse_exact_images` (`true`) | No (replaces only later duplicates within the same request) | `image_reuse.py` |
| Model proxy | One-shot "keep head & tail, trim middle" image pruning when request body exceeds 4.5 MB; window stays locked until next native compaction | `images.trim_oversized_requests` (`true`) | Once per native compaction cycle | `model_proxy._prepare_responses_body` |
| Model proxy | Prune middle images down to ~40 within the same locked window when inline image count exceeds 60 | `images.trim_max_inline_images` (`60`, `0` disables; requires `trim_oversized_requests=true`) | Once per threshold breach | `model_proxy._prepare_responses_body` |
| Model proxy | Further prune and retry if upstream returns `image_compression_failed` | Same as above | Only on the failed request | `model_proxy._complete` |
| Model proxy | History projection: replace non-anchor steps in closed blocks with text summaries and image placeholders (block size `B=8` or trailing `k` steps) | `features.lean_projection` (`false`), `lean_projection_block`, `lean_projection_k` | Once every `B` steps in block mode | `lean_projection.py`; see [Lean Context](../lean_context.md) |
| Model proxy | Decision reminder injection: injected only on `/responses/compact` when projection is off, or placed in fixed block slots in block mode | Follows `lean_projection` | No | `codex._task_decision_context`, `lean_projection.place_block_reminders` |
| Native Codex | Automatic context compaction (`/responses/compact`) | Built into Codex | Yes (establishes a new prefix after compaction) | Native Codex |

Proxy image trimming and `lean_projection` operate independently: `lean_projection` proactively summarizes older history steps to reduce per-request context tokens, whereas proxy trimming only drops middle images when a request would otherwise be rejected by the upstream relay—leaving all text, numbers, and trajectories intact and inserting placeholder text with the `observation_id` and `camera_id` so the model can fetch any needed view via `history_view_observation`. When `trim_oversized_requests` is disabled, requests are forwarded untouched; if an oversized request triggers an upstream `image_compression_failed` error, the proxy does not retry the identical payload and terminates the episode as an unscored model-request failure. The active setting is recorded under `runtime/model_image_policy.oversized_trim`, and each trimming event logs a `model_images_trimmed` record detailing the trigger reason, window bounds, prior window, and byte counts before and after trimming.

### Known Upstream Limits

Empirical observations show that the upstream relay enforces a hard limit on the **number of inline images per request (64 images)** rather than raw byte size alone: requests exceeding 9 MB were accepted when carrying 64 or fewer images, whereas 3–4 MB requests carrying 65 or more images failed with `image_compression_failed`. Consequently, the 4.5 MB byte threshold alone could prune requests that upstream would have accepted while failing to catch compact requests with more than 64 images until after a failed attempt.

`images.trim_max_inline_images` (default `60`) addresses this by triggering the same "keep head & tail, trim middle" window whenever the inline image count exceeds 60, pruning the count down to roughly 40 images and logging `model_images_trimmed.reason` as `"image_count"` (or `"body_bytes_and_image_count"` when both limits are exceeded) alongside `images_forwarded`. The 4.5 MB byte threshold remains in place; the net impact on `image_compression_failed` frequency, prompt-cache hit rates, and task scores remains to be evaluated in controlled benchmarks.

### Prompt Cache Hit Observations

The following figures come directly from upstream `usage.cached_tokens / input_tokens` telemetry rather than offline simulation:

- Appending the decision reminder to the very end of every `/responses` request broke strict prefix extension and reduced the token-weighted cache hit rate to roughly **35%**; restricting reminder injection to `/responses/compact` requests restored the cache hit rate to roughly **77%**.
- On the specific request where proxy image trimming first triggers, the cache hit rate drops as expected, then recovers on subsequent requests because the trimmed window remains locked until the next native compaction.
- Because these observations span batches with different task sets and concurrent Harness updates, they represent operational telemetry rather than a single-variable controlled ablation and do not measure task success rates.

## Evidence of Actual Model Inputs

### Optional Adaptive v2 Feedback

`images.feedback_policy="all"` remains the default. When `"adaptive"` is selected, routine completed single-action responses and standard template terminal responses can omit individual camera views—including the head camera and active wrist cameras—whose pixels have not changed significantly relative to the last view delivered to the model. Every observation still archives all original camera JPEGs, and omitted views return the current `observation_id`, baseline `observation_id`, and ready-to-call `history_view_observation` arguments. Initial deliveries, explicit observations or history queries, failed or uncertain executions, and handoff or support-change verifications always return full camera views. A template's `completed` status only indicates that execution reached its boundary for visual review; it does not prove that an object remains grasped, was stably placed, or satisfied the task.

Delivery baselines are established exclusively from full, uncropped original views returned to the primary actor model; program-internal observations, numeric-only responses, cropped views, and skipped views never advance the baseline. Any mismatch in run, episode, or environment identity, image dimensions, camera calibration, or context epoch forces full re-delivery, and context compaction or session replacement clears all baselines. An agent backend with `context_delivery="full"` (the default for Claude Code and OpenCode) runs with `Harness.context_epoch_volatile`: every call opens a fresh context epoch, so every actor delivery carries all camera views and complete static metadata, capability summaries and program logs. The pixel-difference thresholds in `auriga/perception/view_selection.py` only measure low visual change and do not guarantee unchanged task semantics.

Baselines are currently recorded when the tool response payload is constructed, **which is prior to an upstream provider delivery acknowledgment**. If the final actor projection fails or returns `success=false`, the Harness advances to a unique `tool_output_failed` epoch; if writing the tool reply to Codex RPC fails, `RecordedClient._reply_request` advances to a unique `tool_reply_delivery_uncertain` epoch. The next feedback response then re-delivers all camera views and restores pending program supplementary images and logs even within the same session. Context compactions, native window compactions, and app-server restarts likewise invalidate prior baselines. Audit tools must still cross-check local delivery records against actual outbound provider requests rather than assuming that a recorded baseline proves upstream receipt.

Adaptive deduplication relies on the actor's `direct` tool transport path. Under `code_mode`, a native RPC success could still be reduced to text-only output, stored-only output, or a host error before reaching the model; without typed image delivery confirmation, visual deduplication baselines cannot be safely maintained.

`tests/test_codex_image_transport.py` uses an isolated temporary `CODEX_HOME`, `--no-daemon app-server`, a clean subprocess environment, and a loopback mock Responses server to verify the actual HTTP POST tool catalog, standalone image items, decoded source bytes, and text logs. This test validates the native-to-model-request boundary without launching a live model or robot simulator. Program environment initialization permits both `sys.prefix` and the base interpreter prefix to support symlinked virtual environments while keeping the private run parent directory unreadable.

Image re-encoding alters byte representations without changing pixel coordinates. Delivery records track both representations separately:

- `rgb_ref` and `source_sha256`: the archived source JPEG (or historical image artifact).
- `sha256` and `encoding`: the SHA-256 digest, MIME type, dimensions, quality, and encoder version of the exact bytes transmitted to the model.
- Source image reference, ROI bounds, and `pixel_transform` for cropped views.

Transmitted WebP payloads are not duplicated as separate image artifacts in the Store because the native tool responses and archived model requests already preserve the exact transmitted bytes. `scripts/audit_run.py` supports both legacy byte-identical records and the two-stage JPEG/WebP records, cross-checking them against actual tool responses. The archived JPEG hash must never be conflated with the transmitted WebP hash.

## Validation and Measured Compression

Running the full encoding pipeline over 24 `640 × 480` RoboDojo camera frames yields:

| Stage | Total Size | Reduction vs. Legacy PNG |
|---|---:|---:|
| Raw PNG | 9,180,221 B | Baseline |
| Archived JPEG 95 | 2,100,382 B | −77.12% |
| WebP 85 encoded from archived JPEG 95 | 736,630 B | −91.98% |

Median transport encoding latency is roughly **21 ms per image** (measured on this 24-image sample, not a full-episode latency benchmark). Mean RGB SSIM across the two-stage lossy pipeline is **0.9372**, which is a signal-fidelity metric rather than a percentage of retained task information or a predictor of task success.

Across 11 consecutive frame pairs from a recorded headphone-manipulation sequence, raw PNG, JPEG 95, and JPEG 95 → WebP 85 all pass the visual tracking thresholds, with maximum single-step displacement deviations of `0`, `0.1281`, and `0.3224` pixels, respectively (and a 1-pixel vertical difference in the final ROI for the WebP branch).

Compressing image file sizes at fixed pixel dimensions does not proportionally reduce model visual token billing; no claims of reduced API token charges, higher task success rates, or lower end-to-end episode wall-clock times are made from byte compression alone.

Run the image pipeline test suite with:

```sh
PYTHONDONTWRITEBYTECODE=1 mamba run -n Auriga python -m unittest discover -s tests -p test_images.py -v
PYTHONDONTWRITEBYTECODE=1 mamba run -n Auriga python -m unittest discover -s tests -v
```

These tests use synthetic images alongside real Pillow, OpenCV, and FFmpeg binaries to verify three-camera capture, hard-link deduplication, immutable historical reads, crop coordinate transforms, JPEG ingestion for pi0.5, WebP transport payloads, encoder fallbacks, frozen worker settings, frame retention on video export failure, and temporary-frame cleanup upon verified export success.

Pillow `11.3.0` in the `RoboDojo` environment and Pillow `12.1.1` in the `Auriga` environment produce bit-identical JPEG outputs across all 24 test frames.

### Video Compaction Measurements

Running offline video compaction on an isolated copy of a completed episode (77 observations across 3 cameras = 231 unique `640 × 480` frames stored as JPEG 95) yields:

| Metric | Measured Value |
|---|---:|
| Total source JPEG 95 size | 18,975,352 B |
| Total H.264 archive size | 3,874,037 B |
| RGB storage reduction | 79.58% |
| End-to-end compaction wall-clock time (including full decode verification) | 67.80 s |
| Median single-frame reconstruction read latency | 93.06 ms |
| Mean RGB SSIM of reconstructed PNGs vs. source JPEGs | 0.9759 |
| 24-image sample: median first encode / cache-hit latency | 17.98 ms / 0.030 ms |

All 231 reconstructed frames preserved their `640 × 480` dimensions and matched their recorded source SHA-256 digests. Across the 11 headphone-motion frame pairs, both the JPEG 95 and video-reconstructed branches passed all visual tracking thresholds with identical final ROIs and a maximum single-step displacement difference of `0.2313` pixels.
