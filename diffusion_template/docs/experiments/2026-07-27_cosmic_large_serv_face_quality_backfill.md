# Cosmic-large Serv face-quality backfill

Date: 27 July 2026

## Objective

Apply the same compact no-reference face-quality evaluation used for the Neb
baseline to the four cosmic-large dataset-usage validation runs. Each immutable
validation Comet run must receive seven scalar curves at the same 13 steps,
without per-step table assets. One CSV asset per run retains all 1,248
per-image rows for API access without adding default report curves.

## Planned execution

The original plan used two independent one-A100 Serv jobs in parallel, with
two Comet runs sequentially per job:

- `cosmic_large_face_quality_uniform_highest_1gpu`: distinct-uniform, then
  distinct-highest.
- `cosmic_large_face_quality_top3_minface_1gpu`: top-three score-weighted
  `_r2`, then self-reference minimum-face-256.

Both jobs use the existing Serv Conda environment
`/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS`.
Pinned PyIQA-only packages are loaded from
`/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/python_overlays/pyiqa-0.1.15`;
the Conda environment itself is not modified. A short shared lock serializes
model-weight cache initialization only. Metric calculation remains parallel.

Both submissions were rejected before job creation with
`WORKSPACE_GPU_LIMIT_REACHED_ONLY_0_FREE`. No Serv GPU job ran.

## Executed fallback

Under an explicit user-authorized exception, four CPU download workers on Serv
staged all 4,992 Comet images (7.1 GB). Four local `scp -3` relays then copied
one run apiece to Neb without copying either host's SSH or application
credentials. Neb independently verified every transferred file's recorded
size, SHA-256, and PIL decodability.

Neb's free H100 then processed the four runs sequentially under PGID `812861`
using its existing `photomaker_NS` and PyIQA 0.1.15 setup. The launcher emitted
`NEB_FACE_QUALITY_FOUR_RUN_COMPLETE`.

## Immutable Comet targets

| Dataset-use arm | Validation Comet ID |
|---|---|
| Distinct uniform | `ced6658b5b12484a9e003fe47cd0c2bf` |
| Distinct highest score | `ddaeb234353b45a1ae6763f5d8a1c81f` |
| Distinct top-three softmax `_r2` | `b9751dc78c3b460c9b2ebc50d61b2036` |
| Self-reference minimum-face-256 | `e44bd0b7434348fa868844e96d704fca` |

All four targets were observed in `jul-comet-large-testing` with exactly 96
images at steps 0, 1k, 2k, 3k, 4k, 6k, 8k, 10k, 12k, 14k, 16k, 18k, and 20k.
Before writing, none had compact `face_quality/` metrics, legacy
`manual_val/face_quality/` metrics, or face-quality table assets.

## Metrics and invariants

The seven Comet curves are:

- `face_quality/face_detection_rate`
- `face_quality/topiq_face_mean`
- `face_quality/topiq_face_p10`
- `face_quality/topiq_face_coverage`
- `face_quality/topiq_mean`
- `face_quality/musiq_mean`
- `face_quality/maniqa_mean`

The scorer uses InsightFace detection on CPU, the largest detected face, 25%
padding on each side, a square 512×512 crop, and PyIQA 0.1.15 models
`topiq_nr-face`, `topiq_nr`, `musiq`, and `maniqa-pipal`. Existing equal scalar
values are idempotent; conflicting or duplicate values abort. The tool does not
upload CSV or JSON tables.

The per-image data is uploaded once as
`face_quality_details__per_image_metrics.csv`, with logical namespace
`face_quality_details`. Its metadata records the source experiment, all steps,
row count, image count per step, and SHA-256. It is an ordinary Comet asset,
not a Comet table, so it is hidden from the default report while remaining
available through the asset API. Reruns require exact hash, size, and row-count
agreement.

## Results and audit

An independent Comet API audit found, for every immutable validation run:

- exactly 7 `face_quality/` metric names and 91 scalar points;
- exactly one point at each of the 13 requested steps for every metric;
- the original 96 validation images at every step, unchanged;
- zero `manual_val/face_quality/` metrics and zero per-step table assets;
- one downloaded and hash-verified 1,248-row per-image CSV.

| Arm | Per-image asset ID | CSV SHA-256 prefix |
|---|---|---|
| Distinct uniform | `b17bc3566b144e6d8baea69743df19be` | `45833fbdc811` |
| Distinct highest | `9302b8fe4ecc4233b044e03fca8ede72` | `2ecdb2371bfc` |
| Top-three softmax `_r2` | `21b6741314e54849985c45560f084836` | `aaeab7296e1c` |
| Self-reference min-face-256 | `f2e9b085ee9d40e4b0787c18f68bec12` | `da6715e2ffdc` |

The result JSONs, per-image CSVs, and scorer logs are retained locally under
`analysis/assets/face_quality/cosmic_large_four_20260727/`. The durable
experiment record is
`experiments/cosmic_large_continuation/serv_four_validation_face_quality_backfill.json`.

The post-run credential audit found no `.env`, key, credential-like filename,
credential pattern, or symlink in either transfer staging tree. Serv and Neb
retain distinct machine-local `.env` fingerprints; no credential was copied
between them.
