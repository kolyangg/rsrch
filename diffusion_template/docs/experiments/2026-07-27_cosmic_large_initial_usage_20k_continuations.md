# Cosmic Large initial-usage continuations to 20k

Date: 27 July 2026

## Objective and controls

Continue the baseline and four controlled Cosmic Large dataset-policy runs from
their full epoch-8 checkpoints (4,000 optimizer steps) to 20,000 steps. The
model and optimizer/scheduler state are resumed exactly. All arms retain
eligible SA-only branched attention: branched attention enabled, branched CA
disabled, `pipeline.pose_adapt_ratio=0`, and
`pipeline.ca_mixing_for_face=false`.

The inherited 12-image/500-step training validation is disabled. Training is
split into 2,000-step segments and the sealed full-96 protocol runs after steps
6k, 8k, 10k, 12k, 14k, 16k, 18k, and 20k (batch size 12, eight batches). Each
training segment appends to the original training Comet experiment; each
full-96 gate appends to the arm's existing validation-only Comet experiment.

## Started runs

| Arm | Machine / live job | Training Comet | Full-96 Comet |
|---|---|---|---|
| Distinct uniform reference | Serv `lm-mpi-job-bb07b32f-2e2e-4c63-943c-d880274e92eb` | `288ebfe3ccf74d5ea328a55b3abe31cb` | `ced6658b5b12484a9e003fe47cd0c2bf` |
| Distinct highest-score reference | Serv `lm-mpi-job-50741d46-69db-4fc3-a467-64b419230efe` | `fc3dec2223e84d49aa7c711fda968135` | `ddaeb234353b45a1ae6763f5d8a1c81f` |
| Distinct top-3 softmax reference | Serv `lm-mpi-job-6f171c44-2c62-4ea1-a9f0-891906b09d52` | `b7821337e24e49f388450c103553a9da` | `b9751dc78c3b460c9b2ebc50d61b2036` |
| Self-reference, target face ≥256 px | Serv `lm-mpi-job-a958a020-cd0e-4623-b428-98c5b07a0d5e` | `c6979abd46754e4ca43fae87df77eeff` | `e44bd0b7434348fa868844e96d704fca` |
| Initial-branch self-reference baseline | Neb process group `387209` | `aa982105aad148bf9b2a30d3fc2149f1` | `658d22341cf24accb5a3890869e76c28` |

All five passed the 64-sample dataset preflight and immutable source/full-96
record checks before entering model startup. The first two Serv submissions
per arm exited before training because the prerequisite checker understood only
the older single-checkpoint record fields. Compatibility for the equivalent
multi-checkpoint step-4000 fields was added and verified read-only against all
four records before the live jobs above were submitted.

## Comet trajectory repair

At 09:37 BST, a direct immutable-key audit established:

- baseline is complete with exactly 96 images and both full-96 metrics at
  0/1k/2k/3k/4k/6k/8k/10k/12k/14k/16k/18k/20k;
- highest-score, top-3 softmax, and minimum-face validation IDs each contain
  exactly 96 images and both metrics through their live 16k frontier;
- uniform contains exactly 96 images through 10k. Its original Serv chain
  stopped only because one step-10k REST download was transiently truncated
  during post-upload verification. The server-side asset had the correct
  full size, a fresh download matched all local pixels, and both 10k metrics
  became visible in the same validation ID.

Uniform was resumed from the byte-verified epoch-20/10k checkpoint on Neb,
process group `637653`, after a replacement Serv submission was rejected for
zero free workspace GPUs. It retains training Comet
`288ebfe3ccf74d5ea328a55b3abe31cb` and validation Comet
`ced6658b5b12484a9e003fe47cd0c2bf`. The continuation launcher now retries the
complete Comet pixel-and-metric verifier, not just the HTTP export command, so
a transient truncated download or delayed metric cannot terminate recovery.

`c6979abd46754e4ca43fae87df77eeff` is the historical minimum-face **training**
Comet and correctly retains its original 12-image panels. The canonical
full-96 trajectory for that arm is the separate, previously requested
validation-only ID `e44bd0b7434348fa868844e96d704fca`; all checkpoint
validation remains together in that one ID.

At 11:28 BST, a fresh audit executed from Serv against the five immutable
validation keys found:

- baseline, highest-score, top-3, and minimum-face complete through 20k, with
  exactly 96 images and exactly one identity/text metric at every requested
  step;
- uniform complete and verified through 16k and training toward 18k on Neb.

No relog was performed because every completed uniform step is already intact.
A relog is warranted only if the live job finishes and the immutable-key audit
still shows a deficient step.

Uniform subsequently completed 18k and 20k on Neb. A final immutable-key audit
found exactly 96 images and exactly one identity/text metric at every requested
step from 0 through 20k for all five arms.

## Offline face-quality metric pilot

The original Neb self-reference baseline validation experiment
`658d22341cf24accb5a3890869e76c28` in `jul-comet-large-testing` now also
contains no-reference face-quality metrics at
0/1k/2k/3k/4k/6k/8k/10k/12k/14k/16k/18k/20k. The scorer selects the largest
InsightFace detection, adds 25% padding on each side, makes a square 512px
crop, and evaluates four pinned PyIQA 0.1.15 models:
`topiq_nr-face`, `topiq_nr`, `musiq`, and `maniqa-pipal`. The full summaries
and per-image outputs remain local. Comet exposes only seven decision-oriented
curves in the separate `face_quality/` section: face-detection rate,
TOPIQ-Face mean/p10/coverage, and TOPIQ/MUSIQ/MANIQA means.

The final immutable-key audit found exactly seven names and one value for every
name at all 13 steps (91 scalar points total), while the original 96 images and
identity/text metrics remained unchanged. The earlier 21-series
`manual_val/face_quality/` layout and its 26 per-step JSON/CSV table assets
were deleted. Face detection ranged from 92/96 to 96/96; TOPIQ-Face's stricter
internal aligner produced scores for 79/96 to 89/96. Its coverage is therefore
a first-class metric rather than allowing the quality mean to hide rejected
malformed or small faces.

Observed IQA means are mixed rather than uniformly improving with training.
This pilot should be judged from the full trajectories, especially p10 and
coverage, before applying it to the four Serv arms. Full local results are in
`analysis/assets/face_quality/neb_baseline_658d22341cf24accb5a3890869e76c28/`.
A mistakenly targeted Uniform-arm download was stopped before scoring or any
Comet write; a post-run audit confirms that its validation key has no
face-quality metrics.

## Files

- Orchestrator:
  `launchers/active/run_rhca_cosmic_initial_usage_continue_20k_1gpu.sh`
- Appended-step verifier:
  `tools/inference/verify_appended_full96_step.py`
- Face-quality scorer:
  `tools/inference/calculate_face_quality_metrics.py`
- Idempotent Comet backfill:
  `tools/comet/backfill_face_quality_metrics.py`
- Neb baseline launcher:
  `launchers/neb/backfill_neb_baseline_full96_face_quality.sh`
- Local immutable plans and job IDs:
  `experiments/cosmic_large_continuation/*.json`
