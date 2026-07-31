# Current project handoff

**Last updated:** 31 July 2026

**Repository:** `/home/kolyangg/rsrch_apr_test`

**Primary project:** `diffusion_template/`

**Branch:** `test`

**Current local and Serv commit:** `c04970f342a186d1092f07f9a08d7d8a797383e8`

This is the required starting document for a new agent. It summarizes the
research question, experiment history, reliable results, current code and
machine state, and the highest-value next work. Detailed evidence remains in
the linked reports.

## Read this first

The project tests whether PhotoMaker identity-conditioned generation can be
improved by explicit branched attention (BA). The core invariant is that
target queries must be able to consume identity/reference information through
the intended branched self-attention and cross-attention paths. A run that
looks good because it removes effective reference conditioning is a useful
ablation, but it is not a successful BA result.

For all currently eligible experiments:

```text
use_branched_attention = true
pipeline.pose_adapt_ratio = 0.0
pipeline.ca_mixing_for_face = false
reference_face_kv_weight = 1.0
```

The most recent full-Cosmic experiments have branched self-attention enabled
and branched cross-attention disabled. They establish results for the
reference-conditioned **SA-only BA protocol**, not a combined SA+CA design.

Do not change validation prompts, seeds, reference images, bboxes, validation
base, scheduler, inference steps, CFG, or metrics silently. Exact
comparability is part of correctness.

## Executive state

- Tasks A–D and the subsequent full-Cosmic reference-policy experiments are
  complete.
- Four final 4,000-step full-Cosmic training arms and their matched
  0/1k/2k/3k/4k full-96 validations are complete and integrity-verified.
- The initial-usage baseline and four dataset-policy arms are complete through
  20k, with exactly 96 validation images and one identity/text metric at all
  13 requested steps.
- All five initial-usage validation IDs now have seven decision-oriented
  `face_quality/` curves at all 13 steps plus an API-only per-image CSV, with
  no per-step table assets.
- The complete 20k comparison finds no promotion candidate. Top-three
  score-weighted distinct references are the best visual/identity compromise;
  highest-score distinct is the lower-confidence second. Every arm finishes
  below its peak identity score, and action/small-face attachment failures
  remain widespread.
- The strongest result in the final matrix is the **complete existing 256px
  reference asset (configured as 40% margin), pose-first captions, step
  3,000**. It reaches full-96 identity similarity `0.3606`.
- This is a matrix winner, not a production promotion. All four arms retain a
  repeatable Jisoo-specific malformed-face cluster.
- The nominal 40%/60% context and 256px/512px controls did not add source
  information: 40% and 60% are almost always the same full 256px input, and
  512px is an upscale. Legacy captions slightly improved text similarity but
  reduced identity similarity.
- Every final arm peaks on identity at step 3,000 and declines at step 4,000.
- The full-Cosmic data pipeline is mechanically healthy: 22,140 accepted
  records, deterministic reference transforms, propagated bboxes,
  target/reference inequality checks, CUDA ONNX Runtime, and reproducible
  full-96 validation.
- The best next dataset experiment is a clean highest-versus-top-three,
  fixed-256-versus-scale-curriculum factorial with one accepted-target
  manifest and no self-reference fallback. If longer training is requested,
  probe top-three first with early full-96 gates; do not run unchecked to 50k.

## Large Dataset same-ID 40k run — 27 July 2026

### New curated-dataset singleton switch — 31 July 2026

The training-ready successor is sealed at
`/home/niko/rsrch/dataset_publish/releases/v2`; the portable relative link
`dataset_publish/current -> releases/v2` selects it. Source
`current/pytorch_default.env` for the default 192px, no-singleton,
fixed-full96-disjoint manifest: 349,348 images / 68,648 identities. The
explicit 256px alternative contains 295,867 / 62,673. All selected captions
fit both SDXL tokenizers within 77 tokens and contain exactly one lowercase
`img`; all 386,092 release images were fully decoded and SHA-256 sealed.
`dataset_manifest.json` records every relative image path, size, hash, policy,
split hash, and validation audit. The include-singletons file is ablation-only.

The curated 486,103-image release is available on Neb at
`/home/niko/rsrch/dataset_publish/releases/v1`. It has one hard-linked
1024-square image tree and two loader-compatible manifests:

- `filtered_ids3_exclude_singletons.json`: 449,600 images / 77,050 IDs;
- `filtered_ids3_include_singletons.json`: 486,103 images / 113,553 IDs,
  including 36,503 true one-image identities.

Source `pytorch_exclude_singletons.env` or
`pytorch_include_singletons.env` from that release to set the manifest, image
root, and `LARGE_DATASET_SINGLETON_REFERENCE_POLICY` together. The
`LargeDatasetTrain` default remains fail-closed (`error`). The explicit `self`
mode retains distinct references for multi-image identities and uses the
target itself only where a true singleton has no alternative. A local/Neb
smoke test verified exclude mode selects a distinct reference, include mode
fails under the default policy, and include+`self` loads a singleton.

### Big Celebs training path — prepared 31 July 2026

The dedicated `big_celebs` dataset path is prepared but no training run has
been launched. `BigCelebsTrain` reuses the Large Dataset target/reference and
transform behavior while failing closed on the sealed-release contract:
distinct same-ID references, exact `{new_face_crop, text}` records, in-bounds
faces with minimum side 192px, and exactly one lowercase `img` trigger.

The Neb launcher
`launchers/neb/start_rhca_big_celebs_sameid_40k.sh` pins release `v2`, manifest
SHA-256 `f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa`,
and `dataset_manifest.json`; it does not follow the movable `current` symlink.
Its preflight verifies the READY seal, sealed variant counts and policy, the
selected and full image-tree path sets, every metadata record, and 64 decoded
target/distinct-reference pairs before creating a Comet run. The complete v2
preflight passed for 349,348 images / 68,648 identities, and the actual loader
initialized the full manifest and loaded three boundary samples successfully.

`src/configs/big_celebs_rhca_40k.yaml` inherits the Large Dataset model and
fixed full-96 validation configuration and changes only
`train_dataset_name=big_celebs`. The launcher supplies the current standard
2,000-step epoch length × 20 epochs (40k total), validation/checkpoint gates
every 2,000 steps, `pose_adapt_ratio=0`, and `ca_mixing_for_face=false`. The
prepared immutable experiment spec is
`experiments/big_celebs/rhca_big_celebs_sameid_40k_full96_r1.json`. Before a
real launch, sync these new files to Neb; during startup verify the new Comet
key, step-0 96-image validation and face-quality output, 840/840 processor
tensors in the optimizer, and the first optimizer step.

`rhca_large_dataset_sameid_40k_full96_r4` was stopped by user request on
28 July 2026. It kept the
exact eligible SA-only BA model used by the recent Cosmic Large matrix and
changes only the dataset to the adjusted identity-aware Large Dataset:
47,500 1024px images, 2,561 explicit identities, and a uniformly sampled
distinct same-ID reference for every target.

- 40,000 optimizer steps, batch size 2
- validation at step 0 and every 2,000 steps through 40k
- fixed full-96 panel and default face-crop quality metrics
- terminated Neb launcher PGID `963959`; terminated training/metric PGID
  `964138`; the GPU process list was empty after SIGTERM
- Comet
  [`a99db1fb953d4511827672380e6c1645`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/a99db1fb953d4511827672380e6c1645)

Startup passed a 64/64 dataset decode preflight, exact transfer
reconciliation, ONNX CUDA, 840/840 processor-in-optimizer, 96/96 step-0
generation, face-quality scoring on all 96 inputs, and multiple optimizer
steps. Three preserved zero-step startup records exposed and fixed,
respectively, the missing Neb CUDA library path, an unsynchronized audited
validation runtime patch, and a CPU-only PyIQA subprocess. Full details and
all immutable failed IDs are in
[the run report](../experiments/2026-07-27_large_dataset_sameid_40k.md).

### Two-GPU Serv mirror

`rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu` is running as a
two-GPU mirror with the same model, dataset policy, local batch size 2,
40,000 synchronized optimizer updates per rank, and full-96 validation every
2,000 steps. The controlled differences are Serv, world size 2, and global
batch size 4.

- current Serv continuation job:
  `lm-mpi-job-79007b8b-a9f0-41db-a15a-802ffea65658`
- Comet:
  [`db32f157e75a4798b2dfa530477c66d6`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/db32f157e75a4798b2dfa530477c66d6)
- startup passed the exact manifest and 64/64 decode preflight, CUDA ONNX
  Runtime, PyIQA 0.1.15, 840/840 processor-in-optimizer, and rank-0/rank-1
  DDP epoch synchronization;
- the original job completed all 96 step-0 images and face-quality metrics but
  stalled on its first training batch with zero updates and no checkpoint;
- the original job `lm-mpi-job-3809c1e1-9749-4dd6-9ef9-7fcc0f84e3e4`
  was stopped, while its step-0 Comet artifacts were preserved;
- the first recovery trained through step 2,000 and completed all 96 validation
  images and face-quality metrics, but rank 1 entered epoch 5 while rank 0
  wrote the checkpoint; mismatched NCCL sequence 32048 timed out and left a
  truncated, unloadable epoch-4 checkpoint;
- the repaired trainer holds all ranks around main-only checkpoint/logging and
  writes checkpoints by atomic replacement;
- the replay job reconstructed 0→2k without duplicate validation or Comet
  events, verified the full optimizer checkpoint, resumed epoch 5, and
  completed training plus full-96 validation at step 4,000;
- fresh-container model initialization is opt-in serialized across ranks to
  prevent the observed concurrent 891 MB artifact-cache race;
- after the complete step-4k validation and intact atomic epoch-8 checkpoint,
  rank 1 entered its epoch-9 iterator while rank 0 blocked at the next
  rank-0-only Comet writer boundary; this was a logging-boundary stall, not
  the prior checkpoint/NCCL race;
- the current recovery keeps two-GPU training continuous from step 4k to 40k
  with 2k checkpoints, then evaluates every 6k–40k checkpoint in fresh
  single-process full-96 invocations that append to the same immutable Comet
  key;
- on 28 July, the missing live validation was traced to that intentional
  deferred-validation mode rather than loss of all Comet telemetry: training
  curves were present through step 13,650+, while validation assets stopped at
  4k. Two non-disruptive one-GPU sidecars now restore live full-96 validation
  in the same Comet run: arm 0
  `lm-mpi-job-2e42c27d-d4b0-4524-b728-2758be257aea` covers
  6k,10k,...,38k and arm 1
  `lm-mpi-job-e2a7254f-1754-43d4-861a-fee26db1eabe` covers
  8k,12k,...,40k. They leave the two-GPU trainer untouched and publish
  completion markers that prevent the deferred loop from duplicating work;
- both current ranks synchronized at epoch 9, 840/840 processor tensors remain
  in the optimizer, and the first new optimizer update completed with reduced
  loss `0.043631`.

The exact launcher/YAML hashes and live startup evidence are in
[the Serv mirror report](../experiments/2026-07-27_large_dataset_sameid_serv_2gpu.md).

## Cosmic Large initial-usage continuations — 27 July 2026

The initial-usage baseline plus four dataset-policy arms completed their
4,000-step training and sealed 0/1k/2k/3k/4k full-96 validation. They continued
from the full epoch-8 optimizer/scheduler checkpoints to 20,000 steps. The old
12-image validation every 500 steps was disabled. Training runs
in 2,000-step segments, followed by sealed 96-image validation at
6k/8k/10k/12k/14k/16k/18k/20k. Training metrics append to the original
training Comet keys; validation metrics/images append to the existing
validation-only Comet keys.

| Arm | Machine/job history | Training / validation Comet keys |
|---|---|---|
| Initial self-reference baseline `_r2` | Neb PGID `387209` | `aa982105aad148bf9b2a30d3fc2149f1` / `658d22341cf24accb5a3890869e76c28` |
| Uniform distinct reference | Serv `lm-mpi-job-bb07b32f-2e2e-4c63-943c-d880274e92eb`, recovered on Neb PGID `637653` | `288ebfe3ccf74d5ea328a55b3abe31cb` / `ced6658b5b12484a9e003fe47cd0c2bf` |
| Highest-score distinct reference | Serv `lm-mpi-job-50741d46-69db-4fc3-a467-64b419230efe` | `fc3dec2223e84d49aa7c711fda968135` / `ddaeb234353b45a1ae6763f5d8a1c81f` |
| Top-three score-weighted distinct reference `_r2` | Serv `lm-mpi-job-6f171c44-2c62-4ea1-a9f0-891906b09d52` | `b7821337e24e49f388450c103553a9da` / `b9751dc78c3b460c9b2ebc50d61b2036` |
| Self-reference with 256px minimum target face | Serv `lm-mpi-job-a958a020-cd0e-4623-b428-98c5b07a0d5e` | `c6979abd46754e4ca43fae87df77eeff` / `e44bd0b7434348fa868844e96d704fca` |

All five passed immutable source/evaluation checks and a 64/64 dataset
preflight, resumed their exact existing training Comet experiments, and
entered model startup. Startup monitoring then stopped as requested. Details,
including preserved failed pre-training submissions caused by an older
full-96 record-field assumption, are in
[the 20k continuation report](../experiments/2026-07-27_cosmic_large_initial_usage_20k_continuations.md).

All five are now complete through 20k. Direct immutable-key audits found
exactly 96 images and exactly one identity/text metric at every requested step
from 0 through 20k. No completed uniform step was relogged because every
server-side step was already intact.

Uniform stopped after producing 10k because a
post-upload REST export transiently downloaded one truncated PNG. Its
server-side asset, all 96 local pixels, and both 10k metrics subsequently
verified exactly in the same validation ID. Uniform recovered from its
byte-identical 10k checkpoint on Neb under PGID `637653` and completed 20k,
retaining training key `288ebfe3ccf74d5ea328a55b3abe31cb` and validation key
`ced6658b5b12484a9e003fe47cd0c2bf`.

The minimum-face training key `c6979abd46754e4ca43fae87df77eeff`
intentionally contains the old 12-image panels. Its single canonical full-96
trajectory is validation key `e44bd0b7434348fa868844e96d704fca`.
The local continuation launcher now retries full Comet pixel/metric
verification, including transiently truncated downloads and delayed metrics.

All five validation keys now have the 27 July offline face-quality metrics at
all 13 steps. Their separate `face_quality/` sections each have exactly seven
curves: face-detection rate, TOPIQ-Face mean/p10/coverage, and
TOPIQ/MUSIQ/MANIQA means. Each also has one API-only 1,248-row per-image CSV;
there are no legacy `manual_val/face_quality/` series or per-step table assets.

The completed five-run comparison uses all 96 images at
0/4k/8k/12k/16k/20k. Quantitatively:

- top-three is the best distinct arm at 20k on identity (`0.2703`) and
  TOPIQ-Face mean (`0.6531`);
- highest has the strongest late broad distinct quality and the only
  meaningful 14–20k identity rebound (`0.2324 -> 0.2646`);
- uniform reaches the best text score (`27.1631`) but the worst 20k identity
  (`0.2428`);
- the 256px self-reference arm peaks at `0.3467` identity at 4k and falls to
  `0.2647` at 20k despite leading face-IQA means;
- every 20k endpoint is below that arm's best identity gate.

A matched visual audit of all 480 20k images ranks top-three first and highest
second, but neither is promotion-quality. Jumping, dancing, skiing, and crying
still show pasted, stretched, duplicated, or misplaced facial regions.
TOPIQ-Face coverage and p10 are more useful coherence guards than its mean;
the generic TOPIQ/MUSIQ/MANIQA models, although evaluated on the same padded
face crop, and saturated face detection can reward a crisp but grossly
malformed face. None of these IQA values was calculated on the whole image.
The full decision record and 97-page comparison PDF are in
[the 20k analysis](../experiments/2026-07-27_cosmic_large_initial_usage_20k_analysis.md).

## Completed initial-usage Cosmic Large 4k matrix — 26 July 2026

A controlled matrix reproduced only the Cosmic Large
portion of the initial `test` branch at
`6782e9d62345fe910633cc8ceec0e2fda6ec2fd1`: legacy captions, historical bbox
gate, target-as-reference, and no minimum face size. The current eligible
SA-only BA model is fixed across all arms; a composed-config comparison against
the current adapted run found no architecture, optimizer, loss, pipeline, or
BA-flag differences.

| Arm | Machine/job | Immutable training Comet key |
|---|---|---|
| Initial self-reference baseline `rhca_cosmic_initial_selfref_4k_baseline_r2` | Neb PID `196928`, PGID `196733` | `aa982105aad148bf9b2a30d3fc2149f1` |
| Uniform distinct reference | `lm-mpi-job-8f161a20-3303-40e2-8884-8c137348d9bb` | `288ebfe3ccf74d5ea328a55b3abe31cb` |
| Highest-score distinct reference | `lm-mpi-job-acd898ba-b09a-46e4-a8b5-4becae1b1280` | `fc3dec2223e84d49aa7c711fda968135` |
| Top-three score-weighted distinct reference `_r2` | `lm-mpi-job-f2a4b83f-ab44-4717-82b8-cd085307db3f` | `b7821337e24e49f388450c103553a9da` |
| Self-reference with 256px minimum target face | `lm-mpi-job-ca0acbd0-7433-42da-bcc1-39ab72a38272` | `c6979abd46754e4ca43fae87df77eeff` |

Every run passed a 64/64 decode preflight, explicitly loaded CUDA ONNX Runtime,
verified all 840/840 branched-processor tensors in the optimizer, completed
initial validation, and entered its optimizer loop. Startup monitoring then
stopped. Baseline and distinct-reference arms accept 74,754 examples; 59,143
have audited distinct-reference candidates and 15,611 retain the historical
self-reference fallback. The 256px target-face arm accepts 16,168 examples.

Each trainer targets 4,000 optimizer steps with checkpoints every 1,000 steps.
The same machine job then creates a separate Comet experiment and evaluates
steps 0/1k/2k/3k/4k on every sample in sealed
`cosmic_full96_auto_v1` (96 images per step, batch 12).

The Neb baseline's automatic evaluation chain initially stopped before Comet
creation because the configured historical bbox source did not match the
sealed SHA-256. Its manual restart subsequently completed all five endpoints:
40 batches and 480 images under Comet
`658d22341cf24accb5a3890869e76c28`.

Two failed startup identities are intentionally preserved:

- Neb `rhca_cosmic_initial_selfref_4k_baseline`,
  `a42206ee6fd241a4914aabdb436eca7f`, was stopped before step 1 because the
  CUDA provider could not load `libcudnn_adv.so.9`.
- Serv `rhca_cosmic_initial_distinct_top3softmax_4k`,
  job `lm-mpi-job-5295c0a9-49b9-43b0-8013-feabeeebe687`, Comet
  `ec43ee00375f4563b353bf701720c9eb`, stalled in model initialization and was
  deleted before processor installation or step 1. The `_r2` retry has the
  same experiment semantics and disables only the optional C++ stack
  symbolizer.

The canonical design and live provenance are in
[Cosmic Large initial-usage baseline matrix](../experiments/2026-07-26_cosmic_large_initial_usage_baseline_matrix.md).
## Dataset-policy audit addendum — 26 July 2026

A fresh live-manifest geometry audit materially narrows the interpretation of
the final reference-policy matrix:

- Full-Cosmic `face_paths` are already 256x256 face-focused assets.
- The 40% and 60% policies produce the same crop for `99.9922%` of the
  180,623 valid reference candidates; 40% already returns the full source for
  all but 14 candidates.
- The 512px arm upsamples the same at-most-256px source and therefore does not
  test additional native reference detail.
- The final matrix supports using the complete existing 256px reference asset,
  but it does not establish an optimal real-context margin or source
  resolution.
- The manifest still has no stable identity IDs joining 1024px targets:
  22,140 accepted targets map to 22,140 unique target-specific reference
  groups.

Do not run another numeric margin or 512px upscale arm on these assets. For
dataset-policy work, prioritize an audited reference-selection factorial,
stable multi-target identity grouping, target-scale/quality curricula, and
native full-scene references. The full analysis and experiment designs are in
[Cosmic Full dataset usage recommendations](../../analysis/2026-07-26_cosmic_full_dataset_usage_recommendations.md).

## Why Cosmic Large was initially unsuitable and how it became trainable

This is a central project result, not merely data-loader cleanup. The raw
Cosmic Large package could not safely be substituted into the historical
training path.

### Problems found

1. **The historical loader represented the wrong data contract.**
   `src.datasets.cosmic.CosmicDoubledTrain` combines older Cosmic metadata,
   defaults to using the target itself as the reference unless a separate
   mapping is supplied, and cannot consume the new manifest's `face_paths`,
   per-reference bboxes, and scores. It also does not include reference
   transforms in conditioning-cache identity. It remains historical replay
   code and must not be used for new full-Cosmic training.
2. **The raw manifest needed filtering and validation.** It contains 59,143
   input records, small target faces, invalid target/reference boxes, and
   records without a usable reference after filtering. The audited loader
   retains 22,140 targets with a target face of at least 192px. It removed 137
   invalid reference-bbox entries; accepted samples have 2–10 valid reference
   candidates, mean `8.158`.
3. **Target/reference leakage had to fail closed.** Self-reference lets the
   network copy the target rather than learn identity transfer. The new path
   requires a different reference path and raises an error on a collision.
   The earlier one-ID `51.jpg` training/validation overlap was also removed
   for leak-free endpoint comparisons.
4. **Tight reference geometry was unsafe at inference.** Task B showed that a
   tight 256px Cosmic reference can be copied into the target as an oversized,
   displaced, or incomplete face. Centering the same crop on a blank 1024px
   canvas did not fix occupancy; it caused catastrophic failures in about
   10/12 one-ID images. Real surrounding image context, rather than padding,
   was required.
5. **Reference image and bbox transforms could not diverge.** Cropping,
   resizing, and flipping a reference without applying the exact same
   transform to its face box corrupts the spatial BA mask. The policy and flip
   state also have to be part of the conditioning cache key.
6. **Caption order was poorly matched to long Cosmic captions.** The legacy
   order starts with facial appearance, so pose and background can be weakened
   by token truncation. The controlled pose-first mode emits
   `<class> img, pose, background, remaining appearance` and caps at 55 words.
7. **The apparent data throughput problem was partly runtime configuration.**
   `CUDA_LAUNCH_BLOCKING=1`, CPU InsightFace/ONNX Runtime fallback, and the
   initial worker settings made Serv training take 5–7 seconds/step.
8. **The package has a remaining identity-structure limitation.** The 22,140
   accepted targets resolve to 22,140 fallback reference-parent groups, so the
   manifest does not prove multiple target views per stable explicit identity.
   This is not fixed by the loader and limits claims about multi-view identity
   learning.

### Implemented, backward-compatible fix

The trainable path is isolated rather than replacing historical behavior:

- `src/datasets/cosmic_large_adapted.py` reads the real full-Cosmic manifest,
  validates boxes, filters target face size, samples a distinct valid
  reference, exposes paths/identity IDs for audits, and supports legacy or
  pose-first prompts.
- `src/datasets/reference_policy.py` performs one deterministic square crop,
  adds real context, resizes once with exact bbox propagation, optionally
  applies a diagnostic canvas, and returns a cache descriptor that includes
  the policy.
- `src/configs/cosmic_large_adapted_rhca.yaml` selects the isolated loader,
  keeps CA disabled and masked loss at every step, disables the ineffective
  one-ID-style conditioning LRU, and batches frozen conditioning preparation.
- `tools/datasets/preflight_cosmic_large_adapted.py` must pass before a run is
  registered. It checks decoded dimensions, target/reference inequality,
  bboxes, face-area fractions, exactly one PhotoMaker trigger, prompt policy,
  and cache keys on a deterministic sample.
- The successful final recipe uses diverse 1024px scene targets, a different
  same-identity **complete existing 256px reference asset** (reached through
  the nominal 40% policy), pose-first captions, ratio-zero branched SA,
  branched CA off, and full-scene references for validation/inference.
- Serv production uses asynchronous CUDA, ONNX Runtime 1.20.1 with
  `CUDAExecutionProvider`, and two loader workers. It now trains at roughly
  2.06–2.10 seconds/step and fails closed instead of silently accepting CPU
  fallback.
- The canonical full-96 protocol batches 12 prompts for one shared
  identity/reference at a time. Do not use a heterogeneous-reference batch
  until the remaining first-reference spatial setup is made truly per-sample.

Validation of the fix includes a deterministic 64/64 decode preflight, real
two-sample instantiate/collate checks, 22,140 accepted records, four complete
4k training runs without OOM, and four integrity-verified 480-image multistep
full-96 evaluations. This establishes that Cosmic Large is mechanically
trainable through the adapted path. It does not erase the remaining Jisoo
model-quality failure or the dataset's weak explicit multi-view identity
structure.

The full code audit and rationale are in
[Cosmic Large training recommendations](../../analysis/2026-07-25_cosmic_large_training_recommendations_and_experiments.md).

## Current machine and worktree snapshot

This snapshot is informational; always recheck live state before launching.

- Local checkout: `test` at `c04970f...`.
- Serv checkout:
  `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test`, `test` at
  `c04970f...`; the four recorded Cosmic initial-usage continuation jobs are
  complete.
- Neb checkout: `/home/niko/rsrch/diffusion_template`, `test` at
  `c04970f...`; the 20k continuations and baseline face-quality backfill are
  complete. Preserve `.env`, bbox caches, and other machine-local files.
- No resource availability is implied by this snapshot. Recheck Neb GPU
  processes and Serv Running/Pending jobs before launching.
- The worktree contains untracked experiment reports, image assets, and the
  controlled dataset artifact from prior work. They are intentional evidence;
  do not delete or overwrite them. The user has not authorized a blanket
  commit of those files.

Operational entry points:

- [Project tools](../../TOOLS.md)
- [Neb operations](../../LOCAL_NEB_SERVER_OPERATIONS.md)
- [Serv/MLS operations](../../../local_scripts/serv_instructions.MD)
- [Repository rules](../../../AGENTS.md)

## Experiment history and what was learned

### 1. Original one-ID overfit and validation leakage

The original Cosmic one-ID experiment established the malformed or displaced
face failure. A validation image (`51.jpg`) was also present in the training
set. A leak-free launcher was created and the holdout reproduced correctly.
The leakage was a real comparability issue, but removing it did not explain
the main anatomy failure.

Start with the historical
[one-ID handoff](../../2026-07-24_test_branch_one_id_overfit_handoff.md) only
when reconstructing that baseline.

### 2. Tasks A–C: architecture and fixed-checkpoint diagnostics

The consolidated evidence is in
[Tasks A–D results](../../analysis/2026-07-25_cosmic_large_tasks_a_d_results_handoff.md).

| Task | Intervention | Result | Decision |
|---|---|---|---|
| A | Disable branched cross-attention while retaining branched self-attention | Scenes and bodies improved, but about 9/12 faces remained malformed; text `24.7982`, ID `0.1418` | CA amplified global corruption but was not the primary face-local cause |
| B | Reproduce fixed checkpoints exactly, then vary inference reference, CA, CFG, and identity | Tight 256px Cosmic references recreated the pathology on a healthy checkpoint; a full-scene wrong-identity reference produced mostly attached anatomy; CFG 1 collapsed to haze; null identity destroyed the face | Strongest causal evidence that the spatial reference path is active and unsafe for some tight references |
| C | Train only reference-path SA processors with CA disabled | About 9/12 anatomy failures remained; text `24.4779`, ID `0.1484` | Target/noise projection drift was not the primary cause |

Detailed reports:

- [Task A](../experiments/2026-07-25_task_a_cosmic_faceonly_noca_4k_results.md)
- [Task B](../experiments/2026-07-25_task_b_checkpoint_diagnostic_matrix_results.md)
- [Task C](../experiments/2026-07-25_task_c_cosmic_faceonly_noca_refonly_4k_results.md)

Task B passed its reproduction gates at exact filename, file-hash, and decoded
pixel equality before its interventions. Treat its causal conclusion as more
reliable than an uncontrolled visual comparison.

### 3. Task D: controlled target/reference factorial

Task D used one sealed woman-class identity and isolated training target
diversity from training reference format. Every arm used the same full-scene
reference at validation:

| Arm | Training targets | Training references | Text / ID at 4k | Visual result |
|---|---|---|---:|---|
| `multi_full` | Eight distinct scenes | Full scenes | `25.7448 / 0.2357` | Roughly 6–7/12 coherent |
| `multi_cosref` | Eight distinct scenes | Deterministic tight 256px crops | **`26.9297 / 0.3375`** | Best; two hard failures plus milder defects |
| `single_full` | One repeated scene | Full scenes | `25.0182 / 0.1853` | Worst; repeated eye/missing-feature failures |

Immutable Comet keys:

- `multi_full`: `d6363cba32e444469cde81b1d6e291af`
- `multi_cosref`: `3738f67625894b1ba583d3c7eff06c51`
- `single_full`: `ce3256602a7b4f09a82a30db616c3c3e`

Local immutable records:

- [multi_full JSON](../../comet_records/rhca_controlled_identity_factorial_multi_full_4k.json)
- [multi_cosref JSON](../../comet_records/rhca_controlled_identity_factorial_multi_cosref_4k.json)
- [single_full JSON](../../comet_records/rhca_controlled_identity_factorial_single_full_4k.json)

Task D reconciles with Task B by separating stages:

```text
tight crops used during training
    can focus identity learning and suppress nuisance scene context

tight crops injected through the current spatial path at inference
    can be copied or misregistered as literal face geometry
```

This is supported by controlled interventions but is not yet a
layer-by-layer mechanistic proof. Target diversity clearly helped. No Task D
checkpoint passed a 12/12 anatomy gate.

### 4. Initial full-Cosmic adaptation and runtime correction

The full dataset contains 59,143 input rows and 22,140 accepted training
records after the documented filters. Early Serv runs were slow because:

- production inherited `CUDA_LAUNCH_BLOCKING=1`;
- InsightFace fell back to CPU ONNX Runtime;
- the loader did not use the verified worker configuration.

The corrected runtime uses asynchronous CUDA, ONNX Runtime 1.20.1 with
`CUDAExecutionProvider`, and two training workers. Training improved from
roughly 5–7 seconds/step to roughly 2.0–2.1 seconds/step on Serv and about
1.2 seconds/step on Neb. Production jobs must fail closed if CUDA ONNX Runtime
is unavailable.

The 20%-margin pose-first and legacy full-Cosmic endpoints both failed their
canonical 96-image visual gates, primarily on Jisoo, despite plausible
aggregate metrics:

| Run | Comet key | Full-96 text / ID | Result |
|---|---|---:|---|
| 20% pose-first fast | `7c80400b23ba4a1683d4b034abdbb12c` | `27.0207 / 0.3538` | Fail: six clear Jisoo failures |
| 20% legacy fast | `0de9a9858a784373a8871e6b667316e1` | `27.1722 / 0.3374` | Fail: at least seven clear Jisoo failures |

See the
[full-Cosmic 4k/full-96 report](../experiments/2026-07-26_cosmic_large_adaptation_4k_full96_results.md)
for hashes, panels, and the runtime investigation.

### 5. Drift toward plain PhotoMaker and the architectural reset

A fixed-checkpoint `pose_adapt_ratio` sweep progressively replaced spatial
reference-face K/V with target-native face K/V:

| Ratio | Full-96 text / ID | Visual result |
|---:|---:|---|
| 0.35 | `27.0094 / 0.3615` | Residual identity-specific fragments |
| 0.65 | `26.9725 / 0.4016` | Jisoo improved; Jensen still failed |
| 1.00 | `27.1979 / 0.4421` | Every identity at least 11/12 coherent |

Ratio 1.0 looked attractive, and a train-1/validate-1 run reached full-96 ID
`0.5136` with 12/12 coherent images for all eight identities. However, ratio
1.0 gives spatial reference-face K/V zero weight. A matched plain PhotoMaker
control was equally coherent and slightly better on the 12-image text and ID
metrics. The pixels differed, but there was no evidence of useful
reference-conditioned BA contribution.

This was experimental drift toward plain PhotoMaker, not a BA promotion. The
program was reset:

- runs with `pose_adapt_ratio > 0` were stopped;
- CA-mixing experiments were rejected;
- `AGENTS.md`, launchers, and run records were pinned to ratio zero and no CA
  mixing;
- subsequent experiments changed reference formatting, caption policy, or
  resolution while retaining the reference-face K/V path.

The full chronology and JSON/Comet IDs are in the
[ratio-zero reference-policy handoff](../experiments/2026-07-26_ba_ratio_zero_reference_policy_runs_handoff.md).

### 6. Ratio-zero one-ID reference-policy gates

These gates retained the intended BA route:

| Policy | Comet key | Text / ID | Result |
|---|---|---:|---|
| 40% real context, 256px | `9a947bd85a7745e29ddf329b9be16763` | `26.7409 / 0.3076` | Mostly coherent; strong improvement |
| Exact crop centered on blank 1024px canvas | `f03960bfb34a49bdba6e1503aafaf130` | `26.2995 / 0.1377` | Catastrophic in about 10/12 |
| 60% real context, 256px | `b2ef6ed73f164961b111e6c78c742eab` | See immutable record/report | Completed; motivated the full-data margin control |

The canvas experiment rejects padding as a substitute for real context.
Surrounding image content matters.

### 7. Final four-arm full-Cosmic matrix

All four source runs completed on Serv at commit
`cfa4bffebfbb46e324a7b503bdbfd786bea5e6e6`. They used all 22,140 accepted
records, ratio-zero BA, no CA mixing, active branched SA, disabled branched
CA, 840/840 processor parameters in the optimizer, and approximately
2.06–2.10 seconds/step.

| Arm | Source Comet key | Source experiment JSON |
|---|---|---|
| 40% / 256 / pose-first | `1a19fdf2793f413c9336379d3628874d` | [JSON](../../experiment_specs/rhca_cosmic_full_crop40_posefirst_4k_fast_r1.json) |
| 60% / 256 / pose-first | `a96bcbae3d2b4698a43d7ec80457586c` | [JSON](../../experiment_specs/rhca_cosmic_full_crop60_posefirst_4k_fast_r1.json) |
| 40% / 256 / legacy | `92572589d6594cd59749577fc51f5bba` | [JSON](../../experiment_specs/rhca_cosmic_full_crop40_legacy_4k_fast_r1.json) |
| 40% / 512 / pose-first | `c354369af45b4c9da84f1124cf3e9a88` | [JSON](../../experiment_specs/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1.json) |

The corresponding Serv packages are under
`serv_run_packages/<run_name>/`.

### 8. Final multistep full-96 result

Validation commit `c04970f...` evaluates steps 0, 1,000, 2,000, 3,000, and
4,000 in one Comet run. Each arm produced 96 images at every step using batch
size 12. Step 0 is byte-identical across all arms; all source reproduction and
Comet image checks passed.

| Step | 40% / 256 / pose-first | 60% / 256 / pose-first | 40% / 256 / legacy | 40% / 512 / pose-first |
|---:|---:|---:|---:|---:|
| 0 | `26.3205 / 0.2999` | `26.3205 / 0.2999` | `26.3205 / 0.2999` | `26.3205 / 0.2999` |
| 1,000 | `27.1279 / 0.2972` | `27.0369 / 0.2947` | `27.2619 / 0.2961` | `27.0129 / 0.2872` |
| 2,000 | **`26.8722 / 0.3465`** | `27.0072 / 0.3423` | `27.1172 / 0.3353` | `26.9036 / 0.3390` |
| 3,000 | **`26.6846 / 0.3606`** | `26.7720 / 0.3575` | `27.0054 / 0.3457` | `26.7827 / 0.3545` |
| 4,000 | `26.9992 / 0.3422` | `26.8936 / 0.3458` | `27.1810 / 0.3316` | `26.9494 / 0.3418` |

Validation provenance:

| Arm | Validation Comet key | Immutable local record |
|---|---|---|
| 40% / 256 / pose-first | `519f9ecac929417e8073e7b3cc953c2d` | [JSON](../../comet_records/rhca_cosmic_full_crop40_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 60% / 256 / pose-first | `df99f4b0bb9a4676bd6783d1bc611c6b` | [JSON](../../comet_records/rhca_cosmic_full_crop60_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 40% / 256 / legacy | `dfb06576f4104d969b08c59b06ec7834` | [JSON](../../comet_records/rhca_cosmic_full_crop40_legacy_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |
| 40% / 512 / pose-first | `00cfd945fdcf44dbbd8914b42f139300` | [JSON](../../comet_records/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k.json) |

The complete review and contact-sheet links are in
[the final four-run handoff](../experiments/2026-07-26_current_four_full_cosmic_4k_runs_handoff.md).

Observed conclusions:

1. The useful identity gain appears between 1k and 2k.
2. Every arm peaks at 3k and regresses at 4k.
3. The 40%-labelled full-256 / pose-first arm is the best identity/text
   trade-off.
4. The 60% and 512px arms add no native data information: the former is
   almost always the same crop and the latter is an upscale.
5. Legacy captions trade identity for a small text-score gain.
6. All arms avoid the original widespread pasted/displaced face failure.
7. All arms retain a strong Jisoo-specific failure cluster. Marion and small
   action faces also remain weaker, but less catastrophically.

## What the current results do and do not establish

### Established by observed evidence

- Tight reference formatting at inference can causally trigger copied or
  misregistered face structure.
- Real surrounding context is much safer than blank padding.
- Cropped references can still be beneficial during training.
- Diverse target views are better than a repeated single target.
- Aggregate ID similarity can reward identity fragments and cannot replace a
  per-image anatomy review.
- The full-Cosmic loader, crop/bbox propagation, checkpoint evaluation, Comet
  export, and multistep full-96 path work correctly under the audited
  protocol.
- Using the complete existing 256px reference asset is sufficient for this
  matrix. The current assets cannot test wider real context or higher native
  resolution through larger margins or output resizing.
- A 3k stopping point is better than 4k for this matrix.

### Not established

- That the complete-256 reference policy generalizes beyond the eight full-96
  identities.
- That the Jisoo issue is caused by one specific reference image, bbox,
  PhotoMaker embedding, BA layer, or timestep.
- That branched cross-attention cannot be made useful. It was disabled in the
  successful matrix because earlier CA-on runs caused additional corruption.
- That combined SA+CA BA is healthy.
- That a long run will outperform the 3k candidate.
- That the current candidate beats a fully matched plain PhotoMaker baseline
  on full-96 while retaining a demonstrably useful reference-conditioned BA
  contribution.

## Recommended next experiments

### Priority 1 — clean reference-selection × target-scale factorial

Use one immutable accepted-target manifest with **no self-reference fallback**.
Compare highest-score versus top-three score-weighted distinct references,
crossed with:

1. target face ≥256px throughout; and
2. a scale-balanced curriculum that oversamples ≥256px faces for 4–6k, then
   introduces 192–255px faces in balanced bins.

Audit reference candidates jointly on ArcFace score, pose difference,
occlusion, blur, and native resolution. Highest ArcFace alone may select a
near-duplicate view that encourages literal spatial copying. Do not repeat
40%/60% margin or 512px-upscale arms on the current 256px face assets.

### Priority 2 — bounded top-three continuation

If the user wants to test whether the late recovery continues, resume
top-three from its exact 20k checkpoint. Highest-score is an optional second
arm. Give the run a 50k maximum budget but validate at 22/24/28/32k and stop
unless identity exceeds 20k, TOPIQ-Face coverage/p10 do not regress, and the
fixed jumping/dancing/skiing/reading hard set visibly improves. Do not run
either arm unchecked to 50k.

### Priority 3 — reference-conditioned BA routing/alignment

First localize the failure with fixed-checkpoint branched-SA
layer/resolution/timestep-window ablations. Then test one-variable changes:

- a bounded per-layer/timestep gate on the reference-branch residual merge,
  regularized against collapsing the reference contribution to zero; or
- bbox-relative coordinate remapping of reference K/V and branch masks into
  the target-face frame.

Preserve target queries, explicit reference K/V, reference-face K/V weight
1.0, `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and CA-off. Do not use
target K/V substitution as a fix.

### Priority 4 — matched plain PhotoMaker and broader identity gates

Run exact full-96 inputs as plain PhotoMaker against step 0 and the selected BA
checkpoint, then add identities outside full-96 with difficult
hair/occlusion and small/action faces. This separates failures inherited from
PhotoMaker/reference preparation from failures amplified by spatial BA.

## Code and launch entry points

Run all Hydra and training commands from `diffusion_template/`.

Current data/reference policy:

- `src/datasets/cosmic_large_adapted.py`
- `src/datasets/reference_policy.py`
- `src/configs/cosmic_large_adapted_rhca.yaml`
- `tools/datasets/preflight_cosmic_large_adapted.py`

Current training launcher:

- `launchers/active/run_rhca_cosmic_large_adapted_1gpu.sh`

Current full-96 evaluation:

- `launchers/active/run_rhca_cosmic_full96_eval_1gpu.sh`
- `src/configs/cosmic_large_adapted_full96_eval_rhca.yaml`
- `src/configs/cosmic_large_adapted_full96_multistep_eval_rhca.yaml`
- `tools/inference/full96_protocol.py`
- `tools/inference/finalize_multistep_full96_eval_record.py`

Controlled one-ID reference policies:

- `launchers/active/run_rhca_cosmic_one_id_reference_policy_4k_1gpu.sh`
- `src/configs/controlled_identity_reference_policy_rhca.yaml`

Task D controlled factorial:

- `launchers/active/run_rhca_controlled_identity_factorial_4k_1gpu.sh`
- `src/configs/controlled_identity_factorial_rhca.yaml`
- `src/datasets/controlled_identity_factorial.py`

Before changing the attention subsystem, search the relevant files for
`AICODE-NOTE:`, `AICODE-TODO:`, and `AICODE-QUESTION:` anchors. Keep new
behavior behind toggles and verify both old and new composition.

## Experiment and Comet protocol

Every new experiment must have:

1. a unique run name and output directory;
2. a local experiment JSON describing the hypothesis, fixed controls,
   changed variables, machine, launcher/package, gates, and status;
3. `saved/<run_name>/comet_experiment.json` created at startup;
4. an immutable Comet experiment key copied into the local JSON;
5. metrics and images retrieved by immutable key, never by display name;
6. exact image counts and exact requested steps;
7. visual review separated from metric evidence;
8. preserved failed-start records rather than reused contaminated run names.

Use:

```bash
cd /home/kolyangg/rsrch_apr_test/diffusion_template
python tools/comet/comet_experiment.py --help
```

The local `comet_records/` cache is ignored by Git but contains the current
immutable validation records. The durable experiment specifications and Serv
packages are under `experiment_specs/`, `experiments/`, and
`serv_run_packages/`.

## Machine rules

### Neb

- One 80GB GPU; never overlap training and validation.
- A validation pass can consume about 79.3GB even when training itself appears
  to leave headroom.
- Inspect the complete process group and `nvidia-smi` before launching.
- Activate `photomaker_NS`, source `.env` without printing it, then set the
  correct server-local `PM_PATH`.
- Sync code deliberately; do not overwrite credentials or machine-local bbox
  files.

### Serv

- Write only below
  `/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/`.
- Count this project's own Running and Pending A100 requests by actual GPU
  count. The ceiling is six GPUs; other users do not count.
- Pending is a successful submission.
- If MLS rejects/discards a request for allocation limits, do not retry unless
  the user asks.
- Do not set `CUDA_LAUNCH_BLOCKING=1` in production.
- Require the CUDA ONNX Runtime overlay and fail closed rather than accepting
  CPU fallback.
- Do not sync or mutate a code checkout being read by live jobs; previous NFS
  stale-handle failures came from changing shared files during execution.
  Use immutable run packages or wait for jobs to finish.

## Known implementation caveats

- Alternate-base validation previously installed processors before
  propagating architecture flags. Commit `5e55450b...` fixed flag propagation
  and the controlled-validation DataLoader.
- Some historical validation emitted an
  `AttnProcessor2_0 ... has no attribute parameters` warning after installing
  self-attention processors. The SA path was present, but the catch-all can
  hide future partial-installation failures. New critical evaluations should
  assert exact installed processor counts.
- The original 12-image endpoint is not enough for promotion. Eight-identity
  full-96 exposed failures that the Eddie-only panel missed.
- Step-0 images must be identical across matched arms. If they differ, the
  validation contract has drifted.
- Do not invent CA-on weights for a CA-off checkpoint.
- Do not compare 12-image aggregate metrics directly with 96-image metrics.

## Face-quality metric backfill status (27 July 2026)

- Neb baseline validation key `658d22341cf24accb5a3890869e76c28`
  has seven compact `face_quality/` curves at all 13 full-96 steps. Its
  1,248-row per-image CSV is retained locally under
  `analysis/assets/face_quality/neb_baseline_658d22341cf24accb5a3890869e76c28/`
  and attached as API asset `26160c7a6a18404a8087de4bdb67290e`.
- The four Serv validation keys passed preflight with exactly 96 images at all
  13 steps and no existing face-quality metrics/tables. Both planned Serv GPU
  submissions were rejected before job creation with
  `WORKSPACE_GPU_LIMIT_REACHED_ONLY_0_FREE`.
- Under an explicit exception, Serv staged all 4,992 images and transferred
  them to Neb with per-file size/SHA-256/PIL verification. Neb processed the
  four runs sequentially under PGID `812861`.
- All four validation keys now have exactly seven `face_quality/` curves × 13
  steps plus one API-only 1,248-row per-image CSV asset; the independent audit
  found zero legacy metrics and zero table assets.
- A post-run audit found no credential or key file/pattern in either staging
  tree; Serv and Neb retained distinct machine-local `.env` files.
- A bounded Serv compatibility smoke test was prepared for the uniform
  step-0/full-96 panel using one A100 and no Comet writes. Its single submission
  at 14:07 UTC was rejected before job creation with
  `WORKSPACE_GPU_LIMIT_REACHED_ONLY_0_FREE` and cancelled without retry, as
  requested. Record:
  `experiments/cosmic_large_continuation/serv_face_quality_uniform_step0_smoke.json`.
- Durable status:
  `experiments/cosmic_large_continuation/serv_four_validation_face_quality_backfill.json`.

## Default in-pipeline validation (27 July 2026)

- The standard trainer configuration now validates at step 0 and every 2,000
  optimizer steps. The interval must divide `trainer.epoch_len` exactly.
- New training launchers default to 2,000 optimizer steps per epoch, so each
  epoch ends at a validation/checkpoint gate. Historical scripts that address
  immutable 500-step checkpoint epoch numbers pin `TRAIN_EPOCH_LEN=500`.
- Standard Cosmic Large configs use the fixed 96-image `manual_val` panel and
  one generated image per item. Explicit historical one-identity protocols
  remain 12-image exceptions.
- The canonical seven face-quality metrics run at every actual validation
  event by default behind `trainer.face_quality.enabled`. They use the same
  standalone PyIQA 0.1.15 scorer and definitions as the completed backfill.
- Comet receives seven `face_quality/` scalar curves and one API-only
  per-image CSV asset per validation step; no table is created.
- Full configuration and machine-environment behavior are documented in
  `docs/validation_protocol.md`.

## Detailed document index

- [Tasks A–D implementation request](../../analysis/2026-07-24_cosmic_large_next_steps_implementation_handoff.md)
- [Tasks A–D consolidated results](../../analysis/2026-07-25_cosmic_large_tasks_a_d_results_handoff.md)
- [Cosmic Large recommendations and experiment design](../../analysis/2026-07-25_cosmic_large_training_recommendations_and_experiments.md)
- [Experiment launch plan](../../analysis/2026-07-25_cosmic_large_experiment_launch_plan.md)
- [Full-Cosmic 4k/full-96 report](../experiments/2026-07-26_cosmic_large_adaptation_4k_full96_results.md)
- [Architectural reset and ratio-zero runs](../experiments/2026-07-26_ba_ratio_zero_reference_policy_runs_handoff.md)
- [Final four-run multistep full-96 report](../experiments/2026-07-26_current_four_full_cosmic_4k_runs_handoff.md)
- [Initial-usage five-run 20k analysis and full-96 PDF](../experiments/2026-07-27_cosmic_large_initial_usage_20k_analysis.md)
- [Initial-usage 20k continuation provenance](../experiments/2026-07-27_cosmic_large_initial_usage_20k_continuations.md)
- [Serv face-quality backfill](../experiments/2026-07-27_cosmic_large_serv_face_quality_backfill.md)

Representative validation images are stored beside those reports under
`docs/experiments/assets/`. The final four-run multistep contact sheets are
under `docs/experiments/assets/2026-07-26_full96_multistep/`.

## New-agent startup checklist

1. Read this file completely.
2. Read `AGENTS.md` and `TOOLS.md`.
3. Check branch, commit, and dirty status locally and on the target machine.
4. Recheck Neb GPU/process state and Serv Running+Pending allocations.
5. Identify experiments only through their JSON plus immutable Comet key.
6. Inspect the five-run 20k PDF, especially the jumping/dancing/skiing hard
   cases, before proposing a long run.
7. State whether a proposed experiment preserves effective
   reference-conditioned BA.
8. Start with the clean dataset factorial unless new user direction supersedes
   it; if longer training is requested, gate top-three early.
9. Update this file when a material result changes the current decision.
