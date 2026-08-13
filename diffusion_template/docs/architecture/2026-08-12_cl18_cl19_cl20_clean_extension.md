# CL18, CL19 and CL20 clean extension

- **Date:** 12 August 2026
- **Branch:** `kit/e13-family-clean`
- **Immediate parent before this extension:**
  `704d4914554cce6bd3c5b098b02df967de48dd2a`
- **Underlying clean base:**
  `2157eada14824d14019e80f9416e6d736c837306`
  (`code clean-up - restore 1 ref only`, 2 June 2026 21:58 BST)
- **Source inspected:** test working tree at tracked commit
  `ad194a026ab701dd979712d415c487dd536a4645`, including its uncommitted
  corrected-r2 CL15-CL20 snapshot from 11 August 2026
- **Supported additions:** CL18, CL19 and CL20 only
- **State:** implemented and locally audited; no training job launched

## Executive outcome

The clean branch now supports the three requested corrected-r2 experiments
without merging the test branch or importing CL15, CL16, CL17, or later
architecture families. The existing E13, BC_E13 and CL14 recipes remain
available and their sealed generation gate still passes.

The implementation keeps the concerns deliberately separate:

```text
CL14 clean contract
├── CL18: training objective + alternate same-ID spatial reference
├── CL19: self-attention routing equation only
├── CL20: deterministic training dataset schedule only
└── shared subject-owned validation + inherited speed profile
```

Evidence labels used below:

- **[code]** direct inspection or deterministic local source comparison;
- **[record]** immutable local experiment JSON or handoff evidence from the
  source tree;
- **[measured]** a command executed in this clean worktree.

No new image-quality conclusion is made here. The source records establish
that corrected-r2 reached the fixed step-0 96-image validation and entered the
training loop; they do not, by themselves, establish final model quality.

## Fixed contract retained by all three arms

All three configs inherit CL14 and retain:

| Contract field | Value |
|---|---|
| optimizer steps | 24,000 |
| train batch | 2 |
| epoch/checkpoint boundary | 2,000 optimizer steps |
| validation | step 0 and every 2,000 steps |
| panel | fixed `manual_val`, 96 images, one image/item |
| validation base | `SG161222/RealVisXL_V4.0` |
| scheduler / inference | DDIM, 50 steps, CFG 5 |
| PhotoMaker / BA onset | steps 10 / 15 |
| processor copy | strict `legacy_full_copy` |
| shadow PhotoMaker default | enabled |
| reference routing | `pose_adapt_ratio=0` |
| face CA mixing | disabled |
| trainable contract | 2,240 tensors / 219,217,920 parameters |

The shared extension config is only 18 lines. Each leaf is 13-21 lines and
contains the experiment-specific switch rather than copying the CL14 config.

## Experiment-specific changes

### CL18: alternate-view spatial consistency

Config:
`CL18_cosmic_crossview_spatial_consistency_24k.yaml`.

CL18 retains the CL14 model and single-reference inference path. On 25% of
training batches it performs one additional student forward using a distinct
same-identity spatial reference while keeping the target latents, sampled
target noise, timesteps, prompt embeddings, PhotoMaker identity tokens, and
paired reference noise fixed. The ordinary prediction is detached as the
teacher. The face-local auxiliary is:

```text
SmoothL1(student, stop_gradient(teacher))
  + 0.10 * (1 - cosine(student, stop_gradient(teacher)))
```

It is multiplied by `0.05` before being added to the diffusion loss. The
dataset requires at least three reference candidates for the target and emits
one alternate candidate distinct from the primary spatial reference. Both
views independently use the existing CL14 target-frame scale/position policy.

Why: encourage the trained spatial BA path to be less dependent on which
valid same-ID reference view was sampled, without changing PhotoMaker tokens
or adding an inference-time module. **[code]**

Expected runtime effect: inference cost is unchanged. Roughly one quarter of
training batches execute an additional U-Net pass, so CL18 is intentionally
slower than CL14 despite retaining the inherited conditioning and mask-cache
speed improvements. This cost statement is architectural, not a measured A100
benchmark.

### CL19: true soft full-query router

Config:
`CL19_cosmic_true_soft_fullquery_router_24k.yaml`.

CL19 opts every spatial self-attention group (`down0/1/2`, `mid`,
`up0/1/2`) into one routing equation:

1. Compute the complete native target-Q/target-KV attention message.
2. Compute the complete target-Q/reference-KV message. Reference features
   outside the binary reference face mask remain zero sinks exactly as in the
   source implementation.
3. Derive a two-latent-cell cosine transition from the binary target mask.
4. Blend the two complete messages exactly once at the target.

This removes CL14's thresholded, twice-applied boundary weighting. It adds no
parameters; the rank-128 BA and both effective adapter groups remain the same.
The isolated extension wrapper performs the corrected-r2 subject-owned
validation selection and copies the checkpointed router mode/groups into the
otherwise sealed CL14 pipeline. It does not edit the CL14 pipeline
implementation. **[code]**

A deterministic CPU fixture initialized source and clean processors with the
same attention/LoRA state, two masks and input tensor. State tensors and output
were bit-exact (`rtol=0`, `atol=0`); output shape was `(4, 16, 8)` and checksum
sum `-78.71554565429688`. **[measured]**

Expected runtime effect: CL19 computes both complete target messages in each
selected SA processor. It should not be described as a speed improvement; no
new A100 throughput measurement was made.

### CL20: Cosmic/BigCelebs hard-case curriculum

Config:
`CL20_cosmic_bigcelebs_hardcase_curriculum_24k.yaml`.

CL20 leaves the CL14 model, optimizer and loss route off/unchanged. It replaces
only the shuffled Cosmic loader with a hash-sealed sequential 48,000-row
schedule (two rows per optimizer step):

| Phase | Rows | Optimizer steps | Cosmic | BigCelebs |
|---|---:|---:|---:|---:|
| mixed | 0-39,999 | 0-19,999 | 32,000 | 8,000 |
| re-anchor | 40,000-47,999 | 20,000-23,999 | 8,000 | 0 |

The 8,000 BigCelebs rows require identity depth at least six and rotate across
2,667 synthetic-small-face, 2,667 occlusion-caption and 2,666 action-caption
rows. Schedule seed is `200020`. Corrected-r2 schedule SHA-256 is:

`783eb1729871e4ac423c770042315572ee7ea24171797402fc4a565999dd5289`.

The launcher builds the schedule from the sealed inputs and rejects any other
hash. The loader verifies both source-manifest hashes, summary-to-schedule
binding, exact row schema/order, distinct target/reference paths, all image
paths, phase counts, and resume offset. DataLoader shuffling is disabled only
for datasets that explicitly require sequential sampling; historical datasets
remain shuffled. **[code]**

Why: spend a controlled 20% of the first 20k steps on repeated-view hard cases,
then return to Cosmic for the final 4k rather than replacing the stronger
Cosmic distribution.

## Pipeline changes versus dataset changes

This boundary is intentional and should be preserved during later work.

### Pipeline/model side

- `attn_processor_cleanest.py`: defaults-off CL19 full-message router
  (`+141` lines versus the previous clean commit).
- `branched_runtime.py`: installs the router only for the seven declared CL19
  groups (`+22` lines).
- `lora2.py`: CL18 alternate forward/loss and the three small defaults-off
  controls (`+110` lines).
- `lora2_helpers.py`: forwards an explicit paired reference-noise tensor
  (`+2` lines).
- `e13_contract.py`: records/rejects router or cross-view checkpoint drift;
  the off manifest remains the previous hard-v1 manifest.
- `photomaker_branched_cl18_cl20.py`: isolated subject-v2 validation wrapper
  and router factory; the sealed `photomaker_branched_clean.py` and
  `br_pipeline_helpers.py` are not edited.

### Dataset side

- `cosmic_large_adapted.py`: defaults-off second same-ID reference for CL18
  (`+75` lines); CL14 keeps its previous one-reference behavior.
- `cl20_hardcase_curriculum.py`: exact sequential schedule consumer (219
  lines, copied from corrected-r2 source).
- `build_cl20_hardcase_schedule.py`: deterministic schedule builder (277
  lines, copied from corrected-r2 source).
- `bc_e13_schedule_policy.py`: retained torch-free strict BigCelebs manifest
  validation used by the builder.
- `data_utils.py` and `base_trainer.py`: sequential-only loader selection and
  checkpoint/schedule offset validation. These activate only for CL20.

The dataset schedule files are runtime artifacts under
`preflight_records/<run_name>/`; they are not committed.

## Validation subject and metric correction

Corrected-r2 binds PhotoMaker validation conditioning to the face overlapping
the declared reference bbox. The wrapper computes that reference embedding and
passes it explicitly into the unchanged CL14 pipeline; it fails closed if no
face is detected. This changes the multi-face validation reference in the same
way as corrected-r2 while leaving training conditioning at the historical
model default. **[code] [record]**

Corrected-r2 also uses subject-v2 as the primary identity **metric** and retains
the historical max-over-any-face value as an audit metric:

- `id_sim`: generated face selected by overlap with the exact BA generation
  bbox, scored against the sealed subject-v2 reference embedding;
- `id_sim_legacy_best`: best identity over any generated face;
- diagnostics: mask IoU, face count, no-face, unowned and ambiguity rates.

This extension therefore adds `face_subject_selector.py`, the validation
wrapper, `IDSimMaskMatched`, the subject-v2 metric config, and passes the exact
resolved generation/reference boxes to metrics. The precomputed subject-v2
metric artifact must have SHA-256
`e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465`.

Crucially, this policy is scoped to validation through
`validation_args.face_subject_selection_policy=bbox_overlap_v2`; the training
model's Cosmic conditioning remains historical. The precomputed metric
artifact is not reused as conditioning; validation embeddings are derived from
the exact reference pixels and declared bbox, matching the source helper.

## Training-efficiency behavior

The three arms inherit the already documented E13-family speed profile:

| Existing switch | Retained value/effect |
|---|---|
| batched frozen conditioning | enabled |
| unused text-only conditioning | skipped |
| per-forward resized-mask cache | enabled |
| diverse-pair conditioning cache | disabled |
| branch debug outputs | disabled |
| post-backward zero touches | disabled |
| grad-norm reductions | logging steps only |
| face-quality scoring | deferred until successful training |
| CUDA launch mode | asynchronous; blocking mode rejected |
| process count | one Accelerate process / one A100 |

The extension adds no host-device synchronization to the normal CL14 path.
CL20 constructs the deterministic schedule before loading the model and then
uses a sequential DataLoader. CL18's optional second forward and CL19's second
complete attention message are scientific costs and are documented separately
instead of being mislabeled as optimizations.

The launcher now also verifies that a 32-character immutable Comet key exists
in `saved/<run_name>/comet_experiment.json` during startup. If registration
does not complete within ten minutes, it terminates the run before it can
become an untraceable experiment.

## Provenance and source records

The implementation was ported path-by-path; the dirty test branch was never
merged into the clean branch.

| Arm | Corrected source record | Serv job | Immutable Comet key |
|---|---|---|---|
| CL18 r2 | training loop started after fixed step-0/96 | `lm-mpi-job-1c4dd150-9688-4ca0-b678-8f74134a70e7` | `f6530436bf22472c9fb7731d1696c5ab` |
| CL19 r2 | training loop started after fixed step-0/96 | `lm-mpi-job-f1b9d006-208c-4b35-8e4a-ab0ab2f030a9` | `cfeda7b55c174b3c83e8d40537ebb6dd` |
| CL20 r2 | training loop started after fixed step-0/96 | `lm-mpi-job-1e0f08fd-b0d3-4b26-9167-5d55103f442d` | `b05488e2cce94476acc92bcaa21d7362` |

The r1 packages failed before scientific validation and are not evidence for
model behavior. The table reports what the immutable JSON files recorded on
11 August; it is not a live job-status claim.

Corrected-r2 source hashes used for the port audit (the four Python helpers
below are copied byte-for-byte; the clean leaf configs inherit a different
minimal shared base):

| File | SHA-256 |
|---|---|
| source CL18 leaf config | `b78995369194b3ec499f22f4760a70b5ab72f0dcdcf5af5bd8a19ed5ef9cd633` |
| source CL19 leaf config | `9f907615e34cc4877f218985f28b01621858c8e22fad3a082c500622489a3565` |
| source CL20 leaf config | `1b5e28d105333f82416a3d34a7acddb6b678322aecb84575fb1c1d43ff3636c7` |
| `cl20_hardcase_curriculum.py` | `953a8cd0a2449ebee7476de68591377eafb6782d855feec044810cc6a0aab042` |
| `build_cl20_hardcase_schedule.py` | `8866d41cf2fa5d1bf63fd43199f5cc9a58b05b966bd1f8b59ca5aec3f0175ff8` |
| `preflight_cl20_curriculum.py` | `4c33baad24ae3cc15c9ec4ab8bc780e40670e6dfa4d2be05639f69dbe2a2acca` |
| `face_subject_selector.py` | `4e14aa3a62c24ebae7708a9fbfaf32b8e1801f4a9444135b8a448b6f0e8733a4` |

## Verification performed

| Gate | Result |
|---|---|
| changed Python compilation | passed |
| active/Serv launcher shell syntax | passed |
| whitespace/diff hygiene | passed |
| legacy family composition | E13, BC_E13 and CL14 passed |
| new composition | CL18, CL19 and CL20 passed the fail-closed validator |
| trainable declaration | all three resolve to 2,240 / 219,217,920 |
| CL14 sealed generation source/input gate | passed after the extension |
| CL14 processor off route | bit-exact against pre-extension branch state (`rtol=0`, `atol=0`) |
| CL18 dataset smoke | primary and distinct alternate references both realized as 1024 target-frame canvases with valid boxes |
| CL18 loss smoke | weighted auxiliary contributed to total loss and preserved student/aux gradients while telemetry stayed detached |
| CL19 processor source parity | bit-exact state and output fixture passed |
| subject-v2 validation wrapper | source and clean selected indices `[1, 0]`, returned `(2, 1, 512)`, and produced embedding sum `512.0` |
| CL20 schedule/tool source bytes | exact corrected-r2 hashes above |
| CL20 synthetic schedule smoke | two builds were byte-identical; loader accepted 48,000 rows, counted 40,000 Cosmic / 8,000 BigCelebs, and decoded all eight phase-boundary probes |
| `.env` handling | ignored by Git; no credential file staged |

Not performed locally:

- no A100 training, timing benchmark, or MLS submission;
- no real Cosmic/BigCelebs decode preflight because their machine-local paths
  are not present in this checkout's `.env`;
- no 48k real schedule regeneration for the same reason;
- no historical checkpoint/full-96 RGB replay because checkpoints and the
  pinned GPU runtime are not local.

These are explicit server-side pre-launch/startup gates below. Therefore the
strongest current claim is code/config/source parity, not a newly measured
full-image replay.

## Server runbook

Run all commands from `diffusion_template/`. Do not use Neb; it is unavailable.

### 1. Prepare the checkout and environment

```bash
git fetch origin
git switch kit/e13-family-clean
git pull --ff-only origin kit/e13-family-clean
conda activate /absolute/path/to/photomaker_NS
```

Keep machine paths and credentials in the ignored `.env`. Start from
`.env.example` and set at least:

```text
COMET_API_KEY
FACE_QUALITY_SCORER_PYTHON
PM_PATH                         # optional only if model default is valid
COSMIC_LARGE_MANIFEST
COSMIC_LARGE_ROOT
COSMIC_LARGE_EXPECTED_MANIFEST_SHA256
SUBJECT_V2_ID_EMBEDS
BIG_CELEBS_MANIFEST             # CL20 only
BIG_CELEBS_IMAGES               # CL20 only
BIG_CELEBS_EXPECTED_MANIFEST_SHA256  # CL20 only
```

The launcher hard-requires corrected-r2 Cosmic hash
`8ba369ef2fdc0496a0d3d55afb5c7923c1aa299343a676ac6bc0d94f3a3a0196`,
sealed BigCelebs-v2 hash
`f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa`,
and the subject-v2 hash above. Never commit `.env`.

### 2. Run local pre-launch gates

```bash
python tools/validate_e13_family_config.py
python tools/verify_cl14_generation_parity.py

python tools/validate_cl18_cl20_config.py \
  --config-name CL18_cosmic_crossview_spatial_consistency_24k
python tools/validate_cl18_cl20_config.py \
  --config-name CL19_cosmic_true_soft_fullquery_router_24k
python tools/validate_cl18_cl20_config.py \
  --config-name CL20_cosmic_bigcelebs_hardcase_curriculum_24k

bash -n launchers/active/run_e13_family_24k_1gpu.sh
bash -n launchers/serv/start_e13_family_1gpu.sh
```

The production launcher repeats the relevant gates, decodes a configured
Cosmic sample for CL18/19, and builds plus boundary-decodes the exact CL20
schedule before loading the model.

### 3. Direct one-A100 launch

Set one unique `RUN_NAME` and one exact `CONFIG_NAME`; the launcher rejects all
extra Hydra arguments:

```bash
RUN_NAME=CL18_clean_r1 \
CONFIG_NAME=CL18_cosmic_crossview_spatial_consistency_24k \
bash launchers/active/run_e13_family_24k_1gpu.sh
```

Replace both names together for CL19 or CL20. Do not run multiple arms under
one name and do not reuse an existing `saved/<run_name>` or preflight directory.

### 4. MLS/Serv submission

Before submission, inspect this project's Running and Pending MLS jobs and
respect the normal six-A100 ceiling. Use the exact CL18, CL19 or CL20 YAML
linked from `serv_run_packages/README.md`, review its clean-checkout paths and
unique run identity, then submit it with the Serv CLI. Each YAML delegates to
`launchers/serv/start_e13_family_1gpu.sh`, which activates the existing
environment and then calls the same audited launcher. The generic template
remains available only for an intentionally new package identity.

Do not submit as part of this implementation handoff. Submission requires an
intentional later action after resource inspection.

### 5. Startup evidence to retain

For every launched arm, keep:

- `preflight_records/<run_name>/` including the CL20 schedule/summary when
  applicable;
- `saved/<run_name>/comet_experiment.json` with its immutable key;
- the complete step-0 fixed-96 panel;
- the resolved Hydra config and architecture manifest;
- the server stdout/stderr containing `COMET_STARTUP_VERIFIED`.

Before promoting this port as generation-equivalent, compare the step-0 panel
against the corresponding corrected-r2 protocol/checkpoint source or run a
full fixed-96 RGB replay with the immutable checkpoint. Do not substitute
similar prompts, references, bbox caches, scheduler settings, or metrics.

## Handoff checklist

An implementing or operating agent should:

1. Read `docs/handoffs/LATEST.md` first.
2. Keep CL18 training-only and single-reference at inference.
3. Keep CL19 as a one-time blend of two complete messages with binary
   reference sinks.
4. Keep CL20 model mode `off`, DataLoader shuffle false, and the exact schedule
   hash.
5. Keep bbox-owned PhotoMaker selection scoped to validation, and keep the
   subject-v2 metric artifact separate from conditioning.
6. Preserve `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, fixed-96 inputs,
   DDIM50, CFG5, RealVis, and immutable Comet identity.
7. Run the smallest fail-closed gates above before any GPU allocation.
