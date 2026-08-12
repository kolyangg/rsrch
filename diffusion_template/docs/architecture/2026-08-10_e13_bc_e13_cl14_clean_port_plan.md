# Clean E13-family port plan: E13, BC_E13, and CL14

- **Date:** 10 August 2026
- **Status:** implementation plan only; no model, pipeline, trainer, dataset, or
  configuration code has been changed as part of this plan
- **Scope:** create a concise branch based on the last 2 June `main_clean`
  snapshot, support E13, BC_E13, and CL14, preserve the runtime improvements
  used by those runs, and keep later reusable tools/datasets without importing
  later experimental model mechanisms
- **Evidence cutoff:** completed E13, BC_E13, and CL14 runs and repository state
  available on 10 August 2026

## Executive decision

Create a new branch, recommended name `kit/e13-family-clean`, from
`2157eada14824d14019e80f9416e6d736c837306` (`code clean-up - restore 1 ref
only`, 2 June 2026 21:58 BST). Do **not** merge `test`, and do not update the
existing `main_clean` branch until the clean port passes replay gates.

Implement one shared E13-family architecture and expose three thin recipes:

| Recipe | Core architecture | Training dataset policy | Training-only model delta |
|---|---|---|---|
| E13 | shared E13 contract | `LargeDatasetTrain` | none |
| BC_E13 | exactly E13 | `BigCelebsTrain` | none |
| CL14 | exactly E13 | CL9 Cosmic reference policy | `ba_training_mask_feather=2` |

The implementation must keep four concerns independent:

1. **Core model/training contract:** hard spatial BA, trainable ownership,
   optimizer groups, checkpoint fidelity, and CL14's training mask.
2. **Inference/validation pipeline:** single spatial reference, step routing,
   reference preparation, validation shadowing, and versioned validation
   semantics. It must contain no Cosmic- or BigCelebs-specific branch.
3. **Dataset use:** Large Dataset pairing, BigCelebs pairing, and the corrected
   Cosmic reference construction/caption policy. Cosmic transforms must finish
   before the model receives a batch.
4. **Efficiency/runtime:** batched frozen conditioning, removal of unused work,
   asynchronous CUDA, fail-closed CUDA InsightFace/ONNX Runtime, DataLoader
   settings, and deferred face-quality scoring. These switches must be explicit
   and verified separately from scientific behavior.

This boundary is the central design requirement:

```text
thin Hydra recipe
├── shared E13 core contract
├── one dataset recipe
├── one validation profile
└── one explicit efficiency profile

dataset emits final ref image + ref bbox + cache key
                 │
                 ▼
dataset-agnostic model / pipeline
```

## 1. Source-of-truth hierarchy

Use sources in this order. The dirty `test` working tree is a comparison aid,
not an authoritative snapshot.

| Priority | Source | What it is authoritative for |
|---:|---|---|
| 1 | `2157eada14824d14019e80f9416e6d736c837306` | clean June base and single-reference design |
| 2 | `e860f9ed4d021226575845ae24a9fda1e5a3fa58` | strict processor installation, exact trainable contract, schema-v2 checkpoint substrate |
| 3 | `dd65dec312271610e46dd507ee010a5927b8cbda` | effective generic/default adapter scopes and hard-v1 rank override |
| 4 | `8b8b9abd726df111ce725b6c283869c3dd19e6a0` | E13 joint optimizer grouping and PhotoMaker-default validation shadow |
| 5 | `ebf1ac8295f363adb0055cd74db1a96c2ff03a35` | exact successful E13 r4 runtime tree |
| 6 | `ad194a026ab701dd979712d415c487dd536a4645` | BC_E13 source tree and shared E13-family state before the CL14 sealed overlay |
| 7 | sealed revision `c04970f342a186d1092f07f9a08d7d8a797383e8+cl12-cl14-snapshot-v1-20260809` | exact CL14 source, Cosmic loader/policy, leaf config, and mask-feather implementation |

CL14 snapshot provenance:

- source manifest: 1,220 files;
- source-manifest SHA-256:
  `d43fa65815aa4fc4c106f6ed3e939b5dee690f2a2927b43a022a8e6025ccc294`;
- resolved saved-config SHA-256:
  `642cdcb4acd2b4fcf0ef9fd5fadaa5fb5a117c092b1e07394b8e9c2fd406b2c4`;
- 24k weights SHA-256:
  `0de10ec611c8a5e55e0b362ea90fa348fe686ca5d949f762752bc1add7992ed9`.

Do not cherry-pick any of the large source commits wholesale. Each includes
experiments outside this plan. Inspect their relevant hunks with `git show` or
path-limited `git diff`, then reimplement only the ledger items in section 8.

## 2. Observed run provenance

These records establish what the clean port has to reproduce. Historical
identity values use the validation contract active during each run; do not
silently reinterpret them as subject-v2 values.

| Run | Immutable Comet key | Observed endpoint | Source |
|---|---|---:|---|
| E13 r4 | `1cc0a02371094b24a6a02a4cc649f10c` | `id_sim=.399799` at 24k | clean commit `ebf1ac8295f363adb0055cd74db1a96c2ff03a35` |
| BC_E13 r1 | `c138db7c41ae435c8a7560f40cf5f58d` | peak `.399010` at 16k; `.389430` at 24k | commit `ad194a026ab701dd979712d415c487dd536a4645` |
| CL14 r1 | `6fe0028be92242c38056b3d36665fdd6` | `id_sim=.422475` at 24k | sealed CL14 snapshot |

Additional CL14 endpoint evidence `[measured/report]`:

- all `96/96` generated images had detected faces from 6k through 24k;
- 24k text similarity was `26.3413`;
- 24k TOPIQ-Face mean was approximately `.6806`, with p10 approximately
  `.5810`;
- CL14 exceeded CL9's `.41513` identity promotion threshold by about `.00735`.

This is evidence that CL14 is promising, but it does **not** establish that
mask feathering alone caused the endpoint gain or that final IoU/blending was
better. CL14 was not a controlled dataset comparison against E13, and its
historical identity metric used the legacy subject selector.

## 3. Shared E13-family contract

The following must resolve identically for all three recipes except for the
explicit CL14 mask-feather field:

- `ba_architecture_version=hard_replace_v1`;
- branched self-attention enabled and branched cross-attention disabled;
- `train_on_separate_image=true`, `train_ba_all_steps=true`, and
  `train_ba_only=true` under the audited allowlist semantics;
- `branched_attn_weight_mode=noise_and_ref`;
- `branched_attn_new_weight_kind=lora` and
  `ba_training_timestep_policy=uniform_all`;
- reference-only hard routing enforced, with the historical true-reference-key
  mask disabled;
- `ba_patch_top_k=1.0`, `ba_train_top_k=1.0`, `non_ba_train=false`, and
  `strict_face_routing=false`;
- hard-v1 branched LoRA rank `128`;
- generic effective adapter rank `32`, scope `effective_all`;
- PhotoMaker-default effective adapter rank `64`, scope `effective_all`;
- strict, fail-closed processor installation and trainable ownership;
- checkpoint state format `trainable_v2`;
- `pose_adapt_ratio=0.0` and `ca_mixing_for_face=false`;
- `loss_kind=masked_alternating`, `masked_loss_step=1`, `lambda_face=0.1`;
- SDXL base training model with PhotoMaker v2 in bf16, and the historical
  RealVisXL V4 alternate validation base;
- 24,000 optimizer steps, batch size 2, 2,000-step epochs;
- validation and checkpoint at step 0/every 2,000 steps;
- `WarmupHoldCosineLR`: warmup 20, hold 14,000, total 24,000, minimum factor
  `.1`;
- inference: DDIM, 50 steps, CFG 5, PhotoMaker from step 10, BA from step 15;
- inference `mask_expansion_ratio=1.0` and `mask_softness=0.0`;
- fixed full-96 manual validation panel with the historical prompts, seeds,
  references, bboxes, and one image per item;
- `automatic_bboxes=true` with the historical run-specific derived generation
  bbox map; do not substitute a similarly named canonical/manual seed map;
- validation restores the pretrained PhotoMaker-default adapter after loading
  the trained checkpoint, while preserving the trained weights in the saved
  checkpoint;
- validation processor base mode `legacy_full_copy`, with 70 expected stateful
  processors.

Expected trainable ownership:

| Group | Tensors | Parameters | Optimizer LR |
|---|---:|---:|---:|
| hard spatial BA rank 128 | 840 | 127,795,200 | `1e-4` |
| generic effective adapter rank 32 | 700 | 30,474,240 | `1e-4` |
| PhotoMaker-default effective adapter rank 64 | 700 | 60,948,480 | `1e-4` |
| **Total** | **2,240** | **219,217,920** | — |

The strict installer and complete checkpoint schema are correctness fixes, not
optional cleanup. Earlier warning-and-continue installation could encounter a
plain `AttnProcessor2_0`, silently leave broad trainables active, and save an
incomplete checkpoint. The clean branch must fail before optimizer creation if
the expected ownership cannot be proven.

## 4. File-change risk map

The raw counts below compare the current `test` working tree with the 2 June
base. They identify review risk, **not** how much code should be copied. The
current tree also contains post-CL14 subject-v2 and diagnostic work; the exact
CL14 snapshot is narrower.

| File | June lines | Current lines | Raw diff vs June | Clean-port rule |
|---|---:|---:|---:|---|
| `src/model/photomaker_branched/lora2.py` | 688 | 2,341 | `+1700/-47` | highest risk; never copy wholesale; port only E13 contract hooks and CL14 target-mask feather |
| `src/model/photomaker_branched/lora2_helpers.py` | 315 | 964 | `+773/-124` | replace with a hard-v1-focused subset; exclude v2/v3/v4 and identity-CA branches |
| `src/trainer/base_trainer.py` | 968 | 1,486 | `+741/-223` | port validation shadow, strict processor copy, efficiency flags, and required full-96 plumbing only |
| `src/model/photomaker_branched/branched_runtime.py` | 653 | 1,214 | `+658/-97` | retain hard-replace v1 only; exclude residual/anchored/query-adaptive/identity-CA dispatch |
| `src/datasets/cosmic_large_adapted.py` | absent | 577 | `+577` | dataset-only module; preserve exact CL14 behavior, but do not leak its policy into model/pipeline |
| `src/trainer/sdxl_trainers.py` | 822 | 1,040 | `+269/-51` | port only E13 loss/batch fields and validation data propagation |
| `src/model/photomaker_branched/attn_processor_cleanest.py` | 757 | 931 | `+228/-54` | port only audited hard-v1 rank-128 routing and mask/cache behavior |
| `train.py` | 403 | 528 | `+202/-77` | port explicit optimizer groups and expected-contract assertion |
| `src/pipelines/br_pipeline_helpers.py` | 1,148 | 1,169 | `+160/-139` | pipeline-specific review; exact CL14 source predates current subject-v2/visibility-mask additions |
| `src/pipelines/photomaker_branched_clean.py` | 1,333 | 1,341 | `+14/-6` | keep near the June file; only proven single-reference/validation fixes |
| `src/configs/datasets/all_datasets.yaml` | 192 | 331 | `+221/-82` | rebuild a concise registry containing only supported datasets plus deliberately retained helpers |
| `src/datasets/reference_frame.py` | absent | 186 | `+186` | isolated Cosmic reference construction; suitable for a direct, hash-reviewed port |
| `src/datasets/large_dataset.py` | absent | 166 | `+166` | concise direct port after fixture check |
| `src/datasets/big_celebs.py` | absent | 99 | `+99` | concise direct port after sealed-release checks |

The largest files are large mainly because many later mechanisms share the same
switchboard. The clean implementation should move E13-specific ownership and
serialization into a focused helper such as
`src/model/photomaker_branched/e13_contract.py`, leaving `lora2.py` as a small
integration surface.

## 5. Workstream A: core model and training contract

### A1. Processor installation and hard-v1 runtime

Port only the hard-replacement self-attention route used by E13:

- install branched processors before optimizer construction;
- patch the expected self-attention sites only;
- keep target queries and explicit reference K/V routing visible in code;
- reject branched CA for these recipes;
- support rank 128 for the branch projections;
- propagate mask tensors and the exact single-reference batch layout;
- make an installation error fatal under the E13 profile;
- record the ordered processor-name list and a stable hash in the checkpoint
  manifest.

Do not port:

- `residual_sa_v2`;
- `anchored_mix_sa_v3`;
- `query_adaptive_hard_sa_v4`;
- identity CA v2/v3;
- ArcFace auxiliary loss;
- branch-output, true-key-mask, ROI-warp, reference-dropout, or visibility-mask
  experiments not selected by E13/BC_E13/CL14.

### A2. Exact ownership and optimizer groups

Add one allowlist function that derives the 840/700/700 tensor sets from names
and adapter roles. Freeze everything first, enable only this allowlist, then
assert:

- no expected name is absent;
- no unexpected parameter has `requires_grad=True`;
- every trainable appears in exactly one optimizer group;
- tensor and parameter totals match the table in section 3;
- all three groups use `1e-4` unless a future named recipe deliberately changes
  it.

Avoid a generalized multi-architecture role system. It is the main source of
current helper-file size and is unnecessary for the requested family.

### A3. Checkpoint schema

Save every trainable U-Net tensor by its exact name plus a concise architecture
manifest. On load, fail closed on:

- architecture/profile mismatch;
- missing or unexpected names;
- shape mismatch;
- processor-name/hash mismatch;
- adapter-scope or rank mismatch.

Retain schema-v1 loading only as an explicitly labelled historical fallback.
New E13-family checkpoints must always write v2.

### A4. Validation shadow

The trained PhotoMaker-default adapter must be present in checkpoints but
replaced with its pretrained 700-tensor snapshot inside the validation model.
After restoration, log and assert `restored=700`. Copy full processor state only
under `legacy_full_copy` and assert 70 stateful processors.

This behavior belongs to model/trainer validation construction, not to the
dataset and not to the inference pipeline's denoising equation.

### A5. CL14 training-mask delta

Add one defaults-off model argument, `ba_training_mask_feather: int = 0`.
Change only the target-side training `_bbox_to_mask` path:

- feather 0 must be byte-identical to the June binary rectangle;
- feather 2 must use the sealed CL14 inward ramp: outer latent ring `1/3`, next
  ring `2/3`, interior `1`;
- do not change `_bbox_to_ref_mask`;
- do not change inference mask construction;
- reject negative or unreasonably large values.

This is a **model/training** improvement. It is not a Cosmic dataset fix and it
is not an inference-pipeline soft mask.

## 6. Workstream B: inference and validation pipeline

Pipeline changes must be committed separately from dataset and model changes.
The initial implementation should retain the June pipeline and add only fixes
required to reproduce E13-family validation.

### B1. Dataset-agnostic input contract

The pipeline accepts only generic fields:

- final reference image(s);
- final `face_bbox_ref` in reference-image coordinates;
- target `face_bbox_gen`;
- prompt, seed, and optional precomputed identity embedding.

It must never inspect `train_dataset_name`, a Cosmic record, a BigCelebs
manifest, `reference_frame_mode`, caption mode, or reference jitter fields.

### B2. Single spatial reference

The supported E13-family route has exactly one spatial reference latent/KV lane.
Manual validation batches 12 prompts for one identity/reference. Ensure that:

- one reference latent and one reference mask are prepared per identity batch;
- target bboxes remain per output sample;
- a nested list of repeated identity images cannot accidentally create 12
  spatial reference branches;
- heterogeneous-reference batching fails clearly rather than silently taking
  the first item.

Keep PhotoMaker identity-image batching distinct from the one spatial BA
reference. E13/BC_E13/CL14 use one identity reference, but the distinction
should remain explicit.

### B3. Deterministic reference state and step schedule

Prepare reference latent, reference mask, and reference noise once per pipeline
call. Pin the exact step behavior:

- text-only before PhotoMaker step 10;
- PhotoMaker active from step 10;
- BA active from step 15;
- `branched_start_mode=both`;
- 50 DDIM steps, CFG 5.

The June `branched_attn_end_step` API can remain with `None` for minimal diff;
removing an unused parameter is not required for CL14 functionality.

### B4. Face-analysis runtime

The exact run snapshots used CUDA InsightFace/ONNX Runtime with provider order
`CUDAExecutionProvider`, then `CPUExecutionProvider`, and `ctx_id=0`. Production
must fail before training if CUDA provider initialization fails; it must not
silently run the expensive CPU path.

Provider choice can produce numerical differences in embeddings, so it is part
of the named replay/runtime profile, not an invisible machine optimization.

### B5. Versioned validation semantics

Keep two explicit profiles:

- `legacy_replay`: exact historical reference-subject selection, ID embeddings,
  run-specific generated bbox cache, shadow adapter, and `legacy_full_copy`
  behavior;
- `subject_v2`: bbox-overlap subject selection and corrected identity/ownership
  metrics for new reporting.

Do not copy the current post-CL14 target-visibility-mask work into the clean
pipeline. Do not make subject-v2 the hidden behavior of historical configs.
Before a subject-v2 result is published, first replay its historical batch
pixel-exact under `legacy_replay`.

The replay loader must resolve the active bbox file recorded by each immutable
runtime. A previous backfill incorrectly supplied the shared canonical bbox
seed and produced 12/12 pixel mismatches for both E13 and BC_E13 despite using
otherwise plausible settings. Filename similarity is not a sufficient gate.

## 7. Workstream C: datasets and the Cosmic Large fixes

### C1. Large Dataset recipe

Port `LargeDatasetTrain` with its exact E13 behavior:

- 47,500 adjusted 1024-pixel scene images across 2,561 identities;
- distinct same-identity target/reference images;
- target horizontal flip with bbox propagation;
- no independent reference flip;
- final reference image and bbox emitted directly to the generic model input.

### C2. BigCelebs recipe

`BigCelebsTrain` is the same architecture and optimizer contract as E13; only
the training dataset changes. Preserve:

- sealed v2 release: 349,348 images, 68,648 identities;
- manifest SHA-256
  `f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa`;
- minimum face side 192;
- uniform distinct same-identity reference selection;
- target/reference inequality and fail-closed release readiness;
- no reference flip.

The thin BC_E13 leaf config must differ from E13 only in
`train_dataset_name` and explanatory metadata. BigCelebs schedule experiments
may be retained as tools/dataset helpers, but are not part of the BC_E13 recipe.

### C3. Cosmic problems and what CL14 actually includes

The clean port must not summarize all Cosmic experiments as one undifferentiated
"dataset improvement." The exact boundary is:

| Problem | Evidence | Owner of fix | CL14 behavior |
|---|---|---|---|
| 256px tight reference had about 42% face area and was letterboxed/upscaled into the 1024/128-grid spatial lane, versus roughly 9% target faces | `[measured/code]`; about 2.12x linear mismatch | dataset reference construction | compose into a 1024 target-face frame before VAE; sample reference face area in `[0.06, 0.30]` |
| exact target-scale/target-position compositing gave CL2 a positional-copy shortcut and poor identity transfer | `[report/hypothesis]` | dataset reference policy | add position jitter `0.15` and scale jitter rather than exact target matching |
| reference was independently mirrored 50% of the time | `[code]` | dataset sampling | `random_reference_flip=false`; target flipping remains enabled |
| 79%+ of long Cosmic captions were truncated before useful pose/background content | `[report/measured]` | dataset prompt builder | pose-first prompt, capped at 50 words; measured over-77-token rate about 0.7% at cap 50 |
| CL0-CL8 rendered undersized faces for low-face-area validation references | `[measured]` | dataset reference calibration | inherited CL9 calibration; matched-step CL9/CL10/CL11 reached zero `<0.8` undersized faces |
| hard face-box handover produced visible boundary/seating concerns | `[code/hypothesis]` | model training mask | CL14 feather 2; inference mask unchanged |
| `min_face_res=192` removed most full-body Cosmic targets | `[measured]` | separate CL8/CL10 dataset experiment | **not fixed by CL14**; CL14 retains 192 and 22,140 accepted records |
| per-target pseudo-identities / no stable multi-view target groups | `[code/report]` | unavailable metadata/offline grouping | **not fixed by CL14** |
| multi-reference conditioning | separate CL5/CL11/CL12 experiment | dataset/model batching | **not used by CL14**; one reference |

Exact CL14 Cosmic recipe:

```yaml
train_dataset_name: cosmic_large_adapted
datasets:
  train:
    cosmic_large_adapted:
      min_face_res: 192
      reference_crop_margin: null
      reference_content_size: null
      reference_canvas_size: null
      reference_frame_mode: target_face_frame
      reference_frame_fill: edge
      reference_scale_jitter: [0.06, 0.30]
      reference_position_jitter: 0.15
      random_horizontal_flip: true
      random_reference_flip: false
      prompt_mode: pose_first
      prompt_max_words: 50
      num_identity_refs: 1
model:
  ba_training_mask_feather: 2
```

The preflight contract is 22,140 accepted records from 59,143 input records,
one reference per sample, and CL14's recorded median reference face area of
approximately 17.21%.

### C4. Reference transform ownership

`reference_frame.py` should return:

- the final 1024 RGB reference;
- the propagated reference bbox;
- a deterministic descriptor containing realized scale/window/jitter;
- audit telemetry.

The descriptor must be part of `reference_cache_key`. A reference transformed
differently for another target must never hit the same cache entry. The model
must not repeat or reinterpret this geometry.

For minimum behavioral risk, first port the exact sealed CL14 dataset modules.
If unused CL1/CL5/CL8 policy code is later reorganized, do so in a separate
behavior-preserving commit after fixed-seed dataset parity. It is acceptable for
dataset tooling to remain richer than the core model/pipeline.

## 8. Workstream D: efficiency and training speed

Efficiency changes receive their own commit and verification record. They must
not be described as dataset fixes or architecture improvements.

### D1. Exact E13-family efficiency profile

All three recipes should explicitly select the historical E13-family runtime:

```yaml
model:
  skip_unused_text_conditioning: true
  conditioning_cache_enabled: false
  batched_conditioning_preparation: true
  cache_prepared_masks: true
  compute_branch_debug_outputs: false
trainer:
  post_backward_parameter_touch: false
  grad_norm_log_only: true
dataloaders:
  train:
    batch_size: 2
    num_workers: 2
  manual_val:
    batch_size: 12
    num_workers: 1
    pin_memory: false
```

Why each exists:

| Improvement | Removed work | Scientific/numerical status |
|---|---|---|
| skip unused text-only conditioning when BA trains at all timesteps | redundant text encoding and `timestep.item()` host sync | training-only, expected output-neutral for this route |
| batch frozen conditioning for both samples | per-sample text, PhotoMaker ID, InsightFace, reference VAE, and mask calls | exact historical E13-family setting, but not bit-identical to unbatched bf16 GEMMs |
| disable ineffective LRU for diverse datasets | cache bookkeeping with almost no hits | correct for Large/BigCelebs/Cosmic diverse pairs |
| per-forward prepared-mask cache | repeated mask resizing/preparation within one doubled U-Net forward | scoped to one forward; output should be identical |
| disable branch debug tensors | construction of `noise_face`/`noise_bg` after merged prediction | output-neutral unless diagnostics are explicitly requested |
| disable post-backward zero-valued parameter touch | full parameter scan after backward | cannot change that optimizer step |
| log grad norm only on log steps | unnecessary norm reductions | changes telemetry sampling only; no clipping is active |
| manual validation batch 12 | 12 prompts for one identity/reference in one call | must retain single-reference invariant |

The 26 July benchmark measured full-Cosmic training falling from roughly 5
seconds/step to roughly 0.9 seconds/step after batched frozen conditioning. A
same-seed one-step comparison measured loss `0.06430425` batched versus
`0.06425689` legacy, a relative difference of about `0.074%`, with the same
bf16 gradient norm. Therefore:

- call the path *semantically equivalent and historically exact for E13*, not
  bit-identical to unbatched execution;
- keep it enabled for E13, BC_E13, and CL14 because those runs used it;
- keep a defaults-off legacy mode for June behavior and diagnostics.

Do not enable persistent DataLoader workers in the clean recipe. They change
augmentation RNG progression across epochs. Worker count alone was not the
root cause of the original 5-7 second/step slowdown.

### D2. GPU runtime and fail-closed preflight

Port the reusable behavior of the Serv launchers, not their hard-coded machine
paths:

- reject `CUDA_LAUNCH_BLOCKING` when it is anything other than `0`/unset;
- require asynchronous CUDA production execution;
- require ONNX Runtime 1.20.1 (or an intentionally revalidated replacement)
  with `CUDAExecutionProvider` available and loadable;
- require the CUDA/cuDNN shared libraries before starting;
- use `.env` for machine-local overlay/model/dataset locations;
- verify the source manifest before loading credentials;
- fail closed rather than silently selecting CPU InsightFace;
- log provider/version, DataLoader worker count, and efficiency profile at
  startup.

Historical evidence `[measured/report]`: asynchronous CUDA + CUDA ONNX Runtime
+ two workers improved early full-Cosmic Serv runs from approximately 5-7
seconds/step to 2.0-2.1 seconds/step. This environment correction is distinct
from the later 0.9-second batched-conditioning benchmark.

Implement the reusable checks once, for example in
`tools/runtime/verify_training_runtime.py` plus a small launcher library. Do not
copy dozens of per-run Serv start scripts into the clean branch.

### D3. Deferred face-quality scoring

Keep generation at step 0/every 2k, but stage the fixed-96 images during
training and run PyIQA only after successful training/checkpoint completion.
Benefits:

- PyIQA model construction cannot delay or abort optimizer startup;
- scoring failure is nonfatal to completed training artifacts;
- peak scoring memory is isolated from the training model;
- one canonical finalizer handles all three recipes.

The finalizer must verify immutable Comet key, expected steps, 96 images per
step, and uploaded per-image assets before declaring completion.

### D4. Performance verification

On the same available A100 and fixed 100-step fixture:

1. record startup time, first-step time, warm median/p10/p90 seconds per step,
   peak memory, DataLoader wait, and conditioning time;
2. compare the exact efficiency profile with all efficiency flags disabled;
3. require at least a 3x warm-step improvement for diverse batch-2 conditioning
   or explain the regression; the historical improvement was about 5.4x;
4. compare one fixed batch with identical RNG: prompts, masks, identity
   embeddings, reference latents, loss, gradients, and first optimizer delta;
5. require exact equality for mask-cache/debug-removal paths and document the
   expected small tolerance for batched bf16 conditioning;
6. run the full-96 legacy step-0 replay to prove runtime changes did not alter
   inference.

Do not use Neb for this benchmark; it is unavailable under current repository
instructions. Do not add permanent benchmark/test files without explicit user
permission; use existing validators or a temporary smoke script first.

## 9. Implementation ledger and required change comments

Every changed hunk must contain or be immediately adjacent to a searchable
marker. Use the actual implementation date, not necessarily this plan's date.
Suggested format:

```python
# <actual date> - E13C-CORE-03: Restrict ownership to the three E13 adapter
# groups so an installation failure cannot silently train the base U-Net.
```

Imports and declarations belonging to a marked logical block can share its
marker; do not add a comment to every syntax line. The rule is that every diff
hunk maps unambiguously to one ledger item.

| Marker | Work item | Primary files | Benefit/invariant |
|---|---|---|---|
| `E13C-CORE-01` | fail-closed hard-v1 processor installation | `lora2_helpers.py`, `branched_runtime.py` | architecture cannot silently degrade |
| `E13C-CORE-02` | hard-v1 branch rank 128 | `attn_processor_cleanest.py`, model config | reproduces E13 capacity |
| `E13C-CORE-03` | 840/700/700 ownership and optimizer groups | new `e13_contract.py`, `lora2.py`, `train.py` | exact trainable contract |
| `E13C-CORE-04` | schema-v2 checkpoint manifest/load | `e13_contract.py`, `lora2.py` | complete, auditable checkpoints |
| `E13C-CORE-05` | PhotoMaker-default validation shadow and processor copy | `base_trainer.py` | exact E13 validation mechanism |
| `E13C-CORE-06` | CL14 target training-mask feather | `lora2.py` | gradual learned boundary, inference unchanged |
| `E13C-PIPE-01` | one spatial reference for 12-prompt identity batches | pipeline + helper | prevents reference-batch ambiguity |
| `E13C-PIPE-02` | deterministic reference latent/mask/noise setup | pipeline helper | stable replay and no repeated setup |
| `E13C-PIPE-03` | versioned legacy/subject-v2 validation | pipeline, dataset, metrics configs | corrected evaluation without rewriting history |
| `E13C-DATA-01` | Large Dataset loader | `large_dataset.py` | E13 data recipe |
| `E13C-DATA-02` | sealed BigCelebs loader/readiness | `big_celebs.py` | BC_E13 dataset-only transfer |
| `E13C-DATA-03` | Cosmic prompt/reference hygiene | `cosmic_large_adapted.py` | no reference flip; useful caption tokens |
| `E13C-DATA-04` | target-frame scale/position-jitter policy | `reference_frame.py`, Cosmic loader | fixes Cosmic reference-scale mismatch |
| `E13C-PERF-01` | batched frozen conditioning | model helper | removes per-sample frozen encoder calls |
| `E13C-PERF-02` | unused-work and mask-cache switches | model/runtime/trainer | removes output-irrelevant work |
| `E13C-PERF-03` | fail-closed async CUDA/ORT runtime | runtime verifier/launcher lib | avoids silent 5-7 s/step CPU/blocking path |
| `E13C-PERF-04` | deferred face-quality finalizer | trainer config/launcher/tools | isolates scoring from optimizer trajectory |
| `E13C-CFG-01` | shared E13-family 24k config | shared Hydra config | one architecture definition |
| `E13C-CFG-02` | three thin leaf recipes | E13/BC_E13/CL14 YAMLs | dataset/model deltas remain visible |
| `E13C-DOC-01` | provenance and verification ledger | this document/final port document | another agent can audit every delta |

During implementation, add columns to this table for commit, final symbol/line,
verification result, and status. Remove or update stale `AICODE-*` anchors in
any block being replaced.

## 10. Configuration layout

Recommended configuration shape:

```text
configs/
├── e13_family_24k.yaml                 # architecture, optimizer, schedule,
│                                        # validation and efficiency contract
├── E13_large_ds_joint_shadow_sa128_24k.yaml
├── BC_E13_big_celebs_joint_shadow_sa128_24k.yaml
└── CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml
```

The shared file should contain no dataset name. Leaf configs should declare the
dataset explicitly. CL14 alone sets `ba_training_mask_feather=2`.

Keep the current post-CL14 subject-v2 additions out of the historical shared
config. Instead, compose a named validation overlay:

```text
validation/legacy_replay.yaml
validation/subject_v2.yaml
```

For each resolved config, generate a structured projection containing only:

- architecture and routing fields;
- trainable/optimizer contract;
- training schedule/loss;
- pipeline/validation contract;
- efficiency profile;
- dataset class and policy.

Store or print stable hashes of those projections. Required comparisons:

- E13 vs BC_E13 architecture/pipeline/efficiency projections: identical;
- E13 vs BC_E13 full config: only dataset name and comment differ;
- E13 vs CL14 architecture projection: only mask feather differs;
- all three legacy validation projections: identical;
- dataset projections: intentionally different and fully documented.

## 11. Tools, skills, and repository material to retain

Retain selectively:

- `.claude/skills/research-report`;
- `AGENTS.md`, `TOOLS.md`, validation protocol, and stable handoff;
- Comet pull/fetch/immutable-record tools;
- report publisher and Dropbox uploader;
- dataset preflights and manifest/schedule audit tools;
- `measure_face_body_alignment.py`;
- checkpoint evaluator and subject-v2 backfill tooling;
- Large Dataset, BigCelebs, and Cosmic reference-policy helpers;
- one canonical launcher per supported recipe plus common launcher libraries;
- face-quality deferred finalizer.

Do not bring into the clean branch:

- Serv runtime snapshots or `serv_run_packages/` copies;
- outputs, checkpoints, temporary analysis assets, or generated caches;
- CL0-CL13 leaf configs merely because they exist in `test`;
- E14-E24 mechanisms/configs;
- unused experimental attention processors;
- obsolete archived launchers;
- machine paths, credentials, `.env`, or runtime overlays.

Historical experiments remain available on `test` and through immutable Comet
records. The clean branch needs provenance links, not their full runtime debris.

## 12. Branch and commit sequence

Use a separate worktree so the current dirty `test` checkout is never used as
an implementation target:

```bash
git worktree add <sibling-path> -b kit/e13-family-clean \
  2157eada14824d14019e80f9416e6d736c837306
```

Recommended commits, each independently reviewable:

1. `core: add minimal E13 hard-BA ownership and checkpoint contract`
2. `pipeline: add single-reference E13 validation parity`
3. `datasets: add Large Dataset and sealed BigCelebs recipes`
4. `datasets: add CL14 Cosmic reference construction policy`
5. `performance: add exact E13-family efficiency profile and runtime gates`
6. `configs: add E13 BC_E13 and CL14 thin recipes`
7. `tools/docs: add preflights provenance and verification ledger`

Do not combine dataset and pipeline changes in one commit. Do not commit or push
until the user explicitly requests it.

## 13. Verification gates

### Gate 0: base and scope

- clean worktree at the exact June SHA;
- no merge from `test`;
- changed-file allowlist matches the active phase;
- every hunk has an `E13C-*` marker and ledger entry.

### Gate 1: static/configuration

- Python compile/import checks for changed modules;
- Hydra composition for all three recipes and both validation profiles;
- resolved-config projection diffs satisfy section 10;
- shell syntax for canonical launchers;
- `git diff --check` and credential/path scan.

### Gate 2: architecture and optimizer

- processor installation fails closed under a simulated plain-processor error;
- 840/700/700 and 2,240/219,217,920 exact counts;
- every trainable appears once in the optimizer;
- no base U-Net parameter is trainable;
- `pose_adapt_ratio=0`, CA mixing false, branched CA disabled;
- 70 stateful validation processors and 700 shadow-restored default tensors.

### Gate 3: checkpoint fidelity

- save/load round trip with exact tensor equality;
- load the historical E13, BC_E13, and CL14 24k checkpoints;
- reject a deliberately missing tensor, wrong rank, wrong processor list, and
  wrong dataset-independent architecture manifest;
- verify the known checkpoint hashes before loading. E13:
  `4a9d95a3f957609fcf4eb77771f263dec8e71189dc72aae347233091de4249ab`;
  BC_E13:
  `99b305bad425dd07073a4a54e0a978dea0d4a02456c8129eb1b12afbbf5a459e`;
  CL14: the hash in section 1.

### Gate 4: mask behavior

- feather 0 equals the June binary mask exactly;
- feather 2 has `1/3`, `2/3`, `1` inward values and correct clipping at small
  boxes/image edges;
- reference mask is unchanged;
- inference output at step 0 is pixel-identical between CL9 and CL14 under the
  legacy profile.

### Gate 5: dataset parity

- fixed-seed samples compare target path, reference path, pixels, prompt, bboxes,
  identity ID, and cache key against their source snapshot;
- Large/BigCelebs always use distinct same-ID references;
- BigCelebs sealed counts/hash/readiness pass;
- Cosmic accepts 22,140 records, uses one reference, never flips it, caps prompt
  at 50 words, and realizes the configured face-area/position distributions;
- Cosmic reference cache keys change with scale, position, target bbox, and
  flip state;
- no dataset-specific field or conditional appears in model/pipeline code.

### Gate 6: efficiency

- runtime preflight rejects blocking CUDA and unavailable CUDA ORT provider;
- exact efficiency profile logged at startup;
- fixed 100-step benchmark and parity comparison from section D4;
- cache-disabled diverse-data behavior and per-forward mask-cache lifetime
  verified;
- no persistent-worker RNG drift;
- deferred scorer cannot change, delay, or invalidate completed checkpoints.

### Gate 7: validation replay

Run in increasing cost order:

1. one identity / 12 prompts under `legacy_replay`;
2. full 96 at step 0;
3. one historical 24k checkpoint per recipe;
4. subject-v2 only after the corresponding legacy replay passes.

Before step 1, hash and record the exact run-specific active generation-bbox
map and legacy reference-embedding artifact from the immutable source tree.

The strongest gate is exact RGB equality to the immutable historical export.
If exact equality is impossible because of a deliberately changed low-level
library, stop and document the first divergent tensor/image; do not replace the
gate with aggregate-metric similarity.

Preserve the known data-joining conventions: normalize space/underscore output
keys, fetch Comet assets through `tools/comet/comet_experiment.py`, and give
each step its own output directory.

### Gate 8: concise-code review

- compare line counts and path-limited diffs against the June base;
- no unused architecture switch remains in the core model;
- no CL arm number appears in runtime equations except the documented CL14
  config/comment marker;
- pipeline contains no dataset names;
- all excluded mechanisms in section 5 remain absent;
- final Markdown ledger accounts for every changed hunk.

## 14. Confidence and what is not established

| Claim | Confidence | Basis |
|---|---|---|
| E13 and BC_E13 share one architecture and differ only by training dataset | high | resolved config and immutable run records |
| CL14 uses the E13 architecture plus CL9 Cosmic policy and mask feather 2 | high | sealed config/source inspection |
| Cosmic reference-scale calibration fixed the undersized-face failure | high | matched-step CL8 vs CL9/CL10/CL11 intervention |
| reference flip and caption cap are dataset-side fixes | high | loader/config inspection and token audit |
| CL14 should not change inference masks | high | flag is read only by training `_bbox_to_mask` |
| batched conditioning materially improves diverse-dataset throughput | high | 100-step measured benchmark |
| batched and unbatched bf16 conditioning are bit-identical | false | measured `0.074%` one-step loss difference |
| CL14 mask feather alone caused its final identity gain | not established | no identical-dataset, identical-run controlled endpoint isolating only final stochastic trajectory |
| CL14 fixes Cosmic's missing full-body distribution or pseudo-identities | false | it retains `min_face_res=192` and one pseudo-identity per target |
| subject-v2 aggregate ordering is identical to legacy ordering | not established | historical backfill was still being completed/recovered at the evidence cutoff |
| exact E13-family speed on the future clean branch | not established | must be benchmarked on the target A100/runtime after the port |

## 15. Definition of done

The clean branch is ready for review only when:

- the three recipes compose from one shared architecture;
- pipeline, dataset, and efficiency changes are separate commits and separate
  ledger sections;
- historical checkpoints load with exact ownership;
- mask, dataset, performance, and legacy replay gates pass;
- every logical change has an explanatory dated marker;
- the final port document records source, benefit, compatibility, and evidence
  for every marker;
- the dirty `test` worktree and historical branches remain untouched;
- nothing has been pushed without explicit user authorization.

## References

- [`LATEST.md`](../handoffs/LATEST.md)
- [`validation_protocol.md`](../validation_protocol.md)
- [`2026-07-24_quality_neutral_runtime_optimizations.md`](../experiments/2026-07-24_quality_neutral_runtime_optimizations.md)
- [`2026-07-26_cosmic_large_training_throughput_fix.md`](../experiments/2026-07-26_cosmic_large_training_throughput_fix.md)
- [`2026-08-06_cosmic_large_vs_large_dataset_root_cause_and_cl1_cl3_plan.md`](../../analysis/2026-08-06_cosmic_large_vs_large_dataset_root_cause_and_cl1_cl3_plan.md)
- [`2026-08-08_cl_face_scale_root_cause_and_cl8_cl9.md`](../../analysis/2026-08-08_cl_face_scale_root_cause_and_cl8_cl9.md)
- [`2026-08-09_cl8_cl9_face_scale_results_and_cl10_cl11.md`](../../analysis/2026-08-09_cl8_cl9_face_scale_results_and_cl10_cl11.md)
- [`2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md`](../../analysis/2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md)
- [`2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md`](../../analysis/2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md)
