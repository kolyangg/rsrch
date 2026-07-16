# N34/N35 implementation, configuration, and launch handoff

Date: 16 July 2026

Status: code and launch preparation complete; no training runs were started.

## Confirmed machine allocation

The primary parallel allocation is:

| Run | Machine | Local batch | Accumulation | Effective global batch | Planned optimizer steps |
|---|---:|---:|---:|---:|---:|
| N34 | 4 GPUs | 1 per rank | 2 | 8 | 10,000 |
| N35 | 2 GPUs | 1 per rank | 4 | 8 | 10,000 |

This keeps N34 and N35 matched by effective global batch. The launch scripts
scale `trainer.epoch_len` by gradient accumulation, because this trainer counts
microbatches rather than optimizer updates:

- N34: 2,000 microbatches per epoch = 1,000 optimizer steps.
- N35: 4,000 microbatches per epoch = 1,000 optimizer steps.
- Both run 10 epochs and therefore finish at 10,000 optimizer steps.

The writer now converts microbatch indices to optimizer-step indices whenever
gradient accumulation is enabled. Comet validation points therefore remain
`0, 1k, 2k, ... 10k` rather than being inflated by accumulation.

## Primary configurations

### N34: corrected high-resolution QFormer identity residual

Config:

`src/configs/one_id_ba_causal_highres_qformer_N34.yaml`

N34 inherits N29 and keeps its two distinct frozen PhotoMaker QFormer tokens.
The new run changes the residual route and supervision:

- only the six `up_blocks.1` SDXL cross-attention sites are patched;
- all trainable BA parameters and Adam states use FP32;
- each selected site has a bounded sigmoid gate initialized to effective scale
  `0.25`, with maximum scale `1.0`;
- the output adapter remains zero initialized, so step zero is PhotoMaker;
- inference adds the conditional BA delta once after PhotoMaker CFG;
- target masks fail closed instead of turning an invalid bbox into a full-image
  write mask;
- reference and target face detections are required;
- all DDP ranks use the same sampled timestep and active regime;
- correct, null, and wrong identity memories are evaluated on the same noisy
  latent;
- wrong memories are selected globally with an InsightFace semi-hard criterion;
- the decoded identity objective uses differentiable five-landmark alignment;
- it rewards both correct-memory and wrong-memory directional movement;
- low-frequency chroma and grayscale structure are preserved relative to the
  null/PhotoMaker branch;
- checkpoint architecture restoration is strict.

The old direct ID loss inherited from N25 is explicitly disabled, so N34 uses
only the new causal identity objective in addition to diffusion reconstruction.

### N35: canonical eight-part identity memory

Config:

`src/configs/one_id_ba_causal_canonical_parts_N35.yaml`

N35 inherits the complete N34 route and objective. It changes only identity
memory:

1. Detect the reference face and five landmarks.
2. Warp it to a 224x224 canonical face crop.
3. Run the frozen PhotoMaker CLIP vision encoder on the aligned crop.
4. Pool eight ordered regions:
   - global face;
   - left eye/brow;
   - right eye/brow;
   - nose;
   - mouth;
   - left cheek/contour;
   - right cheek/contour;
   - inner face.
5. Fuse each region with the full-precision InsightFace global embedding and
   frozen QFormer context.
6. Produce eight trainable 2048-D memory tokens in FP32.

This is a concise first canonical-memory implementation. It uses aligned CLIP
spatial features plus global recognition identity; it is not yet the more
complex alternative of exposing intermediate spatial features from a dedicated
face-recognition backbone in a separate parallel residual path.

## Main launch scripts

### N34 on the 4-GPU machine

Script:

`serv_new_runs/start_ba_causal_highres_qformer_4gpu_N34.sh`

Launch:

```bash
cd /home/kolyangg/rsrch/diffusion_template
CUDA_VISIBLE_DEVICES=0,1,2,3 \
COMET_API_KEY="..." \
./serv_new_runs/start_ba_causal_highres_qformer_4gpu_N34.sh
```

Defaults:

- run name: `ba_causal_highres_qformer_4gpu_N34`;
- master port: `29534`;
- local batch: `1`;
- local effective batch: `2`;
- global effective batch: `8`;
- 24-image step-zero smoke validation;
- full 96-image validation every 1,000 optimizer steps;
- detached launch with logs in `logs_new_runs/`.

### N35 on the 2-GPU machine

Script:

`serv_new_runs/start_ba_causal_canonical_parts_2gpu_N35.sh`

Launch:

```bash
cd /home/kolyangg/rsrch/diffusion_template
CUDA_VISIBLE_DEVICES=0,1 \
COMET_API_KEY="..." \
./serv_new_runs/start_ba_causal_canonical_parts_2gpu_N35.sh
```

Defaults:

- run name: `ba_causal_canonical_parts_2gpu_N35`;
- master port: `29535`;
- local batch: `1`;
- local effective batch: `4`;
- global effective batch: `8`;
- the same validation and checkpoint schedule as N34;
- detached launch with logs in `logs_new_runs/`.

Set `RUN_FOREGROUND=1` for an attached process. `RUN_NAME`, `MASTER_PORT`,
`PM_PATH`, batch variables, validation batch size, and any Hydra override can
also be supplied externally.

## Optional one-GPU controls

These are short 3,000-optimizer-step controls, not replacements for N34/N35.
They use effective batch 8 through accumulation and validate every 1,000
optimizer steps.

### N34A: corrected route without decoded causal supervision

Script:

`serv_new_runs/start_ba_highres_qformer_nocausal_1gpu_N34A.sh`

This retains:

- six-site high-resolution routing;
- FP32 BA trainables;
- bounded gates;
- post-CFG residual composition;
- fail-closed identity setup.

It disables the correct/null/wrong decoded objective. Comparing N34A with N34
helps separate improvements from route/precision/CFG correction versus causal
supervision.

```bash
CUDA_VISIBLE_DEVICES=0 \
COMET_API_KEY="..." \
./serv_new_runs/start_ba_highres_qformer_nocausal_1gpu_N34A.sh
```

### N34B: all-layer no-causal routing control

Script:

`serv_new_runs/start_ba_alllayers_qformer_nocausal_1gpu_N34B.sh`

N34B calls N34A and changes only:

```text
model.ba_ca_layer_allowlist=null
```

Comparing N34A and N34B isolates high-resolution-only versus all-layer routing
under the same no-causal objective. Stop this run early if the all-layer route
shows stronger chroma, expression, or geometry drift without identity gain.

## Backward-compatible toggles

All new behavior is opt-in. Existing N28-N33 configs and scripts retain their
legacy defaults.

| Toggle | Legacy/default behavior | N34/N35 |
|---|---|---|
| `ba_ca_layer_allowlist` | `null`: patch all eligible CA sites | `[up_blocks.1]` |
| `ba_trainable_dtype` | `model`: use UNet dtype | `fp32` |
| `ba_face_gate_mode` | `legacy_scalar` | `bounded_sigmoid` |
| `ba_face_gate_init` | `1.0` | `0.25` |
| `ba_cfg_composition` | `legacy_guided` | `post_cfg_delta` |
| `ba_target_mask_fail_closed` | `false` | `true` |
| `ba_sync_timestep` | `false` | `true` |
| `ba_require_reference_face` | `false` | `true` |
| `ba_strict_checkpoint_restore` | `false` | `true` |
| `ba_identity_dependence_mode` | `none` or old paired mode | `decoded_causal` |
| `ba_identity_memory_mode` | existing memory mode | QFormer for N34; canonical parts for N35 |

To repeat an old experiment, use its old config and launch script. The new
constructor defaults preserve its old precision, routing, gate, CFG, mask, and
checkpoint behavior.

## Key implementation changes

### Cross-attention routing and precision

Files:

- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/lora2_helpers.py`

Changes:

- semantic prefix/wildcard CA allowlist;
- mixed BF16 frozen-base and FP32 trainable-adapter linear execution;
- FP32 trainable output adapters, K/V adapters, gates, and memory modules;
- bounded effective gate calculation and telemetry;
- explicit multiplication by `has_identity`, guaranteeing a zero residual for
  null memory even after the output adapter becomes nonzero;
- exactly six `up_blocks.1` CA processors selected by N34/N35.

### CFG composition

Files:

- `src/model/photomaker_branched/branched_runtime.py`
- `src/pipelines/br_pipeline_helpers.py`

For CFG inference:

```text
pm_cfg = pm_uncond + cfg * (pm_cond - pm_uncond)
delta = hard_mask * (ba_cond - pm_cond)
final = pm_cfg + ba_residual_scale * delta
```

The BA correction is no longer multiplied by text CFG. The implementation
requires `ba_pm_preservation_mode=hard_epsilon_merge` when this mode is enabled,
so an incomplete configuration fails immediately.

### Causal identity objective

Files:

- `src/loss/id_loss.py`
- `src/model/photomaker_branched/lora2.py`
- `src/trainer/sdxl_trainers.py`

Changes:

- correct, null/PhotoMaker, and wrong-memory predictions share latent,
  timestep, text, mask, and ordinary PhotoMaker conditioning;
- decoded x0 faces use the same target-derived differentiable alignment;
- correct output must move toward the correct reference relative to null;
- wrong output must move toward the selected wrong reference relative to null;
- both outputs must rank their intended identity above the other identity;
- direct correct-identity similarity remains a weak anchor;
- low-frequency chroma and grayscale-gradient structure are preserved relative
  to null;
- the frozen FaceNet recognizer is forced back to evaluation mode on every
  embedding call;
- global reference embeddings are gathered before the expensive UNet/decode
  section, avoiding a late model-internal collective if a rank OOMs;
- causal metrics are logged under `causal_identity/*`.

The causal branch runs only at `t <= 300` by default. Ordinary diffusion loss
remains the main image prior.

### Canonical memory

Files:

- `src/model/photomaker_branched/identity_memory.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/pipelines/br_pipeline_helpers.py`

Training and inference share the same canonical alignment, part ordering, and
resampler. Full-precision InsightFace vectors are retained separately from the
BF16 copies consumed by the frozen PhotoMaker encoder.

### Dataset and negative selection

Files:

- `src/datasets/cosmic.py`
- `src/model/photomaker_branched/lora2_helpers.py`

The active `CosmicLargeTrain` now emits `identity_id`. At present this is the
target entry path, because the dataset JSON has no audited person label.
Negatives are selected using the full-precision InsightFace embedding and a
moderate target similarity instead of choosing the least-similar, easiest
identity.

This path ID prevents selecting the exact same entry, but it is not a guarantee
that two separate LAION entries are different people. An offline identity
cluster/person-ID map remains the most important data improvement after these
runs.

### Checkpoint and logging safety

Files:

- `src/model/photomaker_branched/lora2.py`
- `infer.py`
- `src/trainer/base_trainer.py`

Checkpoints now carry a BA architecture manifest. Strict N34/N35 loading checks:

- memory mode and token count;
- memory geometry and resampler dimensions;
- selected processor names;
- exact trainable processor tensor keys;
- gate and precision modes;
- CFG and hard-mask behavior;
- presence or absence of the memory resampler.

Inference restores all relevant architecture switches before creating modules
and no longer suppresses setup failures for strict checkpoints.

Writer steps are now optimizer steps under manual accumulation.

## Validation completed locally

The following checks passed:

- Python compilation for every modified Python file;
- `git diff --check`;
- shell syntax for all four scripts;
- Hydra composition for N34, N35, N34A overrides, and all-layer overrides;
- every composed model key is accepted by the model constructor;
- local `CosmicLargeTrain` sample includes `identity_id`, target bbox,
  reference bbox, and one reference image;
- canonical reference face detection and five-landmark 224x224 alignment on a
  real local dataset sample;
- eight canonical region tokens have the expected shape;
- canonical resampler produces FP32 `[B, 8, 2048]` memory and FP32 gradients;
- mixed BF16 frozen-base/FP32 adapter linear path and Adam state behavior;
- bounded FP32 gate changes at learning rate `1e-4`;
- allowlist selects exactly six generated `up_blocks.1` CA names;
- null identity memory produces exactly the ordinary attention result after
  the residual output adapter is made nonzero;
- post-CFG composition matches the intended algebra exactly;
- target mask fail-closed and legacy fallback modes;
- differentiable affine face alignment propagates gradients;
- correct/null/wrong causal loss, including pre-gathered reference embeddings,
  propagates gradients to correct and wrong generated branches;
- cached VGGFace2 FaceNet loads and remains in evaluation mode.

The current development host exposes one 16 GB GPU and does not have the
server's configured PhotoMaker NFS checkpoint mounted. Therefore a full
1024x1024 N34/N35 training forward was not launched here. The first action on
each server should be a short end-to-end smoke run before committing to 10k.

## Recommended server smoke sequence

Use a distinct run name and foreground mode. Keep the normal architecture but
reduce the first run to one epoch and a small validation subset.

N34:

```bash
RUN_NAME=ba_causal_highres_qformer_4gpu_N34_smoke \
RUN_FOREGROUND=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
COMET_API_KEY="..." \
./serv_new_runs/start_ba_causal_highres_qformer_4gpu_N34.sh \
trainer.n_epochs=1 \
datasets.val.manual_val.limit=12 \
val_smoke_test_limit=12
```

N35:

```bash
RUN_NAME=ba_causal_canonical_parts_2gpu_N35_smoke \
RUN_FOREGROUND=1 \
CUDA_VISIBLE_DEVICES=0,1 \
COMET_API_KEY="..." \
./serv_new_runs/start_ba_causal_canonical_parts_2gpu_N35.sh \
trainer.n_epochs=1 \
datasets.val.manual_val.limit=12 \
val_smoke_test_limit=12
```

These still execute 1,000 optimizer steps. For a purely mechanical one-update
test, additionally override:

- N34: `trainer.epoch_len=2`;
- N35: `trainer.epoch_len=4`;
- both: `datasets.val.manual_val.limit=6 val_smoke_test_limit=6`.

Before accepting the smoke:

1. Confirm the startup line reports six selected processors.
2. Confirm every trainable BA parameter is FP32.
3. Confirm no target/reference detection failure.
4. Confirm causal metrics appear on low-timestep batches.
5. Confirm gate values move away from exactly `0.25`.
6. Confirm checkpoint epoch 1 reloads into validation without mismatch.
7. Confirm step-zero output is visually PhotoMaker-equivalent.
8. Inspect GPU memory during a causal decode step.

If decoded causal steps OOM, keep local batch 1 and first try:

```text
model.ba_causal_every_n_steps=2
```

This reduces decoded supervision frequency without changing its semantics.

## Initial stop criteria

At 1k and 3k, prioritize:

- correct-versus-null identity gain;
- movement toward the swapped identity under wrong memory;
- face chroma and saturation relative to PhotoMaker;
- facial structure/expression drift;
- bounded gate saturation;
- outside-face change;
- invalid face-detection rate.

Stop a run if identity direction remains absent while residual norms grow, or
if the run develops N31-like chroma/expression drift without a stronger causal
identity signal.

