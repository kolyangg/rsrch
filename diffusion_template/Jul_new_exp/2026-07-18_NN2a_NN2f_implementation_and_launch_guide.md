# NN2a–NN2f implementation and launch guide

Date: 18 July 2026

Status: implemented and statically/processor-smoke validated. GPU training has
not been started from this workspace.

## Outcome

NN2a–NN2f are runnable one-GPU architecture screens. Every run inherits NN1d
and therefore keeps:

- one doubled `[target, reference]` U-Net call;
- all 70 `BranchedAttnProcessor` self-attention sites;
- all 70 `BranchedCrossAttnProcessor` cross-attention sites;
- target-face Q attending reference-face K/V in every active reference lane;
- split target-generation and reference-face prompts;
- direct return of the target epsilon half;
- active but frozen branched-CA cloned weights;
- the N3a masked-alternating diffusion objective;
- no decoded NN1e/f identity loss;
- strict NN1 correctness, input-validity, checkpoint, and processor-restore
  guards;
- text-only inference steps 0–9, PhotoMaker steps 10–14, and BA steps 15–49.

The implementation is one shared, flag-gated extension of
`BranchedAttnProcessor`; there are no per-run processor classes.

## Defaults and rollback

The new model defaults exactly select the old N3a/NN1 forward:

```yaml
model:
  ba_sa_ref_token_mode: full_grid
  ba_sa_face_mode: reference
  ba_sa_ref_layer_scope: all
  ba_sa_roi_grid_size: 8
  ba_sa_core_ratio: 0.7
  ba_sa_mix_init: 0.25
```

With the first three values set to `full_grid`, `reference`, and `all`, the
processor enters an explicit legacy branch. A numerical parity test against
the pre-change source produced maximum absolute difference `0.0`.

Existing N3a and NN1 configs do not enable an NN2 path. They therefore retain
their old architecture and checkpoint state layout. NN2b/c add one trainable
per-head blend vector to each self-attention processor; NN2f adds one
zero-initialized per-head residual-gain vector. These parameters exist only in
the modes that use them.

New strict checkpoints record all six architecture settings and reject resume
under a different NN2 topology. Older NN1 strict checkpoints did not contain
this field and remain backward-compatible.

## Implemented run matrix

| Run | Config | Reference K/V | Target-face arbitration |
|---|---|---|---|
| NN2a | `one_id_ba_NN2a_packed_roi` | dense normalized 8×8 reference bbox ROI | absolute reference BA, matching legacy authority |
| NN2b | `one_id_ba_NN2b_dual_attention` | legacy masked full reference grid | separate target/ref softmaxes; per-head/layer blend initialized to 25% reference |
| NN2c | `one_id_ba_NN2c_roi_dual_attention` | dense normalized 8×8 reference bbox ROI | NN2b dual arbitration |
| NN2d | `one_id_ba_NN2d_upblock_reference` | legacy full grid at enabled sites | target attention in 24 down + 10 mid sites; absolute reference BA in 36 up sites |
| NN2e | `one_id_ba_NN2e_core_ring` | legacy masked full reference grid | reference BA in inner 70% ellipse; target attention in surrounding bbox ring |
| NN2f | `one_id_ba_NN2f_confidence_residual` | dense normalized 8×8 reference bbox ROI | exact target-attention anchor plus zero-init per-head reference residual weighted by inverse attention entropy |

### Normalized ROI

At each self-attention resolution, the implementation finds the validated
reference-mask rectangle, crops the actual reference hidden states, and
bilinearly normalizes the crop to 8×8. The resulting 64 entries are all real
ROI tokens, so no padding or zero-filled outside-grid tokens enter the
softmax. This deliberately removes absolute reference crop location while
retaining spatial identity evidence.

### Dual attention

NN2b/c compute two independent attention distributions:

```text
target_face = Attn(Qtarget_face, Ktarget, Vtarget)
ref_face    = Attn(Qtarget_face, Kreference, Vreference)
face_out    = (1 - sigmoid(head_gate)) * target_face
              + sigmoid(head_gate) * ref_face
```

Each processor is one layer, and its gate has one value per attention head.
The 25% initialization gives target geometry the initial majority while
preserving an active reference lane.

### Confidence residual

NN2f computes normalized inverse entropy from the packed-reference attention
probabilities:

```text
confidence = 1 - H(P_reference) / log(64)
face_out   = target_face
             + tanh(head_gain) * confidence * (ref_face - target_face)
```

`head_gain` initializes to zero, so the first forward is exactly
target-attention anchored. Its derivative at zero is nonzero, allowing the
reference residual to learn. Diffuse reference matches fall back locally to
target geometry.

## Run protocol

| Setting | Value |
|---|---|
| GPUs per run | 1 |
| physical/effective training batch | 2 / 2 |
| gradient accumulation | 1 |
| validation batch | 12 |
| maximum optimizer steps | 20,000 |
| epoch length | 2,000 optimizer steps |
| epochs | 10 |
| validation | full fixed 96 images at step 0 and every 2k |
| early stopping | stop manually when a systematic visual failure is decisive |
| LR | `5e-5`; target/noise clone multiplier `0.25` |

The 20k value is a ceiling, not a requirement to spend compute after an
obvious collapse. Inspect the difficult skiing, chef, night-ride, laughing,
kickboxing, and hair/hand occlusion cases after each validation.

## Launch on the 2-GPU machine

```bash
cd /home/kolyangg/rsrch/diffusion_template
export COMET_API_KEY="..."
export PM_PATH="/path/to/photomaker-v2.bin"

./jul_serv_runs/start_ba_NN2a_packed_roi_1gpu.sh full_step0_val
./jul_serv_runs/start_ba_NN2b_dual_attention_1gpu.sh full_step0_val
```

Defaults: NN2a uses physical GPU 0 and port 29721; NN2b uses GPU 1 and port
29722.

## Launch on the 4-GPU machine

```bash
cd /home/kolyangg/rsrch/diffusion_template
export COMET_API_KEY="..."
export PM_PATH="/path/to/photomaker-v2.bin"

./jul_serv_runs/start_ba_NN2c_roi_dual_attention_1gpu.sh full_step0_val
./jul_serv_runs/start_ba_NN2d_upblock_reference_1gpu.sh full_step0_val
./jul_serv_runs/start_ba_NN2e_core_ring_1gpu.sh full_step0_val
./jul_serv_runs/start_ba_NN2f_confidence_residual_1gpu.sh full_step0_val
```

Defaults: NN2c/d/e/f use physical GPUs 0/1/2/3 and ports
29723/29724/29725/29726.

Every launcher is detached by default and writes a timestamped log under
`logs_new_runs`. Override a GPU in the usual way:

```bash
CUDA_VISIBLE_DEVICES=3 ./jul_serv_runs/start_ba_NN2a_packed_roi_1gpu.sh
```

Additional arguments are passed through as Hydra overrides.

## Startup gates

Do not trust a run unless its log shows:

- `[BA strict install] SA=70 CA=70`;
- `[BA architecture]` with the run's expected face mode, reference-token mode,
  and layer scope;
- no trainable branched-CA categories;
- `sa_face_mix` for NN2b/c only;
- `sa_face_residual` for NN2f only;
- full 96-image step-zero validation;
- the first-prediction fingerprint before optimization advances.

For NN2d, the installed topology remains 70 SA + 70 CA processors. The scope
switch resolves to exactly 36 up-block reference-BA sites and 34 target-owned
down/mid sites, based on the prior strict processor manifest.

## Validation performed locally

Using `/home/kolyangg/anaconda3/envs/photomaker`:

- all six Hydra configs composed and inherited frozen CA, disabled ID loss,
  and strict correctness guards;
- all seven relevant processor paths (including NN2d down and up behavior)
  produced finite outputs and input gradients;
- the defaults-off processor was bit-identical to the pre-change processor;
- Python compilation, JavaScript syntax, and shell syntax checks passed;
- launcher files are executable.

No full SDXL GPU forward or training step was run in this workspace.

## Files

- shared implementation:
  `src/model/photomaker_branched/attn_processor_cleanest.py`;
- constructor/runtime propagation:
  `src/model/photomaker_branched/lora2.py`,
  `src/model/photomaker_branched/branched_runtime.py`, and
  `src/pipelines/br_pipeline_helpers.py`;
- trainability and strict manifests:
  `src/model/photomaker_branched/lora2_helpers.py`;
- configs: `src/configs/one_id_ba_NN2*.yaml`;
- launchers: `jul_serv_runs/start_ba_NN2*.sh`;
- shared 20k runner: `jul_serv_runs/_run_ba_NN2_common_1gpu.sh`;
- interactive comparison:
  `debug_04Jul/ba_architecture_explorer/index.html`.
