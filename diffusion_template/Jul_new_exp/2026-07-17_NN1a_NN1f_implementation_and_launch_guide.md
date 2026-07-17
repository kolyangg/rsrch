# NN1a–NN1f implementation and launch guide

Date: 17 July 2026

Status: implemented and statically validated; GPU training has not been started
from this workspace.

## Outcome

NN1a–NN1f are now runnable one-GPU experiments. All six preserve the original
N3a doubled `[target, reference]` U-Net call, all 70
`BranchedAttnProcessor` sites, all 70 `BranchedCrossAttnProcessor` sites, full
spatial reference K/V, and direct target-half epsilon output.

The implementation does not backport the post-N3a compact-memory, residual,
layer-allowlist, causal-epsilon, or identity-owner architectures.

## Shared fixes: audit issues 1–4

All six configs enable:

```yaml
model:
  ba_correctness_guards: true
  ba_invalid_sample_policy: skip_batch
  ba_strict_processor_restore: true
```

This adds:

1. fatal processor-install errors; exact 70-SA/70-CA class and name checks;
   trainable Q/K/V category manifests before optimizer creation;
2. strict finite, ordered, in-image target/reference bbox validation with no
   all-image fallback;
3. strict reference face-recognition validity with no zero-vector fallback;
4. checkpoint manifests for installed processors and trainable state keys,
   strict validation-copy checks, and exact optimizer-owned processor object
   reattachment assertions.

Rejected data skips the complete microbatch. It does not consume the requested
optimizer-step budget. Rejection metrics are logged separately as
`invalid_sample/target_bbox`, `invalid_sample/reference_bbox`, and
`invalid_sample/reference_recognition`.

The first guarded training prediction logs a SHA-256 fingerprint plus mean and
standard deviation for NN1a parity checks.

## Isolated run differences

| Run | Config | Difference from its comparison anchor |
|---|---|---|
| NN1a | `one_id_ba_NN1a_n3a_replay` | guarded N3a replay |
| NN1b | `one_id_ba_NN1b_schedule_matched` | sample only BA-active inference region, approximately `t<=699` |
| NN1c | `one_id_ba_NN1c_masked_id_prompt` | additive reference-CA mask excludes non-ID prompt tokens |
| NN1d | `one_id_ba_NN1d_frozen_ca` | split branched CA stays active, but all CA clones are frozen |
| NN1e | `one_id_ba_NN1e_frozen_ca_id_loss` | NN1d plus reference-ID cosine loss `0.1` at `t<=400` |
| NN1f | `one_id_ba_NN1f_ref_kv_id_loss` | NN1e with only SA `ref_to_k/v` LoRA tensors trainable |

NN1c leaves target-generation cross-attention unchanged. Under CFG, the
unconditional reference rows keep plain negative-prompt context and allow all
tokens; conditional rows allow only the PhotoMaker ID-token positions.

NN1e/NN1f use only the minimal differentiable FaceNet objective: predicted
`x0` is decoded at low noise, the validated generated target face is compared
with the validated reference face, and gradients flow through the frozen VAE
decode and frozen recognizer to BA parameters.

## Common run protocol

- one Accelerate process and one GPU per run;
- physical batch 2, effective batch 2, accumulation 1;
- 5 epochs × 2,000 optimizer steps = 10,000 steps;
- full fixed 96-image `manual_val` at step 0, 2k, 4k, 6k, 8k, and 10k;
- PhotoMaker starts at inference step 10 and spatial BA starts at step 15;
- LR `5e-5`, target/noise clone LR multiplier `0.25`, clip `1.0`, weight
  decay `1e-2`;
- N3a `masked_alternating` objective and reference crop/downscale jitter.

The launchers are detached by default, write timestamped logs under
`logs_new_runs`, accept `full_step0_val`, and pass any other arguments through
as Hydra overrides. `COMET_API_KEY` must be supplied through the environment;
no credential is embedded.

## Launch on the 2-GPU machine

Activate the machine's PhotoMaker environment first, then:

```bash
cd /home/kolyangg/rsrch/diffusion_template
export COMET_API_KEY="..."
export PM_PATH="/path/to/photomaker-v2.bin"

./jul_serv_runs/start_ba_NN1a_n3a_replay_1gpu.sh full_step0_val
./jul_serv_runs/start_ba_NN1b_schedule_matched_1gpu.sh full_step0_val
```

The defaults assign NN1a to physical GPU 0 and NN1b to physical GPU 1.

## Launch on the 4-GPU machine

```bash
cd /home/kolyangg/rsrch/diffusion_template
export COMET_API_KEY="..."
export PM_PATH="/path/to/photomaker-v2.bin"

./jul_serv_runs/start_ba_NN1c_masked_id_prompt_1gpu.sh full_step0_val
./jul_serv_runs/start_ba_NN1d_frozen_ca_1gpu.sh full_step0_val
./jul_serv_runs/start_ba_NN1e_frozen_ca_id_loss_1gpu.sh full_step0_val
./jul_serv_runs/start_ba_NN1f_ref_kv_id_loss_1gpu.sh full_step0_val
```

The defaults assign these to physical GPUs 0, 1, 2, and 3 respectively. Every
launcher uses a unique master port. Override a mapping with, for example:

```bash
CUDA_VISIBLE_DEVICES=3 ./jul_serv_runs/start_ba_NN1c_masked_id_prompt_1gpu.sh
```

NN1e and NN1f preflight `facenet-pytorch` and the VGGFace2 recognizer weights.
If needed:

```bash
pip install --no-deps facenet-pytorch
```

## Startup gates

Do not trust a run unless its log shows:

- `[BA strict install] SA=70 CA=70`;
- the expected trainable categories for that run;
- no CA trainable category for NN1d–NN1f;
- only `sa_ref_k` and `sa_ref_v` for NN1f;
- full 96-image step-zero validation;
- a first-prediction fingerprint before training advances.

NN1a is the parity gate. Interpret NN1b/c/d only against NN1a, NN1e only
against NN1d, and NN1f only against NN1e.

## Reverting to old behavior

The model changes are defaults-off:

```yaml
model:
  ba_correctness_guards: false
  ba_invalid_sample_policy: legacy
  ba_strict_processor_restore: false
  ba_train_timestep_mode: all
  ba_face_prompt_attention_mask: false
  ba_sa_train_mode: all
  use_id_loss: false
```

Existing N3a launchers/configs therefore retain their old model forward and
checkpoint behavior. The only generic trainer correction is that a skipped
microbatch no longer shortens an epoch's successful-update budget.

## Files

- configs: `src/configs/one_id_ba_NN1*.yaml`;
- launchers: `jul_serv_runs/start_ba_NN1*.sh`;
- shared runner: `jul_serv_runs/_run_ba_NN1_common_1gpu.sh`;
- guards/trainability/input preparation:
  `src/model/photomaker_branched/lora2_helpers.py`;
- timestep sampler, checkpoint manifest, and ID-loss integration:
  `src/model/photomaker_branched/lora2.py`;
- explicit reference-token mask:
  `src/model/photomaker_branched/branched_runtime.py` and
  `attn_processor_cleanest.py`;
- ID objective: `src/loss/id_loss.py`;
- skip-budget and strict validation-copy handling:
  `src/trainer/base_trainer.py` and `src/trainer/sdxl_trainers.py`.

The interactive explorer contains all six implemented configs and links each
clickable NN1 block to the corresponding config, launcher, and source code.

