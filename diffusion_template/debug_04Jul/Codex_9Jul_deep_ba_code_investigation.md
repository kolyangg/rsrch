# Codex 9 Jul: Deep BA Code Investigation and N22 Recommendation

Date: 2026-07-09

## Scope

I investigated the active branched-attention PhotoMaker training/inference path after the N17/N20 results, focusing on:

- loss choice versus the original PhotoMaker recipe;
- whether branched cross-attention follows the BA scheme;
- whether trainable BA cross-attention is likely hurting;
- mask and dimension handling for different target/reference resolutions;
- ID embedding conditioning and face embedding strategy;
- concise code changes with flags so old behavior stays available;
- the next run config and launch script.

Sources checked:

- Local BA code in `src/model/photomaker_branched`, `src/pipelines`, `src/loss`, and `src/trainer`.
- Existing experiment logs/configs in `saved/`, `full_validation_results/`, and `debug_04Jul/`.
- BA plan PDF `/home/kolyangg/rsrch/_ba_scheme/ba_original_plan.pdf`; rendered pages saved under `debug_04Jul/ba_plan_pages/`.
- Official PhotoMaker repo: https://github.com/TencentARC/PhotoMaker
- PhotoMaker paper: https://arxiv.org/abs/2312.04461

## Current Understanding

The project is fine-tuning a PhotoMaker-V2-on-SDXL personalization model using a branched-attention mechanism:

- The UNet batch is doubled into generation/noise and reference branches.
- Branched self-attention uses the generation/noise branch for background and the reference branch for face/identity, with spatial masks selecting where the face branch affects the generation output.
- Branched cross-attention uses separate text-conditioning paths: generation prompt for the generation branch and face prompt for the reference/face branch.
- Recent strong runs freeze branched CA and train only branched SA adapters, because broad trainable CA has repeatedly correlated with worse images.

The N17/N20 data still points to the same core issue: identity can improve, but face placement/composition can get stuck, especially on difficult prompts such as the Keanu subway/rushing prompt. Tuning ID-loss weight alone is too small a lever now.

## Key Experiment Facts

| Run | Key mechanism | Step | Mean ID sim | Main observation |
|---|---:|---:|---:|---|
| N17 `ba_longrun_N17` | CA frozen, blended+ID, train BA all steps | 26000 | 0.3482 | Best metric, but still has face placement/pose failures that persist with training. |
| N20 `ba_combo_id075_16k_N20` | Same as N17 except lower ID-loss weight | 10000 | 0.3238 | Mostly a minor ID-loss weight variant; not a broad mechanism test. |
| N14 `ba_combo_N14` | CA frozen + ID loss | 6000 | 0.3324 | Strong early result. |
| N16 `ba_idloss6k_N16` | CA trainable + ID loss | 6000 | 0.2811 | Trainable CA looked worse. |
| N15 `ba_saonly6k_N15` | SA-only/frozen CA | 6000 | 0.3115 | Often better placement/composition than N17 despite fewer steps. |
| N12 `ba_idembeds_N12` | Intended ID-embeds probe, but CA trainable and SA ID injection was inactive | 3000 | 0.2715 | Not a clean test of ID embeddings. |

## Ranked Findings

### 1. The old ID-embedding and SA runtime knobs were effectively disabled

In the active processor (`attn_processor_cleanest.py`), `POSE_ADAPT_RATIO` was hardcoded to `0.0`, `CA_MIXING_FOR_FACE` to `False`, and SA ID embedding injection was not active. That means previous `pose_adapt_ratio`, `ca_mixing_for_face`, and `use_id_embeds` settings did not actually affect the active clean processor unless older code paths were used.

Why this matters:

- N12 was not a clean failure of ID embeddings. It also trained CA, and the intended `id_to_hidden` pathway was not really active/trainable in the current clean processor.
- N17/N20 used a rigid face branch: face hidden came almost entirely from the reference branch, with no pose adaptation from current noisy target hidden. That is a plausible contributor to the "face stuck in wrong place" behavior.

Change made:

- Added opt-in runtime SA controls behind `ba_enable_runtime_sa_knobs`.
- Old behavior is still the default: if `ba_enable_runtime_sa_knobs=false`, `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and SA ID injection stays off.
- Added a trainable `id_to_hidden` option behind `ba_train_sa_id_embed_proj`.

Relevant code:

- `src/model/photomaker_branched/attn_processor_cleanest.py`: runtime gate and ID injection.
- `src/model/photomaker_branched/branched_runtime.py`: propagates runtime flags to processors.
- `src/model/photomaker_branched/lora2_helpers.py`: can mark `id_to_hidden` trainable.

### 2. Current branched CA follows the BA scheme, but training it broadly is risky

The active CA code matches the BA plan/screenshot mechanically:

| Scheme element | Current code behavior |
|---|---|
| `noise_hidden -> q_bg` | generation/noise hidden computes the background query. |
| `ref_hidden -> q_ref` | reference hidden computes the face/reference query. |
| `gen_prompt -> k_bg/v_bg` | generation prompt conditions the generation/background branch. |
| `face_prompt -> k_ref/v_ref` | face prompt conditions the reference/face branch. |
| CA output | concatenates `[hidden_bg, hidden_ref]`; there is no spatial CA merge page in the original plan. |

Important nuance: the CA face branch output is for the reference half of the doubled batch, not directly a masked merge into the generation half. The later SA path is what spatially uses reference/face hidden for the generated image.

Why trainable CA can hurt:

- CA controls global prompt conditioning, not just masked face pixels.
- If both `ref_to_*` and `noise_to_*` CA clones are trainable, a small identity dataset can distort the prompt-conditioning pathway instead of only improving face identity transfer.
- The empirical comparison supports this: N16 trained CA and underperformed N14, while N14/N17/N20 froze CA and did better.

Change made:

- Added `ba_ca_train_mode: all | ref_only | noise_only` so a future CA experiment can train only the less dangerous side.
- Added `ba_ca_lr_scale` optimizer grouping so CA can be trained at a smaller LR if enabled later.
- N22 still freezes CA. I do not recommend turning on trainable CA until the SA/ID/loss probe is evaluated.

### 3. Mask resizing assumed square attention grids

The active mask prep used `sqrt(seq_len)` for the attention grid. This is fine for current 1024x1024 training/full validation, but unsafe if target or latent grids become non-square. Since reference and target images can start at different resolutions, this assumption should not be left in the core processor.

Change made:

- Added `_infer_spatial_hw(target_len, mask)` to preserve 4D mask aspect ratio when resizing masks into attention grids.
- Square behavior is unchanged for current runs.

This is a safety fix, not expected to explain N17 directly, because recent full-val images are square 1024 outputs.

### 4. `train_ba_all_steps=true` creates a train/inference schedule mismatch

In training, `train_ba_all_steps=true` runs branched attention for all sampled timesteps. In inference, BA only starts after the configured start steps. This can encourage BA to solve early high-noise structure when inference would not yet use it, and may contribute to locked-in face/pose artifacts.

N22 sets `train_ba_all_steps=false` to restore the staged schedule:

- text-only before PhotoMaker start;
- PhotoMaker prompt conditioning before BA start;
- BA only in the later phase.

### 5. Loss: go back to a cleaner PhotoMaker-like masked alternating probe

The PhotoMaker paper describes a masked diffusion loss used with probability, and the local code already has `masked_alternating` via `trainer.masked_loss_step=2`, meaning every second batch uses face-mask-only MSE and the other batch uses full-image MSE.

I do not recommend fine-tuning ID-loss weight at this stage. It is easy to chase metrics while worsening layout/pose. N22 turns off explicit ID loss and uses `masked_alternating` instead.

### 6. ID embedding strategy should be tested cleanly, not through N12

Original PhotoMaker is built around stacked ID embedding in prompt tokens. Our current successful BA runs use `face_embed_strategy=id`, which relies on PhotoMaker prompt embeddings. The old N12 `id_embeds` result is confounded:

- CA was trainable.
- The active clean SA processor did not actually use the runtime ID embedding path.
- `id_to_hidden` was not selected as trainable under the prior trainable-parameter filter.

N22 is a clean test:

- `pipeline.face_embed_strategy=id_embeds`
- `model.use_id_embeds=true`
- `pipeline.use_id_embeds=true`
- `ba_enable_runtime_sa_knobs=true`
- `ba_train_sa_id_embed_proj=true`

### 7. Dimension handling is mostly coherent for current square runs, but still needs caution

Current full-val/training outputs are square 1024. Reference images are prepared/letterboxed into the expected target canvas before the reference latent path. So the recent N17/N20 failure is probably not a simple target/reference resolution mismatch.

The remaining dimension risk is bbox/mask alignment:

- generated face bbox masks must match the generated image coordinate frame;
- reference face masks must match the reference latent coordinate frame;
- attention masks must be resized in 2D, not rasterized as 1D.

The new mask-grid change addresses the third item.

### 8. PhotoMaker null-text dropout is missing, but I did not patch it here

The trainer computes `do_cfg`, but `PhotomakerBranchedLora.forward()` currently deletes it, so classifier-free/null-text style training is not active. The PhotoMaker paper uses null-text replacement as part of its training recipe.

I did not add this in N22 because it changes prompt conditioning and BA branch contracts. It should be a later isolated experiment after N22 clarifies whether the SA runtime/ID/loss route helps.

## Code Changes Made

All behavior-changing code is flag-gated.

| File | Change | Old behavior preserved? |
|---|---|---|
| `src/model/photomaker_branched/attn_processor_cleanest.py` | Added aspect-aware `_infer_spatial_hw`; added opt-in SA runtime knobs; re-enabled optional SA ID injection; optional `ca_mixing_for_face`; fixed misleading CA comment. | Yes, unless `ba_enable_runtime_sa_knobs=true`. |
| `src/model/photomaker_branched/branched_runtime.py` | Propagates `ba_enable_runtime_sa_knobs`, `pose_adapt_ratio`, `ca_mixing_for_face`, `id_alpha`, `use_id_embeds` to processors. | Yes. |
| `src/model/photomaker_branched/lora2.py` | Added config fields `ba_ca_train_mode`, `ba_enable_runtime_sa_knobs`, `ba_train_sa_id_embed_proj`; added CA/noise optimizer groups with `ba_ca_lr_scale`. | Yes, default groups reproduce old behavior when scales are 1.0. |
| `src/model/photomaker_branched/lora2_helpers.py` | Allows `ba_ca_train_mode`; can train SA `id_to_hidden` only when explicitly enabled. | Yes. |
| `src/pipelines/br_pipeline_helpers.py` | Pipeline now carries `ba_enable_runtime_sa_knobs`. | Yes, default false. |
| `serv_new_runs/run_full_validation_steps.sh` | Added optional `EXTRA_INFER_OVERRIDES` pass-through for gated experiments. | Yes, old calls unchanged. |
| `src/configs/one_id_ba_runtime_idemb_alt_N22.yaml` | New N22 config. | New file. |
| `serv_new_runs/start_ba_runtime_idemb_alt_vast_N22.sh` | New N22 Vast launch script. | New file. |

## Recommended Next Run: N22

Before spending GPU on N22, run an inference-only ablation on an existing checkpoint with
`ba_enable_runtime_sa_knobs=true`, `pose_adapt_ratio=0.25`, and `ca_mixing_for_face=true`.
These knobs are runtime forward-path switches, not learned parameters. In the current code they
can affect both inference and training because training uses the same `two_branch_predict` processor
path when BA is active.

Decision gate:

- If inference-only PAR/camix improves the known pose/placement failures without large ID loss, keep them enabled in N22 training so the new adapters are optimized under the same forward path used at inference.
- If inference-only PAR/camix is neutral or worse, do not spend N22 on them; instead run the loss/schedule/ID-embed probe with PAR/camix disabled, or isolate only one knob at a time.

Run N22 before trying trainable CA only after this quick ablation.

Rationale:

- It tests the biggest plausible non-ID-loss causes of N17 failure: too-rigid face branch, inactive ID embedding path, loss/schedule mismatch.
- It keeps CA frozen, because broad CA training is still the most suspicious source of degradation.
- It avoids more ID-loss-weight tuning.
- It is short enough at 10k steps to compare directly against N17/N20 at 10k.

N22 config summary:

| Criterion | N22 value |
|---|---|
| Run name | `ba_runtime_idemb_alt_N22` |
| Config | `src/configs/one_id_ba_runtime_idemb_alt_N22.yaml` |
| Loss | `masked_alternating` |
| Explicit ID loss | off |
| BA schedule | `train_ba_all_steps=false` |
| Branched CA training | off |
| BA weights | `noise_and_ref`, LoRA clones |
| Noise clone LR | `ba_noise_lr_scale=0.1` |
| Face strategy | `id_embeds` |
| SA ID projection | enabled and trainable |
| Pose adapt | `pose_adapt_ratio=0.25` |
| Face K/V mixing | `ca_mixing_for_face=true` |
| Length | 10 epochs x 1000 steps = 10k |

If N22 improves pose/placement but weakens identity, next try should be N23:

- same as N22;
- keep ID loss off initially;
- set `train_branched_ca_lora=true`;
- set `model.ba_ca_train_mode=ref_only`;
- keep `ba_ca_lr_scale=0.1`;
- do only 4k-6k first.

If N22 is worse than N17/N20 in placement, the next isolate should be N23-alt:

- N22 loss/schedule;
- `face_embed_strategy=id`;
- `ba_enable_runtime_sa_knobs=false`;
- no ID embeds.

## How to Run N22

From repo root on the remote/Vast machine:

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker
export COMET_API_KEY=your_comet_key
bash serv_new_runs/start_ba_runtime_idemb_alt_vast_N22.sh
```

If memory is tight:

```bash
bash serv_new_runs/start_ba_runtime_idemb_alt_vast_N22.sh dataloaders.train.batch_size=1
```

The script defaults to:

```bash
PM_PATH=/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin
```

Override it if needed:

```bash
export PM_PATH=/path/to/photomaker-v2.bin
```

## How to Validate N22 Intermediate Checkpoints

Important: `serv_new_runs/run_full_validation_steps.sh` uses `inference/full_val.yaml` and does not automatically reconstruct all settings from `saved/<run>/config.yaml`. For N22, pass explicit inference overrides.

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker

export N22_INFER_OVERRIDES="pipeline.face_embed_strategy=id_embeds pipeline.use_id_embeds=true pipeline.ba_enable_runtime_sa_knobs=true pipeline.pose_adapt_ratio=0.25 pipeline.ca_mixing_for_face=true model.face_embed_strategy=id_embeds model.use_id_embeds=true model.ba_enable_runtime_sa_knobs=true model.ba_train_sa_id_embed_proj=true validation_args.face_embed_strategy=id_embeds"

PYTHON_BIN=/home/kolyangg/anaconda3/envs/photomaker/bin/python \
EXTRA_INFER_OVERRIDES="${N22_INFER_OVERRIDES}" \
BATCH_SIZE=4 \
bash serv_new_runs/run_full_validation_steps.sh ba_runtime_idemb_alt_N22 4000 6000 8000 10000
```

Outputs:

- images: `full_validation_results/ba_runtime_idemb_alt_N22_step<step>/`
- metrics: `full_validation_results/metrics_ba_runtime_idemb_alt_N22_steps.json`

## Base PhotoMaker Baseline Validation

Added script:

```text
serv_new_runs/run_photomaker_baseline_full_validation.sh
```

Purpose: run the same 96-image full validation set with base PhotoMaker V2 only, without a saved BA checkpoint and with branched attention disabled. This keeps the same `inference/full_val.yaml` dataset, prompts, seed `0`, negative prompt, 50 denoising steps, RealVisXL base, and PhotoMaker V2 weights path. Output is saved to:

```text
full_validation_results/photomaker_baseline
```

Run:

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker
PYTHON_BIN=/home/kolyangg/anaconda3/envs/photomaker/bin/python \
BATCH_SIZE=4 \
bash serv_new_runs/run_photomaker_baseline_full_validation.sh
```

The script explicitly passes:

```text
saved_checkpoint=null
validation_args.use_branched_attention=false
validation_args.face_embed_strategy=id
pipeline.face_embed_strategy=id
pipeline.use_id_embeds=false
model.use_id_embeds=false
disable_branched_sa=true
disable_branched_ca=true
```

Metrics are written to:

```text
full_validation_results/metrics_photomaker_baseline.json
```

If the 96 images already exist, the script skips regeneration and recomputes metrics. To overwrite/regenerate:

```bash
FORCE=1 PYTHON_BIN=/home/kolyangg/anaconda3/envs/photomaker/bin/python \
BATCH_SIZE=4 \
bash serv_new_runs/run_photomaker_baseline_full_validation.sh
```

## Verification Done

Passed:

```bash
bash -n serv_new_runs/start_ba_runtime_idemb_alt_vast_N22.sh
bash -n serv_new_runs/run_full_validation_steps.sh
```

Passed:

```bash
/home/kolyangg/anaconda3/envs/photomaker/bin/python -m py_compile \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/model/photomaker_branched/branched_runtime.py \
  src/model/photomaker_branched/lora2.py \
  src/model/photomaker_branched/lora2_helpers.py \
  src/pipelines/br_pipeline_helpers.py \
  infer.py train.py
```

Hydra composition check resolved the key N22 flags:

```text
loss_kind= masked_alternating
pipeline.face_embed_strategy= id_embeds
pipeline.use_id_embeds= True
pipeline.ba_enable_runtime_sa_knobs= True
model.use_id_embeds= True
model.ba_enable_runtime_sa_knobs= True
model.ba_train_sa_id_embed_proj= True
train_ba_all_steps= False
```

Mask shape smoke test:

```text
non-square 48x32 mask -> target 24x16 grid: (2, 1, 384, 1)
square 64x64 mask -> target 32x32 grid: (2, 1, 1024, 1)
```

## Open Caveats

- N22 is not intended to solve trainable CA; it deliberately freezes CA.
- Public PhotoMaker repo is inference/demo-oriented; the training details come mainly from the paper rather than public training code.
- The null-text replacement / CFG-style training path is still not implemented in the active trainer.
- Full-val helper defaults still describe old runs. For gated N22 behavior, use `EXTRA_INFER_OVERRIDES` as shown above.
