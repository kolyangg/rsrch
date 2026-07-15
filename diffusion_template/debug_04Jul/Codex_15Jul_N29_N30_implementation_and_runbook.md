# N29/N30 architecture experiments and runbook

Date: 15 July 2026

## Purpose

N27/N28 showed that a hard-bbox residual over PhotoMaker preserves target geometry, but N28's identity memory is weak: it averages PhotoMaker's two QFormer outputs into one vector and forms four tokens by adding identity-independent learned basis vectors. N29/N30 keep N28's safe target-face CA residual, hard epsilon merge, reference-supervised ID loss, and ordinary PhotoMaker path. They isolate two identity-interface changes without loss-weight or learning-rate tuning.

| Run | BA identity memory | Reference input seen by BA memory | Question answered |
|---|---|---|---|
| N28 control | one averaged vector plus four learned basis offsets | full reference | Existing baseline |
| N29 | PhotoMaker's two genuine, distinct QFormer tokens | full reference | Does preserving identity-specific tokens fix N28's low-rank memory? |
| N30 | same two genuine QFormer tokens | square hard-bbox crop with 10% padding | Does removing reference framing/background improve identity without transferring pose? |

N30 changes only the BA memory input. PhotoMaker's normal ID-prompt fusion still receives the original full reference, and the hard target bbox is unchanged.

## Code and switches

- `src/model/photomaker_branched/model_v2_NS.py`: `extract_id_features(reduce="tokens")` returns pre-average QFormer output with shape `[B, 2, 2048]` for one reference.
- `src/model/photomaker_branched/attn_processor_cleanest.py`: `ba_identity_memory_mode=qformer_tokens` consumes those tokens directly in target-face CA.
- `src/model/photomaker_branched/identity_memory.py`: square bbox-centered reference crop used only by N30.
- `src/model/photomaker_branched/lora2_helpers.py` and `src/pipelines/br_pipeline_helpers.py`: matching train and inference memory preparation.
- `infer.py`: restores the three new architecture fields from the checkpoint's adjacent `config.yaml`.
- `src/trainer/base_trainer.py`: optional capped step-0 validation through the normal validation loader.

Compatibility defaults preserve earlier runs:

```yaml
model:
  ba_identity_memory_mode: mean_plus_basis
  ba_identity_image_mode: full_reference
  ba_identity_crop_padding: 0.10
```

New behavior is opt-in:

```yaml
model:
  ba_identity_memory_mode: qformer_tokens
  ba_identity_image_mode: full_reference  # N29
  # ba_identity_image_mode: bbox_normalized  # N30
```

Configs:

- `src/configs/one_id_ba_qformer_idtokens_N29.yaml`
- `src/configs/one_id_ba_bboxnorm_idtokens_N30.yaml`

## Validation contract

Both launchers use the same generation contract as `inference/full_val.yaml`:

- full `manual_val`: 8 identities x 12 prompts = 96 images;
- seed 0, one image per prompt;
- fixed `pm96_bboxes_new.json` target boxes; no automatic bbox pass;
- 50 inference steps, guidance scale 5, RealVisXL V4 validation base;
- validation batch size 12;
- full validation and checkpoint every 2,000 training steps;
- Comet metrics and images at steps 0, 2k, 4k, 6k, 8k, and 10k.

Step 0 is a 24-image smoke validation, not the complete set. It uses the same loader, ordering, batch size, seeds, boxes, pipeline, and metrics as full validation, but stops after two batches. The flag is enabled by default:

```bash
VAL_SMOKE_TEST=true
```

Disable only the step-0 pass with `VAL_SMOKE_TEST=false`. Later 96-image validations are unaffected. The old full step-0 behavior remains available with `validate_before_training=true val_smoke_test=false`.

The composed training and inference datasets were compared sample by sample: all 96 prompts, identities, seeds, reference boxes, and generation boxes match. Runtime generation settings also match. The saved config restores `face_embed_strategy=id_embeds` and each run's new architecture before standalone inference loads its checkpoint.

## Start both runs

The scripts detach themselves with `nohup`, use different ports, and save timestamped logs. Do not add another outer `nohup`.

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker
export COMET_API_KEY="<your-key>"

bash serv_new_runs/start_ba_qformer_idtokens_serv_N29.sh
bash serv_new_runs/start_ba_bboxnorm_idtokens_serv_N30.sh
```

Defaults are N29 on GPU 0 and N30 on GPU 1. Explicit equivalent commands are:

```bash
CUDA_VISIBLE_DEVICES=0 bash serv_new_runs/start_ba_qformer_idtokens_serv_N29.sh
CUDA_VISIBLE_DEVICES=1 bash serv_new_runs/start_ba_bboxnorm_idtokens_serv_N30.sh
```

The launch output prints the PID and exact log path. Logs are under `logs_new_runs/ba_qformer_idtokens_N29_*.log` and `logs_new_runs/ba_bboxnorm_idtokens_N30_*.log`. The first log lines confirm smoke status and validation schedule.

To skip the smoke pass:

```bash
VAL_SMOKE_TEST=false CUDA_VISIBLE_DEVICES=0 bash serv_new_runs/start_ba_qformer_idtokens_serv_N29.sh
```

## Standalone full inference

Training already validates all 96 images every 2k. After checkpoints are copied into `saved/<run>/`, the matching standalone scripts recreate all five checkpoints and metrics with batch size 12:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  bash serv_new_runs/run_full_validation_N29.sh

CUDA_VISIBLE_DEVICES=1 PYTHON_BIN=/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  bash serv_new_runs/run_full_validation_N30.sh
```

Outputs:

- `full_validation_results/ba_qformer_idtokens_N29_steps/`
- `full_validation_results/ba_bboxnorm_idtokens_N30_steps/`

Each contains per-step subfolders and `metrics_<run>_steps.json`.

## Decision at 10k

Use images first and metrics only for large differences.

1. N29 versus N28 tests token construction. A useful N29 result changes facial identity more consistently than N28 while retaining N28's head position, body alignment, pose, and non-face content.
2. N30 versus N29 tests reference normalization. Improvement must be reference-consistent facial structure/details, not a stronger generic face or copied reference pose/expression.
3. Inspect the established hard cases at every 2k checkpoint: Keanu rushing; Jisoo skiing, laughing, and kickboxing; Marion crying/laughing/night ride; Lex dancing/night ride.
4. If N29 improves over N28 and N30 adds no value, continue the genuine-token architecture longer with full references.
5. If N30 clearly improves identity without geometry regressions, use bbox-normalized memory as the long-run base.
6. If both remain PhotoMaker-like through 10k despite nonzero residual/K/V norms, do not tune ID-loss weights first. The next architectural step is explicit correct/wrong/null-reference dependence training or identity-part tokens, because the optimizer is still allowed to bypass the residual.

New Comet diagnostics log target-ID K/V LoRA norm, face-delta output norm, and mean residual gate. A zero/flat face-delta norm indicates a dead branch; growing norms with no identity-specific visual change indicate that the memory/objective, rather than optimization speed, remains the bottleneck.

## Checks completed

- Python compilation for every touched module.
- Shell syntax and executable bits for all four scripts.
- Hydra composition for N29 and N30, including the smoke flag and 2k/10k schedule.
- Actual validation collator: 96 samples in eight batches; smoke pass is exactly 24 samples in two batches.
- Training/full-inference sample and runtime-setting parity.
- Target-face processor test: zero initialization is exactly PhotoMaker, changed identities alter only the hard-mask region, and outside-mask output remains exact.
- Bbox-memory crop test: square padded output and invalid-box rejection.
