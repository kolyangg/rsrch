# Codex 8 Jul — N20 vs N17 config comparison

Sources compared:

- `serv_new_runs/start_ba_longrun_vast_N17.sh`
- `serv_new_runs/start_ba_combo_id075_16k_vast_N20.sh`

Short version: **N20 is intentionally N17 with lower ID-loss pressure and a shorter, denser checkpoint schedule.** It is not a new BA architecture. The goal is to preserve the N17 10k-12k identity gains while reducing late-training over-strength artifacts such as pasted/canonical face, long neck / misplaced face, and prop/occlusion collisions.

## Key differences

| Area | N17 | N20 | Practical meaning |
|---|---|---|---|
| Run name | `ba_longrun_N17` | `ba_combo_id075_16k_N20` | New run is separate and will save under a different folder. |
| Total training steps | `trainer.epoch_len=2000`, `trainer.n_epochs=13` = **26000** | `trainer.epoch_len=1000`, `trainer.n_epochs=16` = **16000** | N20 stops before the late N17 region where visual collapses appeared. |
| Checkpoint / val cadence | every **2000** steps | every **1000** steps | N20 gives finer checkpoint selection around the expected 10k-14k sweet spot. |
| ID loss weight | `+model.id_loss_weight=0.1` | `+model.id_loss_weight=0.075` | Main experimental change: reduce cumulative identity pressure by 25%. |
| CUDA device setting | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"` | N20 can be redirected by exporting `CUDA_VISIBLE_DEVICES`; default is still GPU 0. |

## Full launch override table

| Config / launch field | N17 | N20 | Same? | Note |
|---|---|---|---|---|
| shell safety | `set -euo pipefail` | `set -euo pipefail` | yes | Same script safety. |
| `HYDRA_FULL_ERROR` | `1` | `1` | yes | Same. |
| `CUDA_LAUNCH_BLOCKING` | `${CUDA_LAUNCH_BLOCKING:-0}` | `${CUDA_LAUNCH_BLOCKING:-0}` | yes | Same default. |
| `PM_PATH` default | `/mnt/.../PhotoMaker-V2/photomaker-v2.bin` | `/mnt/.../PhotoMaker-V2/photomaker-v2.bin` | yes | Same PhotoMaker V2 init. |
| `COMET_API_KEY` default | same baked key | same baked key | yes | Same. |
| `ACCELERATE_LOG_LEVEL` | `error` | `error` | yes | Same. |
| `TRANSFORMERS_VERBOSITY` | `error` | `error` | yes | Same. |
| `DIFFUSERS_VERBOSITY` | `error` | `error` | yes | Same. |
| `PYTHONWARNINGS` | `ignore::FutureWarning` | `ignore::FutureWarning` | yes | Same. |
| `COMET_DISABLE_AUTO_LOGGING` | `1` | `1` | yes | Same. |
| `COMET_LOGGING_CONSOLE` | `ERROR` | `ERROR` | yes | Same. |
| `CUDA_VISIBLE_DEVICES` | fixed `0` | `${CUDA_VISIBLE_DEVICES:-0}` | no | N20 keeps default GPU 0 but allows override. |
| accelerate config | `src/configs/ddp/accelerate.yaml` | `src/configs/ddp/accelerate.yaml` | yes | Same launcher config. |
| `--num_processes` | `1` | `1` | yes | Same single-GPU run. |
| train entrypoint | `train.py` | `train.py` | yes | Same. |
| Hydra config | `--config-name=one_id_09Feb_testing` | `--config-name=one_id_09Feb_testing` | yes | Same base config. |
| `datasets` | `all_datasets` | `all_datasets` | yes | Same. |
| `train_dataset_name` | `cosmic_large_vast` | `cosmic_large_vast` | yes | Same train data. |
| `datasets.train.cosmic_large_vast.num_refs` | `1` | `1` | yes | Same one-reference setup. |
| `val_datasets_names` | `[manual_val_two]` | `[manual_val_two]` | yes | Same validation set. |
| `trainer.epoch_len` | `2000` | `1000` | no | N20 halves epoch/checkpoint interval. |
| `trainer.n_epochs` | `13` | `16` | no | N20 total is 16k, not 26k. |
| total steps | `26000` | `16000` | no | Derived from `epoch_len * n_epochs`. |
| `dataloaders.train.batch_size` | `1` | `1` | yes | Same memory-safe ID-loss batch size. |
| `dataloaders.train.num_workers` | `12` | `12` | yes | Same. |
| `model.rank` | `32` | `32` | yes | Same LoRA rank. |
| `model.photomaker_path` | `${PM_PATH}` | `${PM_PATH}` | yes | Same. |
| `+model.ba_uncond_face_fix` | `true` | `true` | yes | Same unconditional face fix. |
| `+model.ba_face_prompt_mode` | `id_only` | `id_only` | yes | Same face prompt mode. |
| `+model.use_id_loss` | `true` | `true` | yes | Same auxiliary identity loss enabled. |
| `+model.id_loss_weight` | `0.1` | `0.075` | no | Main experimental change. |
| `+model.id_loss_max_timestep` | `500` | `500` | yes | Same low-noise gate. |
| `validation_args.num_images_per_prompt` | `1` | `1` | yes | Same. |
| `lr_scheduler.warmup_steps` | `200` | `200` | yes | Same. |
| `model.weight_dtype` | `bf16` | `bf16` | yes | Same. |
| `pipeline.variant` | `null` | `null` | yes | Same. |
| `dataloaders.manual_val_two.batch_size` | `4` | `4` | yes | Same val batch size. |
| `datasets.val.manual_val_two.limit` | `24` | `24` | yes | Same small in-run val limit. |
| `val_debug` | `false` | `false` | yes | Same. |
| `branched_attn_weight_mode` | `noise_and_ref` | `noise_and_ref` | yes | Same BA weighting mode. |
| `branched_attn_new_weight_kind` | `lora` | `lora` | yes | Same trainable weight type. |
| `lr_for_lora` | `1e-4` | `1e-4` | yes | Same LR. |
| `+ba_noise_lr_scale` | `0.1` | `0.1` | yes | Same lower LR for noise branch. |
| `trainer.max_grad_norm` | `1.0` | `1.0` | yes | Same grad clipping. |
| `optimizer.weight_decay` | `1e-3` | `1e-3` | yes | Same regularization. |
| `loss_kind` | `blended_masked` | `blended_masked` | yes | Same blended masked loss. |
| `lambda_face` | `0.15` | `0.15` | yes | Same face-region weighting. |
| `automatic_bboxes` | `true` | `true` | yes | Same bbox source. |
| `automatic_bboxes_every_val` | `false` | `false` | yes | Same. |
| `force_log_first_auto_bbox` | `true` | `true` | yes | Same. |
| `train_branched_ca_lora` | `false` | `false` | yes | Same important stabilizer: frozen branched cross-attn. |
| `ba_patch_top_k` | `1.0` | `1.0` | yes | Same. |
| `ba_train_top_k` | `1.0` | `1.0` | yes | Same. |
| `non_ba_train` | `false` | `false` | yes | Same: do not train non-BA path. |
| `train_ba_only` | `true` | `true` | yes | Same. |
| `trainer.masked_loss_step` | `2` | `2` | yes | Same. |
| `train_ba_all_steps` | `true` | `true` | yes | Same. |
| validation base model | `SG161222/RealVisXL_V4.0` | `SG161222/RealVisXL_V4.0` | yes | Same RealVis validation. |
| `metrics` | `all_metrics` | `all_metrics` | yes | Same metric set. |
| `writer` | `cometml` | `cometml` | yes | Same logger. |
| `writer.run_name` | `ba_longrun_N17` | `ba_combo_id075_16k_N20` | no | Separate Comet/saved run. |
| pass-through args | `"$@"` | `"$@"` | yes | Same ability to append overrides. |

## Interpretation

N20 is a controlled follow-up to N17, not a broad search. The only model-loss change is:

```text
+model.id_loss_weight: 0.1 -> 0.075
```

The schedule change is equally important:

```text
N17: 2000 x 13 = 26000, checkpoints every 2000
N20: 1000 x 16 = 16000, checkpoints every 1000
```

This means N20 should be judged by intermediate checkpoints, especially around **10k, 12k, 14k, and 16k**, not only by the last checkpoint.

Recommended N20 launch:

```bash
cd /home/kolyangg/rsrch/diffusion_template
bash serv_new_runs/start_ba_combo_id075_16k_vast_N20.sh
```

Recommended full-val checkpoints after N20 finishes:

```bash
BATCH_SIZE=4 bash serv_new_runs/run_full_validation_steps.sh ba_combo_id075_16k_N20 8000 10000 12000 14000 16000
```
