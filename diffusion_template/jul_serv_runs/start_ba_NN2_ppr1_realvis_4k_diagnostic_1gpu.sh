#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29624}"
RUN_NAME="${RUN_NAME:-ba_NN2_ppr1_realvis_6k_diagnostic}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_6k_diagnostic}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-false}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs_new_runs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"

if [[ -n "${PHOTOMAKER_ENV_BIN:-}" ]]; then
  if [[ ! -x "${PHOTOMAKER_ENV_BIN}/python" ]]; then
    echo "Invalid PHOTOMAKER_ENV_BIN: ${PHOTOMAKER_ENV_BIN}" >&2
    exit 2
  fi
  export PATH="${PHOTOMAKER_ENV_BIN}:${PATH}"
elif [[ "${CONDA_DEFAULT_ENV:-}" != *photomaker* ]]; then
  for candidate in \
    "${HOME}/anaconda3/envs/photomaker/bin" \
    "${HOME}/conda_env/photomaker_NS/bin"; do
    if [[ -x "${candidate}/python" ]]; then
      export PATH="${candidate}:${PATH}"
      break
    fi
  done
fi
if ! python -c 'import torch, diffusers' >/dev/null 2>&1; then
  echo "Activate the PhotoMaker conda environment or set PHOTOMAKER_ENV_BIN." >&2
  exit 2
fi

if [[ -z "${PM_PATH:-}" && -f "/home/niko/models/PhotoMaker-V2/photomaker-v2.bin" ]]; then
  PM_PATH="/home/niko/models/PhotoMaker-V2/photomaker-v2.bin"
fi
PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
if [[ ! -f "${PM_PATH}" ]]; then
  echo "PhotoMaker checkpoint not found: ${PM_PATH}" >&2
  exit 2
fi

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  for candidate in \
    "${PROJECT_DIR}/saved/ba_NN2_ppr1_realvis_1gpu/checkpoint-epoch3.pth" \
    "${PROJECT_DIR}/saved/ba_NN2_ppr1_1gpu/checkpoint-epoch3.pth"; do
    if [[ -f "${candidate}" ]]; then
      CHECKPOINT_PATH="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CHECKPOINT_PATH:-}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to the NN2-PPR 6k checkpoint-epoch3.pth." >&2
  exit 2
fi
CHECKPOINT_PATH="$(cd -- "$(dirname -- "${CHECKPOINT_PATH}")" && pwd)/$(basename -- "${CHECKPOINT_PATH}")"

if [[ "${RUN_FOREGROUND:-0}" != "1" && "${DETACHED_RUN:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  echo "Starting NN2-PPR 6k A-E diagnostic matrix on GPU ${CUDA_VISIBLE_DEVICES}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Output: ${OUTPUT_DIR}"
  echo "Log: ${LOG_FILE}"
  DETACHED_RUN=1 \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" PM_PATH="${PM_PATH}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MASTER_PORT="${MASTER_PORT}" \
    RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT}" LOG_DIR="${LOG_DIR}" LOG_FILE="${LOG_FILE}" \
    nohup bash "$0" "$@" >"${LOG_FILE}" 2>&1 </dev/null &
  echo "PID: $!"
  echo "Follow with: tail -f ${LOG_FILE}"
  exit 0
fi

cd "${PROJECT_DIR}"
export HYDRA_FULL_ERROR=1
export FACEANALYSIS_CPU="${FACEANALYSIS_CPU:-1}"

ACCELERATE_LOG_LEVEL=error \
TRANSFORMERS_VERBOSITY=error \
DIFFUSERS_VERBOSITY=error \
PYTHONWARNINGS="ignore::FutureWarning" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
accelerate launch --config_file=src/configs/ddp/accelerate.yaml \
  --main_process_ip=127.0.0.1 --main_process_port="${MASTER_PORT}" \
  --num_processes=1 train.py \
  --config-name=one_id_ba_NN2_ppr1 \
  datasets=all_datasets \
  train_dataset_name=cosmic_large_neb \
  datasets.train.cosmic_large_neb.num_refs=1 \
  +datasets.train.cosmic_large_neb.ref_crop_margin_min=0.2 \
  +datasets.train.cosmic_large_neb.ref_crop_margin_max=0.6 \
  +datasets.train.cosmic_large_neb.ref_downscale_jitter=0.5 \
  val_datasets_names='[manual_val]' \
  datasets.val.manual_val.limit=96 \
  datasets.val.manual_val.seeds='[0]' \
  dataloaders.manual_val.batch_size=12 \
  dataloaders.manual_val.num_workers=1 \
  trainer.epoch_len=2000 \
  trainer.n_epochs=10 \
  trainer.seed=0 \
  dataloaders.train.batch_size=2 \
  dataloaders.train.grad_accum_enabled=false \
  dataloaders.train.batch_size_eff=2 \
  dataloaders.train.num_workers=6 \
  model.rank=32 \
  model.photomaker_path="${PM_PATH}" \
  model.weight_dtype=bf16 \
  pipeline.variant=null \
  validation_args.num_images_per_prompt=1 \
  validation_args.num_inference_steps=50 \
  validation_args.guidance_scale=5 \
  lr_scheduler.warmup_steps=2000 \
  lr_for_lora=5e-5 \
  trainer.max_grad_norm=1.0 \
  optimizer.weight_decay=1e-2 \
  automatic_bboxes=false \
  automatic_bboxes_every_val=false \
  force_log_first_auto_bbox=false \
  trainer.masked_loss_step=2 \
  pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0 \
  metrics=all_metrics \
  writer=console \
  writer.run_name="${RUN_NAME}" \
  validation_only=true \
  continue_run=false \
  saved_checkpoint="${CHECKPOINT_PATH}" \
  ppr_checkpoint_require_nonzero=true \
  strict_checkpoint_model_config=true \
  ppr_expected_checkpoint_epoch=3 \
  ppr_diagnostic_matrix=true \
  ppr_diagnostic_output_dir="${OUTPUT_DIR}" \
  ppr_diagnostic_overwrite="${OVERWRITE_OUTPUT}" \
  "$@"
