#!/usr/bin/env bash
set -euo pipefail

# N26: N25 plus reference-only branched CA training at 0.1x LR; CA noise clones stay frozen.

RUN_NAME="ba_staged_caref_N26"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs_new_runs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"

if [[ "${RUN_FOREGROUND:-0}" != "1" && "${DETACHED_RUN:-0}" != "1" ]]; then
    mkdir -p "${LOG_DIR}"
    echo "Starting ${RUN_NAME} detached"
    echo "Log: ${LOG_FILE}"
    DETACHED_RUN=1 LOG_DIR="${LOG_DIR}" LOG_FILE="${LOG_FILE}" \
        nohup bash "${SCRIPT_PATH}" "$@" >"${LOG_FILE}" 2>&1 </dev/null &
    echo "PID: $!"
    echo "Follow with: tail -f ${LOG_FILE}"
    exit 0
fi

cd "${PROJECT_DIR}"
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29526}"

PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
COMET_API_KEY="${COMET_API_KEY:-}"
export PM_PATH COMET_API_KEY

if [[ -z "${COMET_API_KEY}" ]]; then
    echo "COMET_API_KEY is not set." >&2
    exit 2
fi

ACCELERATE_LOG_LEVEL=error \
TRANSFORMERS_VERBOSITY=error \
DIFFUSERS_VERBOSITY=error \
PYTHONWARNINGS="ignore::FutureWarning" \
COMET_DISABLE_AUTO_LOGGING=1 \
COMET_LOGGING_CONSOLE=ERROR \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
accelerate launch --config_file=src/configs/ddp/accelerate.yaml \
    --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" \
    --num_processes=1 train.py \
    --config-name=one_id_ba_staged_caref_N26 \
    datasets=all_datasets \
    train_dataset_name=cosmic_large \
    datasets.train.cosmic_large.num_refs=1 \
    val_datasets_names='[manual_val_two]' \
    trainer.epoch_len=1000 \
    trainer.n_epochs=10 \
    dataloaders.train.batch_size=2 \
    dataloaders.train.num_workers=12 \
    model.rank=32 \
    model.photomaker_path="${PM_PATH}" \
    model.weight_dtype=bf16 \
    +model.ba_uncond_face_fix=true \
    +model.ba_face_prompt_mode=id_only \
    validation_args.num_images_per_prompt=1 \
    lr_scheduler.warmup_steps=200 \
    pipeline.variant=null \
    dataloaders.manual_val_two.batch_size=8 \
    datasets.val.manual_val_two.limit=24 \
    val_debug=false \
    lr_for_lora=1e-4 \
    trainer.max_grad_norm=1.0 \
    optimizer.weight_decay=1e-3 \
    automatic_bboxes=true \
    automatic_bboxes_every_val=false \
    force_log_first_auto_bbox=true \
    ba_patch_top_k=1.0 \
    ba_train_top_k=1.0 \
    non_ba_train=false \
    train_ba_only=true \
    trainer.masked_loss_step=2 \
    pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0 \
    metrics=all_metrics \
    writer=cometml writer.run_name="${RUN_NAME}" \
    "$@"
