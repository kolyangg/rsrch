#!/usr/bin/env bash
set -euo pipefail

# N31: 2-GPU counterfactual identity-dependence run, 10k optimizer steps.

RUN_NAME="ba_identity_dependence_2gpu_N31"
VAL_SMOKE_TEST="${VAL_SMOKE_TEST:-true}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs_new_runs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"

if [[ "${RUN_FOREGROUND:-0}" != "1" && "${DETACHED_RUN:-0}" != "1" ]]; then
    mkdir -p "${LOG_DIR}"
    echo "Starting ${RUN_NAME} detached on GPUs ${CUDA_VISIBLE_DEVICES:-0,1}"
    echo "Log: ${LOG_FILE}"
    DETACHED_RUN=1 LOG_DIR="${LOG_DIR}" LOG_FILE="${LOG_FILE}" \
        VAL_SMOKE_TEST="${VAL_SMOKE_TEST}" NUM_PROCESSES="${NUM_PROCESSES}" \
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
        VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
        nohup bash "${SCRIPT_PATH}" "$@" >"${LOG_FILE}" 2>&1 </dev/null &
    echo "PID: $!"
    echo "Follow with: tail -f ${LOG_FILE}"
    exit 0
fi

cd "${PROJECT_DIR}"
export HYDRA_FULL_ERROR=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29531}"
export FACEANALYSIS_CPU="${FACEANALYSIS_CPU:-1}"

PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
COMET_API_KEY="${COMET_API_KEY:-}"
export PM_PATH COMET_API_KEY
if [[ -z "${COMET_API_KEY}" ]]; then
    echo "COMET_API_KEY is not set." >&2
    exit 2
fi

echo "Training: ranks=${NUM_PROCESSES} local_batch=${TRAIN_BATCH_SIZE} global_batch=$((NUM_PROCESSES * TRAIN_BATCH_SIZE))"
echo "Validation: step0=24 aggregate images; full=96 every 2000 steps; local_batch=${VAL_BATCH_SIZE_PER_GPU}"

ACCELERATE_LOG_LEVEL=error \
TRANSFORMERS_VERBOSITY=error \
DIFFUSERS_VERBOSITY=error \
PYTHONWARNINGS="ignore::FutureWarning" \
COMET_DISABLE_AUTO_LOGGING=1 \
COMET_LOGGING_CONSOLE=ERROR \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
accelerate launch --config_file=src/configs/ddp/accelerate.yaml \
    --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" \
    --num_processes="${NUM_PROCESSES}" train.py \
    --config-name=one_id_ba_identity_dependence_N31 \
    datasets=all_datasets \
    train_dataset_name=cosmic_large \
    datasets.train.cosmic_large.num_refs=1 \
    val_datasets_names='[manual_val]' \
    +datasets.val.manual_val.restrict_ids_to_gen_bbox=true \
    datasets.val.manual_val.limit=null \
    dataloaders.manual_val.batch_size="${VAL_BATCH_SIZE_PER_GPU}" \
    dataloaders.manual_val.num_workers=1 \
    trainer.epoch_len=2000 \
    trainer.n_epochs=15 \
    validate_before_training=false \
    val_smoke_test="${VAL_SMOKE_TEST}" \
    val_smoke_test_limit=24 \
    validation_enable_vae_slicing=true \
    dataloaders.train.batch_size="${TRAIN_BATCH_SIZE}" \
    dataloaders.train.num_workers=6 \
    model.rank=32 \
    model.photomaker_path="${PM_PATH}" \
    model.weight_dtype=bf16 \
    +model.ba_uncond_face_fix=true \
    +model.ba_face_prompt_mode=id_only \
    validation_args.num_images_per_prompt=1 \
    validation_args.num_inference_steps=50 \
    validation_args.guidance_scale=5 \
    lr_scheduler.warmup_steps=200 \
    pipeline.variant=null \
    val_debug=false \
    lr_for_lora=1e-4 \
    trainer.max_grad_norm=1.0 \
    optimizer.weight_decay=1e-3 \
    automatic_bboxes=false \
    automatic_bboxes_every_val=false \
    force_log_first_auto_bbox=false \
    ba_patch_top_k=1.0 \
    ba_train_top_k=1.0 \
    non_ba_train=false \
    train_ba_only=true \
    trainer.masked_loss_step=2 \
    pretrained_model_for_validation_name_or_path=SG161222/RealVisXL_V4.0 \
    metrics=all_metrics \
    writer=cometml writer.run_name="${RUN_NAME}" \
    "$@"

