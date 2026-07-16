#!/usr/bin/env bash
set -euo pipefail

# N35: N34's corrected route/objective with canonical eight-part identity
# memory. Two ranks, local batch 1, accumulation 4 -> global batch 8.

RUN_NAME="${RUN_NAME:-ba_causal_canonical_parts_2gpu_N35}"
VAL_SMOKE_TEST="${VAL_SMOKE_TEST:-true}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
LOCAL_EFFECTIVE_BATCH="${LOCAL_EFFECTIVE_BATCH:-4}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-3}"
if (( LOCAL_EFFECTIVE_BATCH % TRAIN_BATCH_SIZE != 0 )); then
    echo "LOCAL_EFFECTIVE_BATCH must be divisible by TRAIN_BATCH_SIZE." >&2
    exit 2
fi
ACCUM_STEPS=$((LOCAL_EFFECTIVE_BATCH / TRAIN_BATCH_SIZE))
MICROBATCHES_PER_EPOCH=$((1000 * ACCUM_STEPS))
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
        RUN_NAME="${RUN_NAME}" VAL_SMOKE_TEST="${VAL_SMOKE_TEST}" \
        NUM_PROCESSES="${NUM_PROCESSES}" TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
        LOCAL_EFFECTIVE_BATCH="${LOCAL_EFFECTIVE_BATCH}" \
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
export MASTER_PORT="${MASTER_PORT:-29535}"
export FACEANALYSIS_CPU="${FACEANALYSIS_CPU:-1}"

PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
COMET_API_KEY="${COMET_API_KEY:-}"
export PM_PATH COMET_API_KEY
if [[ -z "${COMET_API_KEY}" ]]; then
    echo "COMET_API_KEY is not set." >&2
    exit 2
fi

echo "N35: N34 route plus five-landmark canonical eight-part identity memory"
echo "Training: ranks=${NUM_PROCESSES} local_batch=${TRAIN_BATCH_SIZE} accumulation=${ACCUM_STEPS} global_effective=$((NUM_PROCESSES * LOCAL_EFFECTIVE_BATCH))"
echo "Validation: step0 smoke=${VAL_SMOKE_TEST}; full 96 images every 1000 optimizer steps; total=10000 optimizer steps"

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
    --config-name=one_id_ba_causal_canonical_parts_N35 \
    datasets=all_datasets \
    train_dataset_name=cosmic_large \
    datasets.train.cosmic_large.num_refs=1 \
    val_datasets_names='[manual_val]' \
    +datasets.val.manual_val.restrict_ids_to_gen_bbox=true \
    datasets.val.manual_val.limit=null \
    dataloaders.manual_val.batch_size="${VAL_BATCH_SIZE_PER_GPU}" \
    dataloaders.manual_val.num_workers=1 \
    trainer.epoch_len="${MICROBATCHES_PER_EPOCH}" \
    trainer.n_epochs=10 \
    validate_before_training=false \
    val_smoke_test="${VAL_SMOKE_TEST}" \
    val_smoke_test_limit=24 \
    validation_enable_vae_slicing=true \
    dataloaders.train.batch_size="${TRAIN_BATCH_SIZE}" \
    dataloaders.train.grad_accum_enabled=true \
    dataloaders.train.batch_size_eff="${LOCAL_EFFECTIVE_BATCH}" \
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
