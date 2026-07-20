#!/usr/bin/env bash
set -euo pipefail

# Shared N37-style runner for NN1a-NN1f. Invoke a named launcher, not this file.
: "${NN1_CONFIG_NAME:?NN1_CONFIG_NAME is required}"
: "${NN1_RUN_NAME_DEFAULT:?NN1_RUN_NAME_DEFAULT is required}"
: "${NN1_DEFAULT_GPU:?NN1_DEFAULT_GPU is required}"
: "${NN1_DEFAULT_PORT:?NN1_DEFAULT_PORT is required}"
: "${NN1_DESCRIPTION:?NN1_DESCRIPTION is required}"
: "${NN1_LAUNCHER_PATH:?NN1_LAUNCHER_PATH is required}"

RUN_NAME="${RUN_NAME:-${NN1_RUN_NAME_DEFAULT}}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
LOCAL_EFFECTIVE_BATCH="${LOCAL_EFFECTIVE_BATCH:-2}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-12}"
OPTIMIZER_STEPS_PER_EPOCH="${OPTIMIZER_STEPS_PER_EPOCH:-2000}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
WARMUP_OPTIMIZER_STEPS="${WARMUP_OPTIMIZER_STEPS:-2000}"
FULL_STEP0_VAL="${FULL_STEP0_VAL:-true}"
VALIDATION_MODEL="${NN1_VALIDATION_MODEL:-SG161222/RealVisXL_V4.0}"
TRAIN_DATASET_NAME="${NN1_TRAIN_DATASET_NAME:-cosmic_large}"
TRAIN_SEED="${TRAIN_SEED:-0}"
VAL_SEEDS="${VAL_SEEDS:-[0]}"
HYDRA_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "full_step0_val" ]]; then
        FULL_STEP0_VAL=true
    else
        HYDRA_ARGS+=("${arg}")
    fi
done

if [[ "${FULL_STEP0_VAL}" != "true" ]]; then
    echo "NN1 comparison runs require full step-0 validation; set FULL_STEP0_VAL=true." >&2
    exit 2
fi
if (( NUM_PROCESSES != 1 )); then
    echo "NN1a-NN1f are defined as one-process, one-GPU experiments." >&2
    exit 2
fi
if (( LOCAL_EFFECTIVE_BATCH % TRAIN_BATCH_SIZE != 0 )); then
    echo "LOCAL_EFFECTIVE_BATCH must be divisible by TRAIN_BATCH_SIZE." >&2
    exit 2
fi

ACCUM_STEPS=$((LOCAL_EFFECTIVE_BATCH / TRAIN_BATCH_SIZE))
if (( ACCUM_STEPS > 1 )); then
    GRAD_ACCUM_ENABLED=true
else
    GRAD_ACCUM_ENABLED=false
fi
MICROBATCHES_PER_EPOCH=$((OPTIMIZER_STEPS_PER_EPOCH * ACCUM_STEPS))
TOTAL_OPTIMIZER_STEPS=$((OPTIMIZER_STEPS_PER_EPOCH * NUM_EPOCHS))
SCHEDULER_WARMUP_STEPS=$((WARMUP_OPTIMIZER_STEPS * NUM_PROCESSES))
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs_new_runs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"

if [[ "${RUN_FOREGROUND:-0}" != "1" && "${DETACHED_RUN:-0}" != "1" ]]; then
    mkdir -p "${LOG_DIR}"
    echo "Starting ${RUN_NAME} detached on physical GPU ${CUDA_VISIBLE_DEVICES:-${NN1_DEFAULT_GPU}}"
    echo "Log: ${LOG_FILE}"
    DETACHED_RUN=1 LOG_DIR="${LOG_DIR}" LOG_FILE="${LOG_FILE}" \
        RUN_NAME="${RUN_NAME}" NUM_PROCESSES="${NUM_PROCESSES}" \
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
        LOCAL_EFFECTIVE_BATCH="${LOCAL_EFFECTIVE_BATCH}" \
        VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU}" \
        OPTIMIZER_STEPS_PER_EPOCH="${OPTIMIZER_STEPS_PER_EPOCH}" \
        NUM_EPOCHS="${NUM_EPOCHS}" \
        WARMUP_OPTIMIZER_STEPS="${WARMUP_OPTIMIZER_STEPS}" \
        FULL_STEP0_VAL="${FULL_STEP0_VAL}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${NN1_DEFAULT_GPU}}" \
        MASTER_PORT="${MASTER_PORT:-${NN1_DEFAULT_PORT}}" \
        nohup bash "${NN1_LAUNCHER_PATH}" "${HYDRA_ARGS[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
    echo "PID: $!"
    echo "Follow with: tail -f ${LOG_FILE}"
    exit 0
fi

cd "${PROJECT_DIR}"
export HYDRA_FULL_ERROR=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-${NN1_DEFAULT_PORT}}"
export FACEANALYSIS_CPU="${FACEANALYSIS_CPU:-1}"

PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
COMET_API_KEY="${COMET_API_KEY:-}"
export PM_PATH COMET_API_KEY
if [[ -z "${COMET_API_KEY}" ]]; then
    echo "COMET_API_KEY is not set." >&2
    exit 2
fi
if [[ ! -f "${PM_PATH}" ]]; then
    echo "PhotoMaker checkpoint not found: ${PM_PATH}" >&2
    exit 2
fi

if [[ "${NN1_REQUIRE_ID_LOSS:-0}" == "1" ]]; then
    if ! python -c 'from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained="vggface2")' >/dev/null; then
        echo "This run requires facenet-pytorch and cached/downloadable VGGFace2 weights." >&2
        echo "Install safely with: pip install --no-deps facenet-pytorch" >&2
        exit 2
    fi
fi

echo "${NN1_DESCRIPTION}"
echo "Config: ${NN1_CONFIG_NAME}"
echo "Training: GPU=${CUDA_VISIBLE_DEVICES:-${NN1_DEFAULT_GPU}} dataset=${TRAIN_DATASET_NAME} ranks=1 physical_batch=${TRAIN_BATCH_SIZE} accumulation=${ACCUM_STEPS} effective_batch=${LOCAL_EFFECTIVE_BATCH}"
echo "Budget: ${TOTAL_OPTIMIZER_STEPS} optimizer steps (${NUM_EPOCHS} x ${OPTIMIZER_STEPS_PER_EPOCH})"
echo "Validation: base=${VALIDATION_MODEL}; train_seed=${TRAIN_SEED}; val_seeds=${VAL_SEEDS}; batch=${VAL_BATCH_SIZE_PER_GPU}; full fixed 96 images at step 0 and every ${OPTIMIZER_STEPS_PER_EPOCH} optimizer steps"

ACCELERATE_LOG_LEVEL=error \
TRANSFORMERS_VERBOSITY=error \
DIFFUSERS_VERBOSITY=error \
PYTHONWARNINGS="ignore::FutureWarning" \
COMET_DISABLE_AUTO_LOGGING=1 \
COMET_LOGGING_CONSOLE=ERROR \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${NN1_DEFAULT_GPU}}" \
accelerate launch --config_file=src/configs/ddp/accelerate.yaml \
    --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" \
    --num_processes=1 train.py \
    --config-name="${NN1_CONFIG_NAME}" \
    datasets=all_datasets \
    train_dataset_name="${TRAIN_DATASET_NAME}" \
    "datasets.train.${TRAIN_DATASET_NAME}.num_refs=1" \
    "+datasets.train.${TRAIN_DATASET_NAME}.ref_crop_margin_min=0.2" \
    "+datasets.train.${TRAIN_DATASET_NAME}.ref_crop_margin_max=0.6" \
    "+datasets.train.${TRAIN_DATASET_NAME}.ref_downscale_jitter=0.5" \
    val_datasets_names='[manual_val]' \
    datasets.val.manual_val.limit=96 \
    datasets.val.manual_val.seeds="${VAL_SEEDS}" \
    dataloaders.manual_val.batch_size="${VAL_BATCH_SIZE_PER_GPU}" \
    dataloaders.manual_val.num_workers=1 \
    trainer.epoch_len="${MICROBATCHES_PER_EPOCH}" \
    trainer.n_epochs="${NUM_EPOCHS}" \
    trainer.seed="${TRAIN_SEED}" \
    dataloaders.train.batch_size="${TRAIN_BATCH_SIZE}" \
    dataloaders.train.grad_accum_enabled="${GRAD_ACCUM_ENABLED}" \
    dataloaders.train.batch_size_eff="${LOCAL_EFFECTIVE_BATCH}" \
    dataloaders.train.num_workers=6 \
    model.rank=32 \
    model.photomaker_path="${PM_PATH}" \
    model.weight_dtype=bf16 \
    validation_args.num_images_per_prompt=1 \
    validation_args.num_inference_steps=50 \
    validation_args.guidance_scale=5 \
    lr_scheduler.warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
    pipeline.variant=null \
    val_debug=false \
    lr_for_lora=5e-5 \
    trainer.max_grad_norm=1.0 \
    optimizer.weight_decay=1e-2 \
    automatic_bboxes=false \
    automatic_bboxes_every_val=false \
    force_log_first_auto_bbox=false \
    trainer.masked_loss_step=2 \
    pretrained_model_for_validation_name_or_path="${VALIDATION_MODEL}" \
    metrics=all_metrics \
    writer=cometml writer.run_name="${RUN_NAME}" \
    "${HYDRA_ARGS[@]}"
