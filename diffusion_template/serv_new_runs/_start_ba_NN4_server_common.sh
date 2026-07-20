#!/usr/bin/env bash
set -euo pipefail

: "${NN4_NUM_PROCESSES:?NN4_NUM_PROCESSES is required}"
: "${NN4_RUN_NAME:?NN4_RUN_NAME is required}"
: "${NN4_CUDA_DEVICES:?NN4_CUDA_DEVICES is required}"
: "${NN4_MASTER_PORT:?NN4_MASTER_PORT is required}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
    log "ERROR: $*"
    exit 1
}

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="${candidate}"
            break
        fi
    done
fi
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}" || fail "Cannot activate ${CONDA_ENV_PATH}"

[[ -f "${ENV_FILE}" ]] || fail "Missing ${ENV_FILE}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
: "${COMET_API_KEY:?COMET_API_KEY is missing in ${ENV_FILE}}"
PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
[[ -f "${PM_PATH}" ]] || fail "PhotoMaker checkpoint not found: ${PM_PATH}"

python -c 'from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained="vggface2")' \
    >/dev/null || fail "NN4 requires facenet-pytorch and cached/downloadable VGGFace2 weights"

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/insightface}"
export FACEANALYSIS_CPU="${FACEANALYSIS_CPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-2000}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
TRAIN_SEED="${TRAIN_SEED:-0}"
VAL_SEEDS="${VAL_SEEDS:-[0]}"
NN4_VALIDATION_MODEL="${NN4_VALIDATION_MODEL:-SG161222/RealVisXL_V4.0}"

if (( STEPS_PER_EPOCH * NUM_EPOCHS != 20000 )); then
    fail "NN4 requires exactly 20,000 optimizer steps; got $((STEPS_PER_EPOCH * NUM_EPOCHS))"
fi

cd "${PROJECT_DIR}"
log "NN4 server run: name=${NN4_RUN_NAME} GPUs=${NN4_CUDA_DEVICES} processes=${NN4_NUM_PROCESSES}"
log "Dataset=cosmic_large (NFS paths), batch/rank=${TRAIN_BATCH_SIZE}, global_batch=$((TRAIN_BATCH_SIZE * NN4_NUM_PROCESSES))"
log "Budget=20000 steps; validation_base=${NN4_VALIDATION_MODEL}; 96 images at step 0 and every ${STEPS_PER_EPOCH}"

ACCELERATE_LOG_LEVEL=error \
TRANSFORMERS_VERBOSITY=error \
DIFFUSERS_VERBOSITY=error \
PYTHONWARNINGS="ignore::FutureWarning" \
COMET_DISABLE_AUTO_LOGGING=1 \
COMET_LOGGING_CONSOLE=ERROR \
CUDA_VISIBLE_DEVICES="${NN4_CUDA_DEVICES}" \
accelerate launch --config_file=src/configs/ddp/accelerate.yaml \
    --main_process_ip="${MASTER_ADDR}" \
    --main_process_port="${NN4_MASTER_PORT}" \
    --num_processes="${NN4_NUM_PROCESSES}" train.py \
    --config-name=one_id_ba_NN4_causal_null_up0 \
    datasets=all_datasets \
    train_dataset_name=cosmic_large \
    datasets.train.cosmic_large.num_refs=1 \
    +datasets.train.cosmic_large.ref_crop_margin_min=0.2 \
    +datasets.train.cosmic_large.ref_crop_margin_max=0.6 \
    +datasets.train.cosmic_large.ref_downscale_jitter=0.5 \
    val_datasets_names='[manual_val]' \
    datasets.val.manual_val.limit=96 \
    datasets.val.manual_val.seeds="${VAL_SEEDS}" \
    dataloaders.manual_val.batch_size="${VAL_BATCH_SIZE}" \
    dataloaders.manual_val.num_workers=1 \
    trainer.epoch_len="${STEPS_PER_EPOCH}" \
    trainer.n_epochs="${NUM_EPOCHS}" \
    trainer.seed="${TRAIN_SEED}" \
    dataloaders.train.batch_size="${TRAIN_BATCH_SIZE}" \
    dataloaders.train.grad_accum_enabled=false \
    dataloaders.train.batch_size_eff="${TRAIN_BATCH_SIZE}" \
    dataloaders.train.num_workers=12 \
    model.rank=32 \
    model.photomaker_path="${PM_PATH}" \
    model.weight_dtype=bf16 \
    validation_args.num_images_per_prompt=1 \
    validation_args.num_inference_steps=50 \
    validation_args.guidance_scale=5 \
    lr_scheduler.warmup_steps=2000 \
    pipeline.variant=null \
    val_debug=false \
    lr_for_lora=5e-5 \
    trainer.max_grad_norm=1.0 \
    optimizer.weight_decay=1e-2 \
    automatic_bboxes=false \
    automatic_bboxes_every_val=false \
    force_log_first_auto_bbox=false \
    trainer.masked_loss_step=2 \
    pretrained_model_for_validation_name_or_path="${NN4_VALIDATION_MODEL}" \
    metrics=all_metrics \
    writer=cometml \
    writer.run_name="${NN4_RUN_NAME}" \
    "$@"
