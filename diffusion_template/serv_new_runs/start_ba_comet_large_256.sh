#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
CONDA_ENV_PATH="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS"
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
    log "ERROR: $*"
    exit 1
}

find_conda_base() {
    if command -v conda >/dev/null 2>&1; then
        conda info --base
        return 0
    fi

    if [[ -n "${CONDA_EXE:-}" ]]; then
        dirname "$(dirname "${CONDA_EXE}")"
        return 0
    fi

    local candidate
    for candidate in \
        "${HOME}/miniconda3" \
        "${HOME}/anaconda3" \
        "/opt/conda" \
        "/usr/local/conda"
    do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

log "Launcher started"
log "Project directory: ${PROJECT_DIR}"

CONDA_BASE="$(find_conda_base)" || fail "Could not locate a Conda installation"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
[[ -f "${CONDA_SH}" ]] || fail "Conda init script not found at ${CONDA_SH}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
log "Loaded Conda initialization from ${CONDA_SH}"

conda activate "${CONDA_ENV_PATH}" || fail "Failed to activate Conda env ${CONDA_ENV_PATH}"
log "Conda environment activated: ${CONDA_ENV_PATH}"
log "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"
PYTHON_BIN="$(command -v python || true)"
ACCELERATE_BIN="$(command -v accelerate || true)"
[[ -n "${PYTHON_BIN}" ]] || fail "Python is not available after activating ${CONDA_ENV_PATH}"
[[ -n "${ACCELERATE_BIN}" ]] || fail "accelerate is not available after activating ${CONDA_ENV_PATH}"
log "Python executable: ${PYTHON_BIN}"
log "Accelerate executable: ${ACCELERATE_BIN}"

[[ -f "${ENV_FILE}" ]] || fail "Missing env file: ${ENV_FILE}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
[[ -n "${COMET_API_KEY:-}" ]] || fail "COMET_API_KEY is missing in ${ENV_FILE}"
[[ "${COMET_API_KEY}" != "your_comet_api_key_here" ]] || fail "Replace the placeholder COMET_API_KEY in ${ENV_FILE}"
log "Loaded COMET_API_KEY from ${ENV_FILE}"
[[ -n "${PM_PATH:-}" ]] || fail "PM_PATH is missing in ${ENV_FILE}"
[[ -f "${PM_PATH}" ]] || fail "PhotoMaker checkpoint not found at PM_PATH=${PM_PATH}"
log "Using PhotoMaker checkpoint: ${PM_PATH}"

cd "${PROJECT_DIR}"
log "Changed directory to ${PROJECT_DIR}"
log "Starting training command"

if ACCELERATE_LOG_LEVEL=error \
    TRANSFORMERS_VERBOSITY=error \
    DIFFUSERS_VERBOSITY=error \
    PYTHONWARNINGS="ignore::FutureWarning" \
    COMET_DISABLE_AUTO_LOGGING=1 \
    COMET_LOGGING_CONSOLE=ERROR \
    CUDA_VISIBLE_DEVICES=0 \
    COMET_API_KEY="${COMET_API_KEY}" \
    accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 train.py \
        --config-name=one_id_09Feb_testing \
        datasets=all_datasets \
        train_dataset_name=cosmic_large \
        val_datasets_names='[manual_val]' \
        trainer.epoch_len=2000 \
        dataloaders.train.batch_size=4 \
        dataloaders.train.num_workers=12 \
        model.rank=32 \
        model.photomaker_path="${PM_PATH}" \
        validation_args.num_images_per_prompt=1 \
        lr_scheduler.warmup_steps=2000 \
        model.weight_dtype=bf16 \
        pipeline.variant=null \
        dataloaders.manual_val.batch_size=12 \
        datasets.val.manual_val.limit=96 \
        val_debug=false \
        branched_attn_weight_mode=noise_and_ref \
        branched_attn_new_weight_kind=lora \
        lr_for_lora=1e-4 \
        automatic_bboxes=true \
        automatic_bboxes_every_val=false \
        force_log_first_auto_bbox=true \
        train_branched_ca_lora=true \
        ba_patch_top_k=1.0 \
        ba_train_top_k=1.0 \
        non_ba_train=false \
        train_ba_only=true \
        loss_kind=masked_alternating \
        trainer.masked_loss_step=2 \
        train_ba_all_steps=true \
        train_on_separate_image=true \
        train_dataset_upscale_to_1024=false \ 
        metrics=all_metrics \
        val_datasets_names='[manual_val]' \
        writer=cometml writer.run_name="comet_large_256"; then
    log "Training finished successfully"
else
    status=$?
    log "Training failed with exit code ${status}"
    exit "${status}"
fi
