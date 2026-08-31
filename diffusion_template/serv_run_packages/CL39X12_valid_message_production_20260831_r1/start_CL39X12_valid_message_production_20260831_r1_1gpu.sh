#!/usr/bin/env bash
# Sealed CL39X12 production launch: valid-only message, legacy CL39 confidence.
set -euo pipefail

PACKAGE_ID="CL39X12_valid_message_production_20260831_r1"
OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
SOURCE_PACKAGE="${OWNER_ROOT}/analysis_jobs/CL39X12_valid_message_20260831_r1/package"
SOURCE_ARCHIVE="${SOURCE_PACKAGE}/source_36f6d27.tar.gz"
SOURCE_MANIFEST="${SOURCE_PACKAGE}/source_manifest.json"
SOURCE_SHA256="0b0f126a67513db0b38442f316b02eb894fd33fa116a84f3a2efd646c72a0def"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl39x12_valid_message_20260831_r1/${PACKAGE_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
DATASET_FULL_ROOT="${OWNER_ROOT}/rsrch_test/dataset_full"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

test "${PACKAGE_ID}" = CL39X12_valid_message_production_20260831_r1
test ! -e "${RUNTIME_ROOT}"
test -s "${SOURCE_ARCHIVE}" && test -s "${SOURCE_MANIFEST}"
printf '%s  %s\n' "${SOURCE_SHA256}" "${SOURCE_ARCHIVE}" | sha256sum -c -
mkdir -p "${RUNTIME_ROOT}"
tar --extract --gzip --file="${SOURCE_ARCHIVE}" --directory="${RUNTIME_ROOT}"
ln -s "${DATASET_FULL_ROOT}" "${RUNTIME_ROOT}/dataset_full"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
fi
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" verify \
  --root "${PROJECT_ROOT}" --manifest "${SOURCE_MANIFEST}"

cd "${PROJECT_ROOT}"
set -a
# shellcheck disable=SC1090
source "${OWNER_ROOT}/rsrch_test/diffusion_template/.env"
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export COMET_PROJECT=aug-large-ds CUDA_VISIBLE_DEVICES=0 ACCELERATE_NUM_PROCESSES=1
export PHOTOMAKER_FACEANALYSIS_CPU=1
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RUN_NAME=CL39X12_cosmic_valid_kv_legacy_confidence_24k_full96_r1
export CONFIG_NAME=CL39X12_cosmic_valid_kv_legacy_confidence_24k
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large/${RUN_NAME}.json"
test -s "${PM_PATH}" && test -s "${SUBJECT_V2_ID_EMBEDS}"
test -s "${COSMIC_LARGE_MANIFEST}" && test -s "${EXPERIMENT_SPEC_PATH}"
python tools/preflight_cl39x_cpu.py
exec bash launchers/active/run_CL39X01_CL39X08_cl39_followups_1gpu.sh
