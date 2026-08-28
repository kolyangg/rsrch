#!/usr/bin/env bash
# Serv wrapper for the BC39 ownership-mask recovery experiment.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
RUN_ID="BC39_big_celebs_null_key_confidence_router_24k_full96_r2"
CONFIG_NAME="BC39_big_celebs_null_key_confidence_router_24k"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_clean_full_v1/${RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
DATASET_FULL_ROOT="${OWNER_ROOT}/rsrch_test/dataset_full"
BIG_CELEBS_ROOT="${OWNER_ROOT}/datasets/bigcelebs/releases/v2"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then CONDA_BASE="${candidate}"; break; fi
  done
fi
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" verify \
  --root "${PROJECT_ROOT}" --manifest "${SOURCE_MANIFEST}"

# The sealed source is outside rsrch_test, while fixed validation paths resolve
# through the sibling ../dataset_full tree.
test -d "${DATASET_FULL_ROOT}/val_dataset/references"
if [[ -e "${RUNTIME_ROOT}/dataset_full" && ! -L "${RUNTIME_ROOT}/dataset_full" ]]; then
  echo "Refusing to replace non-symlink dataset path: ${RUNTIME_ROOT}/dataset_full" >&2
  exit 2
fi
ln -sfn "${DATASET_FULL_ROOT}" "${RUNTIME_ROOT}/dataset_full"
cd "${PROJECT_ROOT}"

set -a
# shellcheck disable=SC1090
source "${OWNER_ROOT}/rsrch_test/diffusion_template/.env"
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export BIG_CELEBS_MANIFEST="${BIG_CELEBS_ROOT}/filtered_ids3_adj.json"
export BIG_CELEBS_IMAGES="${BIG_CELEBS_ROOT}/large_dataset"
export BIG_CELEBS_SEAL="${BIG_CELEBS_ROOT}/dataset_manifest.json"
export BIG_CELEBS_DOWNLOAD_LOG="${OWNER_ROOT}/datasets/dataset_tools/download_bigcelebs_v2.log"
export BIG_CELEBS_EXPECTED_MANIFEST_SHA256="f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa"
export BIG_CELEBS_MIN_FACE_RES=192
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export PYTHON_BIN="${CONDA_ENV}/bin/python"
export ACCELERATE_BIN="${CONDA_ENV}/bin/accelerate"
export RUN_NAME="${RUN_ID}"
export CONFIG_NAME
export COMET_PROJECT=aug-large-ds CUDA_VISIBLE_DEVICES=0 ACCELERATE_NUM_PROCESSES=1
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

test -s "${PM_PATH}"
test -s "${BIG_CELEBS_MANIFEST}"
test -d "${BIG_CELEBS_IMAGES}"
test -s "${BIG_CELEBS_SEAL}"
test -s "${BIG_CELEBS_DOWNLOAD_LOG}"
test -s "${SUBJECT_V2_ID_EMBEDS}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
exec bash launchers/active/run_clean_full_config_1gpu.sh
