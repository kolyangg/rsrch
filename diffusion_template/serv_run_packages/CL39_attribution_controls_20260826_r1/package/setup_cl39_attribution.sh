#!/usr/bin/env bash
# Shared sealed environment for CL39 fixed-96 attribution-only controls.
set -euo pipefail

: "${CL39_AUDIT_JOB_TAG:?Set a unique attribution job tag}"
: "${CL39_AUDIT_SOURCE_DIR:?Set the sealed attribution source directory}"

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_attribution_controls_20260826_r1"
PROJECT_ROOT="${TASK_ROOT}/${CL39_AUDIT_SOURCE_DIR}/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
CL39_RUN="CL39_cosmic_null_key_confidence_router_24k_full96_r4"
CL39_SEALED_ROOT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${CL39_RUN}/diffusion_template/saved/${CL39_RUN}"
CL39_CONFIG="CL39_cosmic_null_key_confidence_router_24k"
CL39_COMET_KEY="b1ca0b3da679401c85b991f1bbdf0b2a"
CL39_CHECKPOINT_16K="${CL39_SEALED_ROOT}/checkpoint-epoch8.pth"
CL39_CHECKPOINT_24K="${CL39_SEALED_ROOT}/checkpoint-epoch12.pth"
CL39_CHECKPOINT_16K_SHA256="a598b929e4fbfab7eac0f9474c9c96d1713dbac6224e1de6ffbca4b43ae29e86"
CL39_CHECKPOINT_24K_SHA256="74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  for candidate in /home/jovyan/miniconda3 /home/jovyan/anaconda3 /opt/conda; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      CONDA_BASE="${candidate}"
      break
    fi
  done
fi
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" verify \
  --root "${PROJECT_ROOT}" --manifest "${TASK_ROOT}/${CL39_AUDIT_SOURCE_DIR}_manifest.json"
printf '%s  %s\n' "${CL39_CHECKPOINT_16K_SHA256}" "${CL39_CHECKPOINT_16K}" | sha256sum -c -
printf '%s  %s\n' "${CL39_CHECKPOINT_24K_SHA256}" "${CL39_CHECKPOINT_24K}" | sha256sum -c -

DATASET_LINK="${TASK_ROOT}/${CL39_AUDIT_SOURCE_DIR}/dataset_full"
if [[ -e "${DATASET_LINK}" && ! -L "${DATASET_LINK}" ]]; then
  echo "Refusing to replace non-symlink dataset path: ${DATASET_LINK}" >&2
  exit 2
fi
ln -sfn "${OWNER_ROOT}/rsrch_test/dataset_full" "${DATASET_LINK}"

export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export MPLCONFIGDIR="${TASK_ROOT}/runtime/${CL39_AUDIT_JOB_TAG}/matplotlib"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_ALBUMENTATIONS_UPDATE=1

test -s "${PM_PATH}"
test -s "${COSMIC_LARGE_MANIFEST}"
test -s "${SUBJECT_V2_ID_EMBEDS}"
mkdir -p "${MPLCONFIGDIR}" "${TASK_ROOT}/saved" "${TASK_ROOT}/gates"
cd "${PROJECT_ROOT}"
python -m py_compile \
  src/model/photomaker_branched/lora2.py \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/pipelines/photomaker_branched_clean.py \
  src/trainer/sdxl_trainers.py \
  tools/analysis/cl39_attention_capture.py
