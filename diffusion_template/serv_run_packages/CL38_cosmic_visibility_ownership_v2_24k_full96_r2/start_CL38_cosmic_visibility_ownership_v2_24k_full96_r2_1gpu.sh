#!/usr/bin/env bash
# Serv wrapper rendered once per CL38-CL45 source snapshot.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
RUN_ID="CL38_cosmic_visibility_ownership_v2_24k_full96_r2"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
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
export RUN_NAME="${RUN_ID}"
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large/${RUN_ID}.json"
export COMET_PROJECT=aug-large-ds CUDA_VISIBLE_DEVICES=0 ACCELERATE_NUM_PROCESSES=1
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

case "${RUN_ID}" in
  CL38_*) export CONFIG_NAME=CL38_cosmic_visibility_ownership_v2_24k ;;
  CL39_*) export CONFIG_NAME=CL39_cosmic_null_key_confidence_router_24k ;;
  CL40_*) export CONFIG_NAME=CL40_cosmic_identity_motion_projector_24k ;;
  CL41_*) export CONFIG_NAME=CL41_cosmic_landmark_canonical_kv_24k ;;
  CL42_*) export CONFIG_NAME=CL42_cosmic_component_token_memory_24k ;;
  CL43_*) export CONFIG_NAME=CL43_cosmic_id_adaptive_modulation_24k ;;
  CL44_*) export CONFIG_NAME=CL44_cosmic_semantic_window_gate_24k ;;
  CL45_*) export CONFIG_NAME=CL45_cosmic_ba_pcgrad_24k ;;
  *) echo "Unknown CL38-CL45 run: ${RUN_ID}" >&2; exit 2 ;;
esac

test -s "${PM_PATH}" && test -s "${COSMIC_LARGE_MANIFEST}"
test -s "${SUBJECT_V2_ID_EMBEDS}" && test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
exec bash launchers/active/run_CL38_CL45_cl27_architecture_1gpu.sh
