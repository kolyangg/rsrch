#!/usr/bin/env bash
# Serv wrapper template; package renderer replaces the three @@...@@ values.
set -euo pipefail
OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
RUN_ID="CL39X01_cosmic_valid_key_attention_onebatch_smoke_r2"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl39x_v2/${RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
DATASET_FULL_ROOT="${OWNER_ROOT}/rsrch_test/dataset_full"
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
# Sealed sources sit outside rsrch_test; restore the fixed ../dataset_full
# validation path without copying or modifying the shared data.
test -d "${DATASET_FULL_ROOT}/val_dataset/references"
if [[ -e "${RUNTIME_ROOT}/dataset_full" && ! -L "${RUNTIME_ROOT}/dataset_full" ]]; then
  echo "Refusing to replace non-symlink dataset path: ${RUNTIME_ROOT}/dataset_full" >&2
  exit 2
fi
ln -sfn "${DATASET_FULL_ROOT}" "${RUNTIME_ROOT}/dataset_full"
cd "${PROJECT_ROOT}"
set -a; source "${OWNER_ROOT}/rsrch_test/diffusion_template/.env"; set +a
export ENV_FILE=/dev/null PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME="${RUN_ID}" EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large/${RUN_ID}.json"
export COMET_PROJECT=aug-large-ds CUDA_VISIBLE_DEVICES=0 ACCELERATE_NUM_PROCESSES=1
export AUTOMASK_OS_CACHE_ROOT="${OWNER_ROOT}/dataset_cache/automask_os_v1"
export AUTOMASK_OS_VAL_CACHE_ROOT="${OWNER_ROOT}/dataset_cache/automask_os_val_v1"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
case "${RUN_ID}" in
  CL39X01_*) CONFIG_NAME=CL39X01_cosmic_valid_key_attention_24k ;;
  CL39X02_*) CONFIG_NAME=CL39X02_cosmic_cycle_confidence_24k ;;
  CL39X03_*) CONFIG_NAME=CL39X03_cosmic_stage_split_ot_transport_24k ;;
  CL39X04_*) CONFIG_NAME=CL39X04_cosmic_small_face_roi_route_24k ;;
  CL39X05_*) CONFIG_NAME=CL39X05_cosmic_automask_os_24k ;;
  CL39X06_*) CONFIG_NAME=CL39X06_cosmic_counterfactual_reference_24k ;;
  CL39X07_*) CONFIG_NAME=CL39X07_cosmic_intrinsic_id_sidecar_24k ;;
  CL39X08_*) CONFIG_NAME=CL39X08_cosmic_global_local_balance_24k ;;
  *) echo "Unknown CL39-X run: ${RUN_ID}" >&2; exit 2 ;;
esac
export CONFIG_NAME
if [[ "${RUN_ID}" == *_onebatch_smoke_* ]]; then
  export CL39X_ONEBATCH_SMOKE=1 COSMIC_PREFLIGHT_SAMPLES=8
fi
test -s "${PM_PATH}" && test -s "${COSMIC_LARGE_MANIFEST}"
test -s "${SUBJECT_V2_ID_EMBEDS}" && test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
exec bash launchers/active/run_CL39X01_CL39X08_cl39_followups_1gpu.sh
