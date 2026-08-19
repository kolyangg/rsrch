#!/usr/bin/env bash
# Serv wrapper rendered once per CL30-CL37 run package.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
RUN_ID="CL31_cosmic_attention_ownership_alignment_24k_full96_r1"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl30_cl37_v1/${RUN_ID}"
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
  CL30_*) export CONFIG_NAME=CL30_cosmic_positive_lowband_sameid_24k ;;
  CL31_*) export CONFIG_NAME=CL31_cosmic_attention_ownership_alignment_24k ;;
  CL32_*) export CONFIG_NAME=CL32_cosmic_contact_frequency_surface_24k ;;
  CL33_*) export CONFIG_NAME=CL33_cosmic_visibility_balanced_reconstruction_24k ;;
  CL34_*) export CONFIG_NAME=CL34_cosmic_shared_frequency_calibration_24k ;;
  CL35_*) export CONFIG_NAME=CL35_cosmic_attention_gated_patch_identity_24k ;;
  CL36_*)
    export CONFIG_NAME=CL36_cosmic_ba_arcface_hinge_4k
    export CL36_SOURCE_CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl27_cl29_v4/CL27_cosmic_frequency_surface_energy_24k_full96_r3/diffusion_template/saved/CL27_cosmic_frequency_surface_energy_24k_full96_r3/weights-epoch8.pth"
    export ARCFACE_ONNX_PATH="/home/jovyan/.insightface/models/buffalo_l/w600k_r50.onnx"
    ;;
  CL37_*) export CONFIG_NAME=CL37_cosmic_smallface_roi_teacher_distill_24k ;;
  *) echo "Unknown CL30-CL37 run: ${RUN_ID}" >&2; exit 2 ;;
esac

test -s "${PM_PATH}" && test -s "${COSMIC_LARGE_MANIFEST}"
test -s "${SUBJECT_V2_ID_EMBEDS}" && test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
exec bash launchers/active/run_CL30_CL37_cl27_followups_1gpu.sh
