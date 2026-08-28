#!/usr/bin/env bash
# Assemble and render the completed Serv trainer/YAML CL39 audit.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_attention_audit_serv_final_actual"
PROJECT_ROOT="${TASK_ROOT}/source/diffusion_template"
RUN_NAME="CL39_cosmic_null_key_confidence_router_24k_full96_r4"
CHECKPOINT_DIR="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${RUN_NAME}/diffusion_template/saved/${RUN_NAME}"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

for arm in actual c1 ba_off; do
  test -s "${OWNER_ROOT}/analysis_jobs/CL39_attention_audit_serv_final_${arm}/AUDIT_ARM_COMPLETE"
done
cd "${TASK_ROOT}"
sha256sum -c render_manifest.sha256

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

export HOME="${TASK_ROOT}/home"
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export MPLCONFIGDIR="${TASK_ROOT}/home/.config/matplotlib"
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

ACTUAL_VAL="${TASK_ROOT}/saved/CL39_attention_audit_serv_actual/val_images/manual_val"
C1_VAL="${OWNER_ROOT}/analysis_jobs/CL39_attention_audit_serv_final_c1/saved/CL39_attention_audit_serv_c1/val_images/manual_val"
BA_OFF_VAL="${OWNER_ROOT}/analysis_jobs/CL39_attention_audit_serv_final_ba_off/saved/CL39_attention_audit_serv_ba_off/val_images/manual_val"
ASSEMBLED="${TASK_ROOT}/assembled"
FIGURES="${PROJECT_ROOT}/analysis/assets/cl39_attention_24k_serv_a100"

cd "${PROJECT_ROOT}"
python tools/analysis/assemble_cl39_serv_audit.py \
  --manifest "${TASK_ROOT}/package/report_inputs/sample_manifest.json" \
  --actual-dir "${ACTUAL_VAL}" \
  --c1-dir "${C1_VAL}" \
  --ba-off-dir "${BA_OFF_VAL}" \
  --telemetry-dir "${TASK_ROOT}/telemetry" \
  --reference-root "${TASK_ROOT}/source/dataset_full/val_dataset/references" \
  --output-root "${ASSEMBLED}"

python tools/analysis/analyze_cl39_attention.py render \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-root "${ASSEMBLED}" \
  --figure-dir "${FIGURES}"

test -s "${ASSEMBLED}/summary.json"
test -s "${ASSEMBLED}/per_sample_summary.csv"
test "$(find "${FIGURES}/samples" -type f -name '*.png' | wc -l)" -eq 16
printf 'complete\n' > "${TASK_ROOT}/AUDIT_RENDER_COMPLETE"
