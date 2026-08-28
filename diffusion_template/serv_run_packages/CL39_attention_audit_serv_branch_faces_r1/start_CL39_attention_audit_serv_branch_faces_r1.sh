#!/usr/bin/env bash
# Exact CL39 batch-12 R-on-face validation intervention and branch renderer.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_attention_audit_serv_branch_faces_r1"
PROJECT_ROOT="${TASK_ROOT}/source/diffusion_template"
RUN_NAME="CL39_cosmic_null_key_confidence_router_24k_full96_r4"
CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${RUN_NAME}/diffusion_template/saved/${RUN_NAME}/checkpoint-epoch12.pth"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

cd "${TASK_ROOT}"
sha256sum -c package_manifest.sha256
sha256sum -c source_manifest.sha256
printf '%s  %s\n' \
  '74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07' \
  "${CHECKPOINT}" | sha256sum -c -

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
export CL39_CHECKPOINT="${CHECKPOINT}"
export CL39_AUDIT_SAVE_DIR="${TASK_ROOT}/saved"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export MPLCONFIGDIR="${TASK_ROOT}/home/.config/matplotlib"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_ALBUMENTATIONS_UPDATE=1

cd "${PROJECT_ROOT}"
python -m py_compile \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/trainer/sdxl_trainers.py \
  tools/analysis/cl39_attention_capture.py \
  tools/analysis/render_cl39_branch_faces.py

accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py --config-name=CL39_attention_audit_serv_reference_face

REFERENCE_FACE_VAL="${TASK_ROOT}/saved/CL39_attention_audit_serv_reference_face/val_images/manual_val"
ACTUAL_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_attention_audit_serv_final_actual"
NATIVE_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_attention_audit_serv_final_ba_off"
ACTUAL_VAL="${ACTUAL_ROOT}/saved/CL39_attention_audit_serv_actual/val_images/manual_val"
NATIVE_VAL="${NATIVE_ROOT}/saved/CL39_attention_audit_serv_ba_off/val_images/manual_val"
MANIFEST="${ACTUAL_ROOT}/package/report_inputs/sample_manifest.json"
REFERENCE_ROOT="${ACTUAL_ROOT}/source/dataset_full/val_dataset/references"
OUTPUT_ROOT="${TASK_ROOT}/assembled_branch_faces"
FIGURE_DIR="${PROJECT_ROOT}/analysis/assets/cl39_attention_24k_serv_a100_branch_faces"

test "$(find "${REFERENCE_FACE_VAL}" -type f -name '*.png' | wc -l)" -eq 96
python tools/analysis/render_cl39_branch_faces.py \
  --manifest "${MANIFEST}" \
  --actual-dir "${ACTUAL_VAL}" \
  --native-dir "${NATIVE_VAL}" \
  --reference-face-dir "${REFERENCE_FACE_VAL}" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --figure-dir "${FIGURE_DIR}"

test "$(find "${OUTPUT_ROOT}/samples" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 16
test "$(find "${FIGURE_DIR}/branch_samples" -type f -name '*.png' | wc -l)" -eq 16
test "$(find "${FIGURE_DIR}" -maxdepth 1 -type f -name 'cl39_branch_faces_overview_*.png' | wc -l)" -eq 4
test -s "${OUTPUT_ROOT}/branch_face_metrics.csv"
test -s "${OUTPUT_ROOT}/branch_face_summary.json"

python - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/CL39_attention_audit_serv_branch_faces_r1")
payload = json.loads((root / "assembled_branch_faces/branch_face_summary.json").read_text())
payload.update(
    checkpoint_sha256="74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07",
    generated_count=96,
    batch_size=12,
    status="complete",
)
(root / "branch_face_gate.json").write_text(json.dumps(payload, indent=2) + "\n")
(root / "BRANCH_FACES_COMPLETE").write_text("complete\n")
print(json.dumps(payload, indent=2))
PY
