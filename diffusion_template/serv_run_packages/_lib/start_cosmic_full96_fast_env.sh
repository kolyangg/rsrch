#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set the unique full-96 evaluation run name}"
: "${VALIDATION_SOURCE_RUN:?Set the completed source training run}"
: "${VALIDATION_SOURCE_COMET_KEY:?Set the immutable source Comet key}"

CONDA_ENV="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
ORT_OVERLAY="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_overlays/onnxruntime_gpu_1_20_1"
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
cd "/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test"
if [[ "${CONDA_PREFIX:-}" != "${CONDA_ENV}" ]]; then
  echo "Wrong Conda environment: ${CONDA_PREFIX:-unset}" >&2
  exit 70
fi
if [[ "$(git branch --show-current)" != "test" ]]; then
  echo "Serv evaluation requires the test branch" >&2
  exit 71
fi
cd "${PROJECT_ROOT}"

set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export COSMIC_LARGE_MANIFEST="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data/gathered_data_cosmic_large_filtered.json"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export FULL96_HISTORICAL_MANUAL="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/dataset_full/val_dataset/pm96_bboxes_new.json"
export FULL96_AUTO_SEED="${PROJECT_ROOT}/../dataset_full/val_dataset/pm96_bboxes_new_auto.json"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cublas/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RUN_NAME="${RUN_ID}"
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/serv_run_packages/${RUN_ID}/${RUN_ID}.json"

# 26 Jul 2026 - AICODE-NOTE: full-96 reproductions must use the same promoted
# CUDA/InsightFace runtime as their source 4k run; otherwise a runtime change
# is silently mixed into the checkpoint-quality comparison.
if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production evaluation received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
  exit 72
fi
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${NVIDIA_LIB_ROOT}/cublas/lib/libcublasLt.so.12"

python - <<'PY'
import onnxruntime as ort

if ort.__version__ != "1.20.1":
    raise RuntimeError(f"Unexpected ONNX Runtime version: {ort.__version__}")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is unavailable")
print("ONNX Runtime production provider:", ort.__version__, ort.get_available_providers())
PY

exec bash "${FULL96_ACTIVE_LAUNCHER:-launchers/active/run_rhca_cosmic_full96_eval_1gpu.sh}"
