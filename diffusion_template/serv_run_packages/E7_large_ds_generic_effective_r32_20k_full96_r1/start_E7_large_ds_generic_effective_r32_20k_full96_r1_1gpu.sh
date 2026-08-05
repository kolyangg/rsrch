#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
REMOTE_REPO="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E7_E10_20260804/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
RUN_ID="E7_large_ds_generic_effective_r32_20k_full96_r1"
RUN_NAME="${RUN_ID}"

case "${RUN_ID}" in
  E0_large_ds_base_historical_r4_20k_full96_r1)
    CONFIG_NAME="E0_large_ds_base_historical_20k"
    ;;
  E0_large_ds_base_fixed_baonly_r32_20k_full96_r1)
    CONFIG_NAME="E0_large_ds_base_fixed_20k"
    ;;
  E1_large_ds_truekey_r32_20k_full96_r1)
    CONFIG_NAME="E1_large_ds_truekey_20k"
    ;;
  E2_large_ds_branchout_r32_20k_full96_r1)
    CONFIG_NAME="E2_large_ds_branchout_20k"
    ;;
  E3_large_ds_roiwarp_r32_20k_full96_r1)
    CONFIG_NAME="E3_large_ds_roiwarp_20k"
    ;;
  E4_large_ds_midup_r32_20k_full96_r1)
    CONFIG_NAME="E4_large_ds_midup_20k"
    ;;
  E5_large_ds_infersteps_r32_20k_full96_r1)
    CONFIG_NAME="E5_large_ds_infersteps_20k"
    ;;
  E6_large_ds_fp32_r32_20k_full96_r1)
    CONFIG_NAME="E6_large_ds_fp32_20k"
    ;;
  E7_large_ds_generic_effective_r32_20k_full96_r1)
    CONFIG_NAME="E7_large_ds_generic_effective_20k"
    ;;
  E8_large_ds_generic_ca_r32_20k_full96_r1)
    CONFIG_NAME="E8_large_ds_generic_ca_20k"
    ;;
  E9_large_ds_shared_saout_r32_20k_full96_r1)
    CONFIG_NAME="E9_large_ds_shared_saout_20k"
    ;;
  E10_large_ds_pmdefault_effective_r64_20k_full96_r1)
    CONFIG_NAME="E10_large_ds_pmdefault_effective_20k"
    ;;
  *)
    echo "Unknown August Large Dataset package ID: ${RUN_ID}" >&2
    exit 69
    ;;
esac

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
cd "${REMOTE_REPO}"
if [[ "${CONDA_PREFIX:-}" != "${CONDA_ENV}" ]]; then
  echo "Wrong Conda environment: ${CONDA_PREFIX:-unset}" >&2
  exit 70
fi
if [[ "$(git branch --show-current)" != "test" ]]; then
  echo "Serv experiment requires the test branch" >&2
  exit 71
fi
cd "${PROJECT_ROOT}"

if [[ ! -f .env ]]; then
  echo "Missing machine-local diffusion_template/.env" >&2
  exit 72
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-${OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export LARGE_DATASET_MANIFEST="${OWNER_ROOT}/datasets/dataset_full/filtered_ids3_adj.json"
export LARGE_DATASET_IMAGES="${OWNER_ROOT}/datasets/dataset_full/large_dataset_adj/large_dataset"
export FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export CUDA_VISIBLE_DEVICES="0"
export ACCELERATE_NUM_PROCESSES="1"
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME
export CONFIG_NAME
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/large_dataset/${RUN_NAME}.json"
export COMET_PROJECT="aug-large-ds"

if [[ "1" != "1" || "${ACCELERATE_NUM_PROCESSES}" != "1" ]]; then
  echo "This package requires exactly one A100 and one training process." >&2
  exit 73
fi
if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production job received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
  exit 74
fi

test -s "${PM_PATH}"
test -s "${LARGE_DATASET_MANIFEST}"
test -d "${LARGE_DATASET_IMAGES}"
test -s "${FULL96_BBOX_MANUAL}"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${NVIDIA_LIB_ROOT}/cublas/lib/libcublasLt.so.12"

python - <<'PY'
import ctypes
import importlib.metadata
from pathlib import Path

import onnxruntime as ort

if ort.__version__ != "1.20.1":
    raise RuntimeError(f"Unexpected ONNX Runtime version: {ort.__version__}")
provider = Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"
ctypes.CDLL(str(provider))
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is unavailable")
if importlib.metadata.version("pyiqa") != "0.1.15":
    raise RuntimeError("PyIQA 0.1.15 is required")
print("Serv runtime verified:", ort.__version__, ort.get_available_providers())
PY

exec bash launchers/active/run_E_large_ds_hard_v1_20k_1gpu.sh
