#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/niko/rsrch/diffusion_template"
CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"

# shellcheck disable=SC1090
source "${CONDA_INIT}"
conda activate photomaker_NS
cd "${PROJECT_ROOT}"

set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="/home/niko/models/PhotoMaker-V2/photomaker-v2.bin"
export CUDA_VISIBLE_DEVICES=0
NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DATASET_POLICY_ARM="baseline_self"
export COSMIC_INITIAL_IMAGES_ROOT="/home/niko/datasets"
export COSMIC_LARGE_MANIFEST="/home/niko/datasets/gathered_data_cosmic_large_filtered.json"
export COSMIC_LARGE_ROOT="/home/niko/datasets"
export FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export FULL96_HISTORICAL_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/pm96_bboxes_new.json"
export FULL96_AUTO_SEED="${PROJECT_ROOT}/../dataset_full/val_dataset/pm96_bboxes_new_auto.json"
export FULL96_SOURCE_REPRO_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/pm96_bboxes_new.json"
export RUN_NAME="rhca_cosmic_initial_selfref_4k_baseline_r2"
export EVAL_RUN_NAME="${RUN_NAME}_full96_steps0_1k_2k_3k_4k"
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large_dataset_usage/${RUN_NAME}.json"
export EVAL_EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large_dataset_usage/${EVAL_RUN_NAME}.json"

python - <<'PY'
import ctypes
from pathlib import Path

import onnxruntime as ort

if ort.__version__ != "1.20.1":
    raise RuntimeError(f"Unexpected ONNX Runtime version: {ort.__version__}")
provider = Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"
ctypes.CDLL(str(provider))
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is unavailable")
print("ONNX Runtime CUDA provider library loaded:", provider)
PY

exec bash launchers/active/run_rhca_cosmic_large_dataset_usage_train_full96_1gpu.sh
