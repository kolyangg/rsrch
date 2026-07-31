#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="/home/niko/rsrch/diffusion_template"
# shellcheck disable=SC1090
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
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
export SOURCE_RUN="rhca_cosmic_initial_distinct_uniform_4k"
export SOURCE_TRAIN_COMET_KEY="288ebfe3ccf74d5ea328a55b3abe31cb"
export EVAL_RUN="${SOURCE_RUN}_full96_steps0_1k_2k_3k_4k"
export EVAL_COMET_KEY="ced6658b5b12484a9e003fe47cd0c2bf"
export DATASET_POLICY_ARM="distinct_uniform"
export CONTINUATION_FIRST_ENDPOINT_EPOCH=24
export CONTINUATION_BASE_CHECK_MODE=portable_recovery
export CONTINUATION_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large_continuation/rhca_cosmic_initial_distinct_uniform_continue20k.json"
export COSMIC_INITIAL_IMAGES_ROOT="/home/niko/datasets"
export COSMIC_LARGE_MANIFEST="/home/niko/datasets/gathered_data_cosmic_large_filtered.json"
export FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"

python - <<'PY'
import ctypes
from pathlib import Path
import onnxruntime as ort
assert ort.__version__ == "1.20.1", ort.__version__
provider = Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"
ctypes.CDLL(str(provider))
assert "CUDAExecutionProvider" in ort.get_available_providers()
print("ONNX Runtime CUDA provider loaded:", provider)
PY

exec bash launchers/active/run_rhca_cosmic_initial_usage_continue_20k_1gpu.sh
