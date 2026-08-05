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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PYIQA_SITE="/home/niko/rsrch/metric_envs/pyiqa-0.1.15/lib/python3.10/site-packages"
export PYTHONPATH="${PYIQA_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_PREFIX}/bin/python"
export FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"

# 1 Aug 2026 - Pin both the sealed v2 source and the immutable policy-v1
# schedule. The ArcFace curation caches are offline build inputs only.
BIG_CELEBS_RELEASE_ROOT="/home/niko/rsrch/dataset_publish/releases/v2"
BIG_CELEBS_POLICY_ROOT="/home/niko/rsrch/dataset_publish/sampling_policies/big_celebs_v2_policy_v1"
export BIG_CELEBS_MANIFEST="${BIG_CELEBS_RELEASE_ROOT}/filtered_ids3_adj.json"
export BIG_CELEBS_IMAGES="${BIG_CELEBS_RELEASE_ROOT}/large_dataset"
export BIG_CELEBS_SEAL="${BIG_CELEBS_RELEASE_ROOT}/dataset_manifest.json"
export BIG_CELEBS_EXPECTED_MANIFEST_SHA256="f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa"
export BIG_CELEBS_SAMPLING_PLAN="${BIG_CELEBS_POLICY_ROOT}/train_40k_bs2.jsonl"
export BIG_CELEBS_SAMPLING_PLAN_MANIFEST="${BIG_CELEBS_POLICY_ROOT}/train_40k_bs2_manifest.json"
export BIG_CELEBS_EXPECTED_SAMPLING_PLAN_SHA256="e7041ca446331aeeff89baffe7bf2d678a1763722cd995cb33df8fd19b063b24"
export BIG_CELEBS_MIN_FACE_RES=192
export BIG_CELEBS_SCHEDULE_START_STEP="${BIG_CELEBS_SCHEDULE_START_STEP:-0}"
export BIG_CELEBS_GLOBAL_BATCH_SIZE=2

export RUN_NAME="${RUN_NAME:-rhca_big_celebs_scheduled_v1_40k_full96_r1}"
export EXPERIMENT_SPEC_PATH="${EXPERIMENT_SPEC_PATH:-${PROJECT_ROOT}/experiments/big_celebs/${RUN_NAME}.json}"
export COMET_PROJECT="${COMET_PROJECT:-jul-comet-large-testing-tr}"
export TRAIN_EPOCH_LEN="${TRAIN_EPOCH_LEN:-2000}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"

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

exec bash launchers/active/run_rhca_big_celebs_scheduled_40k_1gpu.sh "$@"
