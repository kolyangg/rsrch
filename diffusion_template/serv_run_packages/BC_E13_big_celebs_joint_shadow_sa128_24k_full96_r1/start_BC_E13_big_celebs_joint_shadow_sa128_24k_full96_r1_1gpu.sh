#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
REMOTE_REPO="${SERV_REPO_ROOT:-${OWNER_ROOT}/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808}"
PROJECT_ROOT="${REMOTE_REPO}/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
RUN_ID="BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1"
CONFIG_NAME="BC_E13_big_celebs_joint_shadow_sa128_24k"

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
if [[ "$(git branch --show-current)" != "test" ]]; then
  echo "BC_E13 BigCelebs requires branch test." >&2
  exit 70
fi
# 08 Aug 2026 - The isolated runtime is a clean committed E13 base plus exactly
# these six local-only BC_E13 artifacts. Reject tracked drift or any extra file.
EXPECTED_RUNTIME_STATUS="$(printf '%s\n' \
  '?? diffusion_template/experiments/big_celebs/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1.json' \
  '?? diffusion_template/launchers/active/run_BC_E13_big_celebs_24k_1gpu.sh' \
  '?? diffusion_template/serv_run_packages/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/run_BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1_1gpu.yaml' \
  '?? diffusion_template/serv_run_packages/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/start_BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1_1gpu.sh' \
  '?? diffusion_template/src/configs/BC_E13_big_celebs_joint_shadow_sa128_24k.yaml' \
  '?? diffusion_template/tools/validate_BC_E13_big_celebs_config.py' \
  | LC_ALL=C sort)"
ACTUAL_RUNTIME_STATUS="$(git status --porcelain=v1 --untracked-files=all | LC_ALL=C sort)"
if [[ "${ACTUAL_RUNTIME_STATUS}" != "${EXPECTED_RUNTIME_STATUS}" ]]; then
  echo "BC_E13 BigCelebs Serv runtime differs from its six-file overlay." >&2
  printf 'Expected status:\n%s\nActual status:\n%s\n' \
    "${EXPECTED_RUNTIME_STATUS}" "${ACTUAL_RUNTIME_STATUS}" >&2
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

# 08 Aug 2026 - Pin the explicit Serv release; there is no movable `current`
# symlink, and a partially extracted release must never reach model startup.
BIG_CELEBS_RELEASE_ROOT="${OWNER_ROOT}/datasets/bigcelebs/releases/v2"
export BIG_CELEBS_MANIFEST="${BIG_CELEBS_RELEASE_ROOT}/filtered_ids3_adj.json"
export BIG_CELEBS_IMAGES="${BIG_CELEBS_RELEASE_ROOT}/large_dataset"
export BIG_CELEBS_SEAL="${BIG_CELEBS_RELEASE_ROOT}/dataset_manifest.json"
export BIG_CELEBS_DOWNLOAD_LOG="${OWNER_ROOT}/datasets/dataset_tools/download_bigcelebs_v2.log"
export BIG_CELEBS_EXPECTED_MANIFEST_SHA256="f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa"
export BIG_CELEBS_MIN_FACE_RES=192

export FULL96_BBOX_MANUAL="${OWNER_ROOT}/datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME="${RUN_ID}"
export CONFIG_NAME
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/big_celebs/${RUN_ID}.json"
export COMET_PROJECT=aug-large-ds

test -s "${PM_PATH}"
test -s "${BIG_CELEBS_MANIFEST}"
test -d "${BIG_CELEBS_IMAGES}"
test -s "${BIG_CELEBS_SEAL}"
test -s "${BIG_CELEBS_DOWNLOAD_LOG}"
test -s "${FULL96_BBOX_MANUAL}"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${PROJECT_ROOT}/src/configs/${CONFIG_NAME}.yaml"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"

if ! grep -qF "BIGCELEBS_V2_DOWNLOAD_COMPLETE" "${BIG_CELEBS_DOWNLOAD_LOG}"; then
  echo "BigCelebs v2 download has not passed its terminal validation." >&2
  exit 73
fi
if ! grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"; then
  echo "LIBSTDCXX_PATH does not expose GLIBCXX_3.4.32: ${LIBSTDCXX_PATH}" >&2
  exit 74
fi
export LD_LIBRARY_PATH="$(dirname "${LIBSTDCXX_PATH}"):${LD_LIBRARY_PATH}"
export LD_PRELOAD="${LIBSTDCXX_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"

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

exec bash launchers/active/run_BC_E13_big_celebs_24k_1gpu.sh
