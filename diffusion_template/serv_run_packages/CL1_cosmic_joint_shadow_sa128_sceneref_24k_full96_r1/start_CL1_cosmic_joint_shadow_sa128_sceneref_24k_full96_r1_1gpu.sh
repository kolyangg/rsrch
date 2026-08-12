#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
: "${RUN_ID:=CL1_cosmic_joint_shadow_sa128_sceneref_24k_full96_r1}"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

case "${RUN_ID}" in
  CL1_cosmic_joint_shadow_sa128_sceneref_24k_full96_r1)
    CONFIG_NAME="CL1_cosmic_joint_shadow_sa128_sceneref_24k" ;;
  CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1)
    CONFIG_NAME="CL2_cosmic_joint_shadow_sa128_facecanon_24k" ;;
  CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1)
    CONFIG_NAME="CL3_cosmic_joint_shadow_sa128_fmtfix_24k" ;;
  *) echo "Unknown CL1-CL3 RUN_ID: ${RUN_ID}" >&2; exit 69 ;;
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

test -s "${SOURCE_MANIFEST}"
test -d "${PROJECT_ROOT}"
# 06 Aug 2026 - AICODE-NOTE: CL runs use per-run immutable snapshots; reject any
# missing, changed, or extra source file before loading credentials.
python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" \
  verify --root "${PROJECT_ROOT}" --manifest "${SOURCE_MANIFEST}"

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

# --- Cosmic Large dataset ---------------------------------------------------
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
# CL1 only: sealed identity grouping from tools/datasets/build_cosmic_identity_assets.py.
export COSMIC_IDENTITY_GROUPS="${OWNER_ROOT}/datasets/cosmic_identity_groups_v1.json"
export COSMIC_IDENTITY_GROUPS_SHA256="REPLACE_WITH_SEALED_SHA256"
# Optional shared control; unset to disable the offline reference identity gate.
# export COSMIC_REFERENCE_ACCEPT_LIST="${OWNER_ROOT}/datasets/cosmic_reference_accept_list_v1.json"
# ---------------------------------------------------------------------------

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
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large/${RUN_ID}.json"
export COMET_PROJECT=aug-large-ds

test -s "${PM_PATH}"
test -s "${COSMIC_LARGE_MANIFEST}"
test -d "${COSMIC_LARGE_ROOT}"
test -s "${FULL96_BBOX_MANUAL}"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${PROJECT_ROOT}/src/configs/${CONFIG_NAME}.yaml"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"
if [[ "${CONFIG_NAME}" == CL1_cosmic_joint_shadow_sa128_sceneref_24k ]]; then
  test -s "${COSMIC_IDENTITY_GROUPS}"
  if [[ "${COSMIC_IDENTITY_GROUPS_SHA256}" == "REPLACE_WITH_SEALED_SHA256" ]]; then
    echo "CL1 requires the sealed identity-groups SHA-256 to be pinned" >&2
    exit 75
  fi
fi

if ! grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"; then
  echo "LIBSTDCXX_PATH does not expose GLIBCXX_3.4.32: ${LIBSTDCXX_PATH}" >&2
  exit 74
fi
export LD_LIBRARY_PATH="$(dirname "${LIBSTDCXX_PATH}"):${LD_LIBRARY_PATH}"
export LD_PRELOAD="${LIBSTDCXX_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"

python - <<'PY_INNER'
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
PY_INNER

exec bash launchers/active/run_CL1_CL3_cosmic_24k_1gpu.sh
