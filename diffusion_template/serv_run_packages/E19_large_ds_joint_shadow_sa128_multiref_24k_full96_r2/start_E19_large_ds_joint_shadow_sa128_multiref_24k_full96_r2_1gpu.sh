#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
: "${RUN_ID:=E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2}"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_e19_e24_v3/${RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

case "${RUN_ID}" in
  E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r1|E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2)
    CONFIG_NAME="E19_large_ds_joint_shadow_sa128_multiref_24k" ;;
  E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r1|E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2)
    CONFIG_NAME="E20_large_ds_joint_shadow_sa128_branchout_r32_24k" ;;
  E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r1|E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2)
    CONFIG_NAME="E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k" ;;
  E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r1|E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2)
    CONFIG_NAME="E22_large_ds_joint_shadow_sa128_arcfaceaux_24k" ;;
  E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r1|E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2)
    CONFIG_NAME="E23_large_ds_joint_shadow_sa128_earlydecay_24k" ;;
  E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r1|E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2)
    CONFIG_NAME="E24_large_ds_joint_shadow_sa128_alternating_24k" ;;
  *) echo "Unknown E19-E24 RUN_ID: ${RUN_ID}" >&2; exit 69 ;;
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
# 06 Aug 2026 - AICODE-NOTE: E19-E24 run from per-run immutable snapshots;
# reject any missing, changed, or extra source file before loading credentials.
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
export LARGE_DATASET_MANIFEST="${OWNER_ROOT}/datasets/dataset_full/filtered_ids3_adj.json"
export LARGE_DATASET_IMAGES="${OWNER_ROOT}/datasets/dataset_full/large_dataset_adj/large_dataset"
export LARGE_DATASET_SCHEDULE_START_ROW=0
export FULL96_BBOX_MANUAL="${OWNER_ROOT}/datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export ARCFACE_ONNX_PATH="/home/jovyan/.insightface/models/buffalo_l/w600k_r50.onnx"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME="${RUN_ID}"
export CONFIG_NAME
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/large_dataset/${RUN_ID}.json"
export COMET_PROJECT=aug-large-ds

test -s "${PM_PATH}"
test -s "${LARGE_DATASET_MANIFEST}"
test -d "${LARGE_DATASET_IMAGES}"
test -s "${FULL96_BBOX_MANUAL}"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${PROJECT_ROOT}/src/configs/${CONFIG_NAME}.yaml"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"
if [[ "${CONFIG_NAME}" == E22_large_ds_joint_shadow_sa128_arcfaceaux_24k ]]; then
  test -s "${ARCFACE_ONNX_PATH}"
  actual_arcface_sha="$(sha256sum "${ARCFACE_ONNX_PATH}" | cut -d' ' -f1)"
  if [[ "${actual_arcface_sha}" != 4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43 ]]; then
    echo "ArcFace model hash mismatch: ${actual_arcface_sha}" >&2
    exit 73
  fi
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

exec bash launchers/active/run_E19_E24_large_ds_24k_1gpu.sh
