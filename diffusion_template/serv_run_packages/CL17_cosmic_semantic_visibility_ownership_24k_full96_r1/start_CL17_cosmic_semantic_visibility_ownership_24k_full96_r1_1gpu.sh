#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
RUN_ID="CL17_cosmic_semantic_visibility_ownership_24k_full96_r1"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl15_cl20_v1/${RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

case "${RUN_ID}" in
  CL15_cosmic_shared_highres_roi_ba_24k_full96_r1)
    CONFIG_NAME="CL15_cosmic_shared_highres_roi_ba_24k" ;;
  CL16_cosmic_clean_multiscale_ref_memory_24k_full96_r1)
    CONFIG_NAME="CL16_cosmic_clean_multiscale_ref_memory_24k" ;;
  CL17_cosmic_semantic_visibility_ownership_24k_full96_r1)
    CONFIG_NAME="CL17_cosmic_semantic_visibility_ownership_24k" ;;
  CL18_cosmic_crossview_spatial_consistency_24k_full96_r1)
    CONFIG_NAME="CL18_cosmic_crossview_spatial_consistency_24k" ;;
  CL19_cosmic_true_soft_fullquery_router_24k_full96_r1)
    CONFIG_NAME="CL19_cosmic_true_soft_fullquery_router_24k" ;;
  CL20_cosmic_bigcelebs_hardcase_curriculum_24k_full96_r1)
    CONFIG_NAME="CL20_cosmic_bigcelebs_hardcase_curriculum_24k" ;;
  *) echo "Unknown CL15-CL20 run: ${RUN_ID}" >&2; exit 69 ;;
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
python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" \
  verify --root "${PROJECT_ROOT}" --manifest "${SOURCE_MANIFEST}"

cd "${PROJECT_ROOT}"
LIVE_ENV="${OWNER_ROOT}/rsrch_test/diffusion_template/.env"
test -s "${LIVE_ENV}"
set -a
# shellcheck disable=SC1090
source "${LIVE_ENV}"
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-${OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"

export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export COSMIC_LARGE_EXPECTED_MANIFEST_SHA256="8ba369ef2fdc0496a0d3d55afb5c7923c1aa299343a676ac6bc0d94f3a3a0196"
BIG_CELEBS_RELEASE_ROOT="${OWNER_ROOT}/datasets/bigcelebs/releases/v2"
export BIG_CELEBS_MANIFEST="${BIG_CELEBS_RELEASE_ROOT}/filtered_ids3_adj.json"
export BIG_CELEBS_IMAGES="${BIG_CELEBS_RELEASE_ROOT}/large_dataset"
export BIG_CELEBS_SEAL="${BIG_CELEBS_RELEASE_ROOT}/dataset_manifest.json"
export BIG_CELEBS_DOWNLOAD_LOG="${OWNER_ROOT}/datasets/dataset_tools/download_bigcelebs_v2.log"
export BIG_CELEBS_EXPECTED_MANIFEST_SHA256="f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa"
export FULL96_BBOX_MANUAL="${OWNER_ROOT}/datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"

export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
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
test "$(sha256sum "${COSMIC_LARGE_MANIFEST}" | cut -d' ' -f1)" = \
  "${COSMIC_LARGE_EXPECTED_MANIFEST_SHA256}"
test -s "${FULL96_BBOX_MANUAL}"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${PROJECT_ROOT}/src/configs/${CONFIG_NAME}.yaml"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"
if ! grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"; then
  echo "LIBSTDCXX_PATH does not expose GLIBCXX_3.4.32" >&2
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
ctypes.CDLL(str(Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"))
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is unavailable")
if importlib.metadata.version("pyiqa") != "0.1.15":
    raise RuntimeError("PyIQA 0.1.15 is required")
print("Serv runtime verified:", ort.__version__, ort.get_available_providers())
PY

if [[ "${CONFIG_NAME}" == CL20_* ]]; then
  test -s "${BIG_CELEBS_MANIFEST}"
  test -d "${BIG_CELEBS_IMAGES}"
  test -s "${BIG_CELEBS_SEAL}"
  test "$(sha256sum "${BIG_CELEBS_MANIFEST}" | cut -d' ' -f1)" = \
    "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}"
  if ! grep -qF "BIGCELEBS_V2_DOWNLOAD_COMPLETE" "${BIG_CELEBS_DOWNLOAD_LOG}"; then
    echo "BigCelebs v2 has no completion marker" >&2
    exit 75
  fi
  SCHEDULE_DIR="${RUNTIME_ROOT}/schedule"
  mkdir -p "${SCHEDULE_DIR}"
  export CL20_SCHEDULE="${SCHEDULE_DIR}/train_48k_bs2.jsonl"
  export CL20_SCHEDULE_SUMMARY="${SCHEDULE_DIR}/train_48k_bs2.summary.json"
  python tools/datasets/build_cl20_hardcase_schedule.py \
    --cosmic-manifest "${COSMIC_LARGE_MANIFEST}" \
    --cosmic-root "${COSMIC_LARGE_ROOT}" \
    --big-manifest "${BIG_CELEBS_MANIFEST}" \
    --big-images-root "${BIG_CELEBS_IMAGES}" \
    --output "${CL20_SCHEDULE}" \
    --summary-output "${CL20_SCHEDULE_SUMMARY}"
  export CL20_SCHEDULE_SHA256="$(sha256sum "${CL20_SCHEDULE}" | cut -d' ' -f1)"
  export CL20_SCHEDULE_START_ROW=0
fi

exec bash launchers/active/run_CL15_CL20_hardcases_24k_1gpu.sh
