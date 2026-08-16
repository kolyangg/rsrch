#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
RUN_ID="CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r2"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl21_cl26_v1/${RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

case "${RUN_ID}" in
  CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r1|CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r2)
    CONFIG_NAME="CL21_cosmic_true_soft_router_resididca_v3_24k" ;;
  CL22_cosmic_visibility_order_router_24k_full96_r1|CL22_cosmic_visibility_order_router_24k_full96_r2)
    CONFIG_NAME="CL22_cosmic_visibility_order_router_24k" ;;
  CL23_cosmic_temporal_frequency_router_24k_full96_r1)
    CONFIG_NAME="CL23_cosmic_temporal_frequency_router_24k" ;;
  CL24_cosmic_pm_boundary_distill_24k_full96_r1)
    CONFIG_NAME="CL24_cosmic_pm_boundary_distill_24k" ;;
  CL25_cosmic_low_noise_id_reward_4k_full96_r1|CL25_cosmic_low_noise_id_reward_4k_full96_r2)
    CONFIG_NAME="CL25_cosmic_low_noise_id_reward_4k" ;;
  CL26_cosmic_anchored_highres_roi_ba_24k_full96_r1|CL26_cosmic_anchored_highres_roi_ba_24k_full96_r2)
    CONFIG_NAME="CL26_cosmic_anchored_highres_roi_ba_24k" ;;
  *) echo "Unknown CL21-CL26 run: ${RUN_ID}" >&2; exit 69 ;;
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
python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" verify \
  --root "${PROJECT_ROOT}" --manifest "${SOURCE_MANIFEST}"
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
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export ARCFACE_ONNX_PATH="/home/jovyan/.insightface/models/buffalo_l/w600k_r50.onnx"
export CL25_SOURCE_CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template/saved/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/weights-epoch12.pth"
export CL25_SOURCE_SHA256="707cff809414414c0c85e6fcdf52845d3655284a68d54ecb0d657236634492d5"

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
test "$(sha256sum "${SUBJECT_V2_ID_EMBEDS}" | cut -d' ' -f1)" = \
  "e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${PROJECT_ROOT}/src/configs/${CONFIG_NAME}.yaml"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"
grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"
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

exec bash launchers/active/run_CL21_CL26_cl19_followups_1gpu.sh
