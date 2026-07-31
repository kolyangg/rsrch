#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
ORT_OVERLAY="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_overlays/onnxruntime_gpu_1_20_1"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
RUN_ID="rhca_cosmic_initial_distinct_top3softmax_4k_r2"

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
  echo "Serv experiment requires the test branch" >&2
  exit 71
fi
cd "${PROJECT_ROOT}"

case "${RUN_ID}" in
  rhca_cosmic_initial_distinct_uniform_4k)
    export DATASET_POLICY_ARM="distinct_uniform"
    ;;
  rhca_cosmic_initial_distinct_highest_4k)
    export DATASET_POLICY_ARM="distinct_highest"
    ;;
  rhca_cosmic_initial_distinct_top3softmax_4k|rhca_cosmic_initial_distinct_top3softmax_4k_r2)
    export DATASET_POLICY_ARM="distinct_top3_softmax"
    ;;
  rhca_cosmic_initial_selfref_minface256_4k)
    export DATASET_POLICY_ARM="target_min256"
    ;;
  *)
    echo "Unsupported Cosmic dataset-policy run: ${RUN_ID}" >&2
    exit 72
    ;;
esac

set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export COSMIC_INITIAL_IMAGES_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data/gathered_data_cosmic_large_filtered.json"
export COSMIC_LARGE_ROOT="${COSMIC_INITIAL_IMAGES_ROOT}"
export FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export FULL96_HISTORICAL_MANUAL="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch/dataset_full/val_dataset/pm96_bboxes_new.json"
export FULL96_AUTO_SEED="${PROJECT_ROOT}/../dataset_full/val_dataset/pm96_bboxes_new_auto.json"
export FULL96_SOURCE_REPRO_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/pm96_bboxes_new.json"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cublas/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RUN_NAME="${RUN_ID}"
export EVAL_RUN_NAME="${RUN_ID}_full96_steps0_1k_2k_3k_4k"
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large_dataset_usage/${RUN_ID}.json"
export EVAL_EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large_dataset_usage/${EVAL_RUN_NAME}.json"

if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production job received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
  exit 73
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

exec bash launchers/active/run_rhca_cosmic_large_dataset_usage_train_full96_1gpu.sh
