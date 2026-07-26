#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
ORT_OVERLAY="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_overlays/onnxruntime_gpu_1_20_1"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
RUN_ID="rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1"

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

set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export COSMIC_LARGE_MANIFEST="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data/gathered_data_cosmic_large_filtered.json"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cublas/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RUN_NAME="${RUN_ID}"
export EXPERIMENT_SPEC_PATH="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1/${RUN_ID}.json"

if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production training received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
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

EXTRA_HYDRA_ARGS=()
case "${RUN_ID}" in
  rhca_cosmic_full_crop20_legacy_4k)
    export EXPERIMENT_ARM="crop20_legacy_4k"
    ;;
  rhca_cosmic_full_crop20_legacy_20k)
    export EXPERIMENT_ARM="crop20_legacy_4k"
    export TRAIN_EPOCHS="40"
    ;;
  rhca_cosmic_full_crop20_posefirst_4k|rhca_cosmic_full_crop20_posefirst_4k_r1|rhca_cosmic_full_crop20_posefirst_4k_batched_r1)
    export EXPERIMENT_ARM="crop20_posefirst_4k"
    ;;
  rhca_cosmic_full_crop40_posefirst_4k|rhca_cosmic_full_crop40_posefirst_4k_fast_r1)
    export EXPERIMENT_ARM="crop40_posefirst_4k"
    ;;
  rhca_cosmic_full_crop60_posefirst_4k|rhca_cosmic_full_crop60_posefirst_4k_fast_r1)
    export EXPERIMENT_ARM="crop60_posefirst_4k"
    ;;
  rhca_cosmic_full_crop40_legacy_4k|rhca_cosmic_full_crop40_legacy_4k_fast_r1)
    export EXPERIMENT_ARM="crop40_legacy_4k"
    ;;
  rhca_cosmic_full_crop40_512_posefirst_4k|rhca_cosmic_full_crop40_512_posefirst_4k_fast_r1)
    export EXPERIMENT_ARM="crop40_512_posefirst_4k"
    ;;
  rhca_cosmic_full_crop20_posefirst_4k_batched_workers2_r2)
    export EXPERIMENT_ARM="crop20_posefirst_4k"
    EXTRA_HYDRA_ARGS+=("dataloaders.train.num_workers=2")
    ;;
  rhca_cosmic_full_posefirst_speed_workers0_50)
    export EXPERIMENT_ARM="crop20_posefirst_4k"
    export TRAIN_EPOCHS="1"
    EXTRA_HYDRA_ARGS+=(
      "trainer.epoch_len=50"
      "dataloaders.train.num_workers=0"
    )
    ;;
  rhca_cosmic_full_canvas1024_posefirst_4k)
    export EXPERIMENT_ARM="canvas1024_posefirst_4k"
    ;;
  rhca_cosmic_full_crop20_posefirst_20k)
    export EXPERIMENT_ARM="crop20_posefirst_20k"
    ;;
  *)
    echo "Unsupported Serv Cosmic run ID: ${RUN_ID}" >&2
    exit 72
    ;;
esac

exec bash launchers/active/run_rhca_cosmic_large_adapted_1gpu.sh \
  dataloaders.train.num_workers=2 \
  "${EXTRA_HYDRA_ARGS[@]}"
