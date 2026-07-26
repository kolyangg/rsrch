#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
RUN_ID="rhca_cosmic_full_crop40_posefirst_4k"

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
export RUN_NAME="${RUN_ID}"
export EXPERIMENT_SPEC_PATH="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/rhca_cosmic_full_crop40_posefirst_4k/${RUN_ID}.json"

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
  rhca_cosmic_full_crop40_posefirst_4k)
    export EXPERIMENT_ARM="crop40_posefirst_4k"
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
  "${EXTRA_HYDRA_ARGS[@]}"
