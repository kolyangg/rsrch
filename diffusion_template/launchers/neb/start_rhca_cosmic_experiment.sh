#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/niko/rsrch/diffusion_template"
CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 one_id <margin40|canvas1024>" >&2
  echo "   or: $0 full <crop20_legacy_4k|crop20_posefirst_4k|canvas1024_posefirst_4k>" >&2
  exit 2
fi

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
export CUDA_VISIBLE_DEVICES=0

case "$1" in
  one_id)
    export REFERENCE_POLICY="$2"
    exec bash launchers/active/run_rhca_cosmic_one_id_reference_policy_4k_1gpu.sh
    ;;
  full)
    export EXPERIMENT_ARM="$2"
    export COSMIC_LARGE_MANIFEST="/home/niko/datasets/gathered_data_cosmic_large_filtered.json"
    export COSMIC_LARGE_ROOT="/home/niko/datasets"
    exec bash launchers/active/run_rhca_cosmic_large_adapted_1gpu.sh
    ;;
  *)
    echo "Unknown experiment family: $1" >&2
    exit 2
    ;;
esac
