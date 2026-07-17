#!/usr/bin/env bash
set -euo pipefail

# 4-GPU machine, physical GPU 2: NN1d plus low-noise reference-ID loss.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN1_CONFIG_NAME="one_id_ba_NN1e_frozen_ca_id_loss"
export NN1_RUN_NAME_DEFAULT="ba_NN1e_frozen_ca_id_loss_1gpu"
export NN1_DEFAULT_GPU="2"
export NN1_DEFAULT_PORT="29615"
export NN1_DESCRIPTION="NN1e: frozen-CA full BA plus decoded reference-ID loss at t<=400"
export NN1_REQUIRE_ID_LOSS="1"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN1_common_1gpu.sh" "$@"
