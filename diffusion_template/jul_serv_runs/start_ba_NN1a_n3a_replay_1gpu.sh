#!/usr/bin/env bash
set -euo pipefail

# 2-GPU machine, physical GPU 0: guarded N3a parity control.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN1_CONFIG_NAME="one_id_ba_NN1a_n3a_replay"
export NN1_RUN_NAME_DEFAULT="ba_NN1a_n3a_replay_1gpu"
export NN1_DEFAULT_GPU="0"
export NN1_DEFAULT_PORT="29611"
export NN1_DESCRIPTION="NN1a: guarded N3a replay; full SA+CA noise_and_ref training"
export NN1_REQUIRE_ID_LOSS="0"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN1_common_1gpu.sh" "$@"
