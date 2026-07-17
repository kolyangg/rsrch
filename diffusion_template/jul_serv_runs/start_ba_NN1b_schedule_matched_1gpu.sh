#!/usr/bin/env bash
set -euo pipefail

# 2-GPU machine, physical GPU 1: NN1a with BA-active timestep sampling.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN1_CONFIG_NAME="one_id_ba_NN1b_schedule_matched"
export NN1_RUN_NAME_DEFAULT="ba_NN1b_schedule_matched_1gpu"
export NN1_DEFAULT_GPU="1"
export NN1_DEFAULT_PORT="29612"
export NN1_DESCRIPTION="NN1b: full BA sampled only in the inference-active timestep region"
export NN1_REQUIRE_ID_LOSS="0"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN1_common_1gpu.sh" "$@"
