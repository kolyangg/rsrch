#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN3B_NUM_PROCESSES=2
export NN3B_RUN_NAME="${RUN_NAME:-ba_NN3b_learned_null_pm_attenuation_nfs_2gpu}"
export NN3B_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NN3B_MASTER_PORT="${MASTER_PORT:-29634}"

source "${SCRIPT_DIR}/_start_ba_NN3b_server_common.sh" "$@"
