#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN6_NUM_PROCESSES=1
export NN6_RUN_NAME="${NN6_RUN_NAME:-ba_NN6a_factorized_identity_only_up0_nfs_1gpu}"
export NN6_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NN6_MASTER_PORT="${MASTER_PORT:-29661}"
source "${SCRIPT_DIR}/_start_ba_NN6a_server_common.sh" "$@"
