#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN5_NUM_PROCESSES=1
export NN5_RUN_NAME="${NN5_RUN_NAME:-ba_NN5b_clean_identity_tokens_nfs_1gpu}"
export NN5_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NN5_MASTER_PORT="${MASTER_PORT:-29651}"
source "${SCRIPT_DIR}/_start_ba_NN5b_server_common.sh" "$@"
