#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN4_NUM_PROCESSES=1
export NN4_RUN_NAME="${RUN_NAME:-ba_NN4_causal_null_up0_nfs_1gpu}"
export NN4_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NN4_MASTER_PORT="${MASTER_PORT:-29641}"

source "${SCRIPT_DIR}/_start_ba_NN4_server_common.sh" "$@"
