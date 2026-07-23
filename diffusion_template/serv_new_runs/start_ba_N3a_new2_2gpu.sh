#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export N3A_NEW_NUM_PROCESSES=2
export N3A_NEW_RUN_NAME="${RUN_NAME:-ba_N3a_new2_nfs_2gpu}"
export N3A_NEW_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export N3A_NEW_MASTER_PORT="${MASTER_PORT:-29684}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"

source "${SCRIPT_DIR}/_start_ba_N3a_new2_server_common.sh" "$@"
