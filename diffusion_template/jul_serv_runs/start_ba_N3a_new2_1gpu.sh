#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN1_CONFIG_NAME="one_id_ba_N3a_new2"
export NN1_RUN_NAME_DEFAULT="ba_N3a_new2_1gpu"
export NN1_DEFAULT_GPU="0"
export NN1_DEFAULT_PORT="29682"
export NN1_DESCRIPTION="N3a_new2: up-only full-grid trainable dual-0.35 BA with PhotoMaker output anchor"
export NN1_REQUIRE_ID_LOSS="0"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
export NN1_TRAIN_DATASET_NAME="cosmic_large_neb"
export NN1_VALIDATION_MODEL="SG161222/RealVisXL_V4.0"

export PM_PATH="${PM_PATH:-/home/niko/models/PhotoMaker-V2/photomaker-v2.bin}"
if [[ -z "${PHOTOMAKER_ENV_BIN:-}" && -x /home/niko/miniconda3/envs/photomaker_NS/bin/python ]]; then
  export PHOTOMAKER_ENV_BIN=/home/niko/miniconda3/envs/photomaker_NS/bin
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export OPTIMIZER_STEPS_PER_EPOCH="${OPTIMIZER_STEPS_PER_EPOCH:-2000}"

source "${SCRIPT_DIR}/start_ba_NN2_ppr1_1gpu.sh" "$@"
