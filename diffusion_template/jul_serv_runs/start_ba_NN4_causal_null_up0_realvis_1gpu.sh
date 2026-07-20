#!/usr/bin/env bash
set -euo pipefail

# NN4: correctness-fixed PPR, neutral reference text, learned-null auxiliary
# supervision, and up_blocks.0-only spatial authority.
# One GPU; batch 2; 20k maximum; fixed 96-image RealVis validation at step 0
# and every 2k optimizer steps.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NN1_CONFIG_NAME="one_id_ba_NN4_causal_null_up0"
export NN1_RUN_NAME_DEFAULT="ba_NN4_causal_null_up0_realvis_1gpu"
export NN1_DEFAULT_GPU="0"
export NN1_DEFAULT_PORT="29640"
export NN1_DESCRIPTION="NN4: CFG/text-isolated learned-null PPR at up_blocks.0"
export NN1_REQUIRE_ID_LOSS="1"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
export NN1_TRAIN_DATASET_NAME="cosmic_large_neb"
export NN1_VALIDATION_MODEL="SG161222/RealVisXL_V4.0"

export PM_PATH="${PM_PATH:-/home/niko/models/PhotoMaker-V2/photomaker-v2.bin}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export LOCAL_EFFECTIVE_BATCH="${LOCAL_EFFECTIVE_BATCH:-2}"
export VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-12}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export OPTIMIZER_STEPS_PER_EPOCH="${OPTIMIZER_STEPS_PER_EPOCH:-2000}"
export FULL_STEP0_VAL="${FULL_STEP0_VAL:-true}"

source "${SCRIPT_DIR}/start_ba_NN2_ppr1_1gpu.sh" "$@"
