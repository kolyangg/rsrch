#!/usr/bin/env bash
set -euo pipefail

# Primary server: NN7a 4k approval run. In-run RealVis validation remains all 96.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NN1_CONFIG_NAME="one_id_ba_NN7a_clean_patch_takeover_up1"
export NN1_RUN_NAME_DEFAULT="ba_NN7a_clean_patch_takeover_up1_1gpu"
export NN1_DEFAULT_GPU="0"
export NN1_DEFAULT_PORT="29670"
export NN1_DESCRIPTION="NN7a: clean CLIP local spatial candidate takeover at up_blocks.1"
export NN1_REQUIRE_ID_LOSS="1"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
export NN1_TRAIN_DATASET_NAME="cosmic_large_neb"
export NN1_VALIDATION_MODEL="SG161222/RealVisXL_V4.0"

export PM_PATH="${PM_PATH:-/home/niko/models/PhotoMaker-V2/photomaker-v2.bin}"
if [[ -z "${PHOTOMAKER_ENV_BIN:-}" && -x /home/niko/miniconda3/envs/photomaker_NS/bin/python ]]; then
  export PHOTOMAKER_ENV_BIN=/home/niko/miniconda3/envs/photomaker_NS/bin
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAIN_BATCH_SIZE="1"
export LOCAL_EFFECTIVE_BATCH="2"
export VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-12}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export OPTIMIZER_STEPS_PER_EPOCH="${OPTIMIZER_STEPS_PER_EPOCH:-2000}"
export FULL_STEP0_VAL="${FULL_STEP0_VAL:-true}"

source "${SCRIPT_DIR}/start_ba_NN2_ppr1_1gpu.sh" \
  datasets.val.manual_val.limit=96 \
  ++datasets.train.cosmic_large_neb.return_counterfactual_ref=true \
  ++datasets.train.cosmic_large_neb.counterfactual_same_class_probability=0.8 \
  ++datasets.train.cosmic_large_neb.counterfactual_max_resample_attempts=20 \
  "$@"
