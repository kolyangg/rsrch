#!/usr/bin/env bash
set -euo pipefail

# NN2 uses the validated NN1 one-GPU runner with a 10 x 2k = 20k budget.
# Runs may be stopped early when the visual failure mode is already decisive.
: "${NN2_CONFIG_NAME:?NN2_CONFIG_NAME is required}"
: "${NN2_RUN_NAME_DEFAULT:?NN2_RUN_NAME_DEFAULT is required}"
: "${NN2_DEFAULT_GPU:?NN2_DEFAULT_GPU is required}"
: "${NN2_DEFAULT_PORT:?NN2_DEFAULT_PORT is required}"
: "${NN2_DESCRIPTION:?NN2_DESCRIPTION is required}"
: "${NN2_LAUNCHER_PATH:?NN2_LAUNCHER_PATH is required}"

export NN1_CONFIG_NAME="${NN2_CONFIG_NAME}"
export NN1_RUN_NAME_DEFAULT="${NN2_RUN_NAME_DEFAULT}"
export NN1_DEFAULT_GPU="${NN2_DEFAULT_GPU}"
export NN1_DEFAULT_PORT="${NN2_DEFAULT_PORT}"
export NN1_DESCRIPTION="${NN2_DESCRIPTION}"
export NN1_REQUIRE_ID_LOSS="0"
export NN1_LAUNCHER_PATH="${NN2_LAUNCHER_PATH}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export OPTIMIZER_STEPS_PER_EPOCH="${OPTIMIZER_STEPS_PER_EPOCH:-2000}"
export FULL_STEP0_VAL="${FULL_STEP0_VAL:-true}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_ba_NN1_common_1gpu.sh" "$@"
