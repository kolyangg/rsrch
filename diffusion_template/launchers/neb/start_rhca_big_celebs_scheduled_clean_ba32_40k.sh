#!/usr/bin/env bash
set -euo pipefail

# Correctness-only BA32 arm. The delegated Neb launcher retains all sealed
# dataset, sampling-plan, ONNX CUDA, fixed-96, and Comet preflights.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CONFIG_NAME="big_celebs_scheduled_rhca_clean_ba32_40k"
export RUN_NAME="${RUN_NAME:-rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1}"
export EXPERIMENT_SPEC_PATH="${EXPERIMENT_SPEC_PATH:-${PROJECT_ROOT}/experiments/big_celebs/${RUN_NAME}.json}"

# Neb exposes one project GPU as index 0. Keep this overrideable because live
# allocation must still be checked immediately before launch.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec bash "${SCRIPT_DIR}/start_rhca_big_celebs_scheduled_sameid_40k.sh" "$@"
