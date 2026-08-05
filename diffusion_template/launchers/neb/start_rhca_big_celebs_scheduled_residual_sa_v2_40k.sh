#!/usr/bin/env bash
set -euo pipefail

# Opt-in residual BA-v2 arm. This delegates all sealed-dataset, pinned-plan,
# ONNX CUDA, full-96, and immutable-Comet preflights to the established Neb
# launcher. To replay the old behavior, use
# start_rhca_big_celebs_scheduled_clean_ba32_40k.sh instead.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CONFIG_NAME="big_celebs_scheduled_rhca_residual_sa_v2_40k"
export RUN_NAME="${RUN_NAME:-rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r6}"
export EXPERIMENT_SPEC_PATH="${EXPERIMENT_SPEC_PATH:-${PROJECT_ROOT}/experiments/big_celebs/${RUN_NAME}.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec bash "${SCRIPT_DIR}/start_rhca_big_celebs_scheduled_sameid_40k.sh" "$@"
