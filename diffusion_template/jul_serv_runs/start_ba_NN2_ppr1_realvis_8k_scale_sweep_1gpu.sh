#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PPR_RUN_MODE=scale_sweep
export RUN_NAME="${RUN_NAME:-ba_NN2_ppr1_realvis_8k_scale_sweep}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_8k_scale_sweep}"

INCLUDE_SCALE6="${INCLUDE_SCALE6:-false}"
if [[ "${INCLUDE_SCALE6,,}" =~ ^(1|true|yes)$ ]]; then
  SCALES="[0,1,2,3,4,6]"
else
  SCALES="[0,1,2,3,4]"
fi

SWAP_SCALE="${SWAP_SCALE:-3}"
SWEEP_ARGS=(
  "ppr_diagnostic_matrix=false"
  "ppr_scale_sweep=true"
  "ppr_scale_sweep_output_dir=${OUTPUT_DIR}"
  "ppr_scale_sweep_overwrite=${OVERWRITE_OUTPUT:-false}"
  "ppr_scale_sweep_scales=${SCALES}"
)
if [[ -n "${SWAP_SCALE}" && "${SWAP_SCALE,,}" != "none" ]]; then
  SWEEP_ARGS+=(
    "ppr_scale_sweep_swap_scale=${SWAP_SCALE}"
    "ppr_scale_sweep_swap_count=${SWAP_COUNT:-12}"
  )
fi

exec bash \
  "${SCRIPT_DIR}/start_ba_NN2_ppr1_realvis_4k_diagnostic_1gpu.sh" \
  "${SWEEP_ARGS[@]}" \
  "$@"
