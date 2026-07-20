#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Inference-only five-way test from the same epoch-4 (8k) checkpoint:
# PM0, R1N1, R2N1, R1N2, R2N2. A separate reference RNG keeps the target
# latent seed unchanged. Tensor traces and all metrics remain per sample.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29627}"
export PPR_RUN_MODE=reference_vs_noise
export RUN_NAME="${RUN_NAME:-ba_NN2_ppr1_realvis_8k_reference_vs_noise}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_8k_reference_vs_noise}"

NOISE_SEEDS="${NOISE_SEEDS:-[918273,271828]}"
LIMIT="${LIMIT:-96}"
BATCH_SIZE="${BATCH_SIZE:-1}"
if [[ $# -gt 0 && "$1" =~ ^[1-9][0-9]*$ ]]; then
  BATCH_SIZE="$1"
  shift
fi

exec bash \
  "${SCRIPT_DIR}/start_ba_NN2_ppr1_realvis_4k_diagnostic_1gpu.sh" \
  "datasets.val.manual_val.limit=${LIMIT}" \
  "dataloaders.manual_val.batch_size=${BATCH_SIZE}" \
  "ppr_diagnostic_matrix=false" \
  "ppr_scale_sweep=false" \
  "ppr_reference_noise_test=true" \
  "ppr_reference_noise_output_dir=${OUTPUT_DIR}" \
  "ppr_reference_noise_overwrite=${OVERWRITE_OUTPUT:-false}" \
  "ppr_reference_noise_seeds=${NOISE_SEEDS}" \
  "$@"
