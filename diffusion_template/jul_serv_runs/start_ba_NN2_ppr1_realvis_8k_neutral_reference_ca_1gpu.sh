#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Inference-only ablation from the NN2-PPR epoch-4 (8k) checkpoint.
# The target PhotoMaker half is unchanged. Only reference-half cross-attention
# receives a zero/null prompt. The same PM0/R1N1/R2N1/R1N2/R2N2 matrix and all
# identity, text, MAE, LPIPS, landmark, seam, and tensor-stage metrics are run.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29628}"
export PPR_RUN_MODE=reference_vs_noise
export REQUIRE_LPIPS="${REQUIRE_LPIPS:-true}"
export RUN_NAME="${RUN_NAME:-ba_NN2_ppr1_realvis_8k_neutral_reference_ca}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_8k_neutral_reference_ca}"

# Optional first positional argument: validation batch size.
# Example: ./start_ba_NN2_ppr1_realvis_8k_neutral_reference_ca_1gpu.sh 12
if [[ $# -gt 0 && "$1" =~ ^[1-9][0-9]*$ ]]; then
  export BATCH_SIZE="$1"
  shift
fi

exec bash \
  "${SCRIPT_DIR}/start_ba_NN2_ppr1_realvis_8k_reference_vs_noise_1gpu.sh" \
  "ppr_reference_ca_mode=zero" \
  "$@"
