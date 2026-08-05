#!/usr/bin/env bash
set -euo pipefail

# Controlled E4 objective arm: one 2,000-step epoch on the same first 4,000
# pinned schedule rows as E3. Only differentiable reference ranking and its
# 50% paired sampling rate change; the delegated launcher retains all sealed
# dataset, ONNX CUDA, fixed-96, Comet, and runtime-integrity preflights.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t ACTIVE_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | awk 'NF {print $1}'
  )
  if (( ${#ACTIVE_GPU_PIDS[@]} > 0 )) \
      && [[ "${ALLOW_NEB_GPU_BUSY:-0}" != "1" ]]; then
    echo "Refusing to overlap the Neb GPU; active compute PIDs: ${ACTIVE_GPU_PIDS[*]}" >&2
    echo "Stop and verify the existing process group first." >&2
    exit 5
  fi
fi

export CONFIG_NAME="big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_2k"
export RUN_NAME="${RUN_NAME:-rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2}"
export EXPERIMENT_SPEC_PATH="${EXPERIMENT_SPEC_PATH:-${PROJECT_ROOT}/experiments/big_celebs/${RUN_NAME}.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAIN_EPOCH_LEN=2000
export TRAIN_EPOCHS=1

exec bash "${SCRIPT_DIR}/start_rhca_big_celebs_scheduled_sameid_40k.sh" "$@"
