#!/usr/bin/env bash
set -euo pipefail

# E5-L40: exact E4 architecture/objective with a 40k ceiling. One continuous
# run preserves optimizer and sampling-plan state while producing independent
# full-96 validations and recoverable checkpoints every 2,000 steps.
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

export CONFIG_NAME="big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_40k"
export RUN_NAME="${RUN_NAME:-rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1}"
export EXPERIMENT_SPEC_PATH="${EXPERIMENT_SPEC_PATH:-${PROJECT_ROOT}/experiments/big_celebs/${RUN_NAME}.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAIN_EPOCH_LEN=2000
export TRAIN_EPOCHS=20

exec bash "${SCRIPT_DIR}/start_rhca_big_celebs_scheduled_sameid_40k.sh" "$@"
