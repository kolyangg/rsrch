#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/start_ba_NN2_ppr1_realvis_1gpu.sh"
RUN_NAME="${RUN_NAME:-ba_NN2_ppr1_realvis_1gpu}"
REVALIDATION_RUN_NAME="${REVALIDATION_RUN_NAME:-${RUN_NAME}_checkpoint4k_revalidation}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
EXPECTED_CHECKPOINT_EPOCH="${EXPECTED_CHECKPOINT_EPOCH:-2}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs_new_runs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_resume4k_$(date +%Y%m%d_%H%M%S).log}"

# Use the same PhotoMaker environment policy as the underlying launcher.
if [[ -n "${PHOTOMAKER_ENV_BIN:-}" ]]; then
  if [[ ! -x "${PHOTOMAKER_ENV_BIN}/python" ]]; then
    echo "Invalid PHOTOMAKER_ENV_BIN: ${PHOTOMAKER_ENV_BIN}" >&2
    exit 2
  fi
  export PATH="${PHOTOMAKER_ENV_BIN}:${PATH}"
elif [[ "${CONDA_DEFAULT_ENV:-}" != *photomaker* ]]; then
  for candidate in \
    "${HOME}/anaconda3/envs/photomaker/bin" \
    "${HOME}/conda_env/photomaker_NS/bin"; do
    if [[ -x "${candidate}/python" ]]; then
      export PATH="${candidate}:${PATH}"
      break
    fi
  done
fi
if ! python -c 'import torch, diffusers' >/dev/null 2>&1; then
  echo "Activate the PhotoMaker conda environment or set PHOTOMAKER_ENV_BIN." >&2
  exit 2
fi

# This is the default checkpoint location on the current one-GPU server.
if [[ -z "${PM_PATH:-}" && -f "/home/niko/models/PhotoMaker-V2/photomaker-v2.bin" ]]; then
  export PM_PATH="/home/niko/models/PhotoMaker-V2/photomaker-v2.bin"
fi

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  for candidate in \
    "${PROJECT_DIR}/saved/ba_NN2_ppr1_realvis_1gpu/checkpoint-epoch2.pth" \
    "${PROJECT_DIR}/saved/ba_NN2_ppr1_1gpu/checkpoint-epoch2.pth"; do
    if [[ -f "${candidate}" ]]; then
      CHECKPOINT_PATH="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CHECKPOINT_PATH:-}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to the NN2-PPR 4k checkpoint-epoch2.pth." >&2
  exit 2
fi
CHECKPOINT_PATH="$(cd -- "$(dirname -- "${CHECKPOINT_PATH}")" && pwd)/$(basename -- "${CHECKPOINT_PATH}")"

if [[ "${RUN_FOREGROUND:-0}" != "1" && "${DETACHED_RUN:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  echo "Starting corrected 4k validation, then NN2-PPR continuation on GPU ${CUDA_VISIBLE_DEVICES}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Log: ${LOG_FILE}"
  DETACHED_RUN=1 \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    RUN_NAME="${RUN_NAME}" \
    REVALIDATION_RUN_NAME="${REVALIDATION_RUN_NAME}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    EXPECTED_CHECKPOINT_EPOCH="${EXPECTED_CHECKPOINT_EPOCH}" \
    LOG_DIR="${LOG_DIR}" LOG_FILE="${LOG_FILE}" \
    nohup bash "$0" "$@" >"${LOG_FILE}" 2>&1 </dev/null &
  echo "PID: $!"
  echo "Follow with: tail -f ${LOG_FILE}"
  exit 0
fi

cd "${PROJECT_DIR}"

# Fail before allocating the model if this is not the expected full 4k
# checkpoint or if the trained PPR lane is absent.
CHECKPOINT_PATH="${CHECKPOINT_PATH}" EXPECTED_CHECKPOINT_EPOCH="${EXPECTED_CHECKPOINT_EPOCH}" \
python - <<'PY'
import os
import torch

path = os.environ["CHECKPOINT_PATH"]
expected_epoch = int(os.environ["EXPECTED_CHECKPOINT_EPOCH"])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if int(checkpoint.get("epoch", -1)) != expected_epoch:
    raise RuntimeError(
        f"Expected epoch {expected_epoch} (4k at 2k/epoch), "
        f"found {checkpoint.get('epoch')}"
    )
for key in ("state_dict", "optimizer", "lr_scheduler", "config"):
    if key not in checkpoint:
        raise RuntimeError(f"Continuation checkpoint is missing {key}")
processors = checkpoint["state_dict"].get("attn_processors", {})
connector_up = [
    state["connector_up.weight"].float()
    for state in processors.values()
    if "connector_up.weight" in state
]
nonzero = sum(int(torch.count_nonzero(tensor).item()) for tensor in connector_up)
l2 = sum(float(tensor.square().sum().item()) for tensor in connector_up) ** 0.5
if not connector_up or nonzero == 0:
    raise RuntimeError("Checkpoint has no trained, nonzero PPR connector_up weights")
print(
    f"[Checkpoint file preflight] epoch={expected_epoch} "
    f"connector_up_tensors={len(connector_up)} nonzero={nonzero} l2={l2:.6f}"
)
PY

COMMON_ARGS=(
  saved_checkpoint="${CHECKPOINT_PATH}"
  ppr_checkpoint_require_nonzero=true
  strict_checkpoint_model_config=true
  "$@"
)

echo "Phase 1/2: weights-only 96-image RealVis validation at checkpoint step 4000"
RUN_FOREGROUND=1 \
RUN_NAME="${REVALIDATION_RUN_NAME}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MASTER_PORT="${MASTER_PORT_VALIDATION:-29622}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
bash "${BASE_LAUNCHER}" \
  validation_only=true \
  continue_run=false \
  "${COMMON_ARGS[@]}"

echo "Phase 2/2: validation passed; restoring optimizer/scheduler and continuing at epoch 3"
CONTINUE_ARGS=()
if [[ -n "${CONTINUE_COMET_ID:-}" ]]; then
  CONTINUE_ARGS+=("cometml_id=${CONTINUE_COMET_ID}")
fi
RUN_FOREGROUND=1 \
RUN_NAME="${RUN_NAME}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MASTER_PORT="${MASTER_PORT_TRAINING:-29623}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
bash "${BASE_LAUNCHER}" \
  validation_only=false \
  continue_run=true \
  "${COMMON_ARGS[@]}" \
  "${CONTINUE_ARGS[@]}"
