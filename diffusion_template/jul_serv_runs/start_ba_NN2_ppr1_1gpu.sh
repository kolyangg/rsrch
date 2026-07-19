#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN1_CONFIG_NAME="one_id_ba_NN2_ppr1"
export NN1_RUN_NAME_DEFAULT="${NN1_RUN_NAME_DEFAULT:-ba_NN2_ppr1_1gpu}"
export NN1_DEFAULT_GPU="${NN1_DEFAULT_GPU:-0}"
export NN1_DEFAULT_PORT="${NN1_DEFAULT_PORT:-29620}"
export NN1_DESCRIPTION="${NN1_DESCRIPTION:-NN2-PPR1: up-block packed-reference residual; frozen split CA}"
export NN1_REQUIRE_ID_LOSS="0"
export NN1_LAUNCHER_PATH="${NN1_LAUNCHER_PATH:-${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")}"

# Prefer the project PhotoMaker environment when the caller has not activated it.
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

# Screening protocol: fixed 96-image validation at 0/2k/4k/6k.
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export OPTIMIZER_STEPS_PER_EPOCH="${OPTIMIZER_STEPS_PER_EPOCH:-2000}"
export FULL_STEP0_VAL="${FULL_STEP0_VAL:-true}"

# Same-base validation is the PPR default; the RealVis wrapper overrides it.
export NN1_VALIDATION_MODEL="${NN1_VALIDATION_MODEL:-null}"
source "${SCRIPT_DIR}/_run_ba_NN1_common_1gpu.sh" "$@"
