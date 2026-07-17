#!/usr/bin/env bash
set -euo pipefail

# 4-GPU machine, physical GPU 0: NN1a with explicit ID-token CA masking.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN1_CONFIG_NAME="one_id_ba_NN1c_masked_id_prompt"
export NN1_RUN_NAME_DEFAULT="ba_NN1c_masked_id_prompt_1gpu"
export NN1_DEFAULT_GPU="0"
export NN1_DEFAULT_PORT="29613"
export NN1_DESCRIPTION="NN1c: N3a full BA with non-ID reference-prompt tokens masked from CA"
export NN1_REQUIRE_ID_LOSS="0"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN1_common_1gpu.sh" "$@"
