#!/usr/bin/env bash
set -euo pipefail

# Optional 1-GPU routing control: same corrected precision/CFG behavior as N34A,
# but restore all target-face CA sites. Limit to 3k and compare against N34A.

RUN_NAME="${RUN_NAME:-ba_alllayers_qformer_nocausal_1gpu_N34B}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env RUN_NAME="${RUN_NAME}" \
    bash "${SCRIPT_DIR}/start_ba_highres_qformer_nocausal_1gpu_N34A.sh" \
    model.ba_ca_layer_allowlist=null \
    "$@"
