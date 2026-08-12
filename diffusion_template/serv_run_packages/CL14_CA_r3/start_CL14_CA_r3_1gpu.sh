#!/usr/bin/env bash
set -euo pipefail
export RUN_ID="CL14_CA_r3"
export CONFIG_ID="CL14_CA"
exec bash "$(dirname "${BASH_SOURCE[0]}")/../CL14_CA_relaunch_common/start_CL14_CA_variant_1gpu.sh"
