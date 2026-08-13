#!/usr/bin/env bash
set -euo pipefail
# 12 Aug 2026 - Training optimization smoke with one proven validation batch.
export RUN_ID="CL14_CA_optimized_speed_smoke_r11"
export CONFIG_ID="CL14_CA_onebatch_smoke"
exec bash "$(dirname "${BASH_SOURCE[0]}")/../CL14_CA_relaunch_common/start_CL14_CA_variant_1gpu.sh"
