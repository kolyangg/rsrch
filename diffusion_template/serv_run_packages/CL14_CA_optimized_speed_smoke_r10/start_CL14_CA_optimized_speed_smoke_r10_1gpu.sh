#!/usr/bin/env bash
set -euo pipefail
# 12 Aug 2026 - Training optimization smoke with fused CA and CUDA rehome.
export RUN_ID="CL14_CA_optimized_speed_smoke_r10"
export CONFIG_ID="CL14_CA_skipval_smoke"
exec bash "$(dirname "${BASH_SOURCE[0]}")/../CL14_CA_relaunch_common/start_CL14_CA_variant_1gpu.sh"
