#!/usr/bin/env bash
set -euo pipefail
# 12 Aug 2026 - Training optimization production run with CL20's Eddie fix.
export RUN_ID="CL14_CA_optimized_r8"
export CONFIG_ID="CL14_CA"
exec bash "$(dirname "${BASH_SOURCE[0]}")/../CL14_CA_relaunch_common/start_CL14_CA_variant_1gpu.sh"
