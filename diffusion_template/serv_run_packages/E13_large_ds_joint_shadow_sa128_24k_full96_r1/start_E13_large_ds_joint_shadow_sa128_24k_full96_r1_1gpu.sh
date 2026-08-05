#!/usr/bin/env bash
set -euo pipefail
export RUN_ID="E13_large_ds_joint_shadow_sa128_24k_full96_r1"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_sources/start_E13_E18_large_ds_24k_1gpu.sh"
