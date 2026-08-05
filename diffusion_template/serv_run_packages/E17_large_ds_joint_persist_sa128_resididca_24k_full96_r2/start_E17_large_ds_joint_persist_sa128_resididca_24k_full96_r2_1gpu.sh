#!/usr/bin/env bash
set -euo pipefail
export RUN_ID="E17_large_ds_joint_persist_sa128_resididca_24k_full96_r2"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_sources/start_E13_E18_large_ds_24k_1gpu.sh"
