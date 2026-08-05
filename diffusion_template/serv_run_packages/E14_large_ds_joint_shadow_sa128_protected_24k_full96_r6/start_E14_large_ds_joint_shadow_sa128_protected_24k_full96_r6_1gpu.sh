#!/usr/bin/env bash
set -euo pipefail
export RUN_ID="E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6"
export SERV_REPO_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_sources/start_E13_E18_large_ds_24k_1gpu.sh"
