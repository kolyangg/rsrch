#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_par65"
export VALIDATION_SOURCE_RUN="rhca_cosmic_full_crop20_posefirst_4k_fast_r3"
export VALIDATION_SOURCE_COMET_KEY="7839bf5f50924f3ab2bb848fd97837e0"
export CANONICAL_FULL96_RUN="rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_eval"
export POSE_ADAPT_RATIO="0.65"
export FULL96_ACTIVE_LAUNCHER="launchers/active/run_rhca_cosmic_pose_adapt_full96_eval_1gpu.sh"

exec bash "$(dirname "${BASH_SOURCE[0]}")/../_lib/start_cosmic_full96_fast_env.sh"
