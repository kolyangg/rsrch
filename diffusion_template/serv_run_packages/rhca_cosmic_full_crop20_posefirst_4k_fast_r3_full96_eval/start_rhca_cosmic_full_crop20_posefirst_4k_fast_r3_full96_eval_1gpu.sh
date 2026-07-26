#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_eval"
export VALIDATION_SOURCE_RUN="rhca_cosmic_full_crop20_posefirst_4k_fast_r3"
export VALIDATION_SOURCE_COMET_KEY="7839bf5f50924f3ab2bb848fd97837e0"
export FULL96_REQUIRE_COMPLETED_EVAL="rhca_cosmic_full_crop20_legacy_4k_full96_eval"

exec bash "$(dirname "${BASH_SOURCE[0]}")/../_lib/start_cosmic_full96_fast_env.sh"
