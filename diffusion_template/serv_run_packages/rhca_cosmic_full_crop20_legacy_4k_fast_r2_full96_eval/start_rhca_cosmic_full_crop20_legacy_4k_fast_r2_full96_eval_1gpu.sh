#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="rhca_cosmic_full_crop20_legacy_4k_fast_r2_full96_eval"
export VALIDATION_SOURCE_RUN="rhca_cosmic_full_crop20_legacy_4k_fast_r2"
export VALIDATION_SOURCE_COMET_KEY="f2cd04577b014e6bb2b98fbea5d5472e"
export FULL96_REQUIRE_COMPLETED_EVAL="rhca_cosmic_full_crop20_legacy_4k_full96_eval"

exec bash "$(dirname "${BASH_SOURCE[0]}")/../_lib/start_cosmic_full96_fast_env.sh"
