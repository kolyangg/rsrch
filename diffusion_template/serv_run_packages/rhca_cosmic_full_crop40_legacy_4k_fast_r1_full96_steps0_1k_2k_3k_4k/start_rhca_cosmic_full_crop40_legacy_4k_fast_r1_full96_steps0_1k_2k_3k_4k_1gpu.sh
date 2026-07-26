#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="rhca_cosmic_full_crop40_legacy_4k_fast_r1_full96_steps0_1k_2k_3k_4k"
export VALIDATION_SOURCE_RUN="rhca_cosmic_full_crop40_legacy_4k_fast_r1"
export VALIDATION_SOURCE_COMET_KEY="92572589d6594cd59749577fc51f5bba"
export FULL96_MULTISTEP=true
export FULL96_SOURCE_REPRO_BBOX_MANUAL="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/dataset_full/val_dataset/pm96_bboxes_new.json"

exec bash "$(dirname "${BASH_SOURCE[0]}")/../_lib/start_cosmic_full96_fast_env.sh"
