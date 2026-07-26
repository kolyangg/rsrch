#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="rhca_cosmic_full_crop60_posefirst_4k_fast_r1_full96_steps0_1k_2k_3k_4k"
export VALIDATION_SOURCE_RUN="rhca_cosmic_full_crop60_posefirst_4k_fast_r1"
export VALIDATION_SOURCE_COMET_KEY="a96bcbae3d2b4698a43d7ec80457586c"
export FULL96_MULTISTEP=true
export FULL96_SOURCE_REPRO_BBOX_MANUAL="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/dataset_full/val_dataset/pm96_bboxes_new.json"

exec bash "$(dirname "${BASH_SOURCE[0]}")/../_lib/start_cosmic_full96_fast_env.sh"
