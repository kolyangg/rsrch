#!/usr/bin/env bash
# N17 inference-only ablation:
#   pose_adapt_ratio=0.25
#   ca_mixing_for_face=false
# Uses the same intermediate full-val helper, seeds, prompts, and checkpoints.

set -euo pipefail

cd /home/kolyangg/rsrch/diffusion_template

export PAR_ONLY_OVERRIDES="pipeline.ba_enable_runtime_sa_knobs=true model.ba_enable_runtime_sa_knobs=true pipeline.pose_adapt_ratio=0.25 pipeline.ca_mixing_for_face=false model.use_id_embeds=false pipeline.use_id_embeds=false"

RESULTS_DIR=/home/kolyangg/rsrch/diffusion_template/full_validation_results/par025_no_camix_ablation \
PYTHON_BIN=/home/kolyangg/anaconda3/envs/photomaker/bin/python \
EXTRA_INFER_OVERRIDES="${PAR_ONLY_OVERRIDES}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
bash serv_new_runs/run_full_validation_steps.sh ba_longrun_N17 10000 26000 "$@"
