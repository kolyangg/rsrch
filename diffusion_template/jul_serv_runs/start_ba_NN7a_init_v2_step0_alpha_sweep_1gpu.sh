#!/usr/bin/env bash
set -euo pipefail

# No-training RealVis sweep: alpha 0.05/0.10/0.20 with a common final cap 0.20.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
SUBSET_SIZE="${SUBSET_SIZE:-24}"
SUBSET_SEED="${SUBSET_SEED:-20260722}"
BATCH_SIZE="${BATCH_SIZE:-12}"

labels=(alpha005 alpha010 alpha020)
logits=(-2.70805020110221 -1.9459101490553132 -1.0986122886681098)

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  logit="${logits[$index]}"
  output_dir="${PROJECT_DIR}/ppr_NN7a_init_v2_step0_${label}_realvis_subset${SUBSET_SIZE}_seed${SUBSET_SEED}"
  RUN_FOREGROUND=1 \
  RUN_NAME="ba_NN7a_init_v2_step0_${label}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MASTER_PORT="$((29680 + index))" \
  bash "${SCRIPT_DIR}/start_ba_NN7a_init_v2_1gpu.sh" \
    validation_only=true \
    continue_run=false \
    +ppr_allow_untrained_validation=true \
    ppr_checkpoint_require_nonzero=false \
    ppr_reference_noise_test=true \
    ppr_reference_noise_scale=1.0 \
    ppr_reference_noise_output_dir="${output_dir}" \
    ppr_reference_noise_overwrite="${OVERWRITE_OUTPUT:-false}" \
    ppr_reference_noise_seeds="${NOISE_SEEDS:-[918273,271828]}" \
    model.ba_gate_init_logit="${logit}" \
    datasets.val.manual_val.limit=96 \
    +datasets.val.manual_val.subset_size="${SUBSET_SIZE}" \
    +datasets.val.manual_val.subset_seed="${SUBSET_SEED}" \
    dataloaders.manual_val.batch_size="${BATCH_SIZE}"
done
