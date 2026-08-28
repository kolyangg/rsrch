#!/usr/bin/env bash
# Full-96 2x2 PM-token/spatial-reference crossing for one declared inference seed.
set -euo pipefail

: "${EVAL_SEED:?Set EVAL_SEED to 1, 2, or 3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_cl39_crossing_multiseed.sh"

run_arm() {
  local label="$1" pm_shift="$2" spatial_shift="$3"
  local run_name="CL39_24k_seed${EVAL_SEED}_cross_${label}_r1"
  local generated_root="${TASK_ROOT}/saved/${run_name}/val_images/manual_val"
  local config_path="${TASK_ROOT}/saved/${run_name}/config.yaml"
  local gate="${TASK_ROOT}/gates/${run_name}.json"
  echo "START_CL39_24K_MULTISEED seed=${EVAL_SEED} arm=${label} pm_shift=${pm_shift} spatial_shift=${spatial_shift}"
  accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
    train.py \
    --config-name="${CL39_CONFIG}" \
    +validation_only=true \
    +validation_epoch=12 \
    'val_datasets_names=[manual_val]' \
    "datasets.val.manual_val.seeds=[${EVAL_SEED}]" \
    'inference_metrics=[]' \
    writer=console \
    "writer.run_name=${run_name}" \
    "trainer.from_pretrained=${CL39_CHECKPOINT_24K}" \
    "trainer.save_dir=${TASK_ROOT}/saved" \
    trainer.face_quality.enabled=false \
    trainer.log_per_image_id_sim_table=false \
    +validation_args.cl39_analysis_enabled=true \
    +validation_args.cl39_analysis_capture=false \
    +validation_args.cl39_analysis_processor_scope=all_hardcase \
    +validation_args.cl39_analysis_branch_mode=actual \
    "+validation_args.cl39_analysis_pm_identity_shift=${pm_shift}" \
    "+validation_args.cl39_analysis_spatial_identity_shift=${spatial_shift}" \
    validation_debug_timing=true

  python - "${EVAL_SEED}" "${label}" "${pm_shift}" "${spatial_shift}" \
    "${run_name}" "${generated_root}" "${config_path}" "${gate}" <<'PY'
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

seed, label, pm_shift, spatial_shift, run_name, generated_root, config_path, gate_path = sys.argv[1:]
seed = int(seed)
generated = sorted(Path(generated_root).glob("step_24000_batch_*/*.png"))
config = OmegaConf.load(config_path)
configured_seeds = list(OmegaConf.select(config, "datasets.val.manual_val.seeds"))
payload = {
    "arm": label,
    "validation_seed": seed,
    "configured_seeds": configured_seeds,
    "pm_identity_shift": int(pm_shift),
    "spatial_identity_shift": int(spatial_shift),
    "wrong_identity_rule": "next identity in sorted fixed-panel order",
    "run_name": run_name,
    "generated_count": len(generated),
    "checkpoint_sha256": "74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07",
    "immutable_parent_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
    "source_manifest_sha256": "9566862387800eded64c8972461b873ddd9ac9c86fd1cd27ae23425a27a2d10f",
    "processor_scope": "all_hardcase",
    "validation_step": 24000,
    "validation_panel": "manual_val fixed-96",
}
if configured_seeds != [seed]:
    raise SystemExit(f"Validation seed composition gate failed: {payload}")
if len(generated) != 96:
    raise SystemExit(f"Expected 96 outputs for {run_name}, found {len(generated)}")
Path(gate_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
  echo "COMPLETE_CL39_24K_MULTISEED seed=${EVAL_SEED} arm=${label}"
}

run_arm A_correct_pm_correct_spatial 0 0
run_arm B_correct_pm_wrong_spatial 0 1
run_arm C_wrong_pm_correct_spatial 1 0
run_arm D_wrong_pm_wrong_spatial 1 1
echo "CL39_24K_IDENTITY_CROSSING_COMPLETE seed=${EVAL_SEED}"
