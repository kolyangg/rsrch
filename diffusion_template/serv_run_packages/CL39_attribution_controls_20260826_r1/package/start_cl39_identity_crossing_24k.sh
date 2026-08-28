#!/usr/bin/env bash
# Full-96 2x2 crossing of PM-token identity and spatial BA reference at CL39-24k.
set -euo pipefail

CL39_AUDIT_JOB_TAG=identity_crossing_24k
CL39_AUDIT_SOURCE_DIR=source_cross_historical_r2
export CL39_AUDIT_JOB_TAG CL39_AUDIT_SOURCE_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_cl39_attribution.sh"

run_arm() {
  local label="$1" pm_shift="$2" spatial_shift="$3"
  local run_name="CL39_24k_cross_${label}_r1"
  local generated_root="${TASK_ROOT}/saved/${run_name}/val_images/manual_val"
  local gate="${TASK_ROOT}/gates/${run_name}.json"
  echo "START_CL39_24K_CROSS arm=${label} pm_shift=${pm_shift} spatial_shift=${spatial_shift}"
  accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
    train.py \
    --config-name="${CL39_CONFIG}" \
    +validation_only=true \
    +validation_epoch=12 \
    'val_datasets_names=[manual_val]' \
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

  python - "${label}" "${pm_shift}" "${spatial_shift}" "${run_name}" "${generated_root}" "${gate}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

label, pm_shift, spatial_shift, run_name, generated_root, gate_path = sys.argv[1:]
generated = sorted(Path(generated_root).glob("step_24000_batch_*/*.png"))
payload = {
    "arm": label,
    "pm_identity_shift": int(pm_shift),
    "spatial_identity_shift": int(spatial_shift),
    "wrong_identity_rule": "next identity in sorted fixed-panel order",
    "run_name": run_name,
    "generated_count": len(generated),
    "checkpoint_sha256": "74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07",
    "immutable_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
    "processor_scope": "all_hardcase",
    "validation_step": 24000,
    "validation_panel": "manual_val fixed-96",
}
if len(generated) != 96:
    raise SystemExit(f"Expected 96 outputs for {run_name}, found {len(generated)}")
if label.startswith("A_"):
    sealed_root = Path("/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_sources_cl38_cl45_v1/CL39_cosmic_null_key_confidence_router_24k_full96_r4/diffusion_template/saved/CL39_cosmic_null_key_confidence_router_24k_full96_r4/val_images/manual_val")
    maes = []
    for path in generated:
        sealed = sealed_root / path.parent.name / path.name
        first = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        second = np.asarray(Image.open(sealed).convert("RGB"), dtype=np.float32) / 255.0
        maes.append(float(np.abs(first - second).mean()))
    payload["sealed_rgb_mae_mean"] = float(np.mean(maes))
    payload["sealed_rgb_mae_max"] = float(np.max(maes))
    if payload["sealed_rgb_mae_max"] > 0.002:
        raise SystemExit(f"Sealed 24k replay gate failed: {payload}")
Path(gate_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
  echo "COMPLETE_CL39_24K_CROSS arm=${label}"
}

run_arm A_correct_pm_correct_spatial 0 0
run_arm B_correct_pm_wrong_spatial 0 1
run_arm C_wrong_pm_correct_spatial 1 0
run_arm D_wrong_pm_wrong_spatial 1 1
echo "CL39_24K_IDENTITY_CROSSING_COMPLETE"
