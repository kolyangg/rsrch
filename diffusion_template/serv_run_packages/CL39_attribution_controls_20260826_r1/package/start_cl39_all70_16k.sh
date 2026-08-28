#!/usr/bin/env bash
# Full-96 CL39-16k actual replay plus all-70 BA-off causal control.
set -euo pipefail

CL39_AUDIT_JOB_TAG=all70_16k
CL39_AUDIT_SOURCE_DIR=source_all70_historical
export CL39_AUDIT_JOB_TAG CL39_AUDIT_SOURCE_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_cl39_attribution.sh"

run_arm() {
  local arm="$1" branch_mode="$2"
  local run_name="CL39_16k_all70_${arm}_attribution_r1"
  local generated_root="${TASK_ROOT}/saved/${run_name}/val_images/manual_val"
  local gate="${TASK_ROOT}/gates/${run_name}.json"
  echo "START_CL39_16K_ALL70 arm=${arm} branch_mode=${branch_mode}"
  accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
    train.py \
    --config-name="${CL39_CONFIG}" \
    +validation_only=true \
    +validation_epoch=8 \
    'val_datasets_names=[manual_val]' \
    'inference_metrics=[]' \
    writer=console \
    "writer.run_name=${run_name}" \
    "trainer.from_pretrained=${CL39_CHECKPOINT_16K}" \
    "trainer.save_dir=${TASK_ROOT}/saved" \
    trainer.face_quality.enabled=false \
    trainer.log_per_image_id_sim_table=false \
    +validation_args.cl39_analysis_enabled=true \
    +validation_args.cl39_analysis_capture=false \
    +validation_args.cl39_analysis_processor_scope=all_hardcase \
    "+validation_args.cl39_analysis_branch_mode=${branch_mode}" \
    validation_debug_timing=true

  python - "${arm}" "${branch_mode}" "${run_name}" "${generated_root}" "${gate}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

arm, branch_mode, run_name, generated_root, gate_path = sys.argv[1:]
generated = sorted(Path(generated_root).glob("step_16000_batch_*/*.png"))
payload = {
    "arm": arm,
    "branch_mode": branch_mode,
    "run_name": run_name,
    "generated_count": len(generated),
    "checkpoint_sha256": "a598b929e4fbfab7eac0f9474c9c96d1713dbac6224e1de6ffbca4b43ae29e86",
    "immutable_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
    "processor_scope": "all_hardcase",
    "validation_step": 16000,
    "validation_panel": "manual_val fixed-96",
}
if len(generated) != 96:
    raise SystemExit(f"Expected 96 outputs for {run_name}, found {len(generated)}")
if arm == "actual":
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
        raise SystemExit(f"Sealed 16k replay gate failed: {payload}")
Path(gate_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
  echo "COMPLETE_CL39_16K_ALL70 arm=${arm}"
}

run_arm actual actual
run_arm ba_off native
echo "CL39_16K_ALL70_COMPLETE"
