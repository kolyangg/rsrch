#!/usr/bin/env bash
# Corrected full-96 A/B/C/D crossing using seed-specific PhotoMaker-only boxes.
set -euo pipefail

: "${EVAL_SEED:?Set EVAL_SEED to 1, 2, or 3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_cl39_dynamic_masks.sh"

validate_auto_bbox_cache() {
  python - "${AUTO_BBOX_JSON}" "${EVAL_SEED}" <<'PY'
import json, sys
from pathlib import Path

path, seed = Path(sys.argv[1]), int(sys.argv[2])
data = json.loads(path.read_text(encoding="utf-8"))
if len(data) != 96:
    raise SystemExit(f"Expected 96 automatic bboxes, found {len(data)}")
wrong = [key for key, value in data.items() if int((value.get("_meta") or {}).get("seed", -1)) != seed]
if wrong:
    raise SystemExit(f"Automatic bbox cache contains wrong-seed entries: {wrong[:8]}")
missing = [key for key, value in data.items() if not (value.get("face_crop_new") or value.get("face_crop_old"))]
if missing:
    raise SystemExit(f"Automatic bbox cache contains missing boxes: {missing[:8]}")
print(f"AUTO_BBOX_CACHE_OK seed={seed} entries={len(data)}")
PY
}

run_arm() {
  local label="$1" pm_shift="$2" spatial_shift="$3" debug_flag="$4"
  local run_name="CL39_24k_seed${EVAL_SEED}_cross_${label}_dynamic_bbox_r1"
  local generated_root="${TASK_ROOT}/saved/${run_name}/val_images/manual_val"
  local config_path="${TASK_ROOT}/saved/${run_name}/config.yaml"
  local gate="${TASK_ROOT}/gates/${run_name}.json"
  echo "START_CL39_24K_DYNAMIC_BBOX seed=${EVAL_SEED} arm=${label} pm_shift=${pm_shift} spatial_shift=${spatial_shift}"
  accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
    train.py \
    --config-name="${CL39_CONFIG}" \
    +validation_only=true \
    +validation_epoch=12 \
    'val_datasets_names=[manual_val]' \
    "datasets.val.manual_val.seeds=[${EVAL_SEED}]" \
    "datasets.val.manual_val.bbox_mask_gen=${BBOX_BASE}" \
    automatic_bboxes=true \
    automatic_bboxes_every_val=false \
    force_log_first_auto_bbox=false \
    "val_debug=${debug_flag}" \
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

  test -s "${AUTO_BBOX_JSON}"
  validate_auto_bbox_cache

  if [[ "${label}" == A_correct_pm_correct_spatial ]]; then
    local overlay_count
    overlay_count="$(find hm_debug -type f -name auto_bbox_overlay.png 2>/dev/null | wc -l)"
    if [[ "${overlay_count}" -ne 96 ]]; then
      echo "Expected 96 PhotoMaker-only auto-bbox overlays, found ${overlay_count}" >&2
      exit 3
    fi
    cp -a hm_debug/. "${AUTO_BBOX_DEBUG}/"
    python "${TASK_ROOT}/package/verify_dynamic_bbox_alignment.py" \
      --bbox-json "${AUTO_BBOX_JSON}" \
      --images-root "${generated_root}" \
      --seed "${EVAL_SEED}" \
      --output "${TASK_ROOT}/gates/seed${EVAL_SEED}_dynamic_bbox_alignment.json"
  fi

  python - "${EVAL_SEED}" "${label}" "${pm_shift}" "${spatial_shift}" \
    "${run_name}" "${generated_root}" "${config_path}" "${gate}" \
    "${BBOX_BASE}" "${AUTO_BBOX_JSON}" "${SOURCE_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

(seed, label, pm_shift, spatial_shift, run_name, generated_root, config_path,
 gate_path, bbox_base, auto_bbox_json, source_manifest_sha256) = sys.argv[1:]
seed = int(seed)
generated = sorted(Path(generated_root).glob("step_24000_batch_*/*.png"))
config = OmegaConf.load(config_path)
configured_seeds = list(OmegaConf.select(config, "datasets.val.manual_val.seeds"))
configured_bbox = Path(str(OmegaConf.select(config, "datasets.val.manual_val.bbox_mask_gen"))).resolve()
auto_path = Path(auto_bbox_json).resolve()
auto_data = json.loads(auto_path.read_text(encoding="utf-8"))
auto_sha = hashlib.sha256(auto_path.read_bytes()).hexdigest()
payload = {
    "arm": label,
    "validation_seed": seed,
    "configured_seeds": configured_seeds,
    "pm_identity_shift": int(pm_shift),
    "spatial_identity_shift": int(spatial_shift),
    "wrong_identity_rule": "next identity in sorted fixed-panel order",
    "run_name": run_name,
    "generated_count": len(generated),
    "automatic_bbox_count": len(auto_data),
    "automatic_bbox_sha256": auto_sha,
    "automatic_bbox_json": str(auto_path),
    "automatic_bbox_source": "matched-seed PhotoMaker-only validation pass",
    "configured_bbox_base": str(configured_bbox),
    "automatic_bboxes": bool(OmegaConf.select(config, "automatic_bboxes")),
    "automatic_bboxes_every_val": bool(OmegaConf.select(config, "automatic_bboxes_every_val")),
    "checkpoint_sha256": "74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07",
    "source_manifest_sha256": source_manifest_sha256,
    "immutable_parent_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
    "processor_scope": "all_hardcase",
    "validation_step": 24000,
    "validation_panel": "manual_val fixed-96",
}
if configured_seeds != [seed]:
    raise SystemExit(f"Validation seed composition gate failed: {payload}")
if configured_bbox != Path(bbox_base).resolve():
    raise SystemExit(f"Seed-specific bbox path gate failed: {payload}")
if not payload["automatic_bboxes"] or payload["automatic_bboxes_every_val"]:
    raise SystemExit(f"Automatic bbox mode gate failed: {payload}")
if len(auto_data) != 96 or len(generated) != 96:
    raise SystemExit(f"Expected 96 dynamic boxes and outputs: {payload}")
Path(gate_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
  echo "COMPLETE_CL39_24K_DYNAMIC_BBOX seed=${EVAL_SEED} arm=${label}"
}

# The A arm creates the seed-specific PhotoMaker-only box cache and 96 overlays.
run_arm A_correct_pm_correct_spatial 0 0 true
# The remaining arms reuse that exact verified seed-specific cache.
run_arm B_correct_pm_wrong_spatial 0 1 false
run_arm C_wrong_pm_correct_spatial 1 0 false
run_arm D_wrong_pm_wrong_spatial 1 1 false
echo "CL39_24K_IDENTITY_CROSSING_DYNAMIC_BBOX_COMPLETE seed=${EVAL_SEED}"
