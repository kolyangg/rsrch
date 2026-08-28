#!/usr/bin/env bash
# Two isolated acceleration workers for already-gated seed-specific bbox caches.
set -euo pipefail

: "${ACCEL_WORKER:?Set ACCEL_WORKER to 1 or 2}"
case "${ACCEL_WORKER}" in
  1) TASKS=("1:D_wrong_pm_wrong_spatial:1:1" "3:C_wrong_pm_correct_spatial:1:0") ;;
  2) TASKS=("2:D_wrong_pm_wrong_spatial:1:1" "3:D_wrong_pm_wrong_spatial:1:1") ;;
  *) echo "Refusing unknown acceleration worker: ${ACCEL_WORKER}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_identity_crossing_dynamic_masks_20260827_r1"

wait_for_bbox_gate() {
  local seed="$1" gate="${TASK_ROOT}/gates/seed${seed}_dynamic_bbox_alignment.json"
  while ! python - "${gate}" "${seed}" <<'PY'
import json, sys
from pathlib import Path

path, seed = Path(sys.argv[1]), int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
ok = (
    data.get("accepted") is True
    and int(data.get("validation_seed", -1)) == seed
    and int(data.get("bbox_count", -1)) == 96
    and int(data.get("image_count", -1)) == 96
    and int(data.get("no_face", -1)) == 0
    and int(data.get("unowned", 999)) <= 2
    and float(data.get("mean_best_iou", 0.0)) >= 0.50
)
raise SystemExit(0 if ok else 1)
PY
  do
    echo "WAITING_FOR_ACCEPTED_DYNAMIC_BBOX_GATE seed=${seed}"
    sleep 20
  done
  echo "DYNAMIC_BBOX_GATE_OK seed=${seed}"
}

run_accelerated_arm() (
  local seed="$1" label="$2" pm_shift="$3" spatial_shift="$4"
  export EVAL_SEED="${seed}"
  export SOURCE_DIR_OVERRIDE="source_accel${ACCEL_WORKER}"
  export RUNTIME_LABEL="accel${ACCEL_WORKER}_seed${seed}"
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/setup_cl39_dynamic_masks.sh"

  wait_for_bbox_gate "${seed}"
  local run_name="CL39_24k_seed${seed}_cross_${label}_dynamic_bbox_accel${ACCEL_WORKER}_r1"
  local generated_root="${TASK_ROOT}/saved/${run_name}/val_images/manual_val"
  local config_path="${TASK_ROOT}/saved/${run_name}/config.yaml"
  local gate="${TASK_ROOT}/gates/${run_name}.json"

  python - "${AUTO_BBOX_JSON}" "${seed}" <<'PY'
import json, sys
from pathlib import Path

path, seed = Path(sys.argv[1]), int(sys.argv[2])
data = json.loads(path.read_text(encoding="utf-8"))
wrong = [k for k, v in data.items() if int((v.get("_meta") or {}).get("seed", -1)) != seed]
missing = [k for k, v in data.items() if not (v.get("face_crop_new") or v.get("face_crop_old"))]
if len(data) != 96 or wrong or missing:
    raise SystemExit(f"Invalid automatic bbox cache: entries={len(data)} wrong={wrong[:4]} missing={missing[:4]}")
PY

  echo "START_ACCELERATED_CL39_ARM worker=${ACCEL_WORKER} seed=${seed} arm=${label}"
  accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
    train.py \
    --config-name="${CL39_CONFIG}" \
    +validation_only=true \
    +validation_epoch=12 \
    'val_datasets_names=[manual_val]' \
    "datasets.val.manual_val.seeds=[${seed}]" \
    "datasets.val.manual_val.bbox_mask_gen=${BBOX_BASE}" \
    automatic_bboxes=true \
    automatic_bboxes_every_val=false \
    force_log_first_auto_bbox=false \
    val_debug=false \
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

  python - "${seed}" "${label}" "${pm_shift}" "${spatial_shift}" \
    "${run_name}" "${generated_root}" "${config_path}" "${gate}" \
    "${BBOX_BASE}" "${AUTO_BBOX_JSON}" "${SOURCE_MANIFEST_SHA256}" "${ACCEL_WORKER}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

(seed, label, pm_shift, spatial_shift, run_name, generated_root, config_path,
 gate_path, bbox_base, auto_bbox_json, source_manifest_sha256, worker) = sys.argv[1:]
seed, pm_shift, spatial_shift, worker = map(int, (seed, pm_shift, spatial_shift, worker))
generated = sorted(Path(generated_root).glob("step_24000_batch_*/*.png"))
config = OmegaConf.load(config_path)
configured_seeds = list(OmegaConf.select(config, "datasets.val.manual_val.seeds"))
configured_bbox = Path(str(OmegaConf.select(config, "datasets.val.manual_val.bbox_mask_gen"))).resolve()
auto_path = Path(auto_bbox_json).resolve()
auto_data = json.loads(auto_path.read_text(encoding="utf-8"))
payload = {
    "arm": label,
    "validation_seed": seed,
    "configured_seeds": configured_seeds,
    "pm_identity_shift": pm_shift,
    "spatial_identity_shift": spatial_shift,
    "wrong_identity_rule": "next identity in sorted fixed-panel order",
    "run_name": run_name,
    "generated_count": len(generated),
    "automatic_bbox_count": len(auto_data),
    "automatic_bbox_sha256": hashlib.sha256(auto_path.read_bytes()).hexdigest(),
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
    "accelerated": True,
    "acceleration_worker": worker,
}
if configured_seeds != [seed] or configured_bbox != Path(bbox_base).resolve():
    raise SystemExit(f"Validation composition gate failed: {payload}")
if not payload["automatic_bboxes"] or payload["automatic_bboxes_every_val"]:
    raise SystemExit(f"Automatic bbox mode gate failed: {payload}")
if len(auto_data) != 96 or len(generated) != 96:
    raise SystemExit(f"Expected 96 dynamic boxes and outputs: {payload}")
Path(gate_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
  echo "COMPLETE_ACCELERATED_CL39_ARM worker=${ACCEL_WORKER} seed=${seed} arm=${label}"
)

for task in "${TASKS[@]}"; do
  IFS=: read -r seed label pm_shift spatial_shift <<<"${task}"
  run_accelerated_arm "${seed}" "${label}" "${pm_shift}" "${spatial_shift}"
done
echo "CL39_DYNAMIC_MASK_ACCELERATOR_COMPLETE worker=${ACCEL_WORKER}"
