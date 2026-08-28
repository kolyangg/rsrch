#!/usr/bin/env bash
# Matched fixed-96 CL19/23/27/39 branch audit, sequentially on one Serv A100.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="BA_lineage_branch_audit_serv_r1"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}"
PROJECT_ROOT="${TASK_ROOT}/source/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

CL19_RUN="CL19_cosmic_true_soft_fullquery_router_24k_full96_r2"
CL23_RUN="CL23_cosmic_temporal_frequency_router_24k_full96_r1"
CL27_RUN="CL27_cosmic_frequency_surface_energy_24k_full96_r3"
CL39_RUN="CL39_cosmic_null_key_confidence_router_24k_full96_r4"

CL19_CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl15_cl20_v1/${CL19_RUN}/diffusion_template/saved/${CL19_RUN}/checkpoint-epoch12.pth"
CL23_CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl21_cl26_v1/${CL23_RUN}/diffusion_template/saved/${CL23_RUN}/checkpoint-epoch12.pth"
CL27_CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl27_cl29_v4/${CL27_RUN}/diffusion_template/saved/${CL27_RUN}/checkpoint-epoch12.pth"
CL39_CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${CL39_RUN}/diffusion_template/saved/${CL39_RUN}/checkpoint-epoch12.pth"

declare -A CHECKPOINTS=(
  [CL19]="${CL19_CHECKPOINT}"
  [CL23]="${CL23_CHECKPOINT}"
  [CL27]="${CL27_CHECKPOINT}"
  [CL39]="${CL39_CHECKPOINT}"
)
declare -A CHECKPOINT_SHA256=(
  [CL19]="07aefcb03e432e84f31556429e0bfe221c23703cbe2164e09fe988f984cd2bd9"
  [CL23]="70201f0e82c9cb24aeb5adc27ad660e5e11aea8d29b6969c449ec39e3c8b379c"
  [CL27]="100072242ca34b2056f512f41a32e7aa8e7b98e4b10146043fd258a410ca8a50"
  [CL39]="74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07"
)
declare -A COMET_KEYS=(
  [CL19]="cfeda7b55c174b3c83e8d40537ebb6dd"
  [CL23]="a9ec9c59d1624c68acb98737dcd65298"
  [CL27]="dbfbf40c3bdd4f70bedc58bda3dfb9cd"
  [CL39]="b1ca0b3da679401c85b991f1bbdf0b2a"
)
declare -A CONFIGS=(
  [CL19]="CL19_cosmic_true_soft_fullquery_router_24k"
  [CL23]="CL23_cosmic_temporal_frequency_router_24k"
  [CL27]="CL27_cosmic_frequency_surface_energy_24k"
  [CL39]="CL39_cosmic_null_key_confidence_router_24k"
)
declare -A SEALED_RUNS=(
  [CL19]="${OWNER_ROOT}/runtime_sources_cl15_cl20_v1/${CL19_RUN}/diffusion_template/saved/${CL19_RUN}"
  [CL23]="${OWNER_ROOT}/runtime_sources_cl21_cl26_v1/${CL23_RUN}/diffusion_template/saved/${CL23_RUN}"
  [CL27]="${OWNER_ROOT}/runtime_sources_cl27_cl29_v4/${CL27_RUN}/diffusion_template/saved/${CL27_RUN}"
  [CL39]="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${CL39_RUN}/diffusion_template/saved/${CL39_RUN}"
)

cd "${TASK_ROOT}"
sha256sum -c package_manifest.sha256
sha256sum -c source_manifest.sha256
for lineage in CL19 CL23 CL27 CL39; do
  printf '%s  %s\n' "${CHECKPOINT_SHA256[${lineage}]}" "${CHECKPOINTS[${lineage}]}" \
    | sha256sum -c -
done

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      CONDA_BASE="${candidate}"
      break
    fi
  done
fi
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export HOME="${TASK_ROOT}/home"
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export MPLCONFIGDIR="${TASK_ROOT}/home/.config/matplotlib"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "${HOME}" "${MPLCONFIGDIR}" "${TASK_ROOT}/saved" "${TASK_ROOT}/gates"

cd "${PROJECT_ROOT}"
python -m py_compile \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/trainer/sdxl_trainers.py \
  tools/analysis/cl39_attention_capture.py \
  tools/analysis/render_ba_lineage_branch_audit.py

run_arm() {
  local lineage="$1"
  local arm="$2"
  local branch_mode="$3"
  local confidence_override="$4"
  local run_name="BA_lineage_${lineage}_${arm}"
  local gate="${TASK_ROOT}/gates/${run_name}.json"
  local generated_root="${TASK_ROOT}/saved/${run_name}/val_images/manual_val"
  if [[ -s "${gate}" ]]; then
    echo "SKIP_COMPLETE ${run_name}"
    return
  fi

  echo "START_ARM lineage=${lineage} arm=${arm} branch=${branch_mode} confidence=${confidence_override}"
  args=(
    --config-name="${CONFIGS[${lineage}]}"
    +validation_only=true
    +validation_epoch=12
    'val_datasets_names=[manual_val]'
    'inference_metrics=[]'
    writer=console
    "writer.run_name=${run_name}"
    "trainer.from_pretrained=${CHECKPOINTS[${lineage}]}"
    "trainer.save_dir=${TASK_ROOT}/saved"
    trainer.face_quality.enabled=false
    trainer.log_per_image_id_sim_table=false
    +validation_args.cl39_analysis_enabled=true
    +validation_args.cl39_analysis_capture=false
    +validation_args.cl39_analysis_processor_scope=all_hardcase
    "+validation_args.cl39_analysis_branch_mode=${branch_mode}"
    validation_debug_timing=true
  )
  if [[ "${confidence_override}" != "none" ]]; then
    args+=("+validation_args.cl39_analysis_confidence_override=${confidence_override}")
  fi
  accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
    train.py "${args[@]}"

  python - "${lineage}" "${arm}" "${branch_mode}" "${confidence_override}" \
    "${run_name}" "${generated_root}" "${SEALED_RUNS[${lineage}]}" \
    "${CHECKPOINT_SHA256[${lineage}]}" "${COMET_KEYS[${lineage}]}" "${gate}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

(
    lineage, arm, branch_mode, confidence_override, run_name,
    generated_root, sealed_run, checkpoint_sha256, comet_key, gate_path,
) = sys.argv[1:]
generated_root = Path(generated_root)
sealed_root = Path(sealed_run) / "val_images" / "manual_val"
generated = sorted(generated_root.glob("step_24000_batch_*/*.png"))
payload = {
    "lineage": lineage,
    "arm": arm,
    "branch_mode": branch_mode,
    "confidence_override": confidence_override,
    "run_name": run_name,
    "generated_count": len(generated),
    "checkpoint_sha256": checkpoint_sha256,
    "immutable_comet_key": comet_key,
    "processor_scope": "all_hardcase",
    "validation_step": 24000,
    "validation_panel": "manual_val fixed-96",
    "batch_size": 12,
}
if len(generated) != 96:
    raise SystemExit(f"Expected 96 outputs for {run_name}, found {len(generated)}")
if arm == "actual":
    maes, maxima, changed = [], [], []
    for path in generated:
        sealed = sealed_root / path.parent.name / path.name
        if not sealed.is_file():
            raise SystemExit(f"Missing sealed actual counterpart: {sealed}")
        first = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        second = np.asarray(Image.open(sealed).convert("RGB"), dtype=np.float32) / 255.0
        difference = np.abs(first - second)
        maes.append(float(difference.mean()))
        maxima.append(float(difference.max()))
        changed.append(float((difference.max(axis=2) > 1.0 / 255.0).mean()))
    payload.update(
        sealed_rgb_mae_mean=float(np.mean(maes)),
        sealed_rgb_mae_max=float(np.max(maes)),
        sealed_max_abs=float(np.max(maxima)),
        sealed_pixel_changed_gt_1_255_mean=float(np.mean(changed)),
    )
    if payload["sealed_rgb_mae_max"] > 0.002:
        raise SystemExit(f"Sealed replay gate failed: {payload}")
Path(gate_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
  echo "COMPLETE_ARM ${run_name}"
}

# CL19 actual is already the trained N + S(R-N) operating point.
run_arm CL19 actual actual none
run_arm CL19 native native none

for lineage in CL23 CL27; do
  run_arm "${lineage}" actual actual none
  run_arm "${lineage}" native native none
  run_arm "${lineage}" reference_face reference_face none
  run_arm "${lineage}" low_only low_only none
  run_arm "${lineage}" high_only high_only none
done

run_arm CL39 actual actual none
run_arm CL39 native native none
run_arm CL39 reference_face reference_face none
run_arm CL39 low_only low_only none
run_arm CL39 high_only high_only none
run_arm CL39 confidence_one actual 1.0

python tools/analysis/render_ba_lineage_branch_audit.py \
  --task-root "${TASK_ROOT}" \
  --manifest "${TASK_ROOT}/package/sample_manifest.json" \
  --reference-root "${TASK_ROOT}/source/dataset_full/val_dataset/references" \
  --output-root "${TASK_ROOT}/assembled"

test "$(find "${TASK_ROOT}/gates" -type f -name 'BA_lineage_*.json' | wc -l)" -eq 18
test "$(find "${TASK_ROOT}/assembled/samples" -type f -name '*.png' | wc -l)" -eq 16
test "$(find "${TASK_ROOT}/assembled/overviews" -type f -name '*.png' | wc -l)" -eq 16
test -s "${TASK_ROOT}/assembled/branch_metrics.csv"
test -s "${TASK_ROOT}/assembled/summary.json"

python - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/BA_lineage_branch_audit_serv_r1")
payload = json.loads((root / "assembled/summary.json").read_text(encoding="utf-8"))
payload.update(status="complete", arm_count=18, generated_image_count=18 * 96)
(root / "AUDIT_COMPLETE.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
(root / "AUDIT_COMPLETE").write_text("complete\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
