#!/usr/bin/env bash
# One of four NFS-claimed CL27/CL39 branch-audit shards.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/BA_lineage_branch_audit_serv_r1"
PROJECT_ROOT="${TASK_ROOT}/source/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

JOB_RUNTIME_ID="${HOSTNAME%-mpimaster-0}"
if [[ "${JOB_RUNTIME_ID}" == "${HOSTNAME}" ]]; then
  JOB_RUNTIME_ID="${HOSTNAME%-mpiworker-*}"
fi
if [[ "${JOB_RUNTIME_ID}" == "${HOSTNAME}" ]]; then
  echo "Cannot derive MLS job runtime ID from ${HOSTNAME}." >&2
  exit 69
fi
CLAIM_ROOT="${TASK_ROOT}/parallel_claims/${JOB_RUNTIME_ID}"
STATUS_ROOT="${TASK_ROOT}/parallel_status/${JOB_RUNTIME_ID}"
mkdir -p "${CLAIM_ROOT}" "${STATUS_ROOT}"
worker_slot=""
for candidate in 0 1 2 3; do
  if mkdir "${CLAIM_ROOT}/worker_${candidate}" 2>/dev/null; then
    worker_slot="${candidate}"
    printf '%s\n' "${HOSTNAME}" > "${CLAIM_ROOT}/worker_${candidate}/hostname.txt"
    break
  fi
done
if [[ -z "${worker_slot}" ]]; then
  echo "No parallel BA audit worker slot is available." >&2
  exit 70
fi
LOG_ROOT="${OWNER_ROOT}/logs/BA_lineage_branch_audit_serv_r1/${JOB_RUNTIME_ID}"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/worker_${worker_slot}.stdout.log") \
     2> >(tee -a "${LOG_ROOT}/worker_${worker_slot}.stderr.log" >&2)

cd "${TASK_ROOT}"
sha256sum -c source_manifest.sha256
sha256sum -c parallel_package_manifest.sha256

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
if [[ -n "${CONDA_BASE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
elif [[ -x "${CONDA_ENV}/bin/python" ]]; then
  export CONDA_PREFIX="${CONDA_ENV}"
  export PATH="${CONDA_ENV}/bin:${PATH}"
else
  echo "Could not locate the pinned photomaker_NS environment." >&2
  exit 70
fi

CL27_RUN="CL27_cosmic_frequency_surface_energy_24k_full96_r3"
CL39_RUN="CL39_cosmic_null_key_confidence_router_24k_full96_r4"
declare -A CHECKPOINTS=(
  [CL27]="${OWNER_ROOT}/runtime_sources_cl27_cl29_v4/${CL27_RUN}/diffusion_template/saved/${CL27_RUN}/checkpoint-epoch12.pth"
  [CL39]="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${CL39_RUN}/diffusion_template/saved/${CL39_RUN}/checkpoint-epoch12.pth"
)
declare -A CHECKPOINT_SHA256=(
  [CL27]="100072242ca34b2056f512f41a32e7aa8e7b98e4b10146043fd258a410ca8a50"
  [CL39]="74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07"
)
declare -A COMET_KEYS=(
  [CL27]="dbfbf40c3bdd4f70bedc58bda3dfb9cd"
  [CL39]="b1ca0b3da679401c85b991f1bbdf0b2a"
)
declare -A CONFIGS=(
  [CL27]="CL27_cosmic_frequency_surface_energy_24k"
  [CL39]="CL39_cosmic_null_key_confidence_router_24k"
)
declare -A SEALED_RUNS=(
  [CL27]="${OWNER_ROOT}/runtime_sources_cl27_cl29_v4/${CL27_RUN}/diffusion_template/saved/${CL27_RUN}"
  [CL39]="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${CL39_RUN}/diffusion_template/saved/${CL39_RUN}"
)

export HOME="${TASK_ROOT}/parallel_home/${JOB_RUNTIME_ID}/worker_${worker_slot}"
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export MPLCONFIGDIR="${HOME}/.config/matplotlib"
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
  tools/analysis/cl39_attention_capture.py

run_arm() {
  local lineage="$1" arm="$2" branch_mode="$3" confidence_override="$4"
  local run_name="BA_lineage_${lineage}_${arm}"
  local gate="${TASK_ROOT}/gates/${run_name}.json"
  local generated_root="${TASK_ROOT}/saved/${run_name}/val_images/manual_val"
  if [[ -s "${gate}" ]]; then
    echo "SKIP_COMPLETE ${run_name}"
    return
  fi
  printf '%s  %s\n' "${CHECKPOINT_SHA256[${lineage}]}" "${CHECKPOINTS[${lineage}]}" \
    | sha256sum -c -
  echo "START_PARALLEL_ARM worker=${worker_slot} lineage=${lineage} arm=${arm}"
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
    "lineage": lineage, "arm": arm, "branch_mode": branch_mode,
    "confidence_override": confidence_override, "run_name": run_name,
    "generated_count": len(generated), "checkpoint_sha256": checkpoint_sha256,
    "immutable_comet_key": comet_key, "processor_scope": "all_hardcase",
    "validation_step": 24000, "validation_panel": "manual_val fixed-96",
    "batch_size": 12, "parallel_worker": True,
}
if len(generated) != 96:
    raise SystemExit(f"Expected 96 outputs for {run_name}, found {len(generated)}")
if arm == "actual":
    maes, maxima, changed = [], [], []
    for path in generated:
        sealed = sealed_root / path.parent.name / path.name
        if not sealed.is_file():
            raise SystemExit(f"Missing sealed counterpart: {sealed}")
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
  echo "COMPLETE_PARALLEL_ARM ${run_name}"
}

wait_for_actual_gate() {
  local lineage="$1" gate="${TASK_ROOT}/gates/BA_lineage_${1}_actual.json"
  for _ in $(seq 1 720); do
    if [[ -s "${gate}" ]]; then
      python - "${gate}" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
if float(payload.get("sealed_rgb_mae_max", 1.0)) > 0.002:
    raise SystemExit(f"Actual replay gate is not valid: {payload}")
PY
      return
    fi
    sleep 10
  done
  echo "Timed out waiting for ${lineage} actual replay gate." >&2
  exit 78
}

case "${worker_slot}" in
  0)
    run_arm CL27 actual actual none
    run_arm CL27 native native none
    run_arm CL27 reference_face reference_face none
    ;;
  1)
    wait_for_actual_gate CL27
    run_arm CL27 low_only low_only none
    run_arm CL27 high_only high_only none
    ;;
  2)
    run_arm CL39 actual actual none
    run_arm CL39 native native none
    run_arm CL39 reference_face reference_face none
    ;;
  3)
    wait_for_actual_gate CL39
    run_arm CL39 low_only low_only none
    run_arm CL39 high_only high_only none
    run_arm CL39 confidence_one actual 1.0
    ;;
esac

printf 'complete\n' > "${STATUS_ROOT}/worker_${worker_slot}.complete"
echo "BA_PARALLEL_WORKER_COMPLETE worker=${worker_slot}"
