#!/usr/bin/env bash
# Five-way replay-gated recovery for the CL23 branch-lineage audit arms.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/BA_lineage_branch_audit_serv_r1"
PROJECT_ROOT="${TASK_ROOT}/source_cl23_historical/diffusion_template"
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
CLAIM_ROOT="${TASK_ROOT}/cl23_recovery_claims/${JOB_RUNTIME_ID}"
STATUS_ROOT="${TASK_ROOT}/cl23_recovery_status/${JOB_RUNTIME_ID}"
mkdir -p "${CLAIM_ROOT}" "${STATUS_ROOT}"
worker_slot=""
for candidate in 0 1 2 3 4; do
  if mkdir "${CLAIM_ROOT}/worker_${candidate}" 2>/dev/null; then
    worker_slot="${candidate}"
    printf '%s\n' "${HOSTNAME}" > "${CLAIM_ROOT}/worker_${candidate}/hostname.txt"
    break
  fi
done
if [[ -z "${worker_slot}" ]]; then
  echo "No CL23 recovery worker slot is available." >&2
  exit 70
fi
LOG_ROOT="${OWNER_ROOT}/logs/BA_lineage_branch_audit_serv_r1/${JOB_RUNTIME_ID}"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/cl23_worker_${worker_slot}.stdout.log") \
     2> >(tee -a "${LOG_ROOT}/cl23_worker_${worker_slot}.stderr.log" >&2)

cd "${TASK_ROOT}"
sha256sum -c cl23_historical_source_manifest.sha256
sha256sum -c cl23_recovery_package_manifest.sha256

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

CL23_RUN="CL23_cosmic_temporal_frequency_router_24k_full96_r1"
CL23_CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl21_cl26_v1/${CL23_RUN}/diffusion_template/saved/${CL23_RUN}/checkpoint-epoch12.pth"
CL23_CHECKPOINT_SHA256="70201f0e82c9cb24aeb5adc27ad660e5e11aea8d29b6969c449ec39e3c8b379c"
CL23_COMET_KEY="a9ec9c59d1624c68acb98737dcd65298"
CL23_CONFIG="CL23_cosmic_temporal_frequency_router_24k"
CL23_SEALED_RUN="${OWNER_ROOT}/runtime_sources_cl21_cl26_v1/${CL23_RUN}/diffusion_template/saved/${CL23_RUN}"

export HOME="${TASK_ROOT}/cl23_recovery_home/${JOB_RUNTIME_ID}/worker_${worker_slot}"
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
  src/model/photomaker_branched/lora2.py \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/trainer/sdxl_trainers.py \
  tools/analysis/render_ba_lineage_branch_audit.py

run_arm() {
  local arm="$1" branch_mode="$2"
  local run_name="BA_lineage_CL23_${arm}"
  local gate="${TASK_ROOT}/gates/${run_name}.json"
  local generated_root="${TASK_ROOT}/saved/${run_name}/val_images/manual_val"
  if [[ -s "${gate}" ]]; then
    echo "SKIP_COMPLETE ${run_name}"
    return
  fi
  printf '%s  %s\n' "${CL23_CHECKPOINT_SHA256}" "${CL23_CHECKPOINT}" | sha256sum -c -
  echo "START_CL23_RECOVERY_ARM worker=${worker_slot} arm=${arm} branch=${branch_mode}"
  accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
    train.py \
    --config-name="${CL23_CONFIG}" \
    +validation_only=true \
    +validation_epoch=12 \
    'val_datasets_names=[manual_val]' \
    'inference_metrics=[]' \
    writer=console \
    "writer.run_name=${run_name}" \
    "trainer.from_pretrained=${CL23_CHECKPOINT}" \
    "trainer.save_dir=${TASK_ROOT}/saved" \
    trainer.face_quality.enabled=false \
    trainer.log_per_image_id_sim_table=false \
    +validation_args.cl39_analysis_enabled=true \
    +validation_args.cl39_analysis_capture=false \
    +validation_args.cl39_analysis_processor_scope=all_hardcase \
    "+validation_args.cl39_analysis_branch_mode=${branch_mode}" \
    validation_debug_timing=true

  python - "${arm}" "${branch_mode}" "${run_name}" "${generated_root}" \
    "${CL23_SEALED_RUN}" "${gate}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

arm, branch_mode, run_name, generated_root, sealed_run, gate_path = sys.argv[1:]
generated_root = Path(generated_root)
sealed_root = Path(sealed_run) / "val_images" / "manual_val"
generated = sorted(generated_root.glob("step_24000_batch_*/*.png"))
payload = {
    "lineage": "CL23",
    "arm": arm,
    "branch_mode": branch_mode,
    "confidence_override": "none",
    "run_name": run_name,
    "generated_count": len(generated),
    "checkpoint_sha256": "70201f0e82c9cb24aeb5adc27ad660e5e11aea8d29b6969c449ec39e3c8b379c",
    "immutable_comet_key": "a9ec9c59d1624c68acb98737dcd65298",
    "processor_scope": "all_hardcase",
    "validation_step": 24000,
    "validation_panel": "manual_val fixed-96",
    "batch_size": 12,
    "cl23_recovery_worker": True,
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
  echo "COMPLETE_CL23_RECOVERY_ARM ${run_name}"
}

wait_for_gate() {
  local gate="$1"
  for _ in $(seq 1 720); do
    if [[ -s "${gate}" ]]; then
      return
    fi
    sleep 10
  done
  echo "Timed out waiting for ${gate}." >&2
  exit 78
}

wait_for_actual_gate() {
  local gate="${TASK_ROOT}/gates/BA_lineage_CL23_actual.json"
  wait_for_gate "${gate}"
  python - "${gate}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if float(payload.get("sealed_rgb_mae_max", 1.0)) > 0.002:
    raise SystemExit(f"Actual replay gate is not valid: {payload}")
PY
}

case "${worker_slot}" in
  0)
    run_arm actual actual
    for arm in native reference_face low_only high_only; do
      wait_for_gate "${TASK_ROOT}/gates/BA_lineage_CL23_${arm}.json"
    done
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
(root / "AUDIT_COMPLETE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
(root / "AUDIT_COMPLETE").write_text("complete\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
    ;;
  1)
    wait_for_actual_gate
    run_arm native native
    ;;
  2)
    wait_for_actual_gate
    run_arm reference_face reference_face
    ;;
  3)
    wait_for_actual_gate
    run_arm low_only low_only
    ;;
  4)
    wait_for_actual_gate
    run_arm high_only high_only
    ;;
esac

printf 'complete\n' > "${STATUS_ROOT}/worker_${worker_slot}.complete"
echo "BA_CL23_RECOVERY_WORKER_COMPLETE worker=${worker_slot}"
