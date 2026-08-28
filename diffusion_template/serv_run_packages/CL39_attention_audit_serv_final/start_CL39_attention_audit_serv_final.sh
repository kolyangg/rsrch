#!/usr/bin/env bash
# Exact CL39 trainer-path audit arm on one Serv A100.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
ARM="${CL39_AUDIT_ARM:?Set CL39_AUDIT_ARM to actual, c1, or ba_off}"
case "${ARM}" in
  actual|c1|ba_off) ;;
  *) echo "Unsupported CL39_AUDIT_ARM=${ARM}" >&2; exit 2 ;;
esac

TASK_NAME="CL39_attention_audit_serv_final_${ARM}"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}"
PROJECT_ROOT="${TASK_ROOT}/source/diffusion_template"
CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/CL39_cosmic_null_key_confidence_router_24k_full96_r4/diffusion_template/saved/CL39_cosmic_null_key_confidence_router_24k_full96_r4/checkpoint-epoch12.pth"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

cd "${TASK_ROOT}"
sha256sum -c audit_manifest.sha256
sha256sum -c insightface_manifest.sha256
printf '%s  %s\n' \
  '74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07' \
  "${CHECKPOINT}" | sha256sum -c -

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

test -s "${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
test -s "${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"

export HOME="${TASK_ROOT}/home"
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export CL39_CHECKPOINT="${CHECKPOINT}"
export CL39_AUDIT_SAVE_DIR="${TASK_ROOT}/saved"
export CL39_TELEMETRY_DIR="${TASK_ROOT}/telemetry"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_ALBUMENTATIONS_UPDATE=1

cd "${PROJECT_ROOT}"
python -m py_compile \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/trainer/sdxl_trainers.py \
  tools/analysis/cl39_attention_capture.py
accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py --config-name="CL39_attention_audit_serv_${ARM}"

python - "${ARM}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

arm = sys.argv[1]
owner = Path("/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev")
root = owner / "analysis_jobs" / f"CL39_attention_audit_serv_final_{arm}"
generated_root = (
    root / "saved" / f"CL39_attention_audit_serv_{arm}" /
    "val_images" / "manual_val"
)
generated = sorted(generated_root.glob("step_24000_batch_*/*.png"))
gate = {
    "arm": arm,
    "generated_count": len(generated),
    "generated_root": str(generated_root),
}
if len(generated) != 96:
    raise SystemExit(f"Expected 96 {arm} outputs, found {len(generated)}")

if arm == "actual":
    sealed_root = (
        owner / "runtime_sources_cl38_cl45_v1" /
        "CL39_cosmic_null_key_confidence_router_24k_full96_r4" /
        "diffusion_template" / "saved" /
        "CL39_cosmic_null_key_confidence_router_24k_full96_r4" /
        "val_images" / "manual_val"
    )
    maes = []
    maxima = []
    changed = []
    for path in generated:
        sealed = sealed_root / path.parent.name / path.name
        if not sealed.is_file():
            raise SystemExit(f"Missing sealed counterpart: {sealed}")
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        b = np.asarray(Image.open(sealed).convert("RGB"), dtype=np.float32) / 255.0
        difference = np.abs(a - b)
        maes.append(float(difference.mean()))
        maxima.append(float(difference.max()))
        changed.append(float((difference.max(axis=2) > 1.0 / 255.0).mean()))
    telemetry_npz = sorted((root / "telemetry").glob("*.npz"))
    telemetry_csv = sorted((root / "telemetry").glob("*.csv"))
    gate.update(
        sealed_root=str(sealed_root),
        rgb_mae_mean=float(np.mean(maes)),
        rgb_mae_max=float(np.max(maes)),
        max_abs=float(np.max(maxima)),
        pixel_changed_gt_1_255_mean=float(np.mean(changed)),
        telemetry_npz_count=len(telemetry_npz),
        telemetry_csv_count=len(telemetry_csv),
    )
    if gate["rgb_mae_max"] > 0.002:
        raise SystemExit(f"Full sealed replay gate failed: {gate}")
    if len(telemetry_npz) != 16 or len(telemetry_csv) != 16:
        raise SystemExit(f"Expected telemetry for 16 selected samples: {gate}")

(root / "audit_gate.json").write_text(
    json.dumps(gate, indent=2) + "\n", encoding="utf-8"
)
(root / "AUDIT_ARM_COMPLETE").write_text("complete\n", encoding="utf-8")
print(f"CL39 audit arm complete: {gate}")
PY
