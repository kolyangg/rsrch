#!/usr/bin/env bash
# One-item fail-closed replay through train.py's exact CL39 validation path.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="CL39_attention_audit_serv_trainer_r4_smoke1"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}"
PROJECT_ROOT="${TASK_ROOT}/source/diffusion_template"
RUN_NAME="CL39_attention_audit_serv_trainer_smoke"
CHECKPOINT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/CL39_cosmic_null_key_confidence_router_24k_full96_r4/diffusion_template/saved/CL39_cosmic_null_key_confidence_router_24k_full96_r4/checkpoint-epoch12.pth"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

cd "${TASK_ROOT}"
sha256sum -c source_manifest.sha256
sha256sum -c insightface_manifest.sha256

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

test -s "${CHECKPOINT}"
test -s "${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
test -s "${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"

export HOME="${TASK_ROOT}/home"
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export CL39_CHECKPOINT="${CHECKPOINT}"
export CL39_AUDIT_SAVE_DIR="${TASK_ROOT}/saved"
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
accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py --config-name=CL39_attention_audit_serv_smoke

python - <<'PY'
import json
from pathlib import Path

import numpy as np
from PIL import Image

root = Path("/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/CL39_attention_audit_serv_trainer_r4_smoke1")
generated = list((root / "saved/CL39_attention_audit_serv_trainer_smoke/val_images/manual_val").glob("step_*_batch_0/Rushing_ma_eddie.png"))
if len(generated) != 1:
    raise SystemExit(f"Expected one trainer-path output, found: {generated}")
sealed = Path(
    "/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/"
    "runtime_sources_cl38_cl45_v1/"
    "CL39_cosmic_null_key_confidence_router_24k_full96_r4/"
    "diffusion_template/saved/"
    "CL39_cosmic_null_key_confidence_router_24k_full96_r4/"
    "val_images/manual_val/step_24000_batch_0/Rushing_ma_eddie.png"
)
a = np.asarray(Image.open(generated[0]).convert("RGB"), dtype=np.float32) / 255.0
b = np.asarray(Image.open(sealed).convert("RGB"), dtype=np.float32) / 255.0
difference = np.abs(a - b)
gate = {
    "generated": str(generated[0]),
    "sealed": str(sealed),
    "rgb_mae": float(difference.mean()),
    "max_abs": float(difference.max()),
    "pixel_changed_gt_1_255": float((difference.max(axis=2) > 1.0 / 255.0).mean()),
}
(root / "sealed_replay_gate.json").write_text(
    json.dumps(gate, indent=2) + "\n", encoding="utf-8"
)
if gate["rgb_mae"] > 0.002:
    raise SystemExit(f"Trainer-path sealed replay gate failed: {gate}")
(root / "TRAINER_REPLAY_SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")
print(f"Trainer-path sealed replay gate passed: {gate}")
PY
