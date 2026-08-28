#!/usr/bin/env bash
# Evaluation-only A100 replay of the CL39 24k attention/confidence audit.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="CL39_attention_audit_serv_a100_r1"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}"
PROJECT_ROOT="${TASK_ROOT}/source/diffusion_template"
CHECKPOINT_DIR="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/CL39_cosmic_null_key_confidence_router_24k_full96_r4/diffusion_template/saved/CL39_cosmic_null_key_confidence_router_24k_full96_r4"
OUTPUT_ROOT="${TASK_ROOT}/artifacts/cl39_attention_24k_serv_a100"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

test -d "${PROJECT_ROOT}"
test -s "${TASK_ROOT}/source_manifest.sha256"
cd "${TASK_ROOT}"
sha256sum -c source_manifest.sha256

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

test -s "${CHECKPOINT_DIR}/checkpoint-epoch12.pth"
test -s "${CHECKPOINT_DIR}/config.yaml"
test -s "${CHECKPOINT_DIR}/comet_experiment.json"
test -s "${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
test -s "${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"

export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PROJECT_ROOT}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

# 25 Aug 2026 - A100 replay keeps the sealed DDIM50/CFG5/seed0 contract and
# runs fully resident on GPU, matching the original Serv validation runtime.
python tools/analysis/analyze_cl39_attention.py generate \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --offload none \
  --force

python - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_jobs/CL39_attention_audit_serv_a100_r1/artifacts/cl39_attention_24k_serv_a100")
status = json.loads((root / "generation_status.json").read_text())
expected = [1, 7, 13, 17, 33, 35, 38, 40, 51, 55, 63, 69, 78, 80, 87, 93]
if status.get("failed") is not None or status.get("completed") != expected:
    raise SystemExit(f"Incomplete CL39 Serv audit: {status}")
(root / "SERV_A100_COMPLETE").write_text("complete\n", encoding="utf-8")
print(f"CL39 Serv A100 audit complete: {len(expected)} samples x 3 arms")
PY
