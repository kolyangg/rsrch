#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
PACKAGE_ID="rhca_large_dataset_sameid_40k_full96_serv_r1_recover"
RUN_NAME="rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu"
COMET_KEY="db32f157e75a4798b2dfa530477c66d6"

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
cd "/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test"
if [[ "${CONDA_PREFIX:-}" != "${CONDA_ENV}" ]]; then
  echo "Wrong Conda environment: ${CONDA_PREFIX:-unset}" >&2
  exit 70
fi
if [[ "$(git branch --show-current)" != "test" ]]; then
  echo "Serv recovery requires the test branch" >&2
  exit 71
fi
cd "${PROJECT_ROOT}"

set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-${OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export LARGE_DATASET_MANIFEST="${OWNER_ROOT}/datasets/dataset_full/filtered_ids3_adj.json"
export LARGE_DATASET_IMAGES="${OWNER_ROOT}/datasets/dataset_full/large_dataset_adj/large_dataset"
export FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export CUDA_VISIBLE_DEVICES="0,1"
export ACCELERATE_NUM_PROCESSES="2"
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME
export COMET_PROJECT="jul-comet-large-testing-tr"
export CONFIG_NAME="large_dataset_rhca_40k"
export TRAIN_EPOCHS="80"

if [[ "${PACKAGE_ID}" != "rhca_large_dataset_sameid_40k_full96_serv_r1_recover" ]]; then
  echo "Unexpected package ID: ${PACKAGE_ID}" >&2
  exit 72
fi
if [[ "${ACCELERATE_NUM_PROCESSES}" != "2" || "${CUDA_VISIBLE_DEVICES}" != "0,1" ]]; then
  echo "This package requires exactly two visible GPUs and two processes." >&2
  exit 73
fi
if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production recovery received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
  exit 74
fi

test -s "${LARGE_DATASET_MANIFEST}"
test -d "${LARGE_DATASET_IMAGES}"
test -s "${FULL96_BBOX_MANUAL}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${NVIDIA_LIB_ROOT}/cublas/lib/libcublasLt.so.12"

echo "e90a9023da2a16dd3d1dae54b57c7f2c03e719ca6af177a04be67ea5f8b8e1a2  src/trainer/base_trainer.py" |
  sha256sum --check --strict
echo "395b7f77881eee609dd6a00264c7c26c09ef7e9deb4e8bc21a806e0d5fadb3b8  src/configs/trainer/photomaker_lora.yaml" |
  sha256sum --check --strict

python - "${RUN_NAME}" "${COMET_KEY}" <<'PY'
import json
import sys
from pathlib import Path

run_name, expected_key = sys.argv[1:]
run_dir = Path("saved") / run_name
record = json.loads((run_dir / "comet_experiment.json").read_text(encoding="utf-8"))
actual_key = (record.get("comet") or {}).get("experiment_key")
if actual_key != expected_key:
    raise SystemExit(f"Comet key mismatch: {actual_key!r}")

metrics = run_dir / "face_quality/manual_val/step_00000000/face_quality_metrics.json"
per_image = run_dir / "face_quality/manual_val/step_00000000/face_quality_per_image.csv"
if not metrics.is_file() or not per_image.is_file():
    raise SystemExit("Completed step-0 face-quality artifacts are missing")
images = list((run_dir / "val_images/manual_val").glob("step_0_batch_*/*.png"))
if len(images) != 96:
    raise SystemExit(f"Expected 96 saved step-0 images, found {len(images)}")
if list(run_dir.glob("checkpoint-epoch*.pth")):
    raise SystemExit("Recovery assumes zero optimizer updates, but a checkpoint exists")
print(
    "RECOVERY_PREFLIGHT_OK "
    f"run={run_name} comet={actual_key} step0_images={len(images)} checkpoints=0"
)
PY

# Rank-0-only validation occurred before worker creation in the failed job.
# The recovery reuses that completed step 0, avoids post-CUDA worker forks,
# and lets DDP account for any rank-local conditional BA parameters.
exec bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh \
  "pipeline.pose_adapt_ratio=0.0" \
  "pipeline.ca_mixing_for_face=false" \
  "trainer.skip_initial_validation=true" \
  "dataloaders.train.num_workers=0" \
  "++ddp_find_unused_parameters=true" \
  "cometml_id=${COMET_KEY}"
