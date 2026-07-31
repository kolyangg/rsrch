#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
PACKAGE_ID="rhca_large_dataset_sameid_40k_full96_serv_r1_continue4k_deferred_val"
RUN_NAME="rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu"
COMET_KEY="db32f157e75a4798b2dfa530477c66d6"
START_EPOCH=8
FINAL_EPOCH=80
EPOCH_LEN=500

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
  echo "Serv continuation requires the test branch" >&2
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
export TRAIN_EPOCH_LEN="${EPOCH_LEN}"

if [[ "${PACKAGE_ID}" != "rhca_large_dataset_sameid_40k_full96_serv_r1_continue4k_deferred_val" ]]; then
  echo "Unexpected package ID: ${PACKAGE_ID}" >&2
  exit 72
fi
if [[ "${ACCELERATE_NUM_PROCESSES}" != "2" || "${CUDA_VISIBLE_DEVICES}" != "0,1" ]]; then
  echo "This package requires exactly two visible GPUs and two processes." >&2
  exit 73
fi
if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production continuation received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
  exit 74
fi

test -s "${LARGE_DATASET_MANIFEST}"
test -d "${LARGE_DATASET_IMAGES}"
test -s "${FULL96_BBOX_MANUAL}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${NVIDIA_LIB_ROOT}/cublas/lib/libcublasLt.so.12"

echo "fdf91ecff26272313a3ecbc4f2190d4e3beece571b579c83e71eec4ba639155d  src/trainer/base_trainer.py" |
  sha256sum --check --strict
echo "1987d8ff26f2bb43ffa9ea63b31a0ddcc36b19cc0d87023a47a04c833b53637b  src/logger/cometml.py" |
  sha256sum --check --strict
echo "0219250219046fa98e8a92d95d986c49ea8580006edee6f88402e9f059d1b46a  train.py" |
  sha256sum --check --strict
echo "bba2300285ae1f7960bdda027d312636c9fde87aba7432dbc36268f3c8af59e5  src/configs/trainer/photomaker_lora.yaml" |
  sha256sum --check --strict

RUN_DIR="saved/${RUN_NAME}"
START_CHECKPOINT="checkpoint-epoch${START_EPOCH}.pth"
START_CHECKPOINT_PATH="${RUN_DIR}/${START_CHECKPOINT}"

python - "${RUN_NAME}" "${COMET_KEY}" "${START_CHECKPOINT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

import torch

run_name, expected_key, checkpoint_path = sys.argv[1:]
run_dir = Path("saved") / run_name
record = json.loads((run_dir / "comet_experiment.json").read_text(encoding="utf-8"))
actual_key = (record.get("comet") or {}).get("experiment_key")
if actual_key != expected_key:
    raise SystemExit(f"Comet key mismatch: {actual_key!r}")
for step in (0, 2000, 4000):
    images = list(
        (run_dir / "val_images/manual_val").glob(f"step_{step}_batch_*/*.png")
    )
    if len(images) != 96:
        raise SystemExit(
            f"Expected 96 saved validation images at step {step}, found {len(images)}"
        )
    metrics = (
        run_dir
        / "face_quality/manual_val"
        / f"step_{step:08d}"
        / "face_quality_metrics.json"
    )
    if not metrics.is_file():
        raise SystemExit(f"Face-quality metrics missing at step {step}")
checkpoint_file = Path(checkpoint_path)
checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
if int(checkpoint.get("epoch", -1)) != 8:
    raise SystemExit(f"Expected epoch-8 checkpoint, found {checkpoint.get('epoch')}")
if not checkpoint.get("optimizer", {}).get("state"):
    raise SystemExit("Epoch-8 checkpoint has no optimizer state")
print(
    "CONTINUE4K_PREFLIGHT_OK "
    f"run={run_name} comet={actual_key} checkpoint_bytes={checkpoint_file.stat().st_size} "
    f"optimizer_states={len(checkpoint['optimizer']['state'])}"
)
PY

COMMON_OVERRIDES=(
  "pipeline.pose_adapt_ratio=0.0"
  "pipeline.ca_mixing_for_face=false"
  "trainer.skip_initial_validation=true"
  "dataloaders.train.num_workers=0"
  "++ddp_find_unused_parameters=true"
  "++serialize_distributed_model_init=true"
)

# 28 Jul 2026 - AICODE-NOTE: Full-96 generation is rank-0-only and its Comet
# asset stream blocked the rank-0 SDK before the next epoch while rank 1
# entered the loader. Train continuously to sealed 2k checkpoints without
# in-process validation; evaluate those checkpoints afterward in fresh
# single-process invocations. This preserves the optimizer trajectory, the
# exact checkpoints/steps, the full-96 contract, and the immutable Comet ID.
if [[ ! -s "${RUN_DIR}/checkpoint-epoch${FINAL_EPOCH}.pth" ]]; then
  echo "RECOVERY_PHASE train_4000_to_40000 writer=cometml validation=deferred"
  export WRITER=cometml
  export TRAIN_EPOCHS="${FINAL_EPOCH}"
  ACCELERATE_NUM_PROCESSES=2 CUDA_VISIBLE_DEVICES=0,1 \
    bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh \
      "${COMMON_OVERRIDES[@]}" \
      "continue_run=true" \
      "saved_checkpoint=${START_CHECKPOINT}" \
      "cometml_id=${COMET_KEY}" \
      "trainer.validation_interval_steps=0" \
      "trainer.face_quality.enabled=false" \
      "trainer.save_period=4" \
      "++weights_only_save_period=4"
fi

python - "${RUN_DIR}" "${FINAL_EPOCH}" <<'PY'
import sys
from pathlib import Path

import torch

run_dir = Path(sys.argv[1])
final_epoch = int(sys.argv[2])
missing = [
    epoch
    for epoch in range(12, final_epoch + 1, 4)
    if not (run_dir / f"checkpoint-epoch{epoch}.pth").is_file()
]
if missing:
    raise SystemExit(f"Missing deferred-validation checkpoints: {missing}")
checkpoint_path = run_dir / f"checkpoint-epoch{final_epoch}.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if int(checkpoint.get("epoch", -1)) != final_epoch:
    raise SystemExit(
        f"Final checkpoint epoch mismatch: {checkpoint.get('epoch')} != {final_epoch}"
    )
print(
    "TRAINING_40K_CHECKPOINTS_OK "
    f"gates={len(range(12, final_epoch + 1, 4))} "
    f"final_bytes={checkpoint_path.stat().st_size}"
)
PY

echo "RECOVERY_PHASE validate_6000_to_40000 writer=cometml mode=single_process"
export WRITER=cometml
export TRAIN_EPOCHS="${FINAL_EPOCH}"
for validation_epoch in $(seq 12 4 "${FINAL_EPOCH}"); do
  validation_step=$((validation_epoch * EPOCH_LEN))
  checkpoint_path="${PROJECT_ROOT}/${RUN_DIR}/checkpoint-epoch${validation_epoch}.pth"
  image_count="$(
    find "${RUN_DIR}/val_images/manual_val" \
      -mindepth 2 -maxdepth 2 \
      -path "*/step_${validation_step}_batch_*/*.png" 2>/dev/null |
      wc -l
  )"
  face_quality_path="${RUN_DIR}/face_quality/manual_val/step_$(printf '%08d' "${validation_step}")/face_quality_metrics.json"
  if [[ "${image_count}" -eq 96 && -s "${face_quality_path}" ]]; then
    echo "DEFERRED_VALIDATION_ALREADY_COMPLETE step=${validation_step}"
    continue
  fi
  if [[ "${image_count}" -ne 0 || -e "${face_quality_path}" ]]; then
    echo "Partial local validation state at step ${validation_step}; refusing duplicate Comet logging." >&2
    exit 76
  fi

  echo "DEFERRED_VALIDATION_START step=${validation_step} checkpoint=${checkpoint_path}"
  ACCELERATE_NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 \
    bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh \
      "pipeline.pose_adapt_ratio=0.0" \
      "pipeline.ca_mixing_for_face=false" \
      "cometml_id=${COMET_KEY}" \
      "++validation_only=true" \
      "++validation_epoch=${validation_epoch}" \
      "trainer.from_pretrained=${checkpoint_path}" \
      "trainer.validation_interval_steps=0" \
      "datasets.val.manual_val.limit=96" \
      "datasets.val.manual_val.bbox_mask_gen=${FULL96_BBOX_MANUAL}" \
      "dataloaders.manual_val.num_workers=0" \
      "++serialize_distributed_model_init=false"

  image_count="$(
    find "${RUN_DIR}/val_images/manual_val" \
      -mindepth 2 -maxdepth 2 \
      -path "*/step_${validation_step}_batch_*/*.png" |
      wc -l
  )"
  if [[ "${image_count}" -ne 96 || ! -s "${face_quality_path}" ]]; then
    echo "Deferred validation integrity check failed at step ${validation_step}." >&2
    exit 77
  fi
  echo "DEFERRED_VALIDATION_COMPLETE step=${validation_step} images=${image_count}"
done

echo "RECOVERY_COMPLETE run=${RUN_NAME} comet=${COMET_KEY} final_step=40000"
