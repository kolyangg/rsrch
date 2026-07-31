#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
REMOTE_REPO="${OWNER_ROOT}/rsrch_test"
PROJECT_ROOT="${REMOTE_REPO}/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
RUN_NAME="rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu"
COMET_KEY="db32f157e75a4798b2dfa530477c66d6"
EPOCH_LEN=500
FINAL_EPOCH=80
ARM="${VALIDATION_ARM:?Set VALIDATION_ARM to 0 or 1}"

if [[ "${ARM}" != "0" && "${ARM}" != "1" ]]; then
  echo "VALIDATION_ARM must be 0 or 1; got ${ARM}" >&2
  exit 64
fi

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
cd "${REMOTE_REPO}"
if [[ "${CONDA_PREFIX:-}" != "${CONDA_ENV}" ]]; then
  echo "Wrong Conda environment: ${CONDA_PREFIX:-unset}" >&2
  exit 70
fi
if [[ "$(git branch --show-current)" != "test" ]]; then
  echo "Live validation sidecar requires the test branch" >&2
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
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME
export COMET_PROJECT="jul-comet-large-testing-tr"
export CONFIG_NAME="large_dataset_rhca_40k"
export TRAIN_EPOCH_LEN="${EPOCH_LEN}"
export TRAIN_EPOCHS="${FINAL_EPOCH}"
export WRITER=cometml

if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production validation received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
  exit 72
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

MAIN_RUN_DIR="saved/${RUN_NAME}"
SIDECAR_SAVE_DIR="saved/live_validation_sidecars/arm${ARM}"
SIDECAR_RUN_DIR="${SIDECAR_SAVE_DIR}/${RUN_NAME}"
mkdir -p \
  "${MAIN_RUN_DIR}/val_images/manual_val" \
  "${SIDECAR_RUN_DIR}/val_images/manual_val"

python - "${MAIN_RUN_DIR}/comet_experiment.json" "${COMET_KEY}" <<'PY'
import json
import sys
from pathlib import Path

record_path, expected_key = sys.argv[1:]
record = json.loads(Path(record_path).read_text(encoding="utf-8"))
actual_key = (record.get("comet") or {}).get("experiment_key")
if actual_key != expected_key:
    raise SystemExit(f"Comet key mismatch: {actual_key!r}")
print(f"LIVE_VALIDATION_COMET_OK key={actual_key}")
PY

if [[ "${ARM}" == "0" ]]; then
  VALIDATION_EPOCHS=(12 20 28 36 44 52 60 68 76)
else
  VALIDATION_EPOCHS=(16 24 32 40 48 56 64 72 80)
fi

publish_step_outputs() {
  local validation_step="$1"
  local side_images="${SIDECAR_RUN_DIR}/val_images/manual_val"
  local main_images="${MAIN_RUN_DIR}/val_images/manual_val"
  local side_quality="${SIDECAR_RUN_DIR}/face_quality/manual_val/step_$(printf '%08d' "${validation_step}")"
  local main_quality="${MAIN_RUN_DIR}/face_quality/manual_val/step_$(printf '%08d' "${validation_step}")"
  local batch_dir

  mkdir -p "${main_images}" "$(dirname "${main_quality}")"
  if [[ -e "${main_quality}" ]]; then
    echo "Main face-quality destination already exists: ${main_quality}" >&2
    exit 75
  fi
  for batch_dir in "${side_images}"/step_"${validation_step}"_batch_*; do
    if [[ ! -d "${batch_dir}" ]]; then
      echo "Missing sidecar validation batch for step ${validation_step}" >&2
      exit 76
    fi
    if [[ -e "${main_images}/$(basename "${batch_dir}")" ]]; then
      echo "Main validation batch already exists: $(basename "${batch_dir}")" >&2
      exit 77
    fi
  done
  for batch_dir in "${side_images}"/step_"${validation_step}"_batch_*; do
    mv "${batch_dir}" "${main_images}/"
  done
  mv "${side_quality}" "${main_quality}"
}

for validation_epoch in "${VALIDATION_EPOCHS[@]}"; do
  validation_step=$((validation_epoch * EPOCH_LEN))
  checkpoint_path="${PROJECT_ROOT}/${MAIN_RUN_DIR}/checkpoint-epoch${validation_epoch}.pth"
  main_quality="${MAIN_RUN_DIR}/face_quality/manual_val/step_$(printf '%08d' "${validation_step}")/face_quality_metrics.json"
  main_image_count="$(
    find "${MAIN_RUN_DIR}/val_images/manual_val" \
      -mindepth 2 -maxdepth 2 \
      -path "*/step_${validation_step}_batch_*/*.png" 2>/dev/null |
      wc -l
  )"
  if [[ "${main_image_count}" -eq 96 && -s "${main_quality}" ]]; then
    echo "LIVE_VALIDATION_ALREADY_COMPLETE arm=${ARM} step=${validation_step}"
    continue
  fi
  if [[ "${main_image_count}" -ne 0 || -e "${main_quality}" ]]; then
    echo "Partial main validation state at step ${validation_step}; refusing duplicate Comet logging." >&2
    exit 78
  fi

  while [[ ! -s "${checkpoint_path}" ]]; do
    echo "LIVE_VALIDATION_WAIT arm=${ARM} step=${validation_step} checkpoint=${checkpoint_path}"
    sleep 30
  done

  side_quality="${SIDECAR_RUN_DIR}/face_quality/manual_val/step_$(printf '%08d' "${validation_step}")/face_quality_metrics.json"
  side_image_count="$(
    find "${SIDECAR_RUN_DIR}/val_images/manual_val" \
      -mindepth 2 -maxdepth 2 \
      -path "*/step_${validation_step}_batch_*/*.png" 2>/dev/null |
      wc -l
  )"
  if [[ "${side_image_count}" -ne 0 || -e "${side_quality}" ]]; then
    echo "Partial sidecar state at step ${validation_step}; refusing duplicate Comet logging." >&2
    exit 79
  fi

  echo "LIVE_VALIDATION_START arm=${ARM} step=${validation_step} checkpoint=${checkpoint_path}"
  bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh \
    "pipeline.pose_adapt_ratio=0.0" \
    "pipeline.ca_mixing_for_face=false" \
    "cometml_id=${COMET_KEY}" \
    "++validation_only=true" \
    "++validation_epoch=${validation_epoch}" \
    "trainer.from_pretrained=${checkpoint_path}" \
    "trainer.save_dir=${SIDECAR_SAVE_DIR}" \
    "trainer.validation_interval_steps=0" \
    "trainer.face_quality.device=cuda" \
    "datasets.val.manual_val.limit=96" \
    "datasets.val.manual_val.bbox_mask_gen=${FULL96_BBOX_MANUAL}" \
    "dataloaders.manual_val.num_workers=0" \
    "automatic_bboxes_every_val=false" \
    "force_log_first_auto_bbox=false" \
    "++serialize_distributed_model_init=false"

  side_image_count="$(
    find "${SIDECAR_RUN_DIR}/val_images/manual_val" \
      -mindepth 2 -maxdepth 2 \
      -path "*/step_${validation_step}_batch_*/*.png" |
      wc -l
  )"
  if [[ "${side_image_count}" -ne 96 || ! -s "${side_quality}" ]]; then
    echo "Sidecar validation integrity check failed at step ${validation_step}." >&2
    exit 80
  fi

  publish_step_outputs "${validation_step}"
  main_image_count="$(
    find "${MAIN_RUN_DIR}/val_images/manual_val" \
      -mindepth 2 -maxdepth 2 \
      -path "*/step_${validation_step}_batch_*/*.png" |
      wc -l
  )"
  if [[ "${main_image_count}" -ne 96 || ! -s "${main_quality}" ]]; then
    echo "Published validation integrity check failed at step ${validation_step}." >&2
    exit 81
  fi
  echo "LIVE_VALIDATION_COMPLETE arm=${ARM} step=${validation_step} images=${main_image_count}"
done

echo "LIVE_VALIDATION_ARM_COMPLETE arm=${ARM} comet=${COMET_KEY}"
