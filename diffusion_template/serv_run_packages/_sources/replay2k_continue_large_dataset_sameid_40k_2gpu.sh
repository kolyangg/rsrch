#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="@@REMOTE_OWNER_ROOT@@"
PROJECT_ROOT="@@REMOTE_PROJECT@@"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
PACKAGE_ID="@@RUN_ID@@"
RUN_NAME="rhca_large_dataset_sameid_40k_full96_serv_r1_2gpu"
COMET_KEY="db32f157e75a4798b2dfa530477c66d6"
CHECKPOINT_NAME="checkpoint-epoch4.pth"
CORRUPT_SHA256="7908840b038d7f6c9cd50d100b30b2e6747a187325f4cfe3f4a2ddd4128a4ef4"

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
cd "@@REMOTE_REPO@@"
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
export CUDA_VISIBLE_DEVICES="@@CUDA_VISIBLE_DEVICES@@"
export ACCELERATE_NUM_PROCESSES="@@NUM_PROCESSES@@"
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME
export COMET_PROJECT="jul-comet-large-testing-tr"
export CONFIG_NAME="large_dataset_rhca_40k"
export TRAIN_EPOCH_LEN="500"

if [[ "${PACKAGE_ID}" != "rhca_large_dataset_sameid_40k_full96_serv_r1_replay2k_continue" ]]; then
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

echo "fdf91ecff26272313a3ecbc4f2190d4e3beece571b579c83e71eec4ba639155d  src/trainer/base_trainer.py" |
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
for step in (0, 2000):
    images = list(
        (run_dir / "val_images/manual_val").glob(
            f"step_{step}_batch_*/*.png"
        )
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
print(
    "REPLAY_CONTINUE_PREFLIGHT_OK "
    f"run={run_name} comet={actual_key} validation_steps=0,2000"
)
PY

RUN_DIR="saved/${RUN_NAME}"
CHECKPOINT_PATH="${RUN_DIR}/${CHECKPOINT_NAME}"
CORRUPT_ARCHIVE="${RUN_DIR}/${CHECKPOINT_NAME}.corrupt_ncc_timeout_${CORRUPT_SHA256:0:12}"
if [[ -f "${CHECKPOINT_PATH}" ]]; then
  echo "${CORRUPT_SHA256}  ${CHECKPOINT_PATH}" | sha256sum --check --strict
  if [[ -e "${CORRUPT_ARCHIVE}" ]]; then
    echo "Corrupt-checkpoint archive already exists: ${CORRUPT_ARCHIVE}" >&2
    exit 75
  fi
  mv "${CHECKPOINT_PATH}" "${CORRUPT_ARCHIVE}"
elif [[ ! -f "${CORRUPT_ARCHIVE}" ]]; then
  echo "Neither the known corrupt checkpoint nor its archive exists." >&2
  exit 76
fi

COMMON_OVERRIDES=(
  "pipeline.pose_adapt_ratio=0.0"
  "pipeline.ca_mixing_for_face=false"
  "trainer.skip_initial_validation=true"
  "dataloaders.train.num_workers=0"
  "++ddp_find_unused_parameters=true"
  "++serialize_distributed_model_init=true"
)

# The watchdog interrupted the first epoch-4 checkpoint write, so exact
# optimizer continuation requires deterministic replay. Reuse the original
# Comet initialization path because it precedes dataloader construction, but
# suppress metric/image events and do not repeat completed validation.
echo "RECOVERY_PHASE replay_0_to_2000 writer=cometml events=suppressed validation=disabled"
export WRITER=cometml
export TRAIN_EPOCHS=4
bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh \
  "${COMMON_OVERRIDES[@]}" \
  "+writer.suppress_events=true" \
  "cometml_id=${COMET_KEY}" \
  "trainer.validation_interval_steps=0"

python - "${CHECKPOINT_PATH}" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
required = {"epoch", "state_dict", "optimizer", "lr_scheduler", "config"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"Replayed checkpoint is missing: {sorted(missing)}")
if int(checkpoint["epoch"]) != 4:
    raise SystemExit(f"Expected epoch 4, found {checkpoint['epoch']}")
if not checkpoint["optimizer"].get("state"):
    raise SystemExit("Replayed checkpoint has no optimizer state")
print(
    "REPLAY_CHECKPOINT_OK "
    f"path={path} epoch={checkpoint['epoch']} "
    f"optimizer_states={len(checkpoint['optimizer']['state'])}"
)
PY

# Resume online logging at step 2,000 in the original immutable Comet run.
echo "RECOVERY_PHASE continue_2000_to_40000 writer=cometml validation=full96_every2000"
export WRITER=cometml
export TRAIN_EPOCHS=80
exec bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh \
  "${COMMON_OVERRIDES[@]}" \
  "continue_run=true" \
  "saved_checkpoint=${CHECKPOINT_NAME}" \
  "cometml_id=${COMET_KEY}"
