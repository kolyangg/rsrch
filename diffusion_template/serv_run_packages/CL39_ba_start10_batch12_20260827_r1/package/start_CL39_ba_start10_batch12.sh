#!/usr/bin/env bash
# CL39 validation-only replay with PM and BA sharing denoising step 10.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_ba_start10_batch12_20260827_r1"
SOURCE_DIR="${CL39_SOURCE_DIR:-source_r2}"
CONFIG_NAME="${CL39_CONFIG_NAME:-CL39_ba_start10_batch12}"
RUN_NAME="${CL39_RUN_NAME:-CL39_24k_batch12_ba_start10_r2}"
EXPECTED_COUNT="${CL39_EXPECTED_COUNT:-12}"
MANIFEST_NAME="${CL39_MANIFEST_NAME:-source_r2_manifest.json}"
RUNTIME_DIR="${CL39_RUNTIME_DIR:-runtime}"
PROJECT_ROOT="${TASK_ROOT}/${SOURCE_DIR}/diffusion_template"
PACKAGE_ROOT="${TASK_ROOT}/package"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
CL39_RUN="CL39_cosmic_null_key_confidence_router_24k_full96_r4"
CL39_SEALED_ROOT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${CL39_RUN}/diffusion_template/saved/${CL39_RUN}"
CL39_CHECKPOINT="${CL39_SEALED_ROOT}/checkpoint-epoch12.pth"
CL39_CHECKPOINT_SHA256="74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07"
CL39_COMET_KEY="b1ca0b3da679401c85b991f1bbdf0b2a"
BASELINE_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_attribution_controls_20260826_r1/saved/CL39_24k_cross_A_correct_pm_correct_spatial_r1/val_images/manual_val"
GENERATED_ROOT="${TASK_ROOT}/saved/${RUN_NAME}/val_images/manual_val"
RUNTIME_ROOT="${TASK_ROOT}/${RUNTIME_DIR}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  for candidate in /home/jovyan/miniconda3 /home/jovyan/anaconda3 /opt/conda; do
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

python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" verify \
  --root "${PROJECT_ROOT}" --manifest "${TASK_ROOT}/${MANIFEST_NAME}"
printf '%s  %s\n' "${CL39_CHECKPOINT_SHA256}" "${CL39_CHECKPOINT}" | sha256sum -c -

DATASET_LINK="${TASK_ROOT}/${SOURCE_DIR}/dataset_full"
if [[ -e "${DATASET_LINK}" && ! -L "${DATASET_LINK}" ]]; then
  echo "Refusing to replace non-symlink dataset path: ${DATASET_LINK}" >&2
  exit 2
fi
ln -sfn "${OWNER_ROOT}/rsrch_test/dataset_full" "${DATASET_LINK}"

export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export CL39_CHECKPOINT
export CL39_BA10_SAVE_DIR="${TASK_ROOT}/saved"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_ALBUMENTATIONS_UPDATE=1

test -s "${PM_PATH}"
test -s "${BASELINE_ROOT}/step_24000_batch_0/Angry_man__eddie.png"
mkdir -p "${MPLCONFIGDIR}" "${TASK_ROOT}/saved" "${TASK_ROOT}/gates"
cd "${PROJECT_ROOT}"
python -m py_compile src/pipelines/br_pipeline_helpers.py
if [[ -f src/datasets/manual_val_subset.py ]]; then
  python -m py_compile src/datasets/manual_val_subset.py
fi
python "${PACKAGE_ROOT}/check_selector_contract.py"

accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py --config-name="${CONFIG_NAME}" 2>&1 | tee "${RUNTIME_ROOT}/validation.log"

python - "${GENERATED_ROOT}" "${BASELINE_ROOT}" "${TASK_ROOT}/gates/${RUN_NAME}.json" \
  "${TASK_ROOT}/${MANIFEST_NAME}" "${RUNTIME_ROOT}/validation.log" \
  "${EXPECTED_COUNT}" "${RUN_NAME}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

generated_root, baseline_root, gate_path, manifest_path, log_path = map(Path, sys.argv[1:6])
expected_count = int(sys.argv[6])
run_name = sys.argv[7]
generated = sorted(generated_root.glob("step_24000_batch_*/*.png"))
baseline_all = sorted(baseline_root.glob("step_24000_batch_*/*.png"))
if len(generated) != expected_count:
    raise SystemExit(f"Expected {expected_count} generated outputs, found {len(generated)}")
baseline_by_name = {path.name: path for path in baseline_all}
if len(baseline_by_name) != len(baseline_all):
    raise SystemExit("Baseline PNG filenames are not unique")
if len({path.name for path in generated}) != len(generated):
    raise SystemExit("Generated PNG filenames are not unique")
missing = sorted(path.name for path in generated if path.name not in baseline_by_name)
if missing:
    raise SystemExit(f"Generated/baseline filename join is incomplete: {missing}")
baseline = [baseline_by_name[path.name] for path in generated]

maes = []
for new_path, old_path in zip(generated, baseline):
    new = np.asarray(Image.open(new_path).convert("RGB"), dtype=np.float32) / 255.0
    old = np.asarray(Image.open(old_path).convert("RGB"), dtype=np.float32) / 255.0
    maes.append(float(np.abs(new - old).mean()))
log_text = log_path.read_text(encoding="utf-8", errors="replace")
if "[Switch] step 10 → BOTH" not in log_text:
    raise SystemExit("Missing proof that denoising step 10 switched to BOTH")
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
payload = {
    "run_name": run_name,
    "validation_only": True,
    "validation_step": 24000,
    "validation_indices": list(range(expected_count)),
    "generated_count": len(generated),
    "baseline_count": len(baseline),
    "filename_join": "exact after sorted filename; space/underscore normalization reserved for metric lookup",
    "photomaker_start_step": 10,
    "branched_attn_start_step": 10,
    "branched_active_steps": 40,
    "num_inference_steps": 50,
    "mode_switch_proof": "[Switch] step 10 → BOTH",
    "paired_rgb_mae_mean": float(np.mean(maes)),
    "paired_rgb_mae_min": float(np.min(maes)),
    "paired_rgb_mae_max": float(np.max(maes)),
    "checkpoint_sha256": "74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07",
    "immutable_parent_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
    "source_manifest_sha256": manifest_sha,
    "baseline": "accepted CL39 24k crossing A; original PM@10, BA@15",
}
gate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "CL39_BA_START10_COMPLETE run=${RUN_NAME} count=${EXPECTED_COUNT}"
