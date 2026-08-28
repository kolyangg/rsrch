#!/usr/bin/env bash
# Single-Comet validation-only replay of all CL39 checkpoints with BA at step 10.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_ba_start10_all_checkpoints_20260827_r1"
PROJECT_ROOT="${TASK_ROOT}/source/diffusion_template"
PACKAGE_ROOT="${TASK_ROOT}/package"
RUNTIME_ROOT="${TASK_ROOT}/runtime"
RUN_NAME="CL39_ba_start10_all_checkpoints_full96_r1"
CONFIG_NAME="CL39_ba_start10_all_checkpoints_full96"
SOURCE_MANIFEST="${TASK_ROOT}/source_manifest.json"
CHECKPOINT_MANIFEST="${PACKAGE_ROOT}/cl39_checkpoint_sha256.txt"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
CL39_PARENT_RUN="CL39_cosmic_null_key_confidence_router_24k_full96_r4"
CL39_CHECKPOINT_ROOT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${CL39_PARENT_RUN}/diffusion_template/saved/${CL39_PARENT_RUN}"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

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
  --root "${PROJECT_ROOT}" --manifest "${SOURCE_MANIFEST}"
(cd "${CL39_CHECKPOINT_ROOT}" && sha256sum -c "${CHECKPOINT_MANIFEST}")

DATASET_LINK="${TASK_ROOT}/source/dataset_full"
if [[ -e "${DATASET_LINK}" && ! -L "${DATASET_LINK}" ]]; then
  echo "Refusing to replace non-symlink dataset path: ${DATASET_LINK}" >&2
  exit 2
fi
ln -sfn "${OWNER_ROOT}/rsrch_test/dataset_full" "${DATASET_LINK}"

set -a
# shellcheck disable=SC1090
source "${OWNER_ROOT}/rsrch_test/diffusion_template/.env"
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export CL39_CHECKPOINT_ROOT
export CL39_BA10_ALL_SAVE_DIR="${TASK_ROOT}/saved"
export COMET_PROJECT=aug-large-ds
export COMET_EXPERIMENT_RECORD_PATH="${TASK_ROOT}/saved/${RUN_NAME}/comet_experiment.json"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_ALBUMENTATIONS_UPDATE=1

test -s "${PM_PATH}"
test -s "${SUBJECT_V2_ID_EMBEDS}"
mkdir -p "${MPLCONFIGDIR}" "${TASK_ROOT}/saved" "${TASK_ROOT}/gates"
cd "${PROJECT_ROOT}"
python -m py_compile src/pipelines/br_pipeline_helpers.py src/trainer/base_trainer.py
python "${PACKAGE_ROOT}/check_selector_contract.py"

set +e
accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py --config-name="${CONFIG_NAME}" \
  2>&1 | tee "${RUNTIME_ROOT}/validation.log" &
TRAIN_PID=$!
set -e

COMET_READY=0
for _ in $(seq 1 300); do
  if [[ -s "${COMET_EXPERIMENT_RECORD_PATH}" ]] && python - "${COMET_EXPERIMENT_RECORD_PATH}" <<'PY'
import json
import sys

key = (json.load(open(sys.argv[1], encoding="utf-8")).get("comet") or {}).get("experiment_key")
raise SystemExit(0 if isinstance(key, str) and len(key) == 32 else 1)
PY
  then
    COMET_READY=1
    echo "COMET_STARTUP_VERIFIED ${COMET_EXPERIMENT_RECORD_PATH}"
    break
  fi
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    wait "${TRAIN_PID}"
    exit $?
  fi
  sleep 2
done
if [[ "${COMET_READY}" -ne 1 ]]; then
  echo "Comet immutable key was not registered within 10 minutes." >&2
  kill "${TRAIN_PID}" 2>/dev/null || true
  wait "${TRAIN_PID}" || true
  exit 78
fi

set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  exit "${TRAIN_STATUS}"
fi

python - "${TASK_ROOT}" "${RUN_NAME}" "${SOURCE_MANIFEST}" "${CHECKPOINT_MANIFEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

task_root = Path(sys.argv[1])
run_name = sys.argv[2]
source_manifest = Path(sys.argv[3])
checkpoint_manifest = Path(sys.argv[4])
run_root = task_root / "saved" / run_name
image_root = run_root / "val_images" / "manual_val"
record = json.loads((run_root / "comet_experiment.json").read_text(encoding="utf-8"))
key = str((record.get("comet") or {}).get("experiment_key") or "")
if len(key) != 32:
    raise SystemExit("Missing immutable Comet key")

steps = list(range(0, 24001, 2000))
by_step = {}
reference_names = None
for step in steps:
    paths = sorted(image_root.glob(f"step_{step}_batch_*/*.png"))
    names = [path.name for path in paths]
    if len(paths) != 96 or len(set(names)) != 96:
        raise SystemExit(f"Step {step}: expected 96 unique PNGs, found {len(paths)}/{len(set(names))}")
    if reference_names is None:
        reference_names = set(names)
    elif set(names) != reference_names:
        raise SystemExit(f"Step {step}: filename panel differs from step 0")
    by_step[str(step)] = {"png_count": 96, "unique_filenames": 96}

log_text = (task_root / "runtime" / "validation.log").read_text(
    encoding="utf-8", errors="replace"
)
if log_text.count("[Switch] step 10 → BOTH") < len(steps):
    raise SystemExit("Missing BA-at-10 runtime proof for one or more checkpoints")

sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
gate = {
    "status": "accepted",
    "run_name": run_name,
    "validation_only": True,
    "single_comet_run": True,
    "comet_experiment_key": key,
    "immutable_parent_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
    "validation_steps": steps,
    "images_per_step": 96,
    "total_pngs": 96 * len(steps),
    "photomaker_start_step": 10,
    "branched_attn_start_step": 10,
    "branched_active_steps": 40,
    "num_inference_steps": 50,
    "pose_adapt_ratio": 0,
    "ca_mixing_for_face": False,
    "per_step": by_step,
    "source_manifest_sha256": sha256(source_manifest),
    "checkpoint_manifest_sha256": sha256(checkpoint_manifest),
}
gate_path = task_root / "gates" / f"{run_name}.json"
gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
print(json.dumps(gate, indent=2))
PY

echo "CL39_BA10_ALL_CHECKPOINTS_COMPLETE run=${RUN_NAME}"
