#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
REMOTE_REPO="${OWNER_ROOT}/runtime_worktrees/rsrch_test_E7_E10_20260804"
PROJECT_ROOT="${REMOTE_REPO}/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
SOURCE_RUN_NAME="E10_large_ds_pmdefault_effective_r64_20k_full96_r1"
SOURCE_RUN_DIR="${PROJECT_ROOT}/saved/${SOURCE_RUN_NAME}"
SOURCE_COMET_KEY="0375f172f75c482f840317ec5ae41c05"
SIDECAR_NAME="E10V_large_ds_dynamicmask_reval_2k20k_full96_r1"
STAGING_ROOT="${PROJECT_ROOT}/saved/${SIDECAR_NAME}"
FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"

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
  echo "E10 validation replacement requires the test branch" >&2
  exit 71
fi
cd "${PROJECT_ROOT}"

if [[ ! -f .env ]]; then
  echo "Missing machine-local diffusion_template/.env" >&2
  exit 72
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-${OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export LARGE_DATASET_MANIFEST="${OWNER_ROOT}/datasets/dataset_full/filtered_ids3_adj.json"
export LARGE_DATASET_IMAGES="${OWNER_ROOT}/datasets/dataset_full/large_dataset_adj/large_dataset"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export CONFIG_NAME="E10_large_ds_pmdefault_effective_20k"
export COMET_PROJECT="aug-large-ds"
export WRITER="console"
export TRAIN_EPOCH_LEN=2000
export TRAIN_EPOCHS=10

if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production validation received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
  exit 73
fi
test -s "${PM_PATH}"
test -s "${LARGE_DATASET_MANIFEST}"
test -d "${LARGE_DATASET_IMAGES}"
test -s "${FULL96_BBOX_MANUAL}"
test -s "${SOURCE_RUN_DIR}/comet_experiment.json"
test -s tools/comet/replace_checkpoint_validation.py
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${NVIDIA_LIB_ROOT}/cublas/lib/libcublasLt.so.12"

python - "${SOURCE_RUN_DIR}" "${SOURCE_COMET_KEY}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_key = sys.argv[2]
record = json.loads((run_dir / "comet_experiment.json").read_text(encoding="utf-8"))
actual_key = str((record.get("comet") or {}).get("experiment_key") or "")
if actual_key != expected_key:
    raise SystemExit(f"E10 Comet key mismatch: {actual_key!r}")
checkpoints = sorted(run_dir.glob("checkpoint-epoch*.pth"))
if len(checkpoints) != 10:
    raise SystemExit(f"Expected ten E10 checkpoints, found {len(checkpoints)}")
for epoch in range(1, 11):
    path = run_dir / f"checkpoint-epoch{epoch}.pth"
    if not path.is_file() or path.stat().st_size < 500_000_000:
        raise SystemExit(f"Missing or truncated checkpoint: {path}")
print(f"E10V_SOURCE_OK key={actual_key} checkpoints={len(checkpoints)}")
PY

if [[ "${E10V_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "E10V_PREFLIGHT_ONLY_COMPLETE"
  exit 0
fi

mkdir -p "${STAGING_ROOT}"
for epoch in $(seq 1 10); do
  step=$((epoch * 2000))
  tag="$(printf '%06d' "${step}")"
  step_root="${STAGING_ROOT}/step_${tag}"
  step_run_name="${SIDECAR_NAME}_step_${tag}"
  step_run_dir="${step_root}/${step_run_name}"
  table_path="${step_run_dir}/validation_tables/id_sim__manual_val__step_${tag}.csv"
  quality_root="${step_run_dir}/face_quality/manual_val/step_$(printf '%08d' "${step}")"
  quality_path="${quality_root}/face_quality_metrics.json"
  quality_csv="${quality_root}/face_quality_per_image.csv"
  quality_manifest="${quality_root}/input_manifest.json"
  bbox_path="${step_root}/bbox_manual_auto.json"
  image_count="$({ find "${step_run_dir}/val_images/manual_val" -mindepth 2 -maxdepth 2 -type f -name '*.png' 2>/dev/null || true; } | wc -l)"

  if [[ "${image_count}" -eq 96 && -s "${table_path}" && -s "${bbox_path}" && ! -s "${quality_path}" ]]; then
    quality_input_count="$({ find "${quality_root}/inputs" -maxdepth 1 -type f -name '*.png' 2>/dev/null || true; } | wc -l)"
    if [[ ! -s "${quality_manifest}" || "${quality_input_count}" -ne 96 ]]; then
      echo "Incomplete reusable face-quality staging at step ${step}" >&2
      exit 74
    fi
    echo "E10V_FACE_QUALITY_RECOVERY_START step=${step} device=cpu inputs=${quality_input_count}"
    "${FACE_QUALITY_SCORER_PYTHON}" tools/inference/calculate_face_quality_metrics.py \
      --manifest "${quality_manifest}" \
      --output-json "${quality_path}" \
      --output-csv "${quality_csv}" \
      --metrics "topiq_nr-face,topiq_nr,musiq,maniqa-pipal" \
      --device cpu \
      --batch-size 8 \
      --crop-padding 0.25 \
      --crop-size 512
    test -s "${quality_path}"
    test -s "${quality_csv}"
    echo "E10V_FACE_QUALITY_RECOVERY_COMPLETE step=${step} device=cpu"
  fi

  if [[ "${image_count}" -eq 96 && -s "${table_path}" && -s "${quality_path}" && -s "${bbox_path}" ]]; then
    echo "E10V_STEP_ALREADY_STAGED step=${step} images=${image_count}"
    continue
  fi
  if [[ -e "${step_root}" ]]; then
    echo "Partial E10V staging exists; refusing to overwrite: ${step_root}" >&2
    exit 74
  fi

  mkdir -p "${step_root}"
  cp -- "${FULL96_BBOX_MANUAL}" "${step_root}/bbox_manual.json"
  export RUN_NAME="${step_run_name}"
  checkpoint_path="${SOURCE_RUN_DIR}/checkpoint-epoch${epoch}.pth"
  save_dir="saved/${SIDECAR_NAME}/step_${tag}"
  echo "E10V_VALIDATION_START step=${step} checkpoint=${checkpoint_path}"
  bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh \
    "pipeline.pose_adapt_ratio=0.0" \
    "pipeline.ca_mixing_for_face=false" \
    "disable_branched_ca=true" \
    "model.ba_enforce_reference_only_hard_route=true" \
    "++validation_only=true" \
    "++validation_epoch=${epoch}" \
    "trainer.from_pretrained=${checkpoint_path}" \
    "trainer.save_dir=${save_dir}" \
    "trainer.face_quality.enabled=true" \
    "trainer.face_quality.device=cpu" \
    "trainer.face_quality.expected_images=96" \
    "datasets.val.manual_val.limit=96" \
    "datasets.val.manual_val.bbox_mask_gen=${step_root}/bbox_manual.json" \
    "dataloaders.manual_val.num_workers=0" \
    "automatic_bboxes=true" \
    "automatic_bboxes_every_val=true" \
    "force_log_first_auto_bbox=false" \
    "validation_args.use_bbox_mask_gen=true" \
    "validation_args.val_debug=false" \
    "val_debug=false" \
    "++serialize_distributed_model_init=false" \
    2>&1 | tee "${step_root}/validation.log"

  image_count="$(find "${step_run_dir}/val_images/manual_val" -mindepth 2 -maxdepth 2 -type f -name '*.png' | wc -l)"
  if [[ "${image_count}" -ne 96 || ! -s "${table_path}" || ! -s "${quality_path}" || ! -s "${bbox_path}" ]]; then
    echo "E10V staged-output integrity failed at step ${step}" >&2
    exit 75
  fi
  echo "E10V_VALIDATION_COMPLETE step=${step} images=${image_count}"
done

python tools/comet/replace_checkpoint_validation.py \
  --experiment-key "${SOURCE_COMET_KEY}" \
  --expected-project "aug-large-ds" \
  --expected-run-name "${SOURCE_RUN_NAME}" \
  --sidecar-name "${SIDECAR_NAME}" \
  --staging-root "${STAGING_ROOT}" \
  --steps "2000,4000,6000,8000,10000,12000,14000,16000,18000,20000" \
  --images-per-step 96 \
  --write

echo "E10V_JOB_COMPLETE comet=${SOURCE_COMET_KEY} replacement_steps=2000-20000"
