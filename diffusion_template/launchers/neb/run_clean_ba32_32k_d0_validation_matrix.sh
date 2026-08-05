#!/usr/bin/env bash
set -euo pipefail

# D0 fixed-checkpoint diagnostic matrix from the 2 Aug architecture plan.
# This is evaluation-only: the clean BA32 weights, fixed manual_val inputs,
# PhotoMaker tokens, prompts, seeds, scheduler, and metrics stay unchanged.
PROJECT_ROOT="${PROJECT_ROOT:-/home/niko/rsrch/diffusion_template}"
CONDA_ROOT="${CONDA_ROOT:-/home/niko/miniconda3}"
PYIQA_VENV="${PYIQA_VENV:-/home/niko/rsrch/metric_envs/pyiqa-0.1.15}"
SOURCE_RUN="${SOURCE_RUN:-rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1}"
SOURCE_COMET_KEY="${SOURCE_COMET_KEY:-700240d8f90b48cfa2cc16f8ff2886b6}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/saved/${SOURCE_RUN}/weights-epoch16.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/d0_clean_ba32_32k}"
RUN_FACE_QUALITY="${RUN_FACE_QUALITY:-true}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-false}"
COMET_PROJECT="${COMET_PROJECT:-jul-comet-large-testing-tr}"
RESUME_D0="${RESUME_D0:-false}"

if [[ "${RUN_FACE_QUALITY}" != "true" && "${RUN_FACE_QUALITY}" != "false" ]]; then
  echo "RUN_FACE_QUALITY must be true or false" >&2
  exit 2
fi
if [[ "${ALLOW_BUSY_GPU}" != "true" && "${ALLOW_BUSY_GPU}" != "false" ]]; then
  echo "ALLOW_BUSY_GPU must be true or false" >&2
  exit 2
fi
if [[ "${RESUME_D0}" != "true" && "${RESUME_D0}" != "false" ]]; then
  echo "RESUME_D0 must be true or false" >&2
  exit 2
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate photomaker_NS
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/launchers/lib/prepare_comet_record.sh"
set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export PM_PATH="/home/niko/models/PhotoMaker-V2/photomaker-v2.bin"
NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

[[ "${CONDA_PREFIX:-}" == "${CONDA_ROOT}/envs/photomaker_NS" ]]
[[ -s "${CHECKPOINT}" ]]
[[ -s "${PM_PATH}" ]]
[[ -f "experiments/big_celebs/d0_clean_ba32_32k_validation_matrix.json" ]]
[[ "$(git rev-parse --abbrev-ref HEAD)" == "test" ]]
if [[ -e "${OUTPUT_ROOT}" && "${RESUME_D0}" != "true" ]]; then
  echo "Refusing to reuse D0 output root: ${OUTPUT_ROOT}" >&2
  exit 3
fi

mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)
if (( ${#gpu_pids[@]} > 0 )) && [[ "${ALLOW_BUSY_GPU}" != "true" ]]; then
  echo "Neb GPU already has compute PIDs: ${gpu_pids[*]}" >&2
  echo "Wait for the active job to exit; do not overlap full-96 validation." >&2
  exit 4
fi

mkdir -p "${OUTPUT_ROOT}/logs"
sha256sum "${CHECKPOINT}" | tee "${OUTPUT_ROOT}/checkpoint.sha256"
printf '%s\n' "${SOURCE_RUN}" > "${OUTPUT_ROOT}/source_run.txt"
printf '%s\n' "${SOURCE_COMET_KEY}" > "${OUTPUT_ROOT}/source_comet_key.txt"

run_arm() {
  local label="$1"
  local processor_base_mode="$2"
  local spatial_condition="$3"
  local arm_dir="${OUTPUT_ROOT}/${label}"
  local log_path="${OUTPUT_ROOT}/logs/${label}.log"
  local spec_path="${PROJECT_ROOT}/experiments/big_celebs/${label}.json"
  local comet_record="${PROJECT_ROOT}/saved/${label}/comet_experiment.json"

  echo "D0_ARM_START label=${label} base=${processor_base_mode} spatial=${spatial_condition}"
  local comet_key
  if [[ -s "${comet_record}" && "${RESUME_D0}" == "true" ]]; then
    comet_key="$(python3 - "${comet_record}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["comet"]["experiment_key"])
PY
)"
    echo "D0_COMET_RESUME label=${label} key=${comet_key}" | tee -a "${log_path}"
  else
    prepare_comet_record "${PROJECT_ROOT}" "${label}" "${spec_path}"
    python tools/comet/log_fixed_checkpoint_validation.py \
      --spec "${spec_path}" \
      --run-name "${label}" \
      --project-name "${COMET_PROJECT}" \
      --step 32000 \
      --initialize-only 2>&1 | tee "${log_path}"
    [[ -s "${comet_record}" ]]
    comet_key="$(python3 - "${comet_record}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["comet"]["experiment_key"])
PY
)"
  fi
  [[ "${#comet_key}" -eq 32 ]]
  echo "D0_COMET_READY label=${label} key=${comet_key}" | tee -a "${log_path}"

  python tools/inference/evaluate_rhca_checkpoint.py \
    --config big_celebs_scheduled_rhca_clean_ba32_40k \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-step 32000 \
    --output-dir "${arm_dir}" \
    --validation-dataset manual_val \
    --limit 96 \
    --batch-size 12 \
    --guidance-scale 5 \
    --photomaker-path "${PM_PATH}" \
    --disable-branched-ca \
    --processor-base-mode "${processor_base_mode}" \
    --reference-condition matched \
    --spatial-reference-condition "${spatial_condition}" \
    --device cuda 2>&1 | tee -a "${log_path}"

  [[ "$(find "${arm_dir}/images" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 96 ]]
  [[ -s "${arm_dir}/run_manifest.json" ]]
  [[ -s "${arm_dir}/per_image.json" ]]

  if [[ "${RUN_FACE_QUALITY}" == "true" ]]; then
    [[ -x "${PYIQA_VENV}/bin/python" ]]
    "${PYIQA_VENV}/bin/python" tools/inference/calculate_face_quality_metrics.py \
      --manifest "${arm_dir}/face_quality_input_manifest.json" \
      --output-json "${arm_dir}/face_quality_metrics.json" \
      --output-csv "${arm_dir}/face_quality_per_image.csv" \
      --metrics topiq_nr-face,topiq_nr,musiq,maniqa-pipal \
      --device cuda \
      --batch-size 8 \
      --crop-padding 0.25 \
      --crop-size 512 2>&1 | tee -a "${log_path}"
  else
    echo "Comet publication requires face-quality outputs; RUN_FACE_QUALITY=false is local-only." >&2
    exit 5
  fi

  python tools/comet/log_fixed_checkpoint_validation.py \
    --spec "${spec_path}" \
    --run-name "${label}" \
    --project-name "${COMET_PROJECT}" \
    --run-id "${comet_key}" \
    --step 32000 \
    --arm-dir "${arm_dir}" 2>&1 | tee -a "${log_path}"
  echo "D0_ARM_COMPLETE label=${label}"
}

run_arm "d0_clean_ba32_32k_legacy_matched" "legacy_full_copy" "matched"
run_arm "d0_clean_ba32_32k_native_matched" "validation_native" "matched"
run_arm "d0_clean_ba32_32k_native_zero_spatial" "validation_native" "zero"
run_arm "d0_clean_ba32_32k_native_shuffle_spatial" "validation_native" "shuffle"

echo "D0_MATRIX_COMPLETE output_root=${OUTPUT_ROOT}"
