#!/usr/bin/env bash
set -euo pipefail

# 09 Aug 2026 - Replays the original full96 Eddie batch under each immutable
# training runtime and requires exact pixels before changing the global
# PhotoMaker identity vector. The corrected arm is never run after a failed gate.
TASK_OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_PACKAGE="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/eddie_revalidation_contract_v2_serv_20260809_r1"
TASK_ROOT="${TASK_OWNER_ROOT}/analysis_sidecars/eddie_revalidation_contract_v2_serv_20260809_r1"
TASK_INPUT="${TASK_ROOT}/inputs"
TASK_OUTPUT="${TASK_ROOT}/outputs"
TASK_ENV="${TASK_OWNER_ROOT}/conda_env/photomaker_NS"
TASK_ORT="${TASK_OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
TASK_PYIQA="${TASK_OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
TASK_NVIDIA="${TASK_ENV}/lib/python3.10/site-packages/nvidia"
TASK_LIBSTDCXX="${TASK_OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34"
TASK_ARCHIVE="${TASK_PACKAGE}/eddie_contract_v2_serv_inputs_20260809.tar.gz"
TASK_ARCHIVE_SHA="e3d98a7b1f21385a4d4ab859bbb6d96d6ce88aeab3b75147216962f829c4571e"
TASK_WRAPPER="${TASK_PACKAGE}/run_corrected_eddie_sidecar.py"
TASK_EVALUATOR="${TASK_PACKAGE}/evaluate_rhca_checkpoint.py"
TASK_VERIFY="${TASK_PACKAGE}/verify_eddie_historical_replay.py"

TASK_E13_PROJECT="${TASK_OWNER_ROOT}/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template"
TASK_BC_PROJECT="${TASK_OWNER_ROOT}/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template"
TASK_CL11_PROJECT="${TASK_OWNER_ROOT}/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template"
TASK_E13_CHECKPOINT="${TASK_E13_PROJECT}/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch12.pth"
TASK_BC_CHECKPOINT="${TASK_BC_PROJECT}/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch12.pth"
TASK_CL11_CHECKPOINT="${TASK_CL11_PROJECT}/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch10.pth"
TASK_BBOX_MANUAL="${TASK_OWNER_ROOT}/datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"

if command -v conda >/dev/null 2>&1; then
  TASK_CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  TASK_CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  for TASK_CONDA_CANDIDATE in "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
    if [[ -f "${TASK_CONDA_CANDIDATE}/etc/profile.d/conda.sh" ]]; then
      TASK_CONDA_BASE="${TASK_CONDA_CANDIDATE}"
      break
    fi
  done
fi
: "${TASK_CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${TASK_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${TASK_ENV}"

export CUDA_VISIBLE_DEVICES=0
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TASK_OWNER_ROOT}/metric_cache/torch"
export PM_PATH="${TASK_OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export FULL96_BBOX_MANUAL="${TASK_BBOX_MANUAL}"
export LD_LIBRARY_PATH="${TASK_NVIDIA}/cublas/lib:${TASK_NVIDIA}/cuda_cupti/lib:${TASK_NVIDIA}/cuda_nvrtc/lib:${TASK_NVIDIA}/cuda_runtime/lib:${TASK_NVIDIA}/cudnn/lib:${TASK_NVIDIA}/cufft/lib:${TASK_NVIDIA}/curand/lib:${TASK_NVIDIA}/cusolver/lib:${TASK_NVIDIA}/cusparse/lib:${TASK_NVIDIA}/nccl/lib:${TASK_NVIDIA}/nvjitlink/lib:${TASK_NVIDIA}/nvtx/lib"
export LD_LIBRARY_PATH="$(dirname "${TASK_LIBSTDCXX}"):${LD_LIBRARY_PATH}"
export LD_PRELOAD="${TASK_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}"

test -s "${TASK_ARCHIVE}"
test -s "${TASK_WRAPPER}"
test -s "${TASK_EVALUATOR}"
test -s "${TASK_VERIFY}"
test -s "${TASK_BBOX_MANUAL}"
test -s "${PM_PATH}"
test -s "${TASK_E13_CHECKPOINT}"
test -s "${TASK_BC_CHECKPOINT}"
test -s "${TASK_CL11_CHECKPOINT}"
[[ "$(sha256sum "${TASK_ARCHIVE}" | cut -d' ' -f1)" == "${TASK_ARCHIVE_SHA}" ]]
[[ "$(sha256sum "${TASK_E13_CHECKPOINT}" | cut -d' ' -f1)" == "4a9d95a3f957609fcf4eb77771f263dec8e71189dc72aae347233091de4249ab" ]]
[[ "$(sha256sum "${TASK_BC_CHECKPOINT}" | cut -d' ' -f1)" == "99b305bad425dd07073a4a54e0a978dea0d4a02456c8129eb1b12afbbf5a459e" ]]
[[ "$(sha256sum "${TASK_CL11_CHECKPOINT}" | cut -d' ' -f1)" == "e65972c8c14b5031f879e1ee8b1e11a707823e0cfccdb80553219fc8069dbb83" ]]

if [[ -e "${TASK_ROOT}/COMPLETE" || -e "${TASK_ROOT}/RUNNING" ]]; then
  echo "Refusing to reuse an active or completed task root: ${TASK_ROOT}" >&2
  exit 76
fi
if [[ -d "${TASK_OUTPUT}" ]] && find "${TASK_OUTPUT}" -type f -print -quit | grep -q .; then
  echo "Refusing to overwrite partial outputs: ${TASK_OUTPUT}" >&2
  exit 77
fi
mkdir -p "${TASK_INPUT}" "${TASK_OUTPUT}"
tar --extract --gzip --file="${TASK_ARCHIVE}" --directory="${TASK_INPUT}"
[[ "$(find "${TASK_INPUT}/historical" -type f -name '*_eddie.png' | wc -l)" -eq 36 ]]
touch "${TASK_ROOT}/RUNNING"

TASK_STAGE="startup"
finish() {
  local task_rc=$?
  printf '%s\n' "${TASK_STAGE}" > "${TASK_ROOT}/LAST_STAGE"
  printf '%s\n' "${task_rc}" > "${TASK_ROOT}/LAST_EXIT_CODE"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${TASK_ROOT}/LAST_FINISHED_AT"
  if [[ "${task_rc}" -eq 0 ]]; then
    mv "${TASK_ROOT}/RUNNING" "${TASK_ROOT}/COMPLETE"
  else
    mv "${TASK_ROOT}/RUNNING" "${TASK_ROOT}/FAILED"
  fi
  exit "${task_rc}"
}
trap finish EXIT

sha256sum \
  "${TASK_ARCHIVE}" \
  "${TASK_WRAPPER}" \
  "${TASK_EVALUATOR}" \
  "${TASK_VERIFY}" \
  "${TASK_INPUT}/eddie_foreground_arcface_embedding.npy" \
  "${TASK_BBOX_MANUAL}" \
  "${PM_PATH}" \
  "${TASK_E13_CHECKPOINT}" \
  "${TASK_BC_CHECKPOINT}" \
  "${TASK_CL11_CHECKPOINT}" \
  > "${TASK_ROOT}/input_sha256.txt"

run_pair() {
  local task_name="$1"
  local task_project="$2"
  local task_config="$3"
  local task_checkpoint="$4"
  local task_step="$5"
  local task_historical="$6"
  local task_pair_root="${TASK_OUTPUT}/${task_name}"

  test -s "${task_project}/src/configs/${task_config}.yaml"
  if [[ -f "${task_project}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${task_project}/.env"
    set +a
  fi

  TASK_STAGE="${task_name}:historical_replay"
  printf '%s\n' "${TASK_STAGE}" | tee "${TASK_ROOT}/CURRENT_STAGE"
  cd "${task_project}"
  PYTHONPATH="${task_project}:${TASK_PYIQA}:${TASK_ORT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TASK_ENV}/bin/python" "${TASK_WRAPPER}" \
      --project-root "${task_project}" \
      --evaluator-path "${TASK_EVALUATOR}" \
      --embedding "${TASK_INPUT}/eddie_foreground_arcface_embedding.npy" \
      --config "${task_config}" \
      --checkpoint "${task_checkpoint}" \
      --checkpoint-step "${task_step}" \
      --identity-condition historical \
      --output-dir "${task_pair_root}/historical_replay"

  TASK_STAGE="${task_name}:exact_pixel_gate"
  printf '%s\n' "${TASK_STAGE}" | tee "${TASK_ROOT}/CURRENT_STAGE"
  "${TASK_ENV}/bin/python" "${TASK_VERIFY}" \
    --replay-dir "${task_pair_root}/historical_replay" \
    --historical-dir "${task_historical}" \
    --output "${task_pair_root}/historical_replay_verification.json"

  TASK_STAGE="${task_name}:corrected_global_identity_condition"
  printf '%s\n' "${TASK_STAGE}" | tee "${TASK_ROOT}/CURRENT_STAGE"
  cd "${task_project}"
  PYTHONPATH="${task_project}:${TASK_PYIQA}:${TASK_ORT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TASK_ENV}/bin/python" "${TASK_WRAPPER}" \
      --project-root "${task_project}" \
      --evaluator-path "${TASK_EVALUATOR}" \
      --embedding "${TASK_INPUT}/eddie_foreground_arcface_embedding.npy" \
      --config "${task_config}" \
      --checkpoint "${task_checkpoint}" \
      --checkpoint-step "${task_step}" \
      --identity-condition corrected \
      --output-dir "${task_pair_root}/corrected_eddie"
}

run_pair \
  E13_24000 \
  "${TASK_E13_PROJECT}" \
  E13_large_ds_joint_shadow_sa128_24k \
  "${TASK_E13_CHECKPOINT}" \
  24000 \
  "${TASK_INPUT}/historical/E13_24000"

run_pair \
  BC_E13_24000 \
  "${TASK_BC_PROJECT}" \
  BC_E13_big_celebs_joint_shadow_sa128_24k \
  "${TASK_BC_CHECKPOINT}" \
  24000 \
  "${TASK_INPUT}/historical/BC_E13_24000"

run_pair \
  CL11_20000 \
  "${TASK_CL11_PROJECT}" \
  CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k \
  "${TASK_CL11_CHECKPOINT}" \
  20000 \
  "${TASK_INPUT}/historical/CL11_20000"

TASK_STAGE="final_hashes"
find "${TASK_OUTPUT}" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${TASK_ROOT}/output_sha256.txt"
printf '%s\n' complete > "${TASK_ROOT}/CURRENT_STAGE"
