#!/usr/bin/env bash
set -euo pipefail

TASK_OWNER_ROOT="@@REMOTE_OWNER_ROOT@@"
TASK_PACKAGE="@@REMOTE_RUN_DIR@@"
TASK_ROOT="${TASK_OWNER_ROOT}/analysis_sidecars/@@RUN_ID@@"
TASK_OUTPUT="${TASK_ROOT}/outputs"
TASK_ENV="${TASK_OWNER_ROOT}/conda_env/photomaker_NS"
TASK_ORT="${TASK_OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
TASK_PYIQA="${TASK_OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
TASK_NVIDIA="${TASK_ENV}/lib/python3.10/site-packages/nvidia"
TASK_LIBSTDCXX="${TASK_OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34"
TASK_PROJECT="${TASK_OWNER_ROOT}/runtime_sources_cl1_cl3_v1/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1/diffusion_template"
TASK_RUNTIME_ROOT="$(dirname "${TASK_PROJECT}")"
TASK_SAVED="${TASK_PROJECT}/saved/CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1"
TASK_CHECKPOINT="${TASK_SAVED}/weights-epoch12.pth"
TASK_CONFIG="CL9_cosmic_joint_shadow_sa128_refscale_24k"
TASK_HISTORICAL="${TASK_SAVED}/post_training_face_quality/download_manifest.json"
TASK_BASELINE="${TASK_OWNER_ROOT}/analysis_sidecars/cl9v_smallface_roi_20260810_r1/outputs/full96_historical_replay"
TASK_LEGACY_EMBEDS="${TASK_RUNTIME_ROOT}/dataset_full/val_dataset/id_embeds_manual_val.pth"
TASK_SUBJECT_EMBEDS="${TASK_OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
TASK_BBOX="${TASK_PACKAGE}/pm96_bboxes_new_auto_cl9.json"
TASK_EVALUATOR="${TASK_PACKAGE}/evaluate_rhca_checkpoint.py"
TASK_WRAPPER="${TASK_PACKAGE}/run_cl9_fixed_checkpoint_sidecar.py"
TASK_VERIFY="${TASK_PACKAGE}/verify_rgb_replay.py"
TASK_SCORE="${TASK_PACKAGE}/score_cl9_subject_v2.py"
TASK_ROI="${TASK_PACKAGE}/run_smallface_roi_refine.py"
TASK_FQ="${TASK_PROJECT}/tools/inference/calculate_face_quality_metrics.py"

if command -v conda >/dev/null 2>&1; then
  TASK_CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  TASK_CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  for TASK_CANDIDATE in "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
    if [[ -f "${TASK_CANDIDATE}/etc/profile.d/conda.sh" ]]; then
      TASK_CONDA_BASE="${TASK_CANDIDATE}"
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
export FULL96_BBOX_MANUAL="${TASK_OWNER_ROOT}/datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export LD_LIBRARY_PATH="${TASK_NVIDIA}/cublas/lib:${TASK_NVIDIA}/cuda_cupti/lib:${TASK_NVIDIA}/cuda_nvrtc/lib:${TASK_NVIDIA}/cuda_runtime/lib:${TASK_NVIDIA}/cudnn/lib:${TASK_NVIDIA}/cufft/lib:${TASK_NVIDIA}/curand/lib:${TASK_NVIDIA}/cusolver/lib:${TASK_NVIDIA}/cusparse/lib:${TASK_NVIDIA}/nccl/lib:${TASK_NVIDIA}/nvjitlink/lib:${TASK_NVIDIA}/nvtx/lib"
export LD_LIBRARY_PATH="$(dirname "${TASK_LIBSTDCXX}"):${LD_LIBRARY_PATH}"
export LD_PRELOAD="${TASK_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${TASK_PROJECT}:${TASK_PYIQA}:${TASK_ORT}${PYTHONPATH:+:${PYTHONPATH}}"

test -s "${TASK_CHECKPOINT}"
test -s "${TASK_HISTORICAL}"
test -s "${TASK_BASELINE}/run_manifest.json"
test -s "${TASK_BASELINE}/per_image.json"
test -s "${TASK_LEGACY_EMBEDS}"
test -s "${TASK_SUBJECT_EMBEDS}"
test -s "${TASK_BBOX}"
test -s "${TASK_EVALUATOR}"
test -s "${TASK_WRAPPER}"
test -s "${TASK_VERIFY}"
test -s "${TASK_SCORE}"
test -s "${TASK_ROI}"
test -s "${TASK_FQ}"
[[ "$(sha256sum "${TASK_CHECKPOINT}" | cut -d' ' -f1)" == "5396993b16ace89908501bfddb2e412e755a3f6478a6449c502062d6ca7357c3" ]]
[[ "$(sha256sum "${TASK_LEGACY_EMBEDS}" | cut -d' ' -f1)" == "23ae97075e967f2bcb790c5094ef350b316249c7023df67a68f735bfebb747c6" ]]
[[ "$(sha256sum "${TASK_SUBJECT_EMBEDS}" | cut -d' ' -f1)" == "e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465" ]]
[[ "$(sha256sum "${TASK_BBOX}" | cut -d' ' -f1)" == "b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d" ]]

if [[ -e "${TASK_ROOT}/COMPLETE" || -e "${TASK_ROOT}/RUNNING" ]]; then
  echo "Refusing to reuse active/completed root: ${TASK_ROOT}" >&2
  exit 76
fi
if [[ -d "${TASK_OUTPUT}" ]] && find "${TASK_OUTPUT}" -type f -print -quit | grep -q .; then
  echo "Refusing to overwrite partial outputs: ${TASK_OUTPUT}" >&2
  exit 77
fi
mkdir -p "${TASK_OUTPUT}"
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

sha256sum "${TASK_CHECKPOINT}" "${TASK_HISTORICAL}" \
  "${TASK_BASELINE}/run_manifest.json" "${TASK_BASELINE}/per_image.json" \
  "${TASK_LEGACY_EMBEDS}" \
  "${TASK_SUBJECT_EMBEDS}" "${TASK_BBOX}" "${TASK_EVALUATOR}" \
  "${TASK_WRAPPER}" "${TASK_VERIFY}" "${TASK_SCORE}" "${TASK_ROI}" \
  > "${TASK_ROOT}/input_sha256.txt"

if [[ -f "${TASK_PROJECT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${TASK_PROJECT}/.env"
  set +a
fi

TASK_STAGE="gate:reuse_exact_full96_historical_replay"
printf '%s\n' "${TASK_STAGE}" | tee "${TASK_ROOT}/CURRENT_STAGE"
cd "${TASK_PROJECT}"
"${TASK_ENV}/bin/python" "${TASK_VERIFY}" \
  --replay-dir "${TASK_BASELINE}" \
  --historical-manifest "${TASK_HISTORICAL}" \
  --expect-count 96 \
  --output "${TASK_OUTPUT}/reused_full96_historical_replay_verification.json"

run_roi() {
  local task_name="$1"
  local task_steps="$2"
  TASK_STAGE="refine:${task_name}"
  printf '%s\n' "${TASK_STAGE}" | tee "${TASK_ROOT}/CURRENT_STAGE"
  cd "${TASK_PROJECT}"
  "${TASK_ENV}/bin/python" "${TASK_ROI}" \
    --project-root "${TASK_PROJECT}" \
    --evaluator "${TASK_EVALUATOR}" \
    --config "${TASK_CONFIG}" \
    --checkpoint "${TASK_CHECKPOINT}" \
    --baseline-dir "${TASK_BASELINE}" \
    --output-dir "${TASK_OUTPUT}/${task_name}" \
    --roi-scale 2.0 \
    --bbox-expansion 1.5 \
    --late-steps "${task_steps}" \
    --feather-fraction 0.12
  "${TASK_ENV}/bin/python" "${TASK_VERIFY}" \
    --replay-dir "${TASK_OUTPUT}/${task_name}" \
    --historical-manifest "${TASK_HISTORICAL}" \
    --skip-indices 5,9,17,21,29,33,41,45,53,57,65,69,77,81,89,93 \
    --expect-count 80 \
    --output "${TASK_OUTPUT}/${task_name}_sentinel_verification.json"
}

run_roi CL9V_smallface_roi_refine_24k_r2 18
run_roi CL9V_smallface_roi_refine_gentle_24k_r2 10

score_run() {
  local task_name="$1"
  local task_dir="$2"
  TASK_STAGE="score:${task_name}"
  printf '%s\n' "${TASK_STAGE}" | tee "${TASK_ROOT}/CURRENT_STAGE"
  cd "${TASK_PROJECT}"
  "${TASK_ENV}/bin/python" "${TASK_SCORE}" \
    --project-root "${TASK_PROJECT}" \
    --run-dir "${task_dir}" \
    --subject-embeddings "${TASK_SUBJECT_EMBEDS}" \
    --legacy-embeddings "${TASK_LEGACY_EMBEDS}" \
    --output-json "${task_dir}/subject_v2_scores.json" \
    --output-csv "${task_dir}/subject_v2_scores.csv"
  "${TASK_ENV}/bin/python" "${TASK_FQ}" \
    --manifest "${task_dir}/face_quality_input_manifest.json" \
    --output-json "${task_dir}/face_quality_topiq_face.json" \
    --output-csv "${task_dir}/face_quality_topiq_face.csv" \
    --metrics topiq_nr-face \
    --device cuda \
    --batch-size 8
}

score_run roi_refine_18 "${TASK_OUTPUT}/CL9V_smallface_roi_refine_24k_r2"
score_run roi_refine_10 "${TASK_OUTPUT}/CL9V_smallface_roi_refine_gentle_24k_r2"

TASK_STAGE="final_hashes"
find "${TASK_OUTPUT}" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${TASK_ROOT}/output_sha256.txt"
printf '%s\n' complete > "${TASK_ROOT}/CURRENT_STAGE"
