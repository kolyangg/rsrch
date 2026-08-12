#!/usr/bin/env bash
set -euo pipefail

# 11 Aug 2026 - One allocation intentionally chains all promotion-gated CL9
# fixed-checkpoint validations. Every arm is scored and sealed before the next
# arm starts; any contract or RGB-sentinel failure stops the chain immediately.
TASK_OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_PACKAGE="${TASK_OWNER_ROOT}/rsrch_test/diffusion_template/serv_run_packages/cl9v_validation_chain_20260811_r4"
TASK_ROOT="${TASK_OWNER_ROOT}/analysis_sidecars/cl9v_validation_chain_20260811_r4"
TASK_OUTPUT="${TASK_ROOT}/outputs"
TASK_STAGES="${TASK_ROOT}/stages"
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
TASK_R3_ROI18="${TASK_OWNER_ROOT}/analysis_sidecars/cl9v_smallface_roi_20260810_r3/outputs/CL9V_smallface_roi_refine_24k_r3"
TASK_R3_ROLL="${TASK_OWNER_ROOT}/analysis_sidecars/cl9v_marion_occlusion_20260810_r3/outputs/CL9V_marion_samefile_roll_24k_r3"
TASK_MARION="${TASK_RUNTIME_ROOT}/dataset_full/val_dataset/references/marion.jpg"
TASK_LEGACY_EMBEDS="${TASK_RUNTIME_ROOT}/dataset_full/val_dataset/id_embeds_manual_val.pth"
TASK_SUBJECT_EMBEDS="${TASK_OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
TASK_BBOX="${TASK_PACKAGE}/pm96_bboxes_new_auto_cl9.json"
TASK_VISIBILITY="${TASK_PACKAGE}/occlusion_visibility_plan_precise.json"
TASK_EVALUATOR="${TASK_PACKAGE}/evaluate_rhca_checkpoint.py"
TASK_WRAPPER="${TASK_PACKAGE}/run_cl9_fixed_checkpoint_sidecar.py"
TASK_PREPARE="${TASK_PACKAGE}/prepare_marion_references.py"
TASK_VERIFY="${TASK_PACKAGE}/verify_rgb_replay.py"
TASK_PAIR_VERIFY="${TASK_PACKAGE}/verify_run_pair_rgb.py"
TASK_VISIBILITY_VERIFY="${TASK_PACKAGE}/validate_occluder_geometry.py"
TASK_SCORE="${TASK_PACKAGE}/score_cl9_subject_v2.py"
TASK_ROI="${TASK_PACKAGE}/run_smallface_roi_refine.py"
TASK_FQ="${TASK_PROJECT}/tools/inference/calculate_face_quality_metrics.py"
TASK_SPEC="${TASK_PACKAGE}/CL9V_validation_chain_24k_20260811_r4.json"
TASK_SMALLFACE_INDICES="5,9,17,21,29,33,41,45,53,57,65,69,77,81,89,93"
TASK_OCCLUDER_INDICES="2,7,14,19,26,31,38,43,50,55,62,67,74,79,86,91"
TASK_MARION_INDICES="84,85,86,87,88,89,90,91,92,93,94,95"

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

for TASK_REQUIRED in \
  "${TASK_CHECKPOINT}" "${TASK_HISTORICAL}" \
  "${TASK_BASELINE}/run_manifest.json" "${TASK_BASELINE}/per_image.json" \
  "${TASK_R3_ROI18}/run_manifest.json" "${TASK_R3_ROLL}/run_manifest.json" \
  "${TASK_MARION}" "${TASK_LEGACY_EMBEDS}" "${TASK_SUBJECT_EMBEDS}" \
  "${TASK_BBOX}" "${TASK_VISIBILITY}" "${TASK_EVALUATOR}" \
  "${TASK_WRAPPER}" "${TASK_PREPARE}" "${TASK_VERIFY}" \
  "${TASK_PAIR_VERIFY}" "${TASK_VISIBILITY_VERIFY}" "${TASK_SCORE}" \
  "${TASK_ROI}" "${TASK_FQ}" "${TASK_SPEC}"; do
  test -s "${TASK_REQUIRED}"
done
[[ "$(sha256sum "${TASK_CHECKPOINT}" | cut -d' ' -f1)" == "5396993b16ace89908501bfddb2e412e755a3f6478a6449c502062d6ca7357c3" ]]
[[ "$(sha256sum "${TASK_MARION}" | cut -d' ' -f1)" == "3884de5c8ca4c97840512c4976daa3cc79bb9e33eef4369c9b6ec93aed3f5a22" ]]
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
mkdir -p "${TASK_OUTPUT}" "${TASK_STAGES}"
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

set_stage() {
  TASK_STAGE="$1"
  printf '%s\n' "${TASK_STAGE}" | tee "${TASK_ROOT}/CURRENT_STAGE"
}

seal_stage() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "${TASK_STAGES}/$1.complete"
}

sha256sum "${TASK_CHECKPOINT}" "${TASK_HISTORICAL}" \
  "${TASK_BASELINE}/run_manifest.json" "${TASK_BASELINE}/per_image.json" \
  "${TASK_R3_ROI18}/run_manifest.json" "${TASK_R3_ROLL}/run_manifest.json" \
  "${TASK_MARION}" "${TASK_LEGACY_EMBEDS}" "${TASK_SUBJECT_EMBEDS}" \
  "${TASK_BBOX}" "${TASK_VISIBILITY}" "${TASK_EVALUATOR}" \
  "${TASK_WRAPPER}" "${TASK_PREPARE}" "${TASK_VERIFY}" \
  "${TASK_PAIR_VERIFY}" "${TASK_VISIBILITY_VERIFY}" "${TASK_SCORE}" \
  "${TASK_ROI}" "${TASK_SPEC}" > "${TASK_ROOT}/input_sha256.txt"

if [[ -f "${TASK_PROJECT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${TASK_PROJECT}/.env"
  set +a
fi

cd "${TASK_PROJECT}"
set_stage "gate:reuse_exact_full96_historical_replay"
"${TASK_ENV}/bin/python" "${TASK_VERIFY}" \
  --replay-dir "${TASK_BASELINE}" \
  --historical-manifest "${TASK_HISTORICAL}" \
  --expect-count 96 \
  --output "${TASK_OUTPUT}/reused_full96_historical_replay_verification.json"
seal_stage "000_baseline_replay"

set_stage "gate:precise_occluder_geometry_preflight"
"${TASK_ENV}/bin/python" "${TASK_VISIBILITY_VERIFY}" \
  --plan "${TASK_VISIBILITY}" \
  --baseline-dir "${TASK_BASELINE}" \
  --output-dir "${TASK_OUTPUT}/occluder_geometry_preflight"
seal_stage "001_occluder_geometry_preflight"

set_stage "prepare:marion_samefile_roll"
"${TASK_ENV}/bin/python" "${TASK_PREPARE}" \
  --project-root "${TASK_PROJECT}" \
  --input "${TASK_MARION}" \
  --output-dir "${TASK_OUTPUT}/marion_transforms"
TASK_TRANSFORMS="${TASK_OUTPUT}/marion_transforms/marion_transform_manifest.json"
seal_stage "002_marion_transform"

score_run() {
  local task_name="$1"
  local task_dir="$2"
  set_stage "score:${task_name}"
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

run_roi() {
  local task_steps="$1"
  local task_seed="$2"
  local task_name="CL9V_smallface_roi_steps${task_steps}_seed${task_seed}_24k_r4"
  local task_dir="${TASK_OUTPUT}/${task_name}"
  set_stage "roi:steps${task_steps}:seed${task_seed}"
  cd "${TASK_PROJECT}"
  "${TASK_ENV}/bin/python" "${TASK_ROI}" \
    --project-root "${TASK_PROJECT}" \
    --evaluator "${TASK_EVALUATOR}" \
    --config "${TASK_CONFIG}" \
    --checkpoint "${TASK_CHECKPOINT}" \
    --baseline-dir "${TASK_BASELINE}" \
    --output-dir "${task_dir}" \
    --roi-scale 2.0 \
    --bbox-expansion 1.5 \
    --late-steps "${task_steps}" \
    --seed-override "${task_seed}" \
    --feather-fraction 0.12
  "${TASK_ENV}/bin/python" "${TASK_VERIFY}" \
    --replay-dir "${task_dir}" \
    --historical-manifest "${TASK_HISTORICAL}" \
    --skip-indices "${TASK_SMALLFACE_INDICES}" \
    --expect-count 80 \
    --output "${task_dir}_sentinel_verification.json"
  if [[ "${task_steps}" -eq 18 && "${task_seed}" -eq 0 ]]; then
    "${TASK_ENV}/bin/python" "${TASK_PAIR_VERIFY}" \
      --first "${task_dir}" \
      --second "${TASK_R3_ROI18}" \
      --expect-count 96 \
      --output "${TASK_OUTPUT}/roi_steps18_seed0_r3_reproduction_gate.json"
  fi
  score_run "${task_name}" "${task_dir}"
  seal_stage "roi_steps${task_steps}_seed${task_seed}"
}

# Priority 1: map the positive ROI transition and test it across diagnostic seeds.
for TASK_SEED in 0 1 2 3; do
  for TASK_STEPS in 14 16 18 20; do
    run_roi "${TASK_STEPS}" "${TASK_SEED}"
  done
done

run_fixed() {
  local task_variant="$1"
  local task_dir="$2"
  shift 2
  set_stage "generate:${task_variant}:$(basename "${task_dir}")"
  cd "${TASK_PROJECT}"
  "${TASK_ENV}/bin/python" "${TASK_WRAPPER}" \
    --variant "${task_variant}" \
    --project-root "${TASK_PROJECT}" \
    --evaluator "${TASK_EVALUATOR}" \
    --config "${TASK_CONFIG}" \
    --checkpoint "${TASK_CHECKPOINT}" \
    --checkpoint-step 24000 \
    --generation-bbox-map "${TASK_BBOX}" \
    --output-dir "${task_dir}" \
    "$@"
}

# Priority 2: precise, per-image oracle geometry. This is deliberately one
# standard-seed arm and remains ineligible for training without its visual gate.
TASK_OCCLUDER_NAME="CL9V_occluder_oracle_geometry_24k_r4"
TASK_OCCLUDER_DIR="${TASK_OUTPUT}/${TASK_OCCLUDER_NAME}"
run_fixed occlusion_ownership "${TASK_OCCLUDER_DIR}" \
  --target-visibility-plan "${TASK_VISIBILITY}"
"${TASK_ENV}/bin/python" "${TASK_VERIFY}" \
  --replay-dir "${TASK_OCCLUDER_DIR}" \
  --historical-manifest "${TASK_HISTORICAL}" \
  --skip-indices "${TASK_OCCLUDER_INDICES}" \
  --expect-count 80 \
  --output "${TASK_OUTPUT}/occluder_oracle_sentinel_verification.json"
score_run "${TASK_OCCLUDER_NAME}" "${TASK_OCCLUDER_DIR}"
seal_stage "occluder_oracle_geometry"

# Priority 3: seed 0 is the completed r3 baseline/roll pair. Generate seeds
# 1-3 by overriding only Marion's 12 rows; all 84 other rows remain exact
# historical sentinels in both original-reference and roll-reference arms.
for TASK_SEED in 1 2 3; do
  TASK_BASE_NAME="CL9V_marion_original_seed${TASK_SEED}_24k_r4"
  TASK_BASE_DIR="${TASK_OUTPUT}/${TASK_BASE_NAME}"
  run_fixed baseline "${TASK_BASE_DIR}" \
    --seed-override "${TASK_SEED}" \
    --seed-override-indices "${TASK_MARION_INDICES}"
  "${TASK_ENV}/bin/python" "${TASK_VERIFY}" \
    --replay-dir "${TASK_BASE_DIR}" \
    --historical-manifest "${TASK_HISTORICAL}" \
    --skip-indices "${TASK_MARION_INDICES}" \
    --expect-count 84 \
    --output "${TASK_OUTPUT}/marion_original_seed${TASK_SEED}_sentinel_verification.json"
  score_run "${TASK_BASE_NAME}" "${TASK_BASE_DIR}"
  seal_stage "marion_original_seed${TASK_SEED}"

  TASK_ROLL_NAME="CL9V_marion_roll_seed${TASK_SEED}_24k_r4"
  TASK_ROLL_DIR="${TASK_OUTPUT}/${TASK_ROLL_NAME}"
  run_fixed marion_roll "${TASK_ROLL_DIR}" \
    --reference-transform-manifest "${TASK_TRANSFORMS}" \
    --seed-override "${TASK_SEED}" \
    --seed-override-indices "${TASK_MARION_INDICES}"
  "${TASK_ENV}/bin/python" "${TASK_VERIFY}" \
    --replay-dir "${TASK_ROLL_DIR}" \
    --historical-manifest "${TASK_HISTORICAL}" \
    --skip-indices "${TASK_MARION_INDICES}" \
    --expect-count 84 \
    --output "${TASK_OUTPUT}/marion_roll_seed${TASK_SEED}_sentinel_verification.json"
  score_run "${TASK_ROLL_NAME}" "${TASK_ROLL_DIR}"
  seal_stage "marion_roll_seed${TASK_SEED}"
done

set_stage "final_hashes"
find "${TASK_OUTPUT}" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${TASK_ROOT}/output_sha256.txt"
printf '%s\n' complete > "${TASK_ROOT}/CURRENT_STAGE"
