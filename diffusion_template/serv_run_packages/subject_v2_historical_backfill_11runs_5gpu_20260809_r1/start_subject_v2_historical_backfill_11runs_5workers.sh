#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="${BACKFILL_TASK_NAME:-subject_v2_historical_backfill_11runs_5gpu_20260809_r1}"
BACKFILL_WAVE="${BACKFILL_WAVE:-base}"
WORKER_COUNT="${BACKFILL_WORKER_COUNT:-5}"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}"
PACKAGE_ROOT="${BACKFILL_PACKAGE_ROOT:-${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package}"
OVERLAY_PROJECT="${PACKAGE_ROOT}/overlay/diffusion_template"
ASSET_ROOT="${PACKAGE_ROOT}/assets"
STAGING_ROOT="${BACKFILL_STAGING_ROOT:-${TASK_ROOT}/staging}"
JOB_RUNTIME_ID="${HOSTNAME%-mpimaster-0}"
if [[ "${JOB_RUNTIME_ID}" == "${HOSTNAME}" ]]; then
  JOB_RUNTIME_ID="${HOSTNAME%-mpiworker-*}"
fi
if [[ "${JOB_RUNTIME_ID}" == "${HOSTNAME}" ]]; then
  echo "Cannot derive the MLS job runtime ID from hostname ${HOSTNAME}." >&2
  exit 69
fi
STATUS_ROOT="${TASK_ROOT}/status/${JOB_RUNTIME_ID}"
CLAIM_ROOT="${TASK_ROOT}/worker_claims/${JOB_RUNTIME_ID}"
DYNAMIC_CLAIM_ROOT="${BACKFILL_DYNAMIC_CLAIM_ROOT:-${CLAIM_ROOT}/run_claims}"
LOG_ROOT="${OWNER_ROOT}/logs/${TASK_NAME}/${JOB_RUNTIME_ID}"

CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
LIBSTDCXX_PATH="${OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34"
PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
DATASET_ROOT="${OWNER_ROOT}/datasets/dataset_full"
BIG_CELEBS_ROOT="${OWNER_ROOT}/datasets/bigcelebs/releases/v2"
BBOX_MANUAL="${DATASET_ROOT}/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
SUBJECT_MANIFEST="${ASSET_ROOT}/id_embeds_manual_val_subject_v2.json"
SUBJECT_ID_EMBEDS="${ASSET_ROOT}/id_embeds_manual_val_subject_v2.pth"
BACKFILL_TOOL="${OVERLAY_PROJECT}/tools/comet/backfill_subject_v2_validation.py"
EVALUATOR="${OVERLAY_PROJECT}/tools/inference/evaluate_rhca_checkpoint.py"
INITIAL_SECONDS_PER_CHECKPOINT=1200

mkdir -p "${CLAIM_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}" "${STAGING_ROOT}"

# 09 Aug 2026 - Five MLS workers execute the same binary entry point.  NFS
# mkdir is the allocation primitive, giving every one-GPU worker exactly one
# non-overlapping, deterministic chain without relying on undocumented rank vars.
worker_slot=""
for ((candidate = 0; candidate < WORKER_COUNT; candidate++)); do
  if mkdir "${CLAIM_ROOT}/worker_${candidate}" 2>/dev/null; then
    worker_slot="${candidate}"
    printf '%s\n' "$(hostname)" > "${CLAIM_ROOT}/worker_${candidate}/hostname.txt"
    break
  fi
done
if [[ -z "${worker_slot}" ]]; then
  echo "No unclaimed worker slot is available under ${CLAIM_ROOT}." >&2
  exit 70
fi

exec > >(tee -a "${LOG_ROOT}/worker_${worker_slot}.stdout.log") \
     2> >(tee -a "${LOG_ROOT}/worker_${worker_slot}.stderr.log" >&2)

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
if [[ -n "${CONDA_BASE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
elif [[ -x "${CONDA_ENV}/bin/python" ]]; then
  # 09 Aug 2026 - Internal MLS SSH workers do not inherit mpimaster's Conda
  # bootstrap. The environment itself is sealed and executable, so activate it
  # directly through PATH/CONDA_PREFIX when the base `conda` command is absent.
  export CONDA_PREFIX="${CONDA_ENV}"
  export PATH="${CONDA_ENV}/bin:${PATH}"
else
  echo "Could not locate Conda or the pinned environment on ${HOSTNAME}." >&2
  exit 70
fi
if [[ "$(readlink -f "$(command -v python)")" != "$(readlink -f "${CONDA_ENV}/bin/python")" ]]; then
  echo "Python interpreter is not the pinned Nasilaev photomaker_NS environment." >&2
  exit 70
fi
if [[ "$(readlink -f "${CONDA_PREFIX:-}")" != "$(readlink -f "${CONDA_ENV}")" ]]; then
  echo "CONDA_PREFIX is not the pinned Nasilaev photomaker_NS environment." >&2
  exit 70
fi

ORIGINAL_PYTHONPATH="${PYTHONPATH:-}"
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
ORIGINAL_LD_PRELOAD="${LD_PRELOAD:-}"

test -s "${BACKFILL_TOOL}"
test -s "${EVALUATOR}"
test -s "${OVERLAY_PROJECT}/src/face_subject_selector.py"
test -s "${SUBJECT_MANIFEST}"
test -s "${SUBJECT_ID_EMBEDS}"
test -s "${BBOX_MANUAL}"
test -s "${PM_PATH}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"

[[ "$(sha256sum "${BACKFILL_TOOL}" | cut -d' ' -f1)" == "2f9947d4e0b02c5f5ffb0fd1eff4d0cdc96e91aa4411ce4b6703a0702b9f1cb7" ]]
[[ "$(sha256sum "${EVALUATOR}" | cut -d' ' -f1)" == "c287b69835e5d2ee8ee5e0749136c0f0ed3f6d199d2f70df0dca484670e2f044" ]]
[[ "$(sha256sum "${OVERLAY_PROJECT}/src/face_subject_selector.py" | cut -d' ' -f1)" == "4e14aa3a62c24ebae7708a9fbfaf32b8e1801f4a9444135b8a448b6f0e8733a4" ]]
[[ "$(sha256sum "${SUBJECT_MANIFEST}" | cut -d' ' -f1)" == "7ccbf6f70cfc921142950c1d8a9149bd191f3ae6a506aab30d7df17031af841f" ]]
[[ "$(sha256sum "${SUBJECT_ID_EMBEDS}" | cut -d' ' -f1)" == "e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465" ]]
[[ "$(sha256sum "${BBOX_MANUAL}" | cut -d' ' -f1)" == "a39645e22b68027175946a028e185b7c5393a7514f5d68c94cd74e7cc9f5e614" ]]
if ! grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"; then
  echo "Pinned libstdc++ does not expose GLIBCXX_3.4.32." >&2
  exit 71
fi

DYNAMIC_RUNS=()
case "${BACKFILL_WAVE}:${worker_slot}" in
  base:0) RUN_CHAIN=(E13_large_ds_joint_shadow_sa128_24k_full96_r4 CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1 CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1) ;;
  base:1) RUN_CHAIN=(BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1 CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1) ;;
  base:2) RUN_CHAIN=(BC_E13_ds1_repeatdepth_balanced_24k_full96_r1 CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1) ;;
  base:3) RUN_CHAIN=(BC_E13_ds3_large_anchor_2to1_24k_full96_r1 CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1) ;;
  base:4) RUN_CHAIN=(CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1 BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1) ;;
  e14_e15:0) RUN_CHAIN=(E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6) ;;
  e14_e15:1) RUN_CHAIN=(E15_large_ds_joint_persist_sa128_protected_24k_full96_r2) ;;
  e14_e22_delayed:0) RUN_CHAIN=(E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6 E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2) ;;
  e14_e22_delayed:1) RUN_CHAIN=(E15_large_ds_joint_persist_sa128_protected_24k_full96_r2 E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2) ;;
  e14_e22_delayed:2) RUN_CHAIN=(E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2 E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2) ;;
  e14_e22_delayed:3) RUN_CHAIN=(E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5 E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2) ;;
  e14_e22_delayed:4) RUN_CHAIN=(E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4) ;;
  cl10_cl14_then_e14_e22:0) RUN_CHAIN=(CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2 E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6 E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2) ;;
  cl10_cl14_then_e14_e22:1) RUN_CHAIN=(CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1 E15_large_ds_joint_persist_sa128_protected_24k_full96_r2 E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2) ;;
  cl10_cl14_then_e14_e22:2) RUN_CHAIN=(CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1 E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2 E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2) ;;
  cl10_cl14_then_e14_e22:3) RUN_CHAIN=(CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1 E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5 E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2) ;;
  cl10_cl14_then_e14_e22:4) RUN_CHAIN=(CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1 E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4) ;;
  recovery_and_priority_8gpu:0) RUN_CHAIN=(CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1) ;;
  recovery_and_priority_8gpu:1) RUN_CHAIN=(BC_E13_ds3_large_anchor_2to1_24k_full96_r1) ;;
  recovery_and_priority_8gpu:2) RUN_CHAIN=(CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1) ;;
  recovery_and_priority_8gpu:3) RUN_CHAIN=(CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2 E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6 E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2) ;;
  recovery_and_priority_8gpu:4) RUN_CHAIN=(CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1 E15_large_ds_joint_persist_sa128_protected_24k_full96_r2 E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2) ;;
  recovery_and_priority_8gpu:5) RUN_CHAIN=(CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1 E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2 E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2) ;;
  recovery_and_priority_8gpu:6) RUN_CHAIN=(CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1 E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5 E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2) ;;
  recovery_and_priority_8gpu:7) RUN_CHAIN=(CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1 E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4) ;;
  dynamic_remaining_8gpu:*)
    RUN_CHAIN=()
    DYNAMIC_RUNS=(
      CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1
      CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1
      CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1
      CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1
      E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6
      E15_large_ds_joint_persist_sa128_protected_24k_full96_r2
      E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2
      E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5
      E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4
      E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2
      E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2
      E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2
      E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2
    )
    ;;
  *) echo "Unexpected backfill wave/worker slot ${BACKFILL_WAVE}:${worker_slot}." >&2; exit 72 ;;
esac

chain_run_count=${#RUN_CHAIN[@]}
if (( ${#DYNAMIC_RUNS[@]} > 0 )); then
  chain_run_count=$(((${#DYNAMIC_RUNS[@]} + WORKER_COUNT - 1) / WORKER_COUNT))
fi
chain_checkpoints=$((chain_run_count * 12))
chain_initial_seconds=$((chain_checkpoints * INITIAL_SECONDS_PER_CHECKPOINT))
printf 'BACKFILL_CHAIN_PLAN worker=%s runs=%s checkpoints=%s initial_eta_seconds=%s host=%s\n' \
  "${worker_slot}" "${chain_run_count}" "${chain_checkpoints}" \
  "${chain_initial_seconds}" "$(hostname)"

configure_run() {
  local run_name="$1"
  case "${run_name}" in
    E13_large_ds_joint_shadow_sa128_24k_full96_r4)
      CONFIG_NAME="E13_large_ds_joint_shadow_sa128_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template"
      EXPECTED_COMET_KEY="1cc0a02371094b24a6a02a4cc649f10c"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1)
      CONFIG_NAME="BC_E13_big_celebs_joint_shadow_sa128_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template"
      EXPECTED_COMET_KEY="c138db7c41ae435c8a7560f40cf5f58d"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    BC_E13_ds1_repeatdepth_balanced_24k_full96_r1)
      CONFIG_NAME="BC_E13_ds1_repeatdepth_balanced_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template"
      EXPECTED_COMET_KEY="b5b23b0ca4b449bc8f4703d6a7334be1"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1)
      CONFIG_NAME="BC_E13_ds2_scene_target_canonical_ref_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template"
      EXPECTED_COMET_KEY="5db54d7d4557487e94251656736843db"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    BC_E13_ds3_large_anchor_2to1_24k_full96_r1)
      CONFIG_NAME="BC_E13_ds3_large_anchor_2to1_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809/diffusion_template"
      EXPECTED_COMET_KEY="43adf33cf7174e89b8fde1cdd640a052"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    CL4_cosmic_joint_shadow_sa128_hygiene_24k_full96_r1)
      CONFIG_NAME="CL4_cosmic_joint_shadow_sa128_hygiene_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="0dd86b436b224f939efa3887ad6acbe2"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k_full96_r1)
      CONFIG_NAME="CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="2851395f018e4613b39a6565a92a89c6"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL6_cosmic_joint_shadow_sa128_boundary_24k_full96_r1)
      CONFIG_NAME="CL6_cosmic_joint_shadow_sa128_boundary_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="ddfdc5f140954c5bb9da880a3d204147"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL7_cosmic_joint_shadow_sa128_altloss_24k_full96_r1)
      CONFIG_NAME="CL7_cosmic_joint_shadow_sa128_altloss_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="9010faa3666f413bbbd1fdf8e7a30825"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL8_cosmic_joint_shadow_sa128_fullbody_24k_full96_r1)
      CONFIG_NAME="CL8_cosmic_joint_shadow_sa128_fullbody_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="a6b5970aa1a24d3490ad08e7994b5f1e"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL9_cosmic_joint_shadow_sa128_refscale_24k_full96_r1)
      CONFIG_NAME="CL9_cosmic_joint_shadow_sa128_refscale_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="81bb311ed70545eda3281c64bc48be47"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k_full96_r2)
      CONFIG_NAME="CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="eba0187806ec476996f5ea4af356361e"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1)
      CONFIG_NAME="CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="32f4ba2a3b3a493f96a3a2345147e84c"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k_full96_r1)
      CONFIG_NAME="CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="5ab75864c2fb4df28decf8d76d8306f8"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL13_cosmic_joint_shadow_sa128_refdropout_24k_full96_r1)
      CONFIG_NAME="CL13_cosmic_joint_shadow_sa128_refdropout_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="248d7967cec4457f91219a9ebec22687"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1)
      CONFIG_NAME="CL14_cosmic_joint_shadow_sa128_softmask_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="6fe0028be92242c38056b3d36665fdd6"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    E14_large_ds_joint_shadow_sa128_protected_24k_full96_r6)
      CONFIG_NAME="E14_large_ds_joint_shadow_sa128_protected_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_E14_r6_deferred_20260805/diffusion_template"
      EXPECTED_COMET_KEY="f53c2a2f130247a1b817c820ba7615ae"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    E15_large_ds_joint_persist_sa128_protected_24k_full96_r2)
      CONFIG_NAME="E15_large_ds_joint_persist_sa128_protected_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template"
      EXPECTED_COMET_KEY="f320234a54624aa6a1a100307691b627"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2)
      CONFIG_NAME="E16_large_ds_joint_persist_sa128_idloss_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_E15_E18_gpu_20260805/diffusion_template"
      EXPECTED_COMET_KEY="4561fb0de8c64b3da8663e3f4c37589c"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    E17_large_ds_joint_persist_sa128_resididca_24k_full96_r5)
      CONFIG_NAME="E17_large_ds_joint_persist_sa128_resididca_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_E17_r5_rmsfix_20260805/diffusion_template"
      EXPECTED_COMET_KEY="08ecedf8e058461abe952077f9623ab8"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    E18_large_ds_joint_persist_sa128_multiref_24k_full96_r4)
      CONFIG_NAME="E18_large_ds_joint_persist_sa128_multiref_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_worktrees/rsrch_test_E18_r4_deferred_20260805/diffusion_template"
      EXPECTED_COMET_KEY="b9e118da6dc94cd9b3849566e18c67ff"
      EXPECTED_GENERATION_BBOX_SHA256="4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099"
      ;;
    E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2)
      CONFIG_NAME="E19_large_ds_joint_shadow_sa128_multiref_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_e19_e24_v3/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="3280232a45ef4ea2ae68c8deff3b81c1"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2)
      CONFIG_NAME="E20_large_ds_joint_shadow_sa128_branchout_r32_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_e19_e24_v3/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="4084c35600ae4ad3904446e5f4d2de92"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2)
      CONFIG_NAME="E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_e19_e24_v3/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="3ef78907f60a4f5cbd7727fc5be7143e"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2)
      CONFIG_NAME="E22_large_ds_joint_shadow_sa128_arcfaceaux_24k"
      PROJECT_ROOT="${OWNER_ROOT}/runtime_sources_e19_e24_v3/${run_name}/diffusion_template"
      EXPECTED_COMET_KEY="5a91be0df76f4966be5c77eee26cfc29"
      EXPECTED_GENERATION_BBOX_SHA256="b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
      ;;
    *) echo "No immutable run mapping for ${run_name}." >&2; return 73 ;;
  esac
  # 09 Aug 2026 - AICODE-NOTE: training used the immutable runtime's derived
  # *_auto.json cache for BA.  The sealed canonical manual map is its seed,
  # not the active generation map; conflating them changes every Eddie image.
  GENERATION_BBOX_MAP="${PROJECT_ROOT}/../dataset_full/val_dataset/pm96_bboxes_new_auto.json"
}

assert_staged_steps() {
  local manifest="$1"
  python - "${manifest}" <<'PY'
import json
import sys

expected = list(range(2_000, 24_001, 2_000))
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("steps") != expected:
    raise SystemExit(
        f"Expected all 12 saved validation steps {expected}, found {payload.get('steps')}"
    )
PY
}

run_backfill() {
  local run_name="$1"
  configure_run "${run_name}"
  local run_dir="${PROJECT_ROOT}/saved/${run_name}"
  local record="${run_dir}/comet_experiment.json"
  local staging="${STAGING_ROOT}/${run_name}"
  local completed_audit="${staging}/replacement_verified.json"
  local legacy_id_embeds="${PROJECT_ROOT}/../dataset_full/val_dataset/id_embeds_manual_val.pth"

  test -d "${PROJECT_ROOT}"
  test -f "${PROJECT_ROOT}/src/configs/${CONFIG_NAME}.yaml"
  test -f "${PROJECT_ROOT}/.env"
  test -s "${record}"
  test -s "${legacy_id_embeds}"
  test -s "${GENERATION_BBOX_MAP}"
  if [[ "$(sha256sum "${GENERATION_BBOX_MAP}" | cut -d' ' -f1)" != "${EXPECTED_GENERATION_BBOX_SHA256}" ]]; then
    echo "Active generation-bbox hash mismatch for ${run_name}." >&2
    return 77
  fi
  if [[ "$(sha256sum "${legacy_id_embeds}" | cut -d' ' -f1)" != "23ae97075e967f2bcb790c5094ef350b316249c7023df67a68f735bfebb747c6" ]]; then
    echo "Legacy validation embedding hash mismatch for ${run_name}." >&2
    return 76
  fi
  local checkpoint_count
  checkpoint_count="$(find "${run_dir}" -maxdepth 1 -type f -name 'weights-epoch*.pth' | wc -l)"
  if [[ "${checkpoint_count}" -ne 12 ]]; then
    echo "${run_name} has ${checkpoint_count}, not 12, weights checkpoints." >&2
    return 74
  fi
  local actual_key
  actual_key="$(python - "${record}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["comet"]["experiment_key"])
PY
)"
  if [[ "${actual_key}" != "${EXPECTED_COMET_KEY}" ]]; then
    echo "Immutable Comet key mismatch for ${run_name}." >&2
    return 75
  fi

  set -a
  # shellcheck disable=SC1090
  source "${PROJECT_ROOT}/.env"
  set +a
  export ENV_FILE=/dev/null
  export PM_PATH
  export FULL96_BBOX_MANUAL="${BBOX_MANUAL}"
  export BIG_CELEBS_MANIFEST="${BIG_CELEBS_ROOT}/filtered_ids3_adj.json"
  export BIG_CELEBS_IMAGES="${BIG_CELEBS_ROOT}/large_dataset"
  export BIG_CELEBS_SEAL="${BIG_CELEBS_ROOT}/dataset_manifest.json"
  export BIG_CELEBS_DOWNLOAD_LOG="${OWNER_ROOT}/datasets/dataset_tools/download_bigcelebs_v2.log"
  export BIG_CELEBS_EXPECTED_MANIFEST_SHA256="f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa"
  export LARGE_DATASET_MANIFEST="${DATASET_ROOT}/filtered_ids3_adj.json"
  export LARGE_DATASET_IMAGES="${DATASET_ROOT}/large_dataset_adj/large_dataset"
  export LARGE_DATASET_EXPECTED_MANIFEST_SHA256="0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50"
  export PM_BACKFILL_PROJECT_ROOT="${PROJECT_ROOT}"
  export PM_EVAL_PROJECT_ROOT="${PROJECT_ROOT}"
  export ACCELERATE_NUM_PROCESSES=1
  export TORCH_DISABLE_ADDR2LINE=1
  export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
  export CLIP_CACHE_DIR="${OWNER_ROOT}/metric_cache/clip"
  export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
  export PYTHONPATH="${OVERLAY_PROJECT}:${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${ORIGINAL_PYTHONPATH:+:${ORIGINAL_PYTHONPATH}}"
  export LD_LIBRARY_PATH="$(dirname "${LIBSTDCXX_PATH}"):${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${ORIGINAL_LD_LIBRARY_PATH:+:${ORIGINAL_LD_LIBRARY_PATH}}"
  export LD_PRELOAD="${LIBSTDCXX_PATH}${ORIGINAL_LD_PRELOAD:+:${ORIGINAL_LD_PRELOAD}}"

  cd "${PROJECT_ROOT}"
  local common_args=(
    --run-dir "${run_dir}"
    --config "${CONFIG_NAME}"
    --experiment-record "${record}"
    --staging-root "${staging}"
    --all-safe-checkpoints
    --subject-manifest "${SUBJECT_MANIFEST}"
    --legacy-id-embeddings "${legacy_id_embeds}"
    --subject-v2-id-embeddings "${SUBJECT_ID_EMBEDS}"
    --evaluator "${EVALUATOR}"
    --generation-bbox-map "${GENERATION_BBOX_MAP}"
    --env-file "${PROJECT_ROOT}/.env"
    --metric-batch-size 16
    --face-quality-batch-size 8
    --initial-seconds-per-checkpoint "${INITIAL_SECONDS_PER_CHECKPOINT}"
  )

  if [[ -s "${completed_audit}" ]]; then
    echo "BACKFILL_RUN_ALREADY_VERIFIED run=${run_name} key=${EXPECTED_COMET_KEY}"
    return 0
  fi
  echo "BACKFILL_RUN_START worker=${worker_slot} run=${run_name} key=${EXPECTED_COMET_KEY}"
  if [[ -s "${staging}/job_manifest.json" ]]; then
    assert_staged_steps "${staging}/job_manifest.json"
    python "${BACKFILL_TOOL}" "${common_args[@]}" --reuse-staging --write
  else
    # 10 Aug 2026 - Resume complete checkpoint manifests after transient Comet
    # failures; the tool quarantines any incomplete step before regenerating it.
    python "${BACKFILL_TOOL}" "${common_args[@]}" --reuse-staging
    assert_staged_steps "${staging}/job_manifest.json"
    python "${BACKFILL_TOOL}" "${common_args[@]}" --reuse-staging --write
  fi
  test -s "${completed_audit}"
  python - "${staging}/job_manifest.json" "${completed_audit}" "${EXPECTED_COMET_KEY}" <<'PY'
import json
import sys

job_path, audit_path, expected_key = sys.argv[1:]
job = json.load(open(job_path, encoding="utf-8"))
audit = json.load(open(audit_path, encoding="utf-8"))
if job.get("status") != "verified_on_comet":
    raise SystemExit(f"Backfill job was not verified on Comet: {job.get('status')}")
if audit.get("experiment_key") != expected_key:
    raise SystemExit("Replacement audit immutable-key mismatch")
if audit.get("replacement_steps") != list(range(2_000, 24_001, 2_000)):
    raise SystemExit("Replacement audit does not cover all 12 saved checkpoints")
PY
  echo "BACKFILL_RUN_COMPLETE worker=${worker_slot} run=${run_name} key=${EXPECTED_COMET_KEY}"
}

worker_started="$(date +%s)"
printf '%s\n' "running" > "${STATUS_ROOT}/worker_${worker_slot}.status"
trap 'printf "%s\n" "failed" > "${STATUS_ROOT}/worker_${worker_slot}.status"' ERR
if (( ${#DYNAMIC_RUNS[@]} > 0 )); then
  # 10 Aug 2026 - AICODE-NOTE: independent one-GPU MLS jobs must share this
  # NFS claim root; job-local claim roots would make every job select CL6.
  dynamic_claim_root="${DYNAMIC_CLAIM_ROOT}"
  mkdir -p "${dynamic_claim_root}"
  while true; do
    claimed_run=""
    for candidate_run in "${DYNAMIC_RUNS[@]}"; do
      if mkdir "${dynamic_claim_root}/${candidate_run}" 2>/dev/null; then
        claimed_run="${candidate_run}"
        printf '%s worker=%s host=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          "${worker_slot}" "$(hostname)" > \
          "${dynamic_claim_root}/${candidate_run}/claim.txt"
        break
      fi
    done
    if [[ -z "${claimed_run}" ]]; then
      break
    fi
    echo "BACKFILL_DYNAMIC_CLAIM worker=${worker_slot} run=${claimed_run}"
    run_backfill "${claimed_run}"
  done
else
  for run_name in "${RUN_CHAIN[@]}"; do
    run_backfill "${run_name}"
  done
fi
worker_elapsed=$(($(date +%s) - worker_started))
printf '%s\n' "complete" > "${STATUS_ROOT}/worker_${worker_slot}.status"
trap - ERR
echo "BACKFILL_WORKER_COMPLETE worker=${worker_slot} elapsed_seconds=${worker_elapsed}"
