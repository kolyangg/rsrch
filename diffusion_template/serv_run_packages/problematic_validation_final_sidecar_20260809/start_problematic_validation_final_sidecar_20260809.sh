#!/usr/bin/env bash
set -euo pipefail

echo "Blocked: superseded Eddie sidecar used validation_native, omitted the shadow default-adapter restore, and batched 1 instead of 12. Repackage the contract-v2 replay before submission." >&2
exit 2

# 09 Aug 2026 - Runs three corrected-reference diagnostics sequentially on one
# A100. Each model is loaded from the exact runtime and weights that produced
# its source validation panel; the only generation change is Eddie's ArcFace
# selection from the background detection to the intended foreground face.
TASK_OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${TASK_OWNER_ROOT}/analysis_sidecars/problematic_validation_20260809"
TASK_ENV="${TASK_OWNER_ROOT}/conda_env/photomaker_NS"
TASK_ORT="${TASK_OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
TASK_PYIQA="${TASK_OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
TASK_NVIDIA="${TASK_ENV}/lib/python3.10/site-packages/nvidia"
TASK_LIBSTDCXX="${TASK_OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34"
TASK_SCRIPT="${TASK_ROOT}/run_corrected_eddie_sidecar.py"
TASK_EMBEDDING="${TASK_ROOT}/eddie_foreground_arcface_embedding.npy"
TASK_BBOX_MANUAL="${TASK_OWNER_ROOT}/datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"

TASK_E13_PROJECT="${TASK_OWNER_ROOT}/runtime_worktrees/rsrch_test_E13_E14_deferred_20260805/diffusion_template"
TASK_BC_PROJECT="${TASK_OWNER_ROOT}/runtime_worktrees/rsrch_test_BC_E13_bigcelebs_20260808/diffusion_template"
TASK_CL11_PROJECT="${TASK_OWNER_ROOT}/runtime_sources_cl1_cl3_v1/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/diffusion_template"
TASK_E13_CHECKPOINT="${TASK_E13_PROJECT}/saved/E13_large_ds_joint_shadow_sa128_24k_full96_r4/weights-epoch12.pth"
TASK_BC_CHECKPOINT="${TASK_BC_PROJECT}/saved/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/weights-epoch12.pth"
TASK_CL11_CHECKPOINT="${TASK_CL11_PROJECT}/saved/CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k_full96_r1/weights-epoch10.pth"

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
export TORCH_HOME="${TASK_OWNER_ROOT}/metric_cache/torch"
export PM_PATH="${TASK_OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export FULL96_BBOX_MANUAL="${TASK_BBOX_MANUAL}"
export LD_LIBRARY_PATH="${TASK_NVIDIA}/cublas/lib:${TASK_NVIDIA}/cuda_cupti/lib:${TASK_NVIDIA}/cuda_nvrtc/lib:${TASK_NVIDIA}/cuda_runtime/lib:${TASK_NVIDIA}/cudnn/lib:${TASK_NVIDIA}/cufft/lib:${TASK_NVIDIA}/curand/lib:${TASK_NVIDIA}/cusolver/lib:${TASK_NVIDIA}/cusparse/lib:${TASK_NVIDIA}/nccl/lib:${TASK_NVIDIA}/nvjitlink/lib:${TASK_NVIDIA}/nvtx/lib"
export LD_LIBRARY_PATH="$(dirname "${TASK_LIBSTDCXX}"):${LD_LIBRARY_PATH}"
export LD_PRELOAD="${TASK_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}"

test -s "${TASK_SCRIPT}"
test -s "${TASK_EMBEDDING}"
test -s "${TASK_BBOX_MANUAL}"
test -s "${TASK_BBOX_MANUAL%.json}_auto.json"
test -s "${PM_PATH}"
test -s "${TASK_E13_CHECKPOINT}"
test -s "${TASK_BC_CHECKPOINT}"
test -s "${TASK_CL11_CHECKPOINT}"
test ! -e "${TASK_ROOT}/COMPLETE"

sha256sum \
  "${TASK_SCRIPT}" \
  "${TASK_EMBEDDING}" \
  "${TASK_BBOX_MANUAL}" \
  "${TASK_BBOX_MANUAL%.json}_auto.json" \
  "${TASK_E13_CHECKPOINT}" \
  "${TASK_BC_CHECKPOINT}" \
  "${TASK_CL11_CHECKPOINT}" \
  > "${TASK_ROOT}/input_sha256.txt"

run_corrected_eddie() {
  local task_project="$1"
  local task_config="$2"
  local task_checkpoint="$3"
  local task_step="$4"
  local task_output="$5"
  test -f "${task_project}/src/configs/${task_config}.yaml"
  test -f "${task_project}/tools/inference/evaluate_rhca_checkpoint.py"
  cd "${task_project}"
  PYTHONPATH="${task_project}:${TASK_PYIQA}:${TASK_ORT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TASK_ENV}/bin/python" "${TASK_SCRIPT}" \
      --project-root "${task_project}" \
      --embedding "${TASK_EMBEDDING}" \
      --config "${task_config}" \
      --checkpoint "${task_checkpoint}" \
      --checkpoint-step "${task_step}" \
      --output-dir "${task_output}" \
      --batch-size 1
}

run_corrected_eddie \
  "${TASK_E13_PROJECT}" \
  E13_large_ds_joint_shadow_sa128_24k \
  "${TASK_E13_CHECKPOINT}" \
  24000 \
  "${TASK_ROOT}/outputs/E13_24000_corrected_eddie"

run_corrected_eddie \
  "${TASK_BC_PROJECT}" \
  BC_E13_big_celebs_joint_shadow_sa128_24k \
  "${TASK_BC_CHECKPOINT}" \
  24000 \
  "${TASK_ROOT}/outputs/BC_E13_24000_corrected_eddie"

run_corrected_eddie \
  "${TASK_CL11_PROJECT}" \
  CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k \
  "${TASK_CL11_CHECKPOINT}" \
  20000 \
  "${TASK_ROOT}/outputs/CL11_20000_corrected_eddie"

find "${TASK_ROOT}/outputs" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${TASK_ROOT}/output_sha256.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${TASK_ROOT}/COMPLETE"
