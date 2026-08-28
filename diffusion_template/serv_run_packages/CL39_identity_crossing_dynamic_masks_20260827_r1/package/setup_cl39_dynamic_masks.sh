#!/usr/bin/env bash
# Shared sealed environment for the corrected CL39 seed-1/2/3 crossing.
set -euo pipefail

: "${EVAL_SEED:?Set EVAL_SEED to 1, 2, or 3}"
case "${EVAL_SEED}" in
  1|2|3) ;;
  *) echo "Refusing unreviewed validation seed: ${EVAL_SEED}" >&2; exit 2 ;;
esac

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/CL39_identity_crossing_dynamic_masks_20260827_r1"
SOURCE_DIR="${SOURCE_DIR_OVERRIDE:-source_seed${EVAL_SEED}}"
PROJECT_ROOT="${TASK_ROOT}/${SOURCE_DIR}/diffusion_template"
SOURCE_MANIFEST="${TASK_ROOT}/${SOURCE_DIR}_manifest.json"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
CL39_RUN="CL39_cosmic_null_key_confidence_router_24k_full96_r4"
CL39_SEALED_ROOT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${CL39_RUN}/diffusion_template/saved/${CL39_RUN}"
CL39_CONFIG="CL39_cosmic_null_key_confidence_router_24k"
CL39_COMET_KEY="b1ca0b3da679401c85b991f1bbdf0b2a"
CL39_CHECKPOINT_24K="${CL39_SEALED_ROOT}/checkpoint-epoch12.pth"
CL39_CHECKPOINT_24K_SHA256="74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07"
BBOX_DIR="${TASK_ROOT}/dynamic_bboxes/seed${EVAL_SEED}"
BBOX_BASE="${BBOX_DIR}/pm96_bboxes_seed${EVAL_SEED}.json"
AUTO_BBOX_JSON="${BBOX_DIR}/pm96_bboxes_seed${EVAL_SEED}_auto.json"
AUTO_BBOX_DEBUG="${TASK_ROOT}/auto_bbox_debug/seed${EVAL_SEED}"
export TASK_ROOT CL39_CONFIG CL39_COMET_KEY CL39_CHECKPOINT_24K CL39_CHECKPOINT_24K_SHA256
export BBOX_DIR BBOX_BASE AUTO_BBOX_JSON AUTO_BBOX_DEBUG

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  for candidate in /home/jovyan/miniconda3 /home/jovyan/anaconda3 /opt/conda; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then CONDA_BASE="${candidate}"; break; fi
  done
fi
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" verify \
  --root "${PROJECT_ROOT}" --manifest "${SOURCE_MANIFEST}"
SOURCE_MANIFEST_SHA256="$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')"
export SOURCE_MANIFEST_SHA256
printf '%s  %s\n' "${CL39_CHECKPOINT_24K_SHA256}" "${CL39_CHECKPOINT_24K}" | sha256sum -c -

DATASET_LINK="${TASK_ROOT}/${SOURCE_DIR}/dataset_full"
test -L "${DATASET_LINK}"
test "$(readlink -f "${DATASET_LINK}")" = "${OWNER_ROOT}/rsrch_test/dataset_full"

export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export MPLCONFIGDIR="${TASK_ROOT}/runtime/${RUNTIME_LABEL:-seed_${EVAL_SEED}}/matplotlib"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_ALBUMENTATIONS_UPDATE=1

test -s "${PM_PATH}"
test -s "${COSMIC_LARGE_MANIFEST}"
test -s "${SUBJECT_V2_ID_EMBEDS}"
mkdir -p "${MPLCONFIGDIR}" "${TASK_ROOT}/saved" "${TASK_ROOT}/gates" "${BBOX_DIR}" "${AUTO_BBOX_DEBUG}"
if [[ ! -e "${BBOX_BASE}" ]]; then
  cp -p "${PROJECT_ROOT}/../dataset_full/val_dataset/pm96_bboxes_new.json" "${BBOX_BASE}"
fi
test -s "${BBOX_BASE}"

cd "${PROJECT_ROOT}"
python -m py_compile \
  src/model/photomaker_branched/lora2.py \
  src/model/photomaker_branched/attn_processor_cleanest.py \
  src/pipelines/photomaker_branched_clean.py \
  src/trainer/sdxl_trainers.py \
  tools/analysis/cl39_attention_capture.py
