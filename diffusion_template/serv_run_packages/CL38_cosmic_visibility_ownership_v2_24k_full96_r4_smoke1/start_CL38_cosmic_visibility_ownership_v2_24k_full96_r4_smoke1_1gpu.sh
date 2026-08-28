#!/usr/bin/env bash
# Bounded diagnostic: prove a recovered CL38-CL44 arm crosses one optimizer step.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
SMOKE_RUN_ID="CL38_cosmic_visibility_ownership_v2_24k_full96_r4_smoke1"
SOURCE_RUN_ID="${SMOKE_RUN_ID%_smoke*}"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl38_cl45_v1/${SOURCE_RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
DATASET_FULL_ROOT="${OWNER_ROOT}/rsrch_test/dataset_full"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
SMOKE_OUTPUT_ROOT="${OWNER_ROOT}/smoke_outputs_cl38_cl45_v5/${SMOKE_RUN_ID}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then CONDA_BASE="${candidate}"; break; fi
  done
fi
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" verify \
  --root "${PROJECT_ROOT}" --manifest "${RUNTIME_ROOT}/source_manifest.json"
test -d "${DATASET_FULL_ROOT}/val_dataset/references"
if [[ -e "${RUNTIME_ROOT}/dataset_full" && ! -L "${RUNTIME_ROOT}/dataset_full" ]]; then
  echo "Refusing to replace non-symlink dataset path: ${RUNTIME_ROOT}/dataset_full" >&2
  exit 2
fi
ln -sfn "${DATASET_FULL_ROOT}" "${RUNTIME_ROOT}/dataset_full"
mkdir -p "${SMOKE_OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export CUDA_VISIBLE_DEVICES=0 ACCELERATE_NUM_PROCESSES=1
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HYDRA_FULL_ERROR=1 ACCELERATE_LOG_LEVEL=error TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error

case "${SOURCE_RUN_ID}" in
  CL38_*) CONFIG_NAME=CL38_cosmic_visibility_ownership_v2_24k ;;
  CL39_*) CONFIG_NAME=CL39_cosmic_null_key_confidence_router_24k ;;
  CL40_*) CONFIG_NAME=CL40_cosmic_identity_motion_projector_24k ;;
  CL41_*) CONFIG_NAME=CL41_cosmic_landmark_canonical_kv_24k ;;
  CL42_*) CONFIG_NAME=CL42_cosmic_component_token_memory_24k ;;
  CL43_*) CONFIG_NAME=CL43_cosmic_id_adaptive_modulation_24k ;;
  CL44_*) CONFIG_NAME=CL44_cosmic_semantic_window_gate_24k ;;
  *) echo "Unknown smoke source: ${SOURCE_RUN_ID}" >&2; exit 2 ;;
esac

if [[ "${SOURCE_RUN_ID}" == CL38_* ]]; then
  python tmp/cl38_processor_preflight.py
fi

exec accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py "--config-name=${CONFIG_NAME}" writer=console \
  "writer.run_name=${SMOKE_RUN_ID}" \
  "metrics.id_sim_subject_v2.id_embeds_pth=${SUBJECT_V2_ID_EMBEDS}" \
  "model.photomaker_path=${PM_PATH}" \
  trainer.skip_initial_validation=true trainer.n_epochs=1 trainer.epoch_len=1 \
  trainer.validation_interval_steps=2000 trainer.save_period=999 \
  trainer.face_quality.enabled=false "trainer.save_dir=${SMOKE_OUTPUT_ROOT}" \
  "hydra.run.dir=${SMOKE_OUTPUT_ROOT}/hydra" hydra.output_subdir=null
