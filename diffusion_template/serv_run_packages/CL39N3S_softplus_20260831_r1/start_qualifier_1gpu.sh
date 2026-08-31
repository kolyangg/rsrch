#!/usr/bin/env bash
# Sealed CL39N3S activity qualifier; production is submitted only after this gate passes.
set -euo pipefail

OWNER_ROOT=/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev
PACKAGE_ROOT="${OWNER_ROOT}/analysis_jobs/CL39N3S_softplus_20260831_r1"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl39n3s_softplus_20260831_r1/CL39N3S_speed_smoke101_noval_r1"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

test ! -e "${RUNTIME_ROOT}"
test "$(sha256sum "${PACKAGE_ROOT}/source_cl39n3s_softplus_r1.tar.gz" | cut -d' ' -f1)" = \
  0be4fad9df802d35c52dc383ad3430acf0051268050645bdef6a082f39af6071
mkdir -p "${RUNTIME_ROOT}"
tar -xzf "${PACKAGE_ROOT}/source_cl39n3s_softplus_r1.tar.gz" -C "${RUNTIME_ROOT}"
ln -s "${OWNER_ROOT}/rsrch_test/dataset_full" "${RUNTIME_ROOT}/dataset_full"

if command -v conda >/dev/null 2>&1; then CONDA_BASE="$(conda info --base)";
elif [[ -n "${CONDA_EXE:-}" ]]; then CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")";
fi
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" verify \
  --root "${PROJECT_ROOT}" --manifest "${PACKAGE_ROOT}/source_manifest.json"
cd "${PROJECT_ROOT}"

set -a
# shellcheck disable=SC1090
source "${OWNER_ROOT}/rsrch_test/diffusion_template/.env"
set +a
export ENV_FILE=/dev/null PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export COSMIC_LARGE_ROOT=/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export ARCFACE_ONNX_PATH="${OWNER_ROOT}/analysis_jobs/CL39_wave_a_20260828_r1/assets/w600k_r50.onnx"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME=CL39N3S_speed_smoke101_noval_r1 CONFIG_NAME=CL39N3S_speed_smoke101_noval
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large/${RUN_NAME}.json"
export PHOTOMAKER_FACEANALYSIS_CPU=1 COMET_PROJECT=aug-large-ds CUDA_VISIBLE_DEVICES=0 ACCELERATE_NUM_PROCESSES=1
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch" HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export PYTHONPATH="${PROJECT_ROOT}:${OWNER_ROOT}/python_overlays/pyiqa-0.1.15:${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec bash launchers/active/run_CL39_next_wave_1gpu.sh
