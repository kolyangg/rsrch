#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
ORT_OVERLAY="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_overlays/onnxruntime_gpu_1_20_1"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
RUN_ID="rhca_cosmic_initial_selfref_minface256_continue20k"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test"
[[ "${CONDA_PREFIX:-}" == "${CONDA_ENV}" ]]
[[ "$(git branch --show-current)" == "test" ]]
cd "${PROJECT_ROOT}"

case "${RUN_ID}" in
  rhca_cosmic_initial_distinct_uniform_continue20k)
    export SOURCE_RUN="rhca_cosmic_initial_distinct_uniform_4k"
    export SOURCE_TRAIN_COMET_KEY="288ebfe3ccf74d5ea328a55b3abe31cb"
    export EVAL_RUN="${SOURCE_RUN}_full96_steps0_1k_2k_3k_4k"
    export EVAL_COMET_KEY="ced6658b5b12484a9e003fe47cd0c2bf"
    export DATASET_POLICY_ARM="distinct_uniform"
    ;;
  rhca_cosmic_initial_distinct_highest_continue20k)
    export SOURCE_RUN="rhca_cosmic_initial_distinct_highest_4k"
    export SOURCE_TRAIN_COMET_KEY="fc3dec2223e84d49aa7c711fda968135"
    export EVAL_RUN="${SOURCE_RUN}_full96_steps0_1k_2k_3k_4k"
    export EVAL_COMET_KEY="ddaeb234353b45a1ae6763f5d8a1c81f"
    export DATASET_POLICY_ARM="distinct_highest"
    ;;
  rhca_cosmic_initial_distinct_top3softmax_r2_continue20k)
    export SOURCE_RUN="rhca_cosmic_initial_distinct_top3softmax_4k_r2"
    export SOURCE_TRAIN_COMET_KEY="b7821337e24e49f388450c103553a9da"
    export EVAL_RUN="${SOURCE_RUN}_full96_steps0_1k_2k_3k_4k"
    export EVAL_COMET_KEY="b9751dc78c3b460c9b2ebc50d61b2036"
    export DATASET_POLICY_ARM="distinct_top3_softmax"
    ;;
  rhca_cosmic_initial_selfref_minface256_continue20k)
    export SOURCE_RUN="rhca_cosmic_initial_selfref_minface256_4k"
    export SOURCE_TRAIN_COMET_KEY="c6979abd46754e4ca43fae87df77eeff"
    export EVAL_RUN="${SOURCE_RUN}_full96_steps0_1k_2k_3k_4k"
    export EVAL_COMET_KEY="e44bd0b7434348fa868844e96d704fca"
    export DATASET_POLICY_ARM="target_min256"
    ;;
  *)
    echo "Unsupported continuation package: ${RUN_ID}" >&2
    exit 72
    ;;
esac

set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export COSMIC_INITIAL_IMAGES_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data/gathered_data_cosmic_large_filtered.json"
export FULL96_BBOX_MANUAL="${PROJECT_ROOT}/../dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export CONTINUATION_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large_continuation/${RUN_ID}.json"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cublas/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

[[ "${CUDA_LAUNCH_BLOCKING:-0}" == "0" ]]
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
python - <<'PY'
import onnxruntime as ort
assert ort.__version__ == "1.20.1", ort.__version__
assert "CUDAExecutionProvider" in ort.get_available_providers()
print("ONNX Runtime production provider:", ort.get_available_providers())
PY

exec bash launchers/active/run_rhca_cosmic_initial_usage_continue_20k_1gpu.sh
