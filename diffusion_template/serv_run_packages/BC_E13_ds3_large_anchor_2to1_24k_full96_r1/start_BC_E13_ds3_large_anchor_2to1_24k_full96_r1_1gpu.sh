#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
REMOTE_REPO="${SERV_REPO_ROOT:-${OWNER_ROOT}/runtime_worktrees/rsrch_test_BC_E13_dataset_20260809}"
PROJECT_ROOT="${REMOTE_REPO}/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
RUN_ID="BC_E13_ds3_large_anchor_2to1_24k_full96_r1"

case "${RUN_ID}" in
  BC_E13_ds1_repeatdepth_balanced_24k_full96_r1)
    CONFIG_NAME="BC_E13_ds1_repeatdepth_balanced_24k"; DATASET_MODE="ds1"
    EXPECTED_SPEC_HASH="91543cc8fef7ae6b918764e84e4bf7fb12fff0dfcbb6346aa048cbca2d6d9473" ;;
  BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1)
    CONFIG_NAME="BC_E13_ds2_scene_target_canonical_ref_24k"; DATASET_MODE="ds2"
    EXPECTED_SPEC_HASH="1a6a7276439cae9d4b7e33b243363c4ac6993e3b54b83a4815c244a0f22bd2bd" ;;
  BC_E13_ds3_large_anchor_2to1_24k_full96_r1)
    CONFIG_NAME="BC_E13_ds3_large_anchor_2to1_24k"; DATASET_MODE="ds3"
    EXPECTED_SPEC_HASH="7872152b0c42e9fc0bab8d86244c85e9915b3844f46a1aa0bdfcf571dc6b0437" ;;
  *) echo "Unknown BC_E13 dataset RUN_ID: ${RUN_ID}" >&2; exit 69 ;;
esac

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
: "${CONDA_BASE:?Could not locate Conda}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REMOTE_REPO}"
if [[ "$(git branch --show-current)" != "test" ]]; then
  echo "BC_E13 dataset runtime must use branch test." >&2
  exit 70
fi
if [[ "$(git rev-parse HEAD)" != "ad194a026ab701dd979712d415c487dd536a4645" ]]; then
  echo "BC_E13 dataset runtime must use the audited BC_E13/E13 base commit." >&2
  exit 71
fi

# Hashes are filled from the reviewed local overlay before package generation.
declare -A REQUIRED_HASHES=(
  [diffusion_template/src/datasets/bc_e13_schedule_policy.py]="f4deaa83d4d396624129dae95295bd04eb72625f2e2d8405601482714b12c75b"
  [diffusion_template/src/datasets/big_celebs_e13_scheduled.py]="d449bbf3e7bc676eaa3d29636b61acec1a4f452d3393b54bbb43566ca875d02c"
  [diffusion_template/tools/datasets/build_bc_e13_dataset_schedule.py]="629a00ce250afb03fd29dd1f31e666d740598970f54eaeffd7eb87b249ed5199"
  [diffusion_template/tools/datasets/preflight_bc_e13_dataset_schedule.py]="f312002a3dc0a708b8d7ab1303d84658f09c924c2fcd1afe1ea51836ba17fb2b"
  [diffusion_template/tools/datasets/finalize_bc_e13_schedule_spec.py]="e13435bf3c63e9309926d8f688397f787e081ac0d2462c911612059f40630f33"
  [diffusion_template/tools/validate_BC_E13_dataset_experiments.py]="09004f69976db9aed2660d40d6795335d3bb1470a6fcd3be2e9c48644c89f1ab"
  [diffusion_template/src/configs/datasets/all_datasets.yaml]="474cfef661a3378e0138b37719b0a385e57d3fd2257035a5c9279148c5a3bb26"
  [diffusion_template/src/configs/BC_E13_big_celebs_joint_shadow_sa128_24k.yaml]="b36488704f2f5045d7ff45d62ba845b424551f9ff164d36621581ee18fb95d4f"
  [diffusion_template/src/configs/BC_E13_ds1_repeatdepth_balanced_24k.yaml]="2efc1d01dfbce8a6826edeff0ee9101c473d441dfce017552d8dc44ec17cc8ac"
  [diffusion_template/src/configs/BC_E13_ds2_scene_target_canonical_ref_24k.yaml]="856e70d4682a87c3c37048e44cf4597d3b8ad6232b2e086c90ab7134167d0200"
  [diffusion_template/src/configs/BC_E13_ds3_large_anchor_2to1_24k.yaml]="d96c7d8e1d0e4e20cfd6ba5f3d7caee7ab1881067068e8f8900d8832a16faa9f"
  [diffusion_template/launchers/active/run_BC_E13_dataset_experiments_24k_1gpu.sh]="4f2c93761b30a12865bfe4714f70c4eae602ad90b341b628cde96d9b48c7578b"
)
for relative_path in "${!REQUIRED_HASHES[@]}"; do
  actual="$(sha256sum "${relative_path}" | cut -d' ' -f1)"
  if [[ "${actual}" != "${REQUIRED_HASHES[${relative_path}]}" ]]; then
    echo "Runtime hash mismatch for ${relative_path}: ${actual}" >&2
    exit 72
  fi
done
CURRENT_SPEC="diffusion_template/experiments/big_celebs/${RUN_ID}.json"
if [[ "$(sha256sum "${CURRENT_SPEC}" | cut -d' ' -f1)" != "${EXPECTED_SPEC_HASH}" ]]; then
  echo "Runtime hash mismatch for ${CURRENT_SPEC}." >&2
  exit 72
fi

cd "${PROJECT_ROOT}"
if [[ ! -f .env ]]; then
  echo "Missing machine-local diffusion_template/.env" >&2
  exit 73
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null

export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-${OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
BIG_CELEBS_RELEASE_ROOT="${OWNER_ROOT}/datasets/bigcelebs/releases/v2"
export BIG_CELEBS_MANIFEST="${BIG_CELEBS_RELEASE_ROOT}/filtered_ids3_adj.json"
export BIG_CELEBS_IMAGES="${BIG_CELEBS_RELEASE_ROOT}/large_dataset"
export BIG_CELEBS_SEAL="${BIG_CELEBS_RELEASE_ROOT}/dataset_manifest.json"
export BIG_CELEBS_DOWNLOAD_LOG="${OWNER_ROOT}/datasets/dataset_tools/download_bigcelebs_v2.log"
export BIG_CELEBS_EXPECTED_MANIFEST_SHA256="f846b8cc8a4ce087c78130beee48a65f1b13560b63e42a9715cb5686526e5efa"
export LARGE_DATASET_MANIFEST="${OWNER_ROOT}/datasets/dataset_full/filtered_ids3_adj.json"
export LARGE_DATASET_IMAGES="${OWNER_ROOT}/datasets/dataset_full/large_dataset_adj/large_dataset"
export LARGE_DATASET_EXPECTED_MANIFEST_SHA256="0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50"
export FULL96_BBOX_MANUAL="${OWNER_ROOT}/datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"

export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
# Direct `python tools/...` entry points import the project `src` package.
# Keep the isolated runtime first while retaining the audited metric overlays.
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME="${RUN_ID}"
export CONFIG_NAME
export BC_E13_DATASET_MODE="${DATASET_MODE}"
export BC_E13_SCHEDULE_START_ROW=0
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/big_celebs/${RUN_ID}.json"
export COMET_PROJECT=aug-large-ds

test -s "${PM_PATH}"
test -s "${BIG_CELEBS_MANIFEST}"
test -d "${BIG_CELEBS_IMAGES}"
test -s "${BIG_CELEBS_SEAL}"
test -s "${BIG_CELEBS_DOWNLOAD_LOG}"
test -s "${FULL96_BBOX_MANUAL}"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${PROJECT_ROOT}/src/configs/${CONFIG_NAME}.yaml"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"
if [[ "${DATASET_MODE}" == "ds3" ]]; then
  test -s "${LARGE_DATASET_MANIFEST}"
  test -d "${LARGE_DATASET_IMAGES}"
fi
if ! grep -qF "BIGCELEBS_V2_DOWNLOAD_COMPLETE" "${BIG_CELEBS_DOWNLOAD_LOG}"; then
  echo "BigCelebs v2 has no terminal completion marker." >&2
  exit 74
fi
if ! grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"; then
  echo "LIBSTDCXX_PATH does not expose GLIBCXX_3.4.32." >&2
  exit 75
fi
export LD_LIBRARY_PATH="$(dirname "${LIBSTDCXX_PATH}"):${LD_LIBRARY_PATH}"
export LD_PRELOAD="${LIBSTDCXX_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"

python - <<'PY'
import ctypes
import importlib.metadata
from pathlib import Path
import onnxruntime as ort
if ort.__version__ != "1.20.1":
    raise RuntimeError(f"Unexpected ONNX Runtime version: {ort.__version__}")
provider = Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"
ctypes.CDLL(str(provider))
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is unavailable")
if importlib.metadata.version("pyiqa") != "0.1.15":
    raise RuntimeError("PyIQA 0.1.15 is required")
print("Serv runtime verified:", ort.__version__, ort.get_available_providers())
PY

# Build and seal the deterministic artifact in this run's isolated data
# directory. The complete path/policy/count scan finishes before Comet exists.
SCHEDULE_DIR="${OWNER_ROOT}/dataset_schedules/bc_e13_20260809/${RUN_ID}"
mkdir -p "${SCHEDULE_DIR}"
export BC_E13_SCHEDULE="${SCHEDULE_DIR}/train_48k_bs2.jsonl"
export BC_E13_SCHEDULE_SUMMARY="${SCHEDULE_DIR}/train_48k_bs2.summary.json"
BUILD_ARGS=(
  --mode "${DATASET_MODE}"
  --big-manifest "${BIG_CELEBS_MANIFEST}"
  --big-images-root "${BIG_CELEBS_IMAGES}"
  --big-manifest-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}"
  --output "${BC_E13_SCHEDULE}"
  --summary-output "${BC_E13_SCHEDULE_SUMMARY}"
)
if [[ "${DATASET_MODE}" == "ds3" ]]; then
  BUILD_ARGS+=(
    --large-manifest "${LARGE_DATASET_MANIFEST}"
    --large-images-root "${LARGE_DATASET_IMAGES}"
    --large-manifest-sha256 "${LARGE_DATASET_EXPECTED_MANIFEST_SHA256}"
  )
fi
python tools/datasets/build_bc_e13_dataset_schedule.py "${BUILD_ARGS[@]}"
export BC_E13_EXPECTED_SCHEDULE_SHA256="$(sha256sum "${BC_E13_SCHEDULE}" | cut -d' ' -f1)"
SUMMARY_SCHEDULE_SHA="$(${CONDA_ENV}/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["schedule"]["sha256"])' "${BC_E13_SCHEDULE_SUMMARY}")"
if [[ "${SUMMARY_SCHEDULE_SHA}" != "${BC_E13_EXPECTED_SCHEDULE_SHA256}" ]]; then
  echo "Generated schedule and summary hashes differ." >&2
  exit 76
fi
echo "Sealed ${DATASET_MODE} schedule: ${BC_E13_EXPECTED_SCHEDULE_SHA256}"
python tools/datasets/finalize_bc_e13_schedule_spec.py \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}" \
  --schedule "${BC_E13_SCHEDULE}" \
  --summary "${BC_E13_SCHEDULE_SUMMARY}" \
  --expected-mode "${DATASET_MODE}"

exec bash launchers/active/run_BC_E13_dataset_experiments_24k_1gpu.sh
