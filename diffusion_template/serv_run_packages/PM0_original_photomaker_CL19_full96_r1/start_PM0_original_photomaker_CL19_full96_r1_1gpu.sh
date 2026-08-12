#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
RUN_ID="PM0_original_photomaker_CL19_full96_r1"
PACKAGE_DIR="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/PM0_original_photomaker_CL19_full96_r1"
SOURCE_RUNTIME="${OWNER_ROOT}/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2"
SOURCE_PROJECT="${SOURCE_RUNTIME}/diffusion_template"
SOURCE_MANIFEST="${SOURCE_RUNTIME}/source_manifest.json"
TARGET_PARENT="${OWNER_ROOT}/runtime_sources_baselines_v1"
TARGET_RUNTIME="${TARGET_PARENT}/${RUN_ID}"
PROJECT_ROOT="${TARGET_RUNTIME}/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

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

test -s "${SOURCE_MANIFEST}"
test -d "${SOURCE_PROJECT}"
test -s "${PACKAGE_DIR}/run_PM0_original_photomaker_CL19_full96_1gpu.sh"
test -s "${PACKAGE_DIR}/PM0_original_photomaker_CL19_full96.yaml"
test -s "${PACKAGE_DIR}/${RUN_ID}.json"

mkdir -p "${TARGET_PARENT}"
if [[ -e "${TARGET_RUNTIME}" ]]; then
  echo "Refusing to reuse baseline runtime: ${TARGET_RUNTIME}" >&2
  exit 76
fi
BUILD_RUNTIME="$(mktemp -d "${TARGET_PARENT}/.${RUN_ID}.building.XXXXXX")"
BUILD_PROJECT="${BUILD_RUNTIME}/diffusion_template"

python - "${SOURCE_PROJECT}" "${SOURCE_MANIFEST}" "${BUILD_PROJECT}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

source_root, manifest_path, target_root = map(Path, sys.argv[1:])
record = json.loads(manifest_path.read_text(encoding="utf-8"))
files = record.get("files")
if record.get("schema_version") != 1 or not isinstance(files, dict):
    raise RuntimeError("Invalid CL19 source manifest")

for relative, expected in sorted(files.items()):
    source = source_root / relative
    target = target_root / relative
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"CL19 source changed during copy: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied = hashlib.sha256(target.read_bytes()).hexdigest()
    if copied != expected:
        raise RuntimeError(f"Copied CL19 source hash mismatch: {relative}")
print(f"PM0_CL19_SOURCE_COPY_OK files={len(files)}")
PY

mkdir -p "${BUILD_PROJECT}/experiments/baselines"
install -m 750 \
  "${PACKAGE_DIR}/run_PM0_original_photomaker_CL19_full96_1gpu.sh" \
  "${BUILD_PROJECT}/launchers/active/run_PM0_original_photomaker_CL19_full96_1gpu.sh"
install -m 640 \
  "${PACKAGE_DIR}/PM0_original_photomaker_CL19_full96.yaml" \
  "${BUILD_PROJECT}/src/configs/PM0_original_photomaker_CL19_full96.yaml"
install -m 640 \
  "${PACKAGE_DIR}/${RUN_ID}.json" \
  "${BUILD_PROJECT}/experiments/baselines/${RUN_ID}.json"
ln -s "${OWNER_ROOT}/rsrch_test/dataset_full" "${BUILD_RUNTIME}/dataset_full"

python "${BUILD_PROJECT}/tools/verify_serv_source_manifest.py" build \
  --root "${BUILD_PROJECT}" \
  --output "${BUILD_RUNTIME}/source_manifest.json" \
  --source-revision "CL19-r2-sealed+PM0-plain-photomaker-validation-only-20260811"
python "${BUILD_PROJECT}/tools/verify_serv_source_manifest.py" verify \
  --root "${BUILD_PROJECT}" --manifest "${BUILD_RUNTIME}/source_manifest.json"
mv "${BUILD_RUNTIME}" "${TARGET_RUNTIME}"

cd "${PROJECT_ROOT}"
LIVE_ENV="${OWNER_ROOT}/rsrch_test/diffusion_template/.env"
test -s "${LIVE_ENV}"
set -a
# shellcheck disable=SC1090
source "${LIVE_ENV}"
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-${OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"

export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export COSMIC_LARGE_EXPECTED_MANIFEST_SHA256="8ba369ef2fdc0496a0d3d55afb5c7923c1aa299343a676ac6bc0d94f3a3a0196"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"

export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME="${RUN_ID}"
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/baselines/${RUN_ID}.json"
export PM_BASELINE_PROJECT_ROOT="${PROJECT_ROOT}"

test -s "${PM_PATH}"
test -s "${COSMIC_LARGE_MANIFEST}"
test -d "${COSMIC_LARGE_ROOT}"
test "$(sha256sum "${COSMIC_LARGE_MANIFEST}" | cut -d' ' -f1)" = \
  "${COSMIC_LARGE_EXPECTED_MANIFEST_SHA256}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"
if ! grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"; then
  echo "LIBSTDCXX_PATH does not expose GLIBCXX_3.4.32" >&2
  exit 74
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
ctypes.CDLL(str(Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"))
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is unavailable")
if importlib.metadata.version("pyiqa") != "0.1.15":
    raise RuntimeError("PyIQA 0.1.15 is required")
print("Serv runtime verified:", ort.__version__, ort.get_available_providers())
PY

exec bash launchers/active/run_PM0_original_photomaker_CL19_full96_1gpu.sh
