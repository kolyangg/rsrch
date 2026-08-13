#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set the unique run identity}"
: "${CONFIG_ID:?Set the Hydra config identity}"
OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
case "${RUN_ID}" in
  # 12 Aug 2026 - Training optimization pair with the complete smoke config.
  CL14_CA_optimized_r11|CL14_CA_optimized_speed_smoke_r12) RUNTIME_SERIES="runtime_sources_cl14_ca_v23" ;;
  # 12 Aug 2026 - Training optimization pair using the proven one-batch warmup.
  CL14_CA_optimized_r10|CL14_CA_optimized_speed_smoke_r11) RUNTIME_SERIES="runtime_sources_cl14_ca_v22" ;;
  # 12 Aug 2026 - Training optimization with fused CA and skip-val CUDA rehome.
  CL14_CA_optimized_r9|CL14_CA_optimized_speed_smoke_r10) RUNTIME_SERIES="runtime_sources_cl14_ca_v21" ;;
  # 12 Aug 2026 - Training optimization with CL20's validation-only Eddie fix.
  CL14_CA_optimized_r8|CL14_CA_optimized_speed_smoke_r9) RUNTIME_SERIES="runtime_sources_cl14_ca_v20" ;;
  # 12 Aug 2026 - Training optimization derived from the proven live r7 source.
  CL14_CA_optimized_r7|CL14_CA_optimized_speed_smoke_r8) RUNTIME_SERIES="runtime_sources_cl14_ca_v19" ;;
  # 12 Aug 2026 - Training optimization series with exact defaults-off graph.
  CL14_CA_optimized_r6|CL14_CA_optimized_speed_smoke_r7) RUNTIME_SERIES="runtime_sources_cl14_ca_v18" ;;
  # 12 Aug 2026 - Training optimization series with one-GPU scalar-gather bypass.
  CL14_CA_optimized_r5|CL14_CA_optimized_speed_smoke_r6) RUNTIME_SERIES="runtime_sources_cl14_ca_v17" ;;
  # 12 Aug 2026 - Training optimization series retaining only safe scalar batching.
  CL14_CA_optimized_r4|CL14_CA_optimized_speed_smoke_r5) RUNTIME_SERIES="runtime_sources_cl14_ca_v16" ;;
  # 12 Aug 2026 - Training optimization series with CL14 loader and CL20 Eddie validation.
  CL14_CA_optimized_r3|CL14_CA_optimized_speed_smoke_r4) RUNTIME_SERIES="runtime_sources_cl14_ca_v15" ;;
  # 12 Aug 2026 - Training optimization smoke after restoring the sealed subject-v2 asset.
  CL14_CA_optimized_speed_smoke_r3) RUNTIME_SERIES="runtime_sources_cl14_ca_v14" ;;
  # 12 Aug 2026 - Training optimization source series with deferred prompt-mask indexing.
  CL14_CA_optimized_r2|CL14_CA_optimized_speed_smoke_r2) RUNTIME_SERIES="runtime_sources_cl14_ca_v13" ;;
  # 12 Aug 2026 - Training optimization source series for the measured pair.
  CL14_CA_optimized_r1|CL14_CA_optimized_speed_smoke_r1) RUNTIME_SERIES="runtime_sources_cl14_ca_v12" ;;
  CL14_CA_onebatch_smoke_r2) RUNTIME_SERIES="runtime_sources_cl14_ca_v11" ;;
  CL14_CA_onebatch_smoke_r1) RUNTIME_SERIES="runtime_sources_cl14_ca_v10" ;;
  CL14_CA_oneval_smoke_r1) RUNTIME_SERIES="runtime_sources_cl14_ca_v9" ;;
  CL14_CA_r7|CL14_CA_skipval_smoke_r5) RUNTIME_SERIES="runtime_sources_cl14_ca_v8" ;;
  CL14_CA_r6|CL14_CA_skipval_smoke_r4) RUNTIME_SERIES="runtime_sources_cl14_ca_v7" ;;
  *) RUNTIME_SERIES="runtime_sources_cl14_ca_v6" ;;
esac
RUNTIME_ROOT="${OWNER_ROOT}/${RUNTIME_SERIES}/${RUN_ID}"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
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
test -d "${PROJECT_ROOT}"
# CL14 parity: each run has its own immutable source, validation mount, and
# machine-local .env, matching the proven CL14 Serv package layout.
python "${PROJECT_ROOT}/tools/verify_serv_source_manifest.py" \
  verify --root "${PROJECT_ROOT}" --manifest "${SOURCE_MANIFEST}"
cd "${PROJECT_ROOT}"
if [[ ! -f .env ]]; then
  echo "Missing machine-local diffusion_template/.env" >&2
  exit 72
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="${OWNER_ROOT}/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-${OWNER_ROOT}/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
export FULL96_BBOX_MANUAL="${OWNER_ROOT}/datasets/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export SUBJECT_V2_ID_EMBEDS="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package/assets/id_embeds_manual_val_subject_v2.pth"
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export RUN_NAME="${RUN_ID}"
export CONFIG_NAME="${CONFIG_ID}"
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large/${RUN_ID}.json"
export COMET_PROJECT=aug-large-ds

test -s "${PM_PATH}"
test -s "${COSMIC_LARGE_MANIFEST}"
test -d "${COSMIC_LARGE_ROOT}"
test -s "${FULL96_BBOX_MANUAL}"
test "$(sha256sum "${SUBJECT_V2_ID_EMBEDS}" | cut -d' ' -f1)" = \
  "e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${PROJECT_ROOT}/src/configs/${CONFIG_NAME}.yaml"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${LIBSTDCXX_PATH}"
INSIGHTFACE_SOURCE="${OWNER_ROOT}/metric_cache/insightface/models/buffalo_l"
INSIGHTFACE_DEFAULT="${HOME}/.insightface/models/buffalo_l"
mkdir -p "${INSIGHTFACE_DEFAULT}"
# 12 Aug 2026 - AICODE-NOTE: Keep CL14's unchanged default InsightFace
# construction; complete any partial worker-home cache from the sealed copy.
cp -a "${INSIGHTFACE_SOURCE}/." "${INSIGHTFACE_DEFAULT}/"
insightface_sha="$({
  find "${INSIGHTFACE_SOURCE}" -maxdepth 1 -type f -printf '%P\n' | LC_ALL=C sort |
    while read -r relative_path; do
      printf '%s  %s\n' \
        "$(sha256sum "${INSIGHTFACE_DEFAULT}/${relative_path}" | cut -d' ' -f1)" \
        "${relative_path}"
    done
} | sha256sum | cut -d' ' -f1)"
test "${insightface_sha}" = "d50ec3e4730b9c16d6ef4867310b8ab48a18ca7e38d5e049ee614e2c60be5208"
if ! grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"; then
  echo "LIBSTDCXX_PATH does not expose GLIBCXX_3.4.32: ${LIBSTDCXX_PATH}" >&2
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
provider = Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"
ctypes.CDLL(str(provider))
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is unavailable")
if importlib.metadata.version("pyiqa") != "0.1.15":
    raise RuntimeError("PyIQA 0.1.15 is required")
print("Serv runtime verified:", ort.__version__, ort.get_available_providers())
PY

exec bash launchers/active/run_CL14_CA_24k_1gpu.sh
