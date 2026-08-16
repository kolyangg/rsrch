#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
RUN_ID="ex_CL14_oldsource_speedcheck_r1"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
RUNTIME_ROOT="${OWNER_ROOT}/runtime_sources_cl1_cl3_v1/CL14_cosmic_joint_shadow_sa128_softmask_24k_full96_r1"
PROJECT_ROOT="${RUNTIME_ROOT}/diffusion_template"
SOURCE_MANIFEST="${RUNTIME_ROOT}/source_manifest.json"
PACKAGE_ROOT="${OWNER_ROOT}/rsrch_test/diffusion_template/serv_run_packages/${RUN_ID}"
SPEC_PATH="${PACKAGE_ROOT}/${RUN_ID}.json"
ORT_OVERLAY="${OWNER_ROOT}/runtime_overlays/onnxruntime_gpu_1_20_1"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"

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

cd "${PROJECT_ROOT}"
python - "${PROJECT_ROOT}" "${SOURCE_MANIFEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, manifest_path = map(Path, sys.argv[1:])
record = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = record["files"]
mutable_prefixes = ("hm_debug/", "outputs/")
ignored_top = {".env", "logs", "saved"}
ignored_parts = {"__pycache__", ".pytest_cache", ".mypy_cache"}

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

actual = {}
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    name = relative.as_posix()
    if relative.parts[0] in ignored_top or any(p in ignored_parts for p in relative.parts):
        continue
    if name.startswith(mutable_prefixes):
        continue
    if path.is_symlink():
        raise RuntimeError(f"Unexpected source symlink: {name}")
    if path.is_file():
        actual[name] = digest(path)
expected = {k: v for k, v in expected.items() if not k.startswith(mutable_prefixes)}
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
changed = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
if missing or extra or changed:
    raise RuntimeError(
        f"Immutable source verification failed: missing={missing[:8]}, "
        f"extra={extra[:8]}, changed={changed[:8]}"
    )
print(
    "Immutable historical CL14 source verified: "
    f"revision={record.get('source_revision')}, files={len(actual)}; "
    "excluded mutable outputs/, hm_debug/"
)
PY

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
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_NUM_PROCESSES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYIQA_OVERLAY}:${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export HF_HOME="${OWNER_ROOT}/model_cache/huggingface"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FACE_QUALITY_SCORER_PYTHON="${CONDA_ENV}/bin/python"
export HYDRA_FULL_ERROR=1
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error
export COMET_DISABLE_AUTO_LOGGING=1
export COMET_LOGGING_CONSOLE=ERROR

test -s "${PM_PATH}"
test -s "${COSMIC_LARGE_MANIFEST}"
test -s "${SPEC_PATH}"
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
grep -aFq "GLIBCXX_3.4.32" "${LIBSTDCXX_PATH}"
export LD_LIBRARY_PATH="$(dirname "${LIBSTDCXX_PATH}"):${LD_LIBRARY_PATH}"
export LD_PRELOAD="${LIBSTDCXX_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"

# Record the assigned device before timing so a hardware anomaly is auditable.
nvidia-smi --query-gpu=name,uuid,pstate,clocks.sm,clocks.mem,power.draw,power.limit \
  --format=csv,noheader

# shellcheck disable=SC1091
source launchers/lib/prepare_comet_record.sh
prepare_comet_record "${PROJECT_ROOT}" "${RUN_ID}" "${SPEC_PATH}"

exec accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  --config-name=CL14_cosmic_joint_shadow_sa128_softmask_24k \
  writer=cometml \
  "writer.run_name=${RUN_ID}" \
  writer.project_name=aug-large-ds \
  "model.photomaker_path=${PM_PATH}" \
  trainer.epoch_len=100 \
  trainer.n_epochs=1 \
  trainer.validation_interval_steps=0 \
  trainer.save_period=999 \
  trainer.face_quality.expected_images=12 \
  weights_only_save_period=0 \
  datasets.val.manual_val.limit=12
