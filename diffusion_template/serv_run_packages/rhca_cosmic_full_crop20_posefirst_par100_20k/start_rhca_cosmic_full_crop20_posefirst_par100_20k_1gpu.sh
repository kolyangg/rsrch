#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
ORT_OVERLAY="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/runtime_overlays/onnxruntime_gpu_1_20_1"
NVIDIA_LIB_ROOT="${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
RUN_ID="rhca_cosmic_full_crop20_posefirst_par100_20k"

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
cd "/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test"
if [[ "${CONDA_PREFIX:-}" != "${CONDA_ENV}" ]]; then
  echo "Wrong Conda environment: ${CONDA_PREFIX:-unset}" >&2
  exit 70
fi
if [[ "$(git branch --show-current)" != "test" ]]; then
  echo "Serv experiment requires the test branch" >&2
  exit 71
fi
cd "${PROJECT_ROOT}"

# 26 Jul 2026 - This arm is intentionally fail-closed until both the source
# full-96 visual gate and the matched long-run control's 4k gate are sealed.
python3 - <<'PY'
import json
from pathlib import Path

run_id = "rhca_cosmic_full_crop20_posefirst_par100_20k"
spec = json.loads(
    Path(f"serv_run_packages/{run_id}/{run_id}.json").read_text()
)
plan = spec["plan"]
source_gate = plan["source_4k_gate"]
control_gate = plan["long_run_control"]
if source_gate.get("visual_result") != "pass":
    raise RuntimeError("Target-native full-96 visual gate is not sealed")
if int(source_gate.get("per_identity_min_coherent") or 0) < 11:
    raise RuntimeError("Target-native full-96 per-identity gate is below 11/12")
expected_full96_key = source_gate.get("full96_comet_experiment_key")
if not expected_full96_key:
    raise RuntimeError("Target-native full-96 immutable Comet key is missing")
expected_checkpoint_sha = source_gate.get("checkpoint_sha256")
if not expected_checkpoint_sha:
    raise RuntimeError("Target-native source checkpoint SHA-256 is missing")
if control_gate.get("visual_result") != "pass":
    raise RuntimeError("Train-0/validate-1 step-4000 visual gate is not sealed")

source_record_path = (
    Path("saved") / source_gate["full96_run_name"] / "comet_experiment.json"
)
source_record = json.loads(source_record_path.read_text())
if source_record["comet"]["experiment_key"] != expected_full96_key:
    raise RuntimeError("Target-native full-96 Comet key does not match the gate")
source_plan = source_record.get("plan") or {}
source_provenance = source_plan.get("source") or {}
if source_provenance.get("comet_experiment_key") != source_gate["comet_experiment_key"]:
    raise RuntimeError("Target-native source training Comet key does not match")
source_result = source_record.get("validation_result") or {}
verification = source_result.get("comet_verification") or {}
if source_result.get("image_count") != 96 or not verification.get("verified"):
    raise RuntimeError("Target-native full-96 result is not finalized")
if source_result.get("intervention_label") != "pose_adapt_ratio=1.0":
    raise RuntimeError("Target-native full-96 intervention provenance is missing")
if source_result.get("checkpoint_sha256") != expected_checkpoint_sha:
    raise RuntimeError("Target-native source checkpoint SHA-256 does not match")

control_dir = Path("saved") / control_gate["run_name"]
control_record = json.loads((control_dir / "comet_experiment.json").read_text())
if control_record["comet"]["experiment_key"] != control_gate["comet_experiment_key"]:
    raise RuntimeError("Long-run control Comet key does not match the gate")
checkpoint = control_dir / control_gate["required_checkpoint"]
if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise RuntimeError("Long-run control step-4000 checkpoint is missing")
images = (
    control_dir
    / "val_images"
    / "manual_val"
    / f"step_{control_gate['required_step']}_batch_0"
)
if len(list(images.glob("*.png"))) != control_gate["required_images"]:
    raise RuntimeError("Long-run control step-4000 images are incomplete")
print("PAR100_LONG_RUN_GATES_OK", source_record_path, checkpoint)
PY

set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin"
export LIBSTDCXX_PATH="${LIBSTDCXX_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/nasilaev/lib/libstdc++.so.6.0.34}"
export COSMIC_LARGE_MANIFEST="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data/gathered_data_cosmic_large_filtered.json"
export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="${ORT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cublas/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RUN_NAME="${RUN_ID}"
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/serv_run_packages/${RUN_ID}/${RUN_ID}.json"
export EXPERIMENT_ARM="crop20_posefirst_par100_20k"

if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "Production training received CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}" >&2
  exit 73
fi
test -f "${NVIDIA_LIB_ROOT}/cudnn/lib/libcudnn_adv.so.9"
test -f "${NVIDIA_LIB_ROOT}/cublas/lib/libcublasLt.so.12"

python - <<'PY'
import onnxruntime as ort

if ort.__version__ != "1.20.1":
    raise RuntimeError(f"Unexpected ONNX Runtime version: {ort.__version__}")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is unavailable")
print("ONNX Runtime production provider:", ort.__version__, ort.get_available_providers())
PY

exec bash launchers/active/run_rhca_cosmic_large_adapted_1gpu.sh \
  dataloaders.train.num_workers=2
