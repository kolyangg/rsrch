#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
TORCH_CACHE="${OWNER_ROOT}/metric_cache/torch"
WORK_ROOT="${PROJECT_ROOT}/comet_data/face_quality_backfill"
RUN_LOG_DIR="${OWNER_ROOT}/logs/cosmic_large_face_quality_uniform_highest_1gpu"
STEPS="0,1000,2000,3000,4000,6000,8000,10000,12000,14000,16000,18000,20000"

case "cosmic_large_face_quality_uniform_highest_1gpu" in
  cosmic_large_face_quality_uniform_highest_1gpu)
    RUN_LABELS=(uniform highest)
    EXPERIMENT_KEYS=(
      ced6658b5b12484a9e003fe47cd0c2bf
      ddaeb234353b45a1ae6763f5d8a1c81f
    )
    ;;
  cosmic_large_face_quality_top3_minface_1gpu)
    RUN_LABELS=(top3softmax_r2 selfref_minface256)
    EXPERIMENT_KEYS=(
      b9751dc78c3b460c9b2ebc50d61b2036
      e44bd0b7434348fa868844e96d704fca
    )
    ;;
  *)
    echo "ERROR: unsupported package id: cosmic_large_face_quality_uniform_highest_1gpu" >&2
    exit 64
    ;;
esac

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
if [[ "${CONDA_PREFIX:-}" != "${CONDA_ENV}" ]]; then
  echo "ERROR: expected active Conda environment ${CONDA_ENV}, found ${CONDA_PREFIX:-<none>}" >&2
  exit 65
fi
cd "${PROJECT_ROOT}"
set -a
source .env
set +a
export PYTHONPATH="${PYIQA_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_CACHE}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

mkdir -p "${TORCH_CACHE}" "${WORK_ROOT}" "${RUN_LOG_DIR}"
python - <<'PY'
import importlib.metadata
import os
import sys

expected_python = (
    "/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/"
    "conda_env/photomaker_NS/bin/python"
)
assert sys.executable == expected_python, (sys.executable, expected_python)
assert importlib.metadata.version("pyiqa") == "0.1.15"
assert os.environ["PYTHONPATH"].split(":")[0].endswith(
    "/python_overlays/pyiqa-0.1.15"
)
print("FACE_QUALITY_ENV_VERIFIED", sys.executable)
PY

echo "c824486618c85c948849969be0681847450d0f1924d98a1cb1be939a6305d482  tools/comet/backfill_face_quality_metrics.py" |
  sha256sum --check --strict
echo "8225a0f009c5c5f588afef63ddcd6db3248e4b442940ce8e5bb65f5e32e78c3a  tools/inference/calculate_face_quality_metrics.py" |
  sha256sum --check --strict

# 27 Jul 2026 - AICODE-NOTE: Both one-GPU jobs share this lock only while
# populating immutable model weights. Scoring remains parallel and each job
# processes its two Comet runs sequentially.
exec 9>"${TORCH_CACHE}/.pyiqa-0.1.15-warm.lock"
flock -x 9
if [[ ! -f "${TORCH_CACHE}/.pyiqa-0.1.15-four-models-ready" ]]; then
  python - <<'PY'
import gc

import pyiqa
import torch

for metric_name in ("topiq_nr-face", "topiq_nr", "musiq", "maniqa-pipal"):
    print("FACE_QUALITY_MODEL_WARM_START", metric_name, flush=True)
    model = pyiqa.create_metric(metric_name, device="cuda")
    model.eval()
    if metric_name == "topiq_nr-face":
        try:
            with torch.inference_mode():
                model(torch.zeros(1, 3, 512, 512, device="cuda"))
        except AssertionError as error:
            print("FACE_QUALITY_EXPECTED_NO_FACE", str(error), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("FACE_QUALITY_MODEL_WARM_COMPLETE", metric_name, flush=True)
PY
  touch "${TORCH_CACHE}/.pyiqa-0.1.15-four-models-ready"
fi
flock -u 9

run_one() {
  local label="$1"
  local experiment_key="$2"
  local log_path="${RUN_LOG_DIR}/${label}.log"
  echo "FACE_QUALITY_RUN_START label=${label} key=${experiment_key}"
  if ! python tools/comet/backfill_face_quality_metrics.py \
    --experiment-key "${experiment_key}" \
    --expected-project jul-comet-large-testing \
    --steps "${STEPS}" \
    --images-per-step 96 \
    --metrics topiq_nr-face,topiq_nr,musiq,maniqa-pipal \
    --work-dir "${WORK_ROOT}/${experiment_key}" \
    --scorer-python "${CONDA_ENV}/bin/python" \
    --device cuda \
    --batch-size 8 \
    --crop-padding 0.25 \
    --crop-size 512 \
    --upload-per-image-asset \
    --write >"${log_path}" 2>&1; then
    echo "ERROR: face-quality backfill failed for ${label}; tail follows" >&2
    tail -100 "${log_path}" >&2
    return 1
  fi
  tail -8 "${log_path}"
  echo "FACE_QUALITY_RUN_COMPLETE label=${label} key=${experiment_key}"
}

for index in 0 1; do
  run_one "${RUN_LABELS[$index]}" "${EXPERIMENT_KEYS[$index]}"
done

echo "SERV_FACE_QUALITY_PAIR_COMPLETE package=cosmic_large_face_quality_uniform_highest_1gpu"
