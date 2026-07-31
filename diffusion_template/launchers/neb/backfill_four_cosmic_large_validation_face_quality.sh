#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/niko/rsrch/diffusion_template}"
CONDA_ROOT="${CONDA_ROOT:-/home/niko/miniconda3}"
PYIQA_VENV="${PYIQA_VENV:-/home/niko/rsrch/metric_envs/pyiqa-0.1.15}"
WORK_ROOT="${WORK_ROOT:-/home/niko/rsrch/face_quality_staging/2026-07-27}"
STEPS="0,1000,2000,3000,4000,6000,8000,10000,12000,14000,16000,18000,20000"

RUN_LABELS=(
  uniform
  highest
  top3softmax_r2
  selfref_minface256
)
EXPERIMENT_KEYS=(
  ced6658b5b12484a9e003fe47cd0c2bf
  ddaeb234353b45a1ae6763f5d8a1c81f
  b9751dc78c3b460c9b2ebc50d61b2036
  e44bd0b7434348fa868844e96d704fca
)

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate photomaker_NS
cd "${PROJECT_ROOT}"
set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES=0
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

[[ "${CONDA_PREFIX:-}" == "/home/niko/miniconda3/envs/photomaker_NS" ]]
[[ -x "${PYIQA_VENV}/bin/python" ]]
"${PYIQA_VENV}/bin/python" - <<'PY'
import importlib.metadata

assert importlib.metadata.version("pyiqa") == "0.1.15"
print("NEB_FACE_QUALITY_ENV_VERIFIED")
PY
echo "c824486618c85c948849969be0681847450d0f1924d98a1cb1be939a6305d482  tools/comet/backfill_face_quality_metrics.py" |
  sha256sum --check --strict
echo "8225a0f009c5c5f588afef63ddcd6db3248e4b442940ce8e5bb65f5e32e78c3a  tools/inference/calculate_face_quality_metrics.py" |
  sha256sum --check --strict

mkdir -p "${WORK_ROOT}/score_logs"
for index in 0 1 2 3; do
  label="${RUN_LABELS[$index]}"
  experiment_key="${EXPERIMENT_KEYS[$index]}"
  log_path="${WORK_ROOT}/score_logs/${label}.log"
  echo "NEB_FACE_QUALITY_RUN_START label=${label} key=${experiment_key}"
  if ! python tools/comet/backfill_face_quality_metrics.py \
    --experiment-key "${experiment_key}" \
    --expected-project jul-comet-large-testing \
    --steps "${STEPS}" \
    --images-per-step 96 \
    --metrics topiq_nr-face,topiq_nr,musiq,maniqa-pipal \
    --work-dir "${WORK_ROOT}/${experiment_key}" \
    --scorer-python "${PYIQA_VENV}/bin/python" \
    --device cuda \
    --batch-size 8 \
    --crop-padding 0.25 \
    --crop-size 512 \
    --upload-per-image-asset \
    --write >"${log_path}" 2>&1; then
    echo "ERROR: Neb face-quality backfill failed for ${label}" >&2
    tail -120 "${log_path}" >&2
    exit 1
  fi
  tail -10 "${log_path}"
  echo "NEB_FACE_QUALITY_RUN_COMPLETE label=${label} key=${experiment_key}"
done

echo "NEB_FACE_QUALITY_FOUR_RUN_COMPLETE"
