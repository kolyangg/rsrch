#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/niko/rsrch/diffusion_template}"
CONDA_ROOT="${CONDA_ROOT:-/home/niko/miniconda3}"
PYIQA_VENV="${PYIQA_VENV:-/home/niko/rsrch/metric_envs/pyiqa-0.1.15}"
EXPERIMENT_KEY="${EXPERIMENT_KEY:-658d22341cf24accb5a3890869e76c28}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/comet_data/face_quality_backfill/${EXPERIMENT_KEY}}"
WRITE="${WRITE:-false}"
REUSE_RESULTS="${REUSE_RESULTS:-false}"
CLEANUP_LEGACY_LAYOUT="${CLEANUP_LEGACY_LAYOUT:-false}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate photomaker_NS
cd "${PROJECT_ROOT}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

set -a
source .env
set +a

if [[ ! -x "${PYIQA_VENV}/bin/python" ]]; then
  mkdir -p "$(dirname "${PYIQA_VENV}")"
  python -m venv --system-site-packages "${PYIQA_VENV}"
fi

if ! "${PYIQA_VENV}/bin/python" -c 'import importlib.metadata; assert importlib.metadata.version("pyiqa") == "0.1.15"' >/dev/null 2>&1; then
  "${PYIQA_VENV}/bin/python" -m pip install \
    --disable-pip-version-check \
    "pyiqa==0.1.15"
fi

extra_args=()
if [[ "${WRITE}" == "true" ]]; then
  extra_args+=(--write)
fi
if [[ "${REUSE_RESULTS}" == "true" ]]; then
  extra_args+=(--reuse-results)
fi
if [[ "${CLEANUP_LEGACY_LAYOUT}" == "true" ]]; then
  extra_args+=(--cleanup-legacy-layout)
fi

python tools/comet/backfill_face_quality_metrics.py \
  --experiment-key "${EXPERIMENT_KEY}" \
  --expected-project jul-comet-large-testing \
  --steps 0,1000,2000,3000,4000,6000,8000,10000,12000,14000,16000,18000,20000 \
  --images-per-step 96 \
  --metrics topiq_nr-face,topiq_nr,musiq,maniqa-pipal \
  --work-dir "${WORK_DIR}" \
  --scorer-python "${PYIQA_VENV}/bin/python" \
  --device cuda \
  --batch-size 8 \
  --crop-padding 0.25 \
  --crop-size 512 \
  "${extra_args[@]}"
