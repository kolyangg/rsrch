#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
PROJECT_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
PYIQA_OVERLAY="${OWNER_ROOT}/python_overlays/pyiqa-0.1.15"
SOURCE_STEP_DIR="${OWNER_ROOT}/face_quality_staging/2026-07-27/ced6658b5b12484a9e003fe47cd0c2bf/images/step_000000"
WORK_DIR="${OWNER_ROOT}/face_quality_smoke_serv/cosmic_large_face_quality_uniform_step0_smoke_1gpu"
DEST_STEP_DIR="${WORK_DIR}/images/step_000000"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ "${CONDA_PREFIX:-}" == "${CONDA_ENV}" ]]
cd "${PROJECT_ROOT}"
set -a
source .env
set +a
export PYTHONPATH="${PYIQA_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OWNER_ROOT}/metric_cache/torch"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

[[ "$(find "${SOURCE_STEP_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 96 ]]
if [[ ! -d "${DEST_STEP_DIR}" ]]; then
  mkdir -p "${DEST_STEP_DIR}"
  cp -a "${SOURCE_STEP_DIR}/." "${DEST_STEP_DIR}/"
fi
[[ "$(find "${DEST_STEP_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 96 ]]

python - <<'PY'
import importlib.metadata
import sys

assert sys.executable == (
    "/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/"
    "conda_env/photomaker_NS/bin/python"
)
assert importlib.metadata.version("pyiqa") == "0.1.15"
print("SERV_FACE_QUALITY_SMOKE_ENV_VERIFIED")
PY
echo "c824486618c85c948849969be0681847450d0f1924d98a1cb1be939a6305d482  tools/comet/backfill_face_quality_metrics.py" |
  sha256sum --check --strict
echo "8225a0f009c5c5f588afef63ddcd6db3248e4b442940ce8e5bb65f5e32e78c3a  tools/inference/calculate_face_quality_metrics.py" |
  sha256sum --check --strict

# This is deliberately dry-run: it exercises the exact production scorer but
# cannot append or replace anything in the existing Comet validation run.
python tools/comet/backfill_face_quality_metrics.py \
  --experiment-key ced6658b5b12484a9e003fe47cd0c2bf \
  --expected-project jul-comet-large-testing \
  --steps 0 \
  --images-per-step 96 \
  --metrics topiq_nr-face,topiq_nr,musiq,maniqa-pipal \
  --work-dir "${WORK_DIR}" \
  --scorer-python "${CONDA_ENV}/bin/python" \
  --device cuda \
  --batch-size 8 \
  --crop-padding 0.25 \
  --crop-size 512 \
  --keep-images

python - "${WORK_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
result = json.loads(
    (work_dir / "results/face_quality_metrics.json").read_text(encoding="utf-8")
)
assert set(result["steps"]) == {"0"}
assert result["steps"]["0"]["image_count"] == 96
assert result["metric_backend"]["pyiqa_version"] == "0.1.15"
with (work_dir / "results/face_quality_per_image.csv").open(
    encoding="utf-8", newline=""
) as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 96
assert {int(row["step"]) for row in rows} == {0}
print("SERV_FACE_QUALITY_SMOKE_RESULTS_VERIFIED images=96 step=0")
PY

echo "SERV_FACE_QUALITY_SMOKE_COMPLETE"
