#!/usr/bin/env bash
set -euo pipefail

OWNER="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
BASE_ROOT="${OWNER}/runtime_sources_cl27_cl29_v1/CL28_cosmic_learnable_frequency_schedule_24k_full96_r1"
BASE="${BASE_ROOT}/diffusion_template"
RUN="CL28_cosmic_learnable_frequency_schedule_24k_full96_r2"
ROOT="${OWNER}/runtime_sources_cl27_cl29_v2/${RUN}"
OVERLAY="${OWNER}/cl28_r2_overlay.tar"
PYTHON="${OWNER}/conda_env/photomaker_NS/bin/python"

test -d "${BASE}"
test -s "${BASE_ROOT}/source_manifest.json"
test "$(sha256sum "${OVERLAY}" | cut -d' ' -f1)" = \
  "941b1350b327754ae778c6e6169412f2d6e4d10e367ceaac88dd1db10a581b2f"
"${PYTHON}" - "${BASE}" "${BASE_ROOT}/source_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
record = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
copied_roots = {
    ".env.example", ".gitignore", "README.md", "TOOLS.md", "infer.py", "train.py",
    "bbox_utils", "experiments", "launchers", "src", "tools",
}
for relative, expected in record["files"].items():
    if relative.split("/", 1)[0] not in copied_roots:
        continue
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Changed sealed CL28 r1 source: {relative}")
PY
if [[ -e "${ROOT}" ]]; then
  echo "Refusing existing runtime ${ROOT}" >&2
  exit 73
fi
mkdir -p "${ROOT}/diffusion_template"
(
  cd "${BASE}"
  tar -cf - .env.example .gitignore README.md TOOLS.md infer.py train.py \
    bbox_utils experiments launchers src tools
) | (cd "${ROOT}/diffusion_template" && tar -xf -)
tar -xf "${OVERLAY}" -C "${ROOT}/diffusion_template"
ln -s "${OWNER}/rsrch_test/dataset_full" "${ROOT}/dataset_full"
"${PYTHON}" "${ROOT}/diffusion_template/tools/verify_serv_source_manifest.py" build \
  --root "${ROOT}/diffusion_template" --output "${ROOT}/source_manifest.json" \
  --source-revision "cl28-r1-category-contract-fix-20260814-v2"
"${PYTHON}" "${ROOT}/diffusion_template/tools/verify_serv_source_manifest.py" verify \
  --root "${ROOT}/diffusion_template" --manifest "${ROOT}/source_manifest.json"
