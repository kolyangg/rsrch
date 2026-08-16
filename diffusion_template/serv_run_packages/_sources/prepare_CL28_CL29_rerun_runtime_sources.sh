#!/usr/bin/env bash
set -euo pipefail

OWNER="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
DESTROOT="${OWNER}/runtime_sources_cl27_cl29_v5"
OVERLAY="${OWNER}/cl28_cl29_rerun_overlay.tar"
PYTHON="${OWNER}/conda_env/photomaker_NS/bin/python"
test "$(sha256sum "${OVERLAY}" | cut -d' ' -f1)" = \
  "d83b863d7f104e152fb32b86563a876b6e3d2cf72dca5003b959bd898be7df7d"

declare -A bases=(
  [CL28_cosmic_learnable_frequency_schedule_24k_full96_r4]="${OWNER}/runtime_sources_cl27_cl29_v3/CL28_cosmic_learnable_frequency_schedule_24k_full96_r3"
  [CL29_cosmic_lowband_causal_contrastive_24k_full96_r3]="${OWNER}/runtime_sources_cl27_cl29_v3/CL29_cosmic_lowband_causal_contrastive_24k_full96_r2"
)
for run in \
  CL28_cosmic_learnable_frequency_schedule_24k_full96_r4 \
  CL29_cosmic_lowband_causal_contrastive_24k_full96_r3; do
  base_root="${bases[${run}]}"
  base="${base_root}/diffusion_template"
  root="${DESTROOT}/${run}"
  test -d "${base}"
  test -s "${base_root}/source_manifest.json"
  "${PYTHON}" - "${base}" "${base_root}/source_manifest.json" <<'PY'
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
        raise RuntimeError(f"Changed sealed source: {relative}")
PY
  if [[ -e "${root}" ]]; then
    echo "Refusing existing runtime ${root}" >&2
    exit 73
  fi
  mkdir -p "${root}/diffusion_template"
  (
    cd "${base}"
    tar -cf - .env.example .gitignore README.md TOOLS.md infer.py train.py \
      bbox_utils experiments launchers src tools
  ) | (cd "${root}/diffusion_template" && tar -xf -)
  tar -xf "${OVERLAY}" -C "${root}/diffusion_template"
  ln -s "${OWNER}/rsrch_test/dataset_full" "${root}/dataset_full"
  "${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" build \
    --root "${root}/diffusion_template" --output "${root}/source_manifest.json" \
    --source-revision "cl28-cl29-training-transition-fix-20260814-v5"
  "${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" verify \
    --root "${root}/diffusion_template" --manifest "${root}/source_manifest.json"
done
