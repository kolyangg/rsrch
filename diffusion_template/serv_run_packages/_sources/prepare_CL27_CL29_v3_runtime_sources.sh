#!/usr/bin/env bash
set -euo pipefail

OWNER="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
DESTROOT="${OWNER}/runtime_sources_cl27_cl29_v3"
OVERLAY="${OWNER}/cl27_cl29_v3_overlay.tar"
PYTHON="${OWNER}/conda_env/photomaker_NS/bin/python"
test "$(sha256sum "${OVERLAY}" | cut -d' ' -f1)" = \
  "986c04932d2877eb019ed120eba209239f6efa25946712d50754ecd4a15d9342"

declare -A bases=(
  [CL27_cosmic_frequency_surface_energy_24k_full96_r2]="${OWNER}/runtime_sources_cl27_cl29_v1/CL27_cosmic_frequency_surface_energy_24k_full96_r1"
  [CL28_cosmic_learnable_frequency_schedule_24k_full96_r3]="${OWNER}/runtime_sources_cl27_cl29_v2/CL28_cosmic_learnable_frequency_schedule_24k_full96_r2"
  [CL29_cosmic_lowband_causal_contrastive_24k_full96_r2]="${OWNER}/runtime_sources_cl27_cl29_v1/CL29_cosmic_lowband_causal_contrastive_24k_full96_r1"
)
for run in \
  CL27_cosmic_frequency_surface_energy_24k_full96_r2 \
  CL28_cosmic_learnable_frequency_schedule_24k_full96_r3 \
  CL29_cosmic_lowband_causal_contrastive_24k_full96_r2; do
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
    --source-revision "cl27-cl29-validation-map-fix-20260814-v3"
  "${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" verify \
    --root "${root}/diffusion_template" --manifest "${root}/source_manifest.json"
done
