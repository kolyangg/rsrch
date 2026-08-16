#!/usr/bin/env bash
set -euo pipefail

OWNER="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
BASE_ROOT="${OWNER}/runtime_sources_cl21_cl26_v1/CL23_cosmic_temporal_frequency_router_24k_full96_r1"
BASE="${BASE_ROOT}/diffusion_template"
DESTROOT="${OWNER}/runtime_sources_cl27_cl29_v1"
OVERLAY="${OWNER}/cl27_cl29_overlay.tar"
PYTHON="${OWNER}/conda_env/photomaker_NS/bin/python"
REVISION="cl23-a9ec9c59+cl27-cl29-20260814-v1"

test -d "${BASE}"
test -s "${BASE_ROOT}/source_manifest.json"
# 14 Aug 2026 - Training writes hm_debug/ and outputs/ into sealed CL23 runtimes;
# verify every copied source byte against the original manifest and omit those artifacts.
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
checked = 0
for relative, expected in record["files"].items():
    if relative.split("/", 1)[0] not in copied_roots:
        continue
    path = root / relative
    if not path.is_file():
        raise RuntimeError(f"Missing sealed CL23 source: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Changed sealed CL23 source: {relative}")
    checked += 1
print(f"Verified immutable CL23 source subset: files={checked}")
PY
test "$(sha256sum "${OVERLAY}" | cut -d' ' -f1)" = \
  "800b167c4a4ac6b41fbb255506c986a0ed8aeb4bdd031b1746b305cc3e824cdc"
mkdir -p "${DESTROOT}"

runs=(
  CL27_cosmic_frequency_surface_energy_24k_full96_r1
  CL28_cosmic_learnable_frequency_schedule_24k_full96_r1
  CL29_cosmic_lowband_causal_contrastive_24k_full96_r1
)
for run in "${runs[@]}"; do
  root="${DESTROOT}/${run}"
  if [[ -e "${root}" ]]; then
    echo "Refusing existing runtime ${root}" >&2
    exit 73
  fi
  mkdir -p "${root}/diffusion_template"
  (
    cd "${BASE}"
    tar -cf - .env.example .gitignore README.md TOOLS.md infer.py train.py \
      bbox_utils experiments launchers src tools
  ) | (cd "${root}/diffusion_template" && tar -xf -)
  tar -xf "${OVERLAY}" -C "${root}/diffusion_template"
  ln -s "${OWNER}/rsrch_test/dataset_full" "${root}/dataset_full"
  "${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" \
    build --root "${root}/diffusion_template" \
    --output "${root}/source_manifest.json" --source-revision "${REVISION}"
  "${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" \
    verify --root "${root}/diffusion_template" \
    --manifest "${root}/source_manifest.json"
done
