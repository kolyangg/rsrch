#!/usr/bin/env bash
set -euo pipefail

OWNER="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
DESTROOT="${OWNER}/runtime_sources_cl21_cl26_v1"
SOURCE_RUN="CL26_cosmic_anchored_highres_roi_ba_24k_full96_r2"
RUN="CL26_cosmic_anchored_highres_roi_ba_24k_full96_r3"
FIX="${OWNER}/cl26_r3_activation_dtype_fix.tar"
PYTHON="${OWNER}/conda_env/photomaker_NS/bin/python"

test "$(sha256sum "${FIX}" | cut -d' ' -f1)" = \
  "fe9c220da9b151518ffb709803df8498c14882a8aaffed1a0383455d62f2a4d8"
source="${DESTROOT}/${SOURCE_RUN}/diffusion_template"
root="${DESTROOT}/${RUN}"
if [[ -e "${root}" ]]; then
  echo "Refusing existing runtime ${root}" >&2
  exit 73
fi
mkdir -p "${root}/diffusion_template"
(
  cd "${source}"
  tar -cf - .env.example .gitignore README.md TOOLS.md infer.py train.py \
    bbox_utils experiments hm_debug launchers src tools
) | (cd "${root}/diffusion_template" && tar -xf -)
tar -xf "${FIX}" -C "${root}/diffusion_template"
ln -s "${OWNER}/rsrch_test/dataset_full" "${root}/dataset_full"
"${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" \
  build --root "${root}/diffusion_template" \
  --output "${root}/source_manifest.json" \
  --source-revision "cl19-cfeda7b5+cl21-cl26-20260813-r3-activation-dtype-fix"
"${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" \
  verify --root "${root}/diffusion_template" \
  --manifest "${root}/source_manifest.json"
