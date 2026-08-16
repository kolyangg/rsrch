#!/usr/bin/env bash
set -euo pipefail

OWNER="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
DESTROOT="${OWNER}/runtime_sources_cl21_cl26_v1"
SOURCE_RUN="CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r1"
RUN="CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r2"
FIX="${OWNER}/cl21_r2_identity_mask_fix.tar"
PYTHON="${OWNER}/conda_env/photomaker_NS/bin/python"

test "$(sha256sum "${FIX}" | cut -d' ' -f1)" = \
  "bc23a990ea6fd9f71a5914df9b240dfdddc6e0cc72f91bcd8c680416d1378aec"
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
  --source-revision "cl19-cfeda7b5+cl21-cl26-20260813-r2-identity-mask-fix"
"${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" \
  verify --root "${root}/diffusion_template" \
  --manifest "${root}/source_manifest.json"
