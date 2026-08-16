#!/usr/bin/env bash
set -euo pipefail

OWNER="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
DESTROOT="${OWNER}/runtime_sources_cl21_cl26_v1"
FIX="${OWNER}/cl22_cl25_cl26_r2_fix.tar"
PYTHON="${OWNER}/conda_env/photomaker_NS/bin/python"
test "$(sha256sum "${FIX}" | cut -d' ' -f1)" = \
  "9dc6c5c152c970d38950905546832067bb12796e4db9c29eb12be76a71da2006"

pairs=(
  "CL22_cosmic_visibility_order_router_24k_full96_r1 CL22_cosmic_visibility_order_router_24k_full96_r2"
  "CL25_cosmic_low_noise_id_reward_4k_full96_r1 CL25_cosmic_low_noise_id_reward_4k_full96_r2"
  "CL26_cosmic_anchored_highres_roi_ba_24k_full96_r1 CL26_cosmic_anchored_highres_roi_ba_24k_full96_r2"
)
for pair in "${pairs[@]}"; do
  read -r source_run run <<<"${pair}"
  source="${DESTROOT}/${source_run}/diffusion_template"
  root="${DESTROOT}/${run}"
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
    --source-revision "cl19-cfeda7b5+cl21-cl26-20260813-r2-startupfix"
  "${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" \
    verify --root "${root}/diffusion_template" \
    --manifest "${root}/source_manifest.json"
done
