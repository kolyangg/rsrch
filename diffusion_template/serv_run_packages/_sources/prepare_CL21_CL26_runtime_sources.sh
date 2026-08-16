#!/usr/bin/env bash
set -euo pipefail

OWNER="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
BASE="${OWNER}/runtime_sources_cl15_cl20_v1/CL19_cosmic_true_soft_fullquery_router_24k_full96_r2/diffusion_template"
DESTROOT="${OWNER}/runtime_sources_cl21_cl26_v1"
OVERLAY="${OWNER}/cl21_cl26_overlay.tar"
PYTHON="${OWNER}/conda_env/photomaker_NS/bin/python"
REVISION="cl19-cfeda7b5+cl21-cl26-20260813-v1"

test "$(sha256sum "${OVERLAY}" | cut -d' ' -f1)" = \
  "31f9bbe81cb44b49d3943d485052296c9d58e38594a65302cc56887a027b3895"
mkdir -p "${DESTROOT}"

runs=(
  CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r1
  CL22_cosmic_visibility_order_router_24k_full96_r1
  CL23_cosmic_temporal_frequency_router_24k_full96_r1
  CL24_cosmic_pm_boundary_distill_24k_full96_r1
  CL25_cosmic_low_noise_id_reward_4k_full96_r1
  CL26_cosmic_anchored_highres_roi_ba_24k_full96_r1
)
for run in "${runs[@]}"; do
  root="${DESTROOT}/${run}"
  if [[ -e "${root}/source_manifest.json" ]]; then
    echo "Refusing existing runtime ${root}" >&2
    exit 73
  fi
  mkdir -p "${root}/diffusion_template"
  (
    cd "${BASE}"
    tar -cf - .env.example .gitignore README.md TOOLS.md infer.py train.py \
      bbox_utils experiments hm_debug launchers src tools
  ) | (cd "${root}/diffusion_template" && tar -xf -)
  tar -xf "${OVERLAY}" -C "${root}/diffusion_template"
  if [[ ! -e "${root}/dataset_full" ]]; then
    ln -s "${OWNER}/rsrch_test/dataset_full" "${root}/dataset_full"
  fi
  "${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" \
    build --root "${root}/diffusion_template" \
    --output "${root}/source_manifest.json" --source-revision "${REVISION}"
  "${PYTHON}" "${root}/diffusion_template/tools/verify_serv_source_manifest.py" \
    verify --root "${root}/diffusion_template" \
    --manifest "${root}/source_manifest.json"
done
