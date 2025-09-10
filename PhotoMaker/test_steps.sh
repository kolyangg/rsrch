#!/usr/bin/env bash
set -euo pipefail

# --- Paths you already use (adjust if your tree differs) ---
IMAGE_FOLDER="../compare/testing/ref2"
PROMPT_FILE="../compare/testing/prompt_one2.txt"
CLASS_FILE="../compare/testing/classes_ref.json"

# Base results folder; each run gets a unique subdir under this
BASE_OUTDIR="../compare/results/PM_upgrade_sweeps"

# Optional: JSON with custom scenarios as argv[1]
# JSON format:
# {
#   "base_output_dir": "../compare/results/PM_upgrade_custom",
#   "scenarios": [
#     {"start_merge_step": 10, "branched_attn_start_step": 15},
#     {"start_merge_step": 20, "branched_attn_start_step": 10}
#   ]
# }
JSON_CFG="${1:-}"

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${BASE_OUTDIR}/run_${timestamp}"
COLLECT_DIR="${RUN_ROOT}/_collected"
mkdir -p "${RUN_ROOT}" "${COLLECT_DIR}"

# If JSON provided, parse scenarios via Python (no jq dependency).
# Otherwise, use the two default sweeps:
#   - start_merge_step=10, branched_attn_start_step in {15,20,25,30}
#   - branched_attn_start_step=10, start_merge_step in {15,20,25,30}
gen_scenarios() {
  if [[ -n "${JSON_CFG}" && -f "${JSON_CFG}" ]]; then
    python3 - "$JSON_CFG" <<'PY'
import json, sys
cfg_path = sys.argv[1]
with open(cfg_path, 'r', encoding='utf-8') as f:
    j = json.load(f)
base = j.get("base_output_dir")
if base:
    print(f"__BASE__ {base}")
for sc in j.get("scenarios", []):
    s = sc.get("start_merge_step")
    b = sc.get("branched_attn_start_step")
    if s is None or b is None: continue
    print(f"{s} {b}")
PY
  else
    # defaults
    # for b in 15 20 25 30; do echo "10 ${b}"; done
    # for s in 15 20 25 30; do echo "${s} 10"; done
    for b in 15 30; do echo "10 ${b}"; done
    for s in 15; do echo "${s} 10"; done
  fi
}

# Apply optional base_output_dir from JSON (if present)
while read -r line; do
  if [[ "${line}" == __BASE__* ]]; then
    BASE_OUTDIR="$(echo "${line}" | awk '{print $2}')"
    RUN_ROOT="${BASE_OUTDIR}/run_${timestamp}"
    COLLECT_DIR="${RUN_ROOT}/_collected"
    mkdir -p "${RUN_ROOT}" "${COLLECT_DIR}"
  else
    SCEN_LIST+=("$line")
  fi
done < <(gen_scenarios)
: "${SCEN_LIST:?No scenarios parsed}"

# Run every scenario
i=0
MANIFEST="${RUN_ROOT}/summary_index.txt"
: > "${MANIFEST}"

for pair in "${SCEN_LIST[@]}"; do
  read -r S B <<<"${pair}"
  TAG="s${S}_b${B}"
  OUTDIR="${RUN_ROOT}/${TAG}"
  mkdir -p "${OUTDIR}"

  echo "[RUN $((++i))/${#SCEN_LIST[@]}] start_merge_step=${S}  branched_attn_start_step=${B}"
  echo "${TAG}: start_merge_step=${S}, branched_attn_start_step=${B}" >> "${MANIFEST}"

  # Run your current script entry with the two knobs varied
  python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
    --image_folder "${IMAGE_FOLDER}" \
    --prompt_file "${PROMPT_FILE}" \
    --class_file "${CLASS_FILE}" \
    --output_dir "${OUTDIR}" \
    --face_embed_strategy id_embeds \
    --save_heatmaps \
    --start_merge_step "${S}" \
    --branched_attn_start_step "${B}" \
    --use_branched_attention | tee "${OUTDIR}/run.log"

  # Collect all mask strips from this run (usually one per prompt/ref)
  shopt -s nullglob
  for f in "${OUTDIR}"/*_mask_evolution.jpg; do
    # Prefix the filename with the scenario tag to disambiguate
    cp -f "$f" "${COLLECT_DIR}/${TAG}__$(basename "$f")"
  done
done

# Build a single multi-page PDF with the strips as-is (no reformatting)
SUMMARY_PDF="${RUN_ROOT}/summary_mask_strips.pdf"
python3 - "${COLLECT_DIR}" "${SUMMARY_PDF}" <<'PY'
import sys, os, glob
from PIL import Image
collect_dir, out_pdf = sys.argv[1], sys.argv[2]
imgs = sorted(glob.glob(os.path.join(collect_dir, "*.jpg")))
if not imgs:
    raise SystemExit("No *_mask_evolution.jpg files found to aggregate.")
pages = [Image.open(p).convert("RGB") for p in imgs]
# Keep each strip exactly as-is; one strip per PDF page
first, rest = pages[0], pages[1:]
first.save(out_pdf, save_all=True, append_images=rest)
print(f"[OK] Summary PDF saved -> {out_pdf}")
PY

echo
echo "Done."
echo "Root folder: ${RUN_ROOT}"
echo "Index:       ${MANIFEST}"
echo "Summary PDF: ${SUMMARY_PDF}"
