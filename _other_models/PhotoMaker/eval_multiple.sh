#!/usr/bin/env bash
set -euo pipefail

# Enable `conda activate` in non-interactive shells
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

conda activate photomaker


# Usage:
#   bash run_multi_pdfs.sh RUNS.json [image_folder] [prompt_file] [class_file] [output_root] [eval_repo]
#
# Notes:
# - Images are generated under: <output_root>/<run_name>/<ref_stem>/
# - Reuse: if <ref_stem>_p0_0.jpg exists in that folder, generation for that ref is skipped.
# - Metrics/PDF are built from a flattened "<run>/_flat" folder (no changes to your eval/pdf code).

JSON=${1:-eval_runs.json}
# If these args are omitted, fall back to JSON top-level fields
IMAGE_FOLDER=${2:-}
PROMPT_FILE=${3:-}
CLASS_FILE=${4:-}
OUTPUT_ROOT=${5:-}
EVAL_REPO=${6:-../persongen}   # must contain src/metrics/eval_NS2.py

# Ensure local package is installed (same as other scripts)
pip uninstall -y photomaker >/dev/null 2>&1 || true
pip install -e .


# Pull globals (and mask opts) from JSON when not provided via args
IFS='|' read -r G_IMAGE_FOLDER G_PROMPT_FILE G_CLASS_FILE G_OUTPUT_ROOT G_IMP_MASK_FOLDER G_USE_MASK_FOLDER G_USE_DYNAMIC_MASK G_FORCE_REPLACE < <(python3 - "$JSON" <<'PY'

import json,sys
cfg=json.load(open(sys.argv[1]))
def g(k, d=""): return cfg.get(k, d)
vals = [
  g("IMAGE_FOLDER",""),
  g("PROMPT_FILE",""),
  g("CLASS_FILE",""),
  g("OUTPUT_ROOT",""),
  g("import_mask_folder",""),
  int(g("use_mask_folder",0)),
  int(g("use_dynamic_mask",0)),
  int(g("force_replace_images",0)),
]
print("|".join(map(str, vals)))
PY
)
IMAGE_FOLDER=${IMAGE_FOLDER:-$G_IMAGE_FOLDER}
PROMPT_FILE=${PROMPT_FILE:-$G_PROMPT_FILE}
CLASS_FILE=${CLASS_FILE:-$G_CLASS_FILE}
OUTPUT_ROOT=${OUTPUT_ROOT:-$G_OUTPUT_ROOT}
: "${IMAGE_FOLDER:?Set IMAGE_FOLDER in JSON or pass as arg}"
: "${PROMPT_FILE:?Set PROMPT_FILE in JSON or pass as arg}"
: "${CLASS_FILE:?Set CLASS_FILE in JSON or pass as arg}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT in JSON or pass as arg}"
IMP_MASK_FOLDER="$G_IMP_MASK_FOLDER"
USE_MASK_FOLDER="$G_USE_MASK_FOLDER"
USE_DYNAMIC_MASK="$G_USE_DYNAMIC_MASK"
FORCE_REPLACE_IMAGES="$G_FORCE_REPLACE"
mkdir -p "$OUTPUT_ROOT"

# Parse runs → one line per run (fields separated by "|")
mapfile -t RUNS < <(python3 - "$JSON" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
for r in cfg.get("runs", []):
    name = r["name"]
    pdf  = r["pdf"]
    vals = [
        name, pdf,
        int(r.get("use_branched_attention", 1)),
        r.get("face_embed_strategy", "face"),
        int(r.get("use_id_embeds", 0)),
        int(r.get("merge_start_step", 10)),
        int(r.get("photomaker_start_step", 10)),
        int(r.get("branched_attn_start_step", 10)),
        int(r.get("ca_mixing_for_face", 0)),
        float(r.get("pose_adapt_ratio", 0.0)),
        int(r.get("force_par_before_pm", 0)),
        int(r.get("num_images_per_prompt", 1)),
    ]
    print("|".join(map(str, vals)))
PY
)

# Collect all references once
mapfile -t REF_IMAGES < <(find "$IMAGE_FOLDER" -maxdepth 1 -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort)
echo "[Init] Found ${#REF_IMAGES[@]} reference images in $IMAGE_FOLDER"

for line in "${RUNS[@]}"; do
  IFS='|' read -r NAME PDF USE_BR STRAT USE_ID MS PMS BAS MIX PAR FPBPM NIMG <<< "$line"
  RUN_DIR="$OUTPUT_ROOT/$NAME"
  mkdir -p "$RUN_DIR"


  # Pre-count how many gens are needed (for nicer logs), honor force flag
  if [ "${FORCE_REPLACE_IMAGES:-0}" -eq 1 ]; then
    GEN_NEED=${#REF_IMAGES[@]}
  else
    GEN_NEED=0
    for ref in "${REF_IMAGES[@]}"; do
      base="$(basename "${ref%.*}")"
      out_img="$RUN_DIR/$base/${base}_p0_0.jpg"
      [ ! -f "$out_img" ] && GEN_NEED=$((GEN_NEED+1))
    done
  fi


  GEN_DONE=0
  if [ "$GEN_NEED" -eq 0 ] && [ "${FORCE_REPLACE_IMAGES:-0}" -ne 1 ]; then
    echo "[OK] $NAME: all images already exist"
  fi

  # Generate per reference (reuse if present)
  for ref in "${REF_IMAGES[@]}"; do
    base="$(basename "${ref%.*}")"
    RDIR="$RUN_DIR/$base"
    if [ "${FORCE_REPLACE_IMAGES:-0}" -eq 1 ]; then rm -rf "$RDIR"; fi
    mkdir -p "$RDIR"
    OUT_IMG="$RDIR/${base}_p0_0.jpg"

    if [ -f "$OUT_IMG" ] && [ "${FORCE_REPLACE_IMAGES:-0}" -ne 1 ]; then
      echo "[SKIP $NAME/$base] exists: $OUT_IMG"
      continue
    fi

    echo "[GEN $NAME/$base] $(($GEN_DONE+1))/$GEN_NEED"
    TMP="$(mktemp -d)"
    cp "$ref" "$TMP/"

    # Build command
    CMD=(python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py
         --image_folder "$TMP"
         --prompt_file "$PROMPT_FILE"
         --class_file "$CLASS_FILE"
         --output_dir "$RDIR"
         --face_embed_strategy "$STRAT"
         --merge_start_step "$MS"
         --photomaker_start_step "$PMS"
         --branched_attn_start_step "$BAS"
         --pose_adapt_ratio "$PAR"
         --ca_mixing_for_face "$MIX"
         --force_par_before_pm "$FPBPM"
         --use_id_embeds "$USE_ID"
         --auto_mask_ref)

    # Mask controls from JSON (same for all runs)
    if [ -n "$IMP_MASK_FOLDER" ]; then
      CMD+=(--import_mask_folder "$IMP_MASK_FOLDER")
    fi
    CMD+=(--use_mask_folder "$USE_MASK_FOLDER")
    CMD+=(--use_dynamic_mask "$USE_DYNAMIC_MASK")
         

    if [ "$USE_BR" -eq 1 ]; then
      CMD+=(--use_branched_attention)
    else
      CMD+=(--no_branched_attention)
    fi

    if [ "$NIMG" -gt 1 ]; then
      CMD+=(--num_images_per_prompt "$NIMG")
    fi

    "${CMD[@]}"
    rm -rf "$TMP"
    GEN_DONE=$((GEN_DONE+1))
  done

  # Flatten all *_p*_* images for metrics/PDF (one flat folder per run)
  FLAT="$RUN_DIR/_flat"
  rm -rf "$FLAT"; mkdir -p "$FLAT"
  find "$RUN_DIR" -mindepth 2 -maxdepth 2 -type f -regextype posix-extended \
    -regex '.*_p[0-9]+_[0-9]+\.(jpg|jpeg|png|webp)$' -print0 | xargs -0 -I{} cp "{}" "$FLAT/"

  # Metrics (single CSV over the entire run; pdf_output4 expects a flat dir)
  MET_CSV="$RUN_DIR/metrics.csv"
  # Switch to metrics env → run → back to photomaker
  conda deactivate || true
  conda activate metrics
  python3 "$EVAL_REPO/src/metrics/eval_NS2.py" \
    --image_folder "$IMAGE_FOLDER" \
    --prompt_file "$PROMPT_FILE" \
    --new_images "$FLAT" \
    --class_file "$CLASS_FILE" \
    --out "$MET_CSV"
  conda deactivate
  conda activate photomaker

  # PDF (one PDF per run config)
  python3 ../compare/testing/pdf_output4.py \
    --image_folder "$IMAGE_FOLDER" \
    --prompt_file "$PROMPT_FILE" \
    --new_images "$FLAT" \
    --metrics_file "$MET_CSV" \
    --output_pdf "$PDF"

  echo "[DONE] $NAME → $(realpath "$PDF")"
done
