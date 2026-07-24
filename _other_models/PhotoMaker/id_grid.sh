#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_pose_id_grid.sh [cfg_json] [image_folder] [prompt_file] [class_file] [output_root] [start_merge_step] [branched_start_step]
#
# Examples:
#   bash run_pose_id_grid.sh pose_id_grid.json ../compare/testing/ref5 ../compare/testing/prompt_one2.txt ../compare/testing/classes_ref.json ../compare/results/POSE_ID_GRID 10 15

CFG=${1:-id_grid2.json}
# IMAGE_FOLDER=${2:-../compare/testing/ref5}
IMAGE_FOLDER=${2:-../compare/testing/references}
# IMAGE_FOLDER=${2:-../compare/testing/ref2} # Keanu
# IMAGE_FOLDER=${2:-../compare/testing/ref3} # Eddie

PROMPT_FILE=${3:-../compare/testing/prompt_one2.txt}
CLASS_FILE=${4:-../compare/testing/classes_ref.json}
# OUTPUT_ROOT=${5:-../compare/results/POSE_ID_GRID_NEW_FIX_MASK} # Keanu
# OUTPUT_ROOT=${5:-../compare/results/POSE_ID_GRID_NEW_FIX_MASK_EDDIE} # Eddie
OUTPUT_ROOT=${5:-../compare/results/POSE_ID_GRID_NEW_FIX_MASK_FULL} # Full
START_MERGE=${6:-10}
BRANCHED_START=${7:-15}
FORCE=${8:-0}                       # 0 = skip if exists, 1 = force regenerate
EVAL_REPO=${9:-../persongen}        # repo that contains src/metrics/eval_NS2.py

# Same preamble as test_upgr2.sh
pip uninstall -y photomaker >/dev/null 2>&1 || true
pip install -e .

mkdir -p "$OUTPUT_ROOT"

# One-shot Python runner does everything: runs all variants and builds the PDF
# python3 pose_id_grid.py \
#   --cfg "$CFG" \
#   --image_folder "$IMAGE_FOLDER" \
#   --prompt_file "$PROMPT_FILE" \
#   --class_file "$CLASS_FILE" \
#   --out_root "$OUTPUT_ROOT" \
#   --start_merge_step "$START_MERGE" \
#   --branched_attn_start_step "$BRANCHED_START"


#
# 1) Generate all images per $CFG (rows × branched columns) for each ref,
#    plus a single NO-BRANCHED run.
# 2) Aggregate existing images → PDF using pose_id_grid.py (standalone).
#

# Parse rows (idx,mix,pose_float,pose_pct) and branched columns with optional per-column overrides

mapfile -t ROWS < <(python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
for i,r in enumerate(c["rows"]):
    mix=int(bool(r["ca_mixing_for_face"])); pose=float(r["pose_adapt_ratio"]); pct=int(round(pose*100))
    print(f"{i},{mix},{pose},{pct}")
PY
)
mapfile -t BCOLS < <(python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
for col in c["columns"]:
    if col.get("use_branched_attention",0)==1:
        # print: slug,face_embed_strategy,photomaker_start_step,merge_start_step,branched_attn_start_step,branched_start_mode,force_par_before_pm

        print(",".join([
            str(col.get('slug','c')),
            str(col.get('face_embed_strategy','face')),
            str(col.get('photomaker_start_step','')),
            str(col.get('merge_start_step','')),
            str(col.get('branched_attn_start_step','')),
            str(col.get('branched_start_mode','')),
            str(col.get('force_par_before_pm','')),
        ]))


PY
)
HAS_NOBR=$(python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print(any(col.get("use_branched_attention",0)==0 for col in c["columns"]))
PY
)

# shopt -s nullglob
# for ref in "$IMAGE_FOLDER"/*.{jpg,jpeg,png,webp}; do

# Collect reference images once and iterate that list (fix for early stop).
mapfile -t REF_IMAGES < <(find "$IMAGE_FOLDER" -maxdepth 1 -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort)
echo "[Init] Found ${#REF_IMAGES[@]} reference images in $IMAGE_FOLDER"


# shopt -s nullglob
# # Collect reference images first so we don't silently no-op when the glob is empty
# REF_IMAGES=( "$IMAGE_FOLDER"/*.{jpg,jpeg,png,webp} )
# if [ ${#REF_IMAGES[@]} -eq 0 ]; then
#   echo "[WARN] No reference images found in: $IMAGE_FOLDER  (accepted: jpg/jpeg/png/webp)"
#   echo "       Pass an explicit image folder as 2nd arg if needed."
# fi
# echo "[Init] Found ${#REF_IMAGES[@]} reference images in $IMAGE_FOLDER"
for ref in "${REF_IMAGES[@]}"; do
  ref_base="$(basename "${ref%.*}")"
  REF_DIR="$OUTPUT_ROOT/$ref_base"
  mkdir -p "$REF_DIR"
  TMP="$(mktemp -d)"; cp "$ref" "$TMP/"

  # ── Pre-count how many generations are needed (checks inside subfolders)
  GEN_NEED=0
  if [ "$HAS_NOBR" = "True" ]; then
    NB_DIR="$REF_DIR/col_nobr"
    NB_IMG="$NB_DIR/${ref_base}_p0_0.jpg"
    if [ "$FORCE" = "1" ] || [ ! -f "$NB_IMG" ]; then GEN_NEED=$((GEN_NEED+1)); fi
  fi
  for row in "${ROWS[@]}"; do
    IFS=, read -r RIDX MIX POSEF POSEPCT <<< "$row"
    for b in "${BCOLS[@]}"; do
      IFS=, read -r SLUG STRAT PM_START MERGE_START_C BR_START BR_MODE <<< "$b"
      OUT="$REF_DIR/r$(printf '%02d' "$RIDX")_c${SLUG}/mix${MIX}_pose$(printf '%02d' "$POSEPCT")"
      OUT_IMG="$OUT/${ref_base}_p0_0.jpg"
      if [ "$FORCE" = "1" ] || [ ! -f "$OUT_IMG" ]; then GEN_NEED=$((GEN_NEED+1)); fi
    done
  done
  GEN_DONE=0
  [ "$GEN_NEED" -eq 0 ] && echo "[OK] $ref_base: nothing to generate (all images exist under $REF_DIR)"


  if [ "$HAS_NOBR" = "True" ]; then
    NB_DIR="$REF_DIR/col_nobr"
    mkdir -p "$NB_DIR"

    NB_IMG="$NB_DIR/${ref_base}_p0_0.jpg"
    if [ "$FORCE" = "1" ]; then rm -f "$NB_IMG" 2>/dev/null || true; fi
    if [ ! -f "$NB_IMG" ]; then
      echo "[GEN $ref_base] generating NO-BRANCHED → $(($GEN_DONE+1))/$GEN_NEED  (left: $(($GEN_NEED-$GEN_DONE-1)))"
      python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
        --image_folder "$TMP" --prompt_file "$PROMPT_FILE" --class_file "$CLASS_FILE" \
        --output_dir "$NB_DIR" --face_embed_strategy id_embeds \
        --merge_start_step "$START_MERGE" --branched_attn_start_step "$BRANCHED_START" \
        --no_branched_attention --auto_mask_ref \
        --import_mask_folder "../compare/testing/gen_masks" \
        --use_mask_folder 1 \
        --use_dynamic_mask 0
        GEN_DONE=$((GEN_DONE+1))
    else
      echo "[SKIP $ref_base] exists: $NB_IMG"
    fi
  fi

  for row in "${ROWS[@]}"; do
    IFS=, read -r RIDX MIX POSEF POSEPCT <<< "$row"
    for b in "${BCOLS[@]}"; do
      IFS=, read -r SLUG STRAT PM_START MERGE_START_C BR_START BR_MODE FPAR <<< "$b"
      OUT="$REF_DIR/r$(printf '%02d' "$RIDX")_c${SLUG}/mix${MIX}_pose$(printf '%02d' "$POSEPCT")"
      mkdir -p "$OUT"

      OUT_IMG="$OUT/${ref_base}_p0_0.jpg"
      # if [ "$FORCE" = "1" ]; then rm -f "$OUT_IMG" 2>/dev/null || true; fi
      # if [ ! -f "$OUT_IMG" ]; then
      if [ "$FORCE" = "1" ]; then rm -f "$OUT_IMG" 2>/dev/null || true; fi
      if [ ! -f "$OUT_IMG" ]; then
        echo "[GEN $ref_base] generating r$(printf '%02d' "$RIDX")/c${SLUG} → $(($GEN_DONE+1))/$GEN_NEED  (left: $(($GEN_NEED-$GEN_DONE-1)))"

        # Effective per-column overrides (fallback to global defaults if empty)
        MERGE_EFF="${MERGE_START_C:-$START_MERGE}"
        PM_EFF="${PM_START:-$START_MERGE}"
        BR_EFF="${BR_START:-$BRANCHED_START}"

        python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
          --image_folder "$TMP" --prompt_file "$PROMPT_FILE" --class_file "$CLASS_FILE" \
          --output_dir "$OUT" \
          --face_embed_strategy "$STRAT" \
          --merge_start_step "$MERGE_EFF" \
          --photomaker_start_step "$PM_EFF" \
          --branched_attn_start_step "$BR_EFF" \
          ${BR_MODE:+--branched_start_mode "$BR_MODE"} \
          ${FPAR:+--force_par_before_pm $FPAR} \
          --use_branched_attention --auto_mask_ref \
          --import_mask_folder "../compare/testing/gen_masks" \
          --use_mask_folder 1 \
          --use_dynamic_mask 0 \
          --pose_adapt_ratio "$POSEF" --ca_mixing_for_face "$MIX"
          GEN_DONE=$((GEN_DONE+1))
        else
          echo "[SKIP $ref_base] exists: $OUT_IMG"
      fi
    done
  done


  # ── (1) Collect ALL *_p0_0.* into a single flat folder with class-leading names that FOLLOW the naming rule
  #     Use an increasing _{img_id} and keep a TSV map from that filename → "subfolder/file".
  FLAT="$REF_DIR/_metrics_flat"
  rm -rf "$FLAT"; mkdir -p "$FLAT"
  ABS_REF_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$REF_DIR")"
  MAP="$REF_DIR/_metrics_map_idx.tsv"; : > "$MAP"
  idx=0
  while IFS= read -r -d '' F; do
    # REL like: r00_cid_new/mix0_pose00/michael_p0_0.jpg  or  col_nobr/michael_p0_0.jpg
    REL="$(python3 -c 'import os,sys; p=sys.argv[1]; root=sys.argv[2]; print(os.path.relpath(p, root).replace(os.sep,"/"))' "$F" "$ABS_REF_DIR")"
    EXT="${F##*.}"
    NEW="${ref_base}_p0_${idx}.${EXT}"          # matches {img_basename}_p{p_idx}_{img_id}.jpg
    ln -sfn "$F" "$FLAT/$NEW"
    printf '%s\t%s\n' "$NEW" "$REL" >> "$MAP"   # map: new_name → subfolder/file
    idx=$((idx+1))
  done < <(find "$ABS_REF_DIR" -type f \( -name "*_p0_0.jpg" -o -name "*_p0_0.jpeg" -o -name "*_p0_0.png" -o -name "*_p0_0.webp" -o -name "*_p0_0.bmp" \) -print0)

  # absolute paths for cross-repo call
  ABS_IMG_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$IMAGE_FOLDER")"
  ABS_PROMPTS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PROMPT_FILE")"
  ABS_CLASS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CLASS_FILE")"
  ABS_FLAT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$FLAT")"
  ABS_OUT_CSV="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$REF_DIR/_metrics.csv")"

  pushd "$EVAL_REPO" >/dev/null


  # ── switch to metrics env for eval_NS2.py
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda deactivate || true
    conda activate metrics
  fi

  # ── (2) Run metrics ONCE over the flat folder; write temp CSV
  python3 src/metrics/eval_NS2.py \
    --image_folder "$ABS_IMG_DIR" \
    --prompt_file "$ABS_PROMPTS" \
    --new_images   "$ABS_FLAT" \
    --class_file   "$ABS_CLASS" \
    --out          "$ABS_OUT_CSV.tmp"
  # ── (3) Rewrite 'generated_file' using the TSV map so each row points to "subfolder/file"
  python3 - "$ABS_OUT_CSV.tmp" "$ABS_OUT_CSV" "$MAP" <<'PY'
import csv,sys
src,dst,map_path = sys.argv[1], sys.argv[2], sys.argv[3]
# Build name→relpath map from TSV
name2rel = {}
with open(map_path, encoding='utf-8') as f:
    for line in f:
        name, rel = line.rstrip('\n').split('\t', 1)
        name2rel[name] = rel
with open(src, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f); rows=list(r); fns=r.fieldnames
for row in rows:
    fn = row.get('generated_file','')   # e.g. michael_p0_17.jpg
    row['generated_file'] = name2rel.get(fn, fn)
with open(dst, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fns); w.writeheader(); w.writerows(rows)
PY

  # ── restore photomaker env after metrics run
  if command -v conda >/dev/null 2>&1; then
    conda deactivate || true
    conda activate photomaker
  fi
  popd >/dev/null

  rm -rf "$TMP"
done

# Aggregate → PDF (overlay metrics)
python3 pose_id_grid.py \
  --cfg "$CFG" \
  --root "$OUTPUT_ROOT" \
  --ref-folder "$IMAGE_FOLDER" \
  --metrics-root "$OUTPUT_ROOT" \
  --pdf "../compare/results/POSE_ID_GRID_NEW_FIX_MASK_FULL/pose_id_grid_new_fix_full.pdf"
