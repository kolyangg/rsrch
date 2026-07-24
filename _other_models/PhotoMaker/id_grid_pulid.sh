#!/usr/bin/env bash
set -euo pipefail

# id_grid_pulid.sh — Mirror of id_grid.sh but drives PuLID via a new
# CLI script that reuses PhotoMaker’s branched attention + masking helpers.

# Usage:
#   bash id_grid_pulid.sh [cfg_json] [image_folder] [prompt_file] [class_file] [output_root] [start_merge_step] [branched_start_step]

CFG=${1:-id_grid2.json}
IMAGE_FOLDER=${2:-../compare/testing/references}
PROMPT_FILE=${3:-../compare/testing/prompt_one2.txt}
CLASS_FILE=${4:-../compare/testing/classes_ref.json}
OUTPUT_ROOT=${5:-../compare/results/POSE_ID_GRID_PULID}
START_MERGE=${6:-1}
BRANCHED_START=${7:-1}
FORCE=${8:-0}
EVAL_REPO=${9:-../persongen}

# Ensure both packages are importable
pip uninstall -y photomaker >/dev/null 2>&1 || true
pip install -e .
pip install -e ../PuLID

mkdir -p "$OUTPUT_ROOT"

# Parse rows/columns like original id_grid.sh
mapfile -t ROWS < <(python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
for i,r in enumerate(c["rows"]):
    mix=int(bool(r.get("ca_mixing_for_face",0)))
    pose=float(r.get("pose_adapt_ratio",0.0)); pct=int(round(pose*100))
    print(f"{i},{mix},{pose},{pct}")
PY
)

mapfile -t BCOLS < <(python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
for col in c["columns"]:
    if col.get("use_branched_attention",0)==1:
        print(",".join([
            str(col.get('slug','c')),
            str(col.get('photomaker_start_step','')),
            str(col.get('merge_start_step','')),
            str(col.get('branched_attn_start_step','')),
            str(col.get('branched_start_mode','')),
        ]))
PY
)

HAS_NOBR=$(python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print(any(col.get("use_branched_attention",0)==0 for col in c["columns"]))
PY
)

# Collect reference images
mapfile -t REF_IMAGES < <(find "$IMAGE_FOLDER" -maxdepth 1 -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort)
echo "[Init] Found ${#REF_IMAGES[@]} reference images in $IMAGE_FOLDER"

for ref in "${REF_IMAGES[@]}"; do
  ref_base="$(basename "${ref%.*}")"
  REF_DIR="$OUTPUT_ROOT/$ref_base"
  mkdir -p "$REF_DIR"
  TMP="$(mktemp -d)"; cp "$ref" "$TMP/"

  GEN_NEED=0
  if [ "$HAS_NOBR" = "True" ]; then
    NB_DIR="$REF_DIR/col_nobr"; NB_IMG="$NB_DIR/${ref_base}_p0_0.jpg"
    if [ "$FORCE" = "1" ] || [ ! -f "$NB_IMG" ]; then GEN_NEED=$((GEN_NEED+1)); fi
  fi
  for row in "${ROWS[@]}"; do
    IFS=, read -r RIDX MIX POSEF POSEPCT <<< "$row"
    for b in "${BCOLS[@]}"; do
      IFS=, read -r SLUG PM_START MERGE_START_C BR_START BR_MODE <<< "$b"
      OUT="$REF_DIR/r$(printf '%02d' "$RIDX")_c${SLUG}/mix${MIX}_pose$(printf '%02d' "$POSEPCT")"
      OUT_IMG="$OUT/${ref_base}_p0_0.jpg"
      if [ "$FORCE" = "1" ] || [ ! -f "$OUT_IMG" ]; then GEN_NEED=$((GEN_NEED+1)); fi
    done
  done
  GEN_DONE=0
  [ "$GEN_NEED" -eq 0 ] && echo "[OK] $ref_base: nothing to generate (all images exist under $REF_DIR)"

  # --- NO BRANCHED ---
  if [ "$HAS_NOBR" = "True" ]; then
    NB_DIR="$REF_DIR/col_nobr"; mkdir -p "$NB_DIR"
    NB_IMG="$NB_DIR/${ref_base}_p0_0.jpg"
    [ "$FORCE" = "1" ] && rm -f "$NB_IMG" 2>/dev/null || true
    if [ ! -f "$NB_IMG" ]; then
      echo "[GEN $ref_base] generating NO-BRANCHED → $(($GEN_DONE+1))/$GEN_NEED"
      python3 ../PuLID/inference/inference_pulid_seed_NS4_upd2.py \
        --image_folder "$TMP" --prompt_file "$PROMPT_FILE" --class_file "$CLASS_FILE" \
        --output_dir "$NB_DIR" --no_branched_attention \
        --use_dynamic_mask 0 --steps 4 --scale 1.2
      GEN_DONE=$((GEN_DONE+1))
    else
      echo "[SKIP $ref_base] exists: $NB_IMG"
    fi
  fi

  # --- BRANCHED variants ---
  for row in "${ROWS[@]}"; do
    IFS=, read -r RIDX MIX POSEF POSEPCT <<< "$row"
    for b in "${BCOLS[@]}"; do
      IFS=, read -r SLUG PM_START MERGE_START_C BR_START BR_MODE <<< "$b"
      OUT="$REF_DIR/r$(printf '%02d' "$RIDX")_c${SLUG}/mix${MIX}_pose$(printf '%02d' "$POSEPCT")"
      mkdir -p "$OUT"
      OUT_IMG="$OUT/${ref_base}_p0_0.jpg"
      [ "$FORCE" = "1" ] && rm -f "$OUT_IMG" 2>/dev/null || true
      if [ ! -f "$OUT_IMG" ]; then
        echo "[GEN $ref_base] generating r$(printf '%02d' "$RIDX")/c${SLUG} → $(($GEN_DONE+1))/$GEN_NEED"
        BR_EFF="${BR_START:-$BRANCHED_START}"
        python3 ../PuLID/inference/inference_pulid_seed_NS4_upd2.py \
          --image_folder "$TMP" --prompt_file "$PROMPT_FILE" --class_file "$CLASS_FILE" \
          --output_dir "$OUT" --use_branched_attention \
          --branched_attn_start_step "$BR_EFF" \
          --use_dynamic_mask 0 --steps 4 --scale 1.2
        GEN_DONE=$((GEN_DONE+1))
      else
        echo "[SKIP $ref_base] exists: $OUT_IMG"
      fi
    done
  done

  # Flatten + metrics (same as id_grid.sh)
  FLAT="$REF_DIR/_metrics_flat"; rm -rf "$FLAT"; mkdir -p "$FLAT"
  ABS_REF_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$REF_DIR")"
  MAP="$REF_DIR/_metrics_map_idx.tsv"; : > "$MAP"
  idx=0
  while IFS= read -r -d '' F; do
    REL="$(python3 -c 'import os,sys; p=sys.argv[1]; root=sys.argv[2]; print(os.path.relpath(p, root).replace(os.sep,"/"))' "$F" "$ABS_REF_DIR")"
    EXT="${F##*.}"; NEW="${ref_base}_p0_${idx}.${EXT}"
    ln -sfn "$F" "$FLAT/$NEW"
    printf '%s\t%s\n' "$NEW" "$REL" >> "$MAP"
    idx=$((idx+1))
  done < <(find "$ABS_REF_DIR" -type f \( -name "*_p0_0.jpg" -o -name "*_p0_0.jpeg" -o -name "*_p0_0.png" -o -name "*_p0_0.webp" -o -name "*_p0_0.bmp" \) -print0)

  ABS_IMG_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$IMAGE_FOLDER")"
  ABS_PROMPTS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PROMPT_FILE")"
  ABS_CLASS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CLASS_FILE")"
  ABS_FLAT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$FLAT")"
  ABS_OUT_CSV="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$REF_DIR/_metrics.csv")"

  pushd "$EVAL_REPO" >/dev/null
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"; conda deactivate || true; conda activate metrics
  fi
  python3 src/metrics/eval_NS2.py \
    --image_folder "$ABS_IMG_DIR" \
    --prompt_file "$ABS_PROMPTS" \
    --new_images   "$ABS_FLAT" \
    --class_file   "$ABS_CLASS" \
    --out          "$ABS_OUT_CSV.tmp"
  python3 - "$ABS_OUT_CSV.tmp" "$ABS_OUT_CSV" "$MAP" <<'PY'
import csv,sys
src,dst,map_path = sys.argv[1], sys.argv[2], sys.argv[3]
name2rel = {}
with open(map_path, encoding='utf-8') as f:
    for line in f:
        name, rel = line.rstrip('\n').split('\t', 1); name2rel[name] = rel
with open(src, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f); rows=list(r); fns=r.fieldnames
for row in rows:
    fn = row.get('generated_file',''); row['generated_file'] = name2rel.get(fn, fn)
with open(dst, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fns); w.writeheader(); w.writerows(rows)
PY
  if command -v conda >/dev/null 2>&1; then
    conda deactivate || true; conda activate photomaker
  fi
  popd >/dev/null

  rm -rf "$TMP"
done

# Aggregate PDF (reuse existing tool)
python3 pose_id_grid.py \
  --cfg "$CFG" \
  --root "$OUTPUT_ROOT" \
  --ref-folder "$IMAGE_FOLDER" \
  --metrics-root "$OUTPUT_ROOT" \
  --pdf "$OUTPUT_ROOT/pose_id_grid_pulid.pdf"

