#!/usr/bin/env bash
# id_grid_pulid.sh - Run id_grid with PuLID backend
set -euo pipefail

# Same interface as PhotoMaker's id_grid.sh
CFG=${1:-id_grid_full.json}
IMAGE_FOLDER=${2:-../compare/testing/references}
PROMPT_FILE=${3:-../compare/testing/prompt_one2.txt}
CLASS_FILE=${4:-../compare/testing/classes_ref.json}
OUTPUT_ROOT=${5:-../compare/results/PULID_BRANCHED}
START_MERGE=${6:-10}
BRANCHED_START=${7:-15}
FORCE=${8:-0}

# Ensure PuLID is ready
pip uninstall -y pulid >/dev/null 2>&1 || true
pip install -e .

mkdir -p "$OUTPUT_ROOT"

# Parse configuration (same as PhotoMaker)
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
        print(",".join([
            str(col.get('slug','c')),
            str(col.get('face_embed_strategy','id_embeds')),
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

# Collect reference images
mapfile -t REF_IMAGES < <(find "$IMAGE_FOLDER" -maxdepth 1 -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort)
echo "[Init] Found ${#REF_IMAGES[@]} reference images in $IMAGE_FOLDER"

# Process each reference image
for ref in "${REF_IMAGES[@]}"; do
  ref_base="$(basename "${ref%.*}")"
  REF_DIR="$OUTPUT_ROOT/$ref_base"
  mkdir -p "$REF_DIR"
  TMP="$(mktemp -d)"; cp "$ref" "$TMP/"

  # Count needed generations
  GEN_NEED=0
  if [ "$HAS_NOBR" = "True" ]; then
    NB_DIR="$REF_DIR/col_nobr"
    NB_IMG="$NB_DIR/${ref_base}_p0_0.jpg"
    if [ "$FORCE" = "1" ] || [ ! -f "$NB_IMG" ]; then GEN_NEED=$((GEN_NEED+1)); fi
  fi
  
  for row in "${ROWS[@]}"; do
    IFS=, read -r RIDX MIX POSEF POSEPCT <<< "$row"
    for b in "${BCOLS[@]}"; do
      IFS=, read -r SLUG STRAT PM_START MERGE_START_C BR_START BR_MODE FPAR <<< "$b"
      OUT="$REF_DIR/r$(printf '%02d' "$RIDX")_c${SLUG}/mix${MIX}_pose$(printf '%02d' "$POSEPCT")"
      OUT_IMG="$OUT/${ref_base}_p0_0.jpg"
      if [ "$FORCE" = "1" ] || [ ! -f "$OUT_IMG" ]; then GEN_NEED=$((GEN_NEED+1)); fi
    done
  done
  
  GEN_DONE=0
  [ "$GEN_NEED" -eq 0 ] && echo "[OK] $ref_base: nothing to generate (all images exist under $REF_DIR)"

  # Generate non-branched baseline if needed
  if [ "$HAS_NOBR" = "True" ]; then
    NB_DIR="$REF_DIR/col_nobr"
    mkdir -p "$NB_DIR"
    NB_IMG="$NB_DIR/${ref_base}_p0_0.jpg"
    
    if [ "$FORCE" = "1" ]; then rm -f "$NB_IMG" 2>/dev/null || true; fi
    if [ ! -f "$NB_IMG" ]; then
      echo "[GEN $ref_base] generating NO-BRANCHED → $(($GEN_DONE+1))/$GEN_NEED"
      python3 pulid_generate_NS2.py \
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

  # Generate branched variants
  for row in "${ROWS[@]}"; do
    IFS=, read -r RIDX MIX POSEF POSEPCT <<< "$row"
    for b in "${BCOLS[@]}"; do
      IFS=, read -r SLUG STRAT PM_START MERGE_START_C BR_START BR_MODE FPAR <<< "$b"
      OUT="$REF_DIR/r$(printf '%02d' "$RIDX")_c${SLUG}/mix${MIX}_pose$(printf '%02d' "$POSEPCT")"
      mkdir -p "$OUT"

      OUT_IMG="$OUT/${ref_base}_p0_0.jpg"
      if [ "$FORCE" = "1" ]; then rm -f "$OUT_IMG" 2>/dev/null || true; fi
      if [ ! -f "$OUT_IMG" ]; then
        echo "[GEN $ref_base] r$(printf '%02d' "$RIDX")/c${SLUG} → $(($GEN_DONE+1))/$GEN_NEED"

        # Effective per-column overrides
        MERGE_EFF="${MERGE_START_C:-$START_MERGE}"
        PM_EFF="${PM_START:-$START_MERGE}"
        BR_EFF="${BR_START:-$BRANCHED_START}"

        python3 pulid_generate_NS2.py \
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
  
  rm -rf "$TMP"
done

echo "[Complete] All images generated in $OUTPUT_ROOT"
