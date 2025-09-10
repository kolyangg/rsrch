#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_mask_grid.sh scenarios_grid.json [image_folder] [prompt_file] [class_file] [output_root]
#
# Defaults mirror test_upgr2.sh paths.

JSON_FILE=${1:-test_steps2.json}
IMAGE_FOLDER=${2:-../compare/testing/ref2}
# IMAGE_FOLDER=${2:-../compare/testing/ref3}
# IMAGE_FOLDER=${2:-../compare/testing/ref1}
PROMPT_FILE=${3:-../compare/testing/prompt_one2.txt}
CLASS_FILE=${4:-../compare/testing/classes_ref.json}
# OUTPUT_ROOT=${5:-../compare/results/PM_upgrade1}
# OUTPUT_ROOT=${5:-../compare/results/PM_upgrade3}
OUTPUT_ROOT=${5:-../compare/results/PM_upgrade2}

# Optional: PDF scale factor (default 2x for higher-res pages)
PDF_SCALE=${6:-2}
export PDF_SCALE


# Ensure local Photomaker is installed (same as test_upgr2.sh)
pip uninstall -y photomaker >/dev/null 2>&1 || true
pip install -e .

mkdir -p "$OUTPUT_ROOT"

# Parse scenarios from JSON into "start,branched" lines
mapfile -t SCENARIOS < <(python3 - "$JSON_FILE" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
for r in j.get("runs", []):
    bsm = r.get("branched_start_mode","both").lower()
    print(f"{int(r['start_merge_step'])},{int(r['branched_attn_start_step'])},{bsm}")
PY
)

# Run each scenario into its own subfolder for clean aggregation
for pair in "${SCENARIOS[@]}"; do
  IFS=, read -r SMS BAS MODE <<< "$pair"
  SUBDIR="$OUTPUT_ROOT/sms${SMS}_bas${BAS}_${MODE}"
  mkdir -p "$SUBDIR"
  echo "[RUN] start_merge_step=$SMS  branched_attn_start_step=$BAS  branched_start_mode=$MODE  → $SUBDIR"



#   python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#     --image_folder "$IMAGE_FOLDER" \
#     --prompt_file "$PROMPT_FILE" \
#     --class_file "$CLASS_FILE" \
#     --output_dir "$SUBDIR" \
#     --face_embed_strategy id_embeds \
#     --save_heatmaps \
#     --start_merge_step "$SMS" \
#     --branched_attn_start_step "$BAS" \
#     --use_branched_attention

  python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
    --image_folder "$IMAGE_FOLDER" \
    --prompt_file "$PROMPT_FILE" \
    --class_file "$CLASS_FILE" \
    --output_dir "$SUBDIR" \
    --face_embed_strategy face \
    --save_heatmaps \
    --start_merge_step "$SMS" \
    --branched_attn_start_step "$BAS" \
    --branched_start_mode "$MODE" \
    --use_branched_attention \
    --auto_mask_ref \
    --pose_adapt_ratio 0.0 \
    --ca_mixing_for_face 0 \
    --use_id_embeds 0

done

# Aggregate *_mask_evolution.jpg strips → PDF with the *same* layout as add_masking.save_heatmap_pdf()
python3 - "$JSON_FILE" "$OUTPUT_ROOT" <<'PY'
import sys, json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

json_path, output_root = sys.argv[1], sys.argv[2]
scenarios = json.load(open(json_path))["runs"]

# Collect rows as (label_text, strip_path), preserving scenario order.
rows = []
for r in scenarios:
    sms = int(r["start_merge_step"])
    bas = int(r["branched_attn_start_step"])
    bsm = r.get("branched_start_mode","both").lower()
    label = f"\"start_merge_step\": {sms}, \"branched_attn_start_step\": {bas}, \"branched_start_mode\": \"{bsm}\""
    subdir = Path(output_root) / f"sms{sms}_bas{bas}_{bsm}"
    
    for fp in sorted(subdir.rglob("*_mask_evolution.jpg")):
        rows.append((label, fp))

if not rows:
    print("[DynamicMask] No heatmap pages to save")
    sys.exit(0)

# ── Layout constants (scaled for higher-res PDF) ─────────────────────────────
scale = float(os.environ.get("PDF_SCALE", "2"))
ROWS_PER_PAGE   = 10
LABEL_W_PX      = int(200 * scale)
ROW_H_PX        = int(150 * scale)
MARGIN          = int(20  * scale)
total_available = int(1400 * scale)
imgs_per_row    = 7
target_img_w    = int((total_available - LABEL_W_PX - 2 * MARGIN) / imgs_per_row)
PAGE_W          = LABEL_W_PX + target_img_w * imgs_per_row + 2 * MARGIN
 

try:
    # scale font as well for crisp labels
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        max(11, int(11 * scale))
    )

except Exception:
    font = ImageFont.load_default()

pages = []
for page_idx in range(0, len(rows), ROWS_PER_PAGE):
    page_rows = rows[page_idx:page_idx + ROWS_PER_PAGE]
    page_h = MARGIN + ROW_H_PX * len(page_rows)
    page = Image.new("RGB", (PAGE_W, page_h), "white")
    draw = ImageDraw.Draw(page)
    y = MARGIN

    for label_text, path in page_rows:
        strip = Image.open(path).convert("RGB")
        orig_w, orig_h = strip.size
        new_w = target_img_w * imgs_per_row
        new_h = int(orig_h * (new_w / orig_w))
        strip = strip.resize((new_w, new_h), Image.LANCZOS)

        # Left label (same wrapping/centering style as add_masking.py)
        max_width = LABEL_W_PX - 20
        words = label_text.replace("_", "_\n").replace(".", ".\n").split("\n")
        lines, current = [], ""
        for w in words:
            test_line = (current + w) if not current else (current + w)
            if font.getlength(test_line) <= max_width:
                current = test_line
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)

        line_height = max(12, int(12 * scale))
        text_block_height = len(lines) * line_height
        start_y = y + (ROW_H_PX - text_block_height) // 2
        for i, line in enumerate(lines):
            draw.text((10, start_y + i * line_height), line, font=font, fill="black")

        page.paste(strip, (LABEL_W_PX, y))
        y += ROW_H_PX

    pages.append(page)

out_dir = Path("hm_debug") / "hm_results"
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "pm_vs_ba_steps.pdf"   # matches the existing message format
pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
print(f"[DynamicMask] Saved heatmap PDF to {pdf_path} with {len(pages)} pages")
PY
