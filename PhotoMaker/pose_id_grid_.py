#!/usr/bin/env python3
import argparse, json, os, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---- args ----
ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--image_folder", required=True)
ap.add_argument("--prompt_file", required=True)
ap.add_argument("--class_file", required=True)
ap.add_argument("--out_root", required=True)
ap.add_argument("--start_merge_step", type=int, default=10)
ap.add_argument("--branched_attn_start_step", type=int, default=15)
args = ap.parse_args()

CFG = json.load(open(args.cfg))
ROWS = CFG["rows"]
COLS = CFG["columns"]

image_paths = sorted([p for p in Path(args.image_folder).glob("*") if p.is_file()])

# Layout tuned to match add_masking.save_heatmap_pdf() spacing
# (left label strip, row height, margins). We only place *final images* here.
LABEL_W_PX = 200
ROW_H_PX   = 170
MARGIN     = 20
COL_HEADER_H = 26

# The inference script saves final images as "{basename}_p{p}_{i}.jpg". We'll take p=0,i=0. :contentReference[oaicite:3]{index=3}
def pick_final(out_dir: Path, ref_base: str):
    # Prefer the exact "{ref}_p0_0.jpg", else first *.jpg in the folder.
    p = out_dir / f"{ref_base}_p0_0.jpg"
    if p.exists(): return p
    imgs = sorted(out_dir.glob("*.jpg"))
    return imgs[0] if imgs else None

# small run helper
def run_infer(tmp_dir: Path, out_dir: Path, ref_img: Path, col: dict, row: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "inference_scripts/inference_pmv2_seed_NS4_upd2.py",
        "--image_folder", str(tmp_dir),
        "--prompt_file",  args.prompt_file,
        "--class_file",   args.class_file,
        "--output_dir",   str(out_dir),
        "--start_merge_step", str(args.start_merge_step),
        "--branched_attn_start_step", str(args.branched_attn_start_step),
        "--auto_mask_ref"
    ]
    # Columns decide branched vs not, face strategy, and id usage
    if col.get("use_branched_attention", 0):
        cmd += ["--use_branched_attention", "--face_embed_strategy", col.get("face_embed_strategy","face")]
    else:
        cmd += ["--no_branched_attention", "--face_embed_strategy", "id_embeds"]

    # Row knobs (only matter when branched is on; harmless otherwise)
    cmd += ["--pose_adapt_ratio", str(row["pose_adapt_ratio"]),
            "--ca_mixing_for_face", "1" if row["ca_mixing_for_face"] else "0"]

    # ID injection toggle (we exposed this earlier)
    if "use_id_embeds" in col:
        cmd += ["--use_id_embeds", "1" if col["use_id_embeds"] else "0"]

    subprocess.run(cmd, check=True)

def build_page_for_ref(ref_img: Path, percell_paths, pdf_pages):
    # Determine cell size based on first available generated image
    sample = next((p for p in percell_paths.values() if isinstance(p, Path) and p and p.exists()), None)
    if sample is None:
        return
    img = Image.open(sample).convert("RGB")
    cell_w, cell_h = img.size

    # total width: left label + Ncols * cell_w + margins
    ncols, nrows = len(COLS), len(ROWS)
    page_w = LABEL_W_PX + ncols * cell_w + 2*MARGIN
    page_h = COL_HEADER_H + nrows * ROW_H_PX + 2*MARGIN

    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    try: font = ImageFont.load_default()
    except: font = None

    # Header: column labels
    x = LABEL_W_PX
    for c in COLS:
        label = c["label"]
        tw, th = draw.textbbox((0,0), label, font=font)[2:]
        draw.text((x + (cell_w - tw)//2, MARGIN + (COL_HEADER_H - th)//2), label, font=font, fill="black")
        x += cell_w

    # Rows
    y = MARGIN + COL_HEADER_H
    for r_idx, row in enumerate(ROWS):
        # left label text (match test_steps2.sh quoted keys style) :contentReference[oaicite:4]{index=4}
        draw.multiline_text((10, y + (ROW_H_PX - 12)//2), row["label"], font=font, fill="black")

        x = LABEL_W_PX
        for c_idx, col in enumerate(COLS):
            img_path = percell_paths.get((r_idx, c_idx))
            if isinstance(img_path, Path) and img_path and img_path.exists():
                im = Image.open(img_path).convert("RGB")
            else:
                im = Image.new("RGB", (cell_w, cell_h), "gray")
            # vertically center the cell inside ROW_H_PX if needed
            top = y + max(0, (ROW_H_PX - cell_h)//2)
            page.paste(im, (x, top))
            x += cell_w
        y += ROW_H_PX

    # Title band at top-left with the reference filename
    title = f"{ref_img.name}"
    draw.text((MARGIN, 2), title, font=font, fill="black")
    pdf_pages.append(page)

# ─────────────────────────────────────────────────────────────────────────────

out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)
pdf_pages = []

for ref_img in image_paths:
    ref_base = ref_img.stem
    ref_dir  = out_root / ref_base
    if ref_dir.exists():
        shutil.rmtree(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    # temp folder holding only this one ref (so the inference loops just this) :contentReference[oaicite:5]{index=5}
    tmp = ref_dir / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref_img, tmp / ref_img.name)

    # Column 1 (REF) has no run; we will paste the original image
    ref_pil = Image.open(ref_img).convert("RGB")

    # Column 2 (NO-BRANCHED) → one run reused for all rows
    nobr_dir = ref_dir / "col_nobr"
    run_infer(tmp, nobr_dir, ref_img, {"use_branched_attention":0}, {"pose_adapt_ratio":0.0, "ca_mixing_for_face":0})
    nobr_img = pick_final(nobr_dir, ref_base)

    # For the other 3 columns, run per row
    cell_map = {}
    for r_idx, row in enumerate(ROWS):
        for c_idx, col in enumerate(COLS):
            if col.get("kind") == "ref":
                cell_map[(r_idx, c_idx)] = ref_img   # we’ll paste ref_pil later
                continue
            if col.get("use_branched_attention", 0) == 0:
                cell_map[(r_idx, c_idx)] = nobr_img
                continue

            sub = ref_dir / f"r{r_idx:02d}_c{col['slug']}"
            # include row params in dir name to keep outputs separate
            sub = sub / f"mix{row['ca_mixing_for_face']}_pose{int(row['pose_adapt_ratio']*100):02d}"
            run_infer(tmp, sub, ref_img, col, row)
            cell_map[(r_idx, c_idx)] = pick_final(sub, ref_base)

    # Create page for this ref (paste the REF / NO-BRANCHED explicitly)
    # Determine cell size with any generated image (fallback to ref image)
    sample_any = next((p for p in cell_map.values() if isinstance(p, Path) and p and p.exists()), None)
    if sample_any is None:
        # fallback to ref size
        cell_w, cell_h = ref_pil.size
    else:
        with Image.open(sample_any) as s: cell_w, cell_h = s.size

    # Make sure REF and NOBR images match cell size
    ref_resized = ref_pil.resize((cell_w, cell_h), Image.LANCZOS)
    if nobr_img and nobr_img.exists():
        nobr_resized = Image.open(nobr_img).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
        nobr_resized.save(nobr_img)  # optional overwrite so paste uses same size

    # swap in resized ref for all REF cells
    for r_idx in range(len(ROWS)):
        cell_map[(r_idx, 0)] = ref_img  # marker; we paste resized version inside builder

    # Build page
    # Replace markers with actual resized images during paste
    # (builder re-opens Paths, so for REF we just use ref_resized directly)
    def _builder_with_ref():
        # Clone a map that points REF cells to a temporary saved image if needed
        tmp_ref = ref_dir / "_ref_resized.jpg"
        ref_resized.save(tmp_ref)
        return {(k if v != ref_img else k): (tmp_ref if v == ref_img else v) for k, v in cell_map.items()}

    build_page_for_ref(ref_img, _builder_with_ref(), pdf_pages)

    shutil.rmtree(tmp)

# Save multi-page PDF
if pdf_pages:
    pdf_path = out_root / "pose_id_grid.pdf"
    pdf_pages[0].save(pdf_path, save_all=True, append_images=pdf_pages[1:])
    print(f"[Grid] Saved PDF → {pdf_path}")
else:
    print("[Grid] Nothing to save")
