#!/usr/bin/env python3
# import argparse, json, os, shutil, subprocess
# from pathlib import Path
# from PIL import Image, ImageDraw, ImageFont

import argparse, json, re
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
import csv


# ---- args ----

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True, help="Grid config JSON with 'rows' and 'columns'")
ap.add_argument("--root", required=True, help="Root folder containing per-reference subfolders with generated images")
ap.add_argument("--pdf", default="../compare/results/POSE_ID_GRID/pose_id_grid_new.pdf", help="Output PDF path")
ap.add_argument("--rows-per-page", type=int, default=10)
ap.add_argument("--row-height", type=int, default=160, help="Row height in px (portrait-friendly)")
ap.add_argument("--label-width", type=int, default=200)
ap.add_argument("--content-width", type=int, default=900, help="Total width (px) available for image columns; ignored if --cell-width > 0")
ap.add_argument("--cell-width", type=int, default=0, help="Fixed width (px) PER column image; if >0 overrides --content-width")
ap.add_argument("--dpi", type=int, default=300, help="PDF resolution (DPI metadata)")
ap.add_argument("--scale", type=float, default=1.0, help="Scale all layout constants (label, margins, fonts, etc.)")
ap.add_argument("--ref-folder", default="../compare/testing/references", help="Folder with original reference images; picks REF by stem")
ap.add_argument("--metrics-root", type=str, default="", help="If set, read per-ref <root>/<ref>/_metrics.csv and overlay id_similarity")
ap.add_argument("--mark-enable", type=int, default=int(os.environ.get("GRID_MARK_ENABLE","1")),
                help="If 1, draw a green border around the top-K scored cells per page (excludes NO-BRANCHED and REF).")
ap.add_argument("--mark-topk", type=int, default=int(os.environ.get("GRID_MARK_TOPK","5")),
                help="How many top images to highlight per page (default 5).")

args = ap.parse_args()


# image_paths = sorted([p for p in Path(args.image_folder).glob("*") if p.is_file()])
CFG  = json.load(open(args.cfg))
ROWS = CFG["rows"]
COLS = CFG["columns"]

SCALE         = max(0.1, float(args.scale))
LABEL_W_PX    = int(round(args.label_width * SCALE))
ROW_H_PX      = int(round(args.row_height * SCALE))
MARGIN        = int(round(20 * SCALE))
COL_HEADER_H  = int(round(26 * SCALE))
ROWS_PER_PAGE = args.rows_per_page
CONTENT_W_PX  = int(round(args.content_width * SCALE))
CELL_W_FIXED  = int(round(args.cell_width * SCALE)) if args.cell_width > 0 else 0



def load_font() -> ImageFont.FreeTypeFont:
    size = max(9, int(round(11 * SCALE)))
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _pick_first_image(folder: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        imgs = sorted(folder.glob(ext))
        if imgs:
            return imgs[0]
    return None

def _pick_by_refbase(folder: Path, ref_base: str):
    p = folder / f"{ref_base}_p0_0.jpg"
    if p.exists():
        return p
    return _pick_first_image(folder)

def _scale_to_fit(img, max_w, max_h):
    w, h = img.size
    if w <= max_w and h <= max_h:
        return img
    s = min(max_w / float(w), max_h / float(h))
    return img.resize((max(1,int(w*s)), max(1,int(h*s))), Image.LANCZOS)

def _row_label_lines(row):
    # Ensure two separate lines exactly as requested
    cmf = row.get("ca_mixing_for_face")
    par = row.get("pose_adapt_ratio")
    return [f"ca_mixing_for_face: {cmf}", f"pose_adapt_ratio: {par}"]



def _extract_row_suffix(row):
    """Match folders we used earlier, e.g., mix{0/1}_pose{00..100}."""
    mix = int(bool(row.get("ca_mixing_for_face")))
    pose = int(round(float(row.get("pose_adapt_ratio", 0.0))*100))
    return f"mix{mix}_pose{pose:02d}"


def build_page_for_ref(ref_dir: Path, ref_base: str, percell_paths, pdf_pages, metrics_dict):
    # Define target cell width (prefer fixed per-column width if provided)
    ncols, nrows = len(COLS), len(ROWS)
    # cell_w = CELL_W_FIXED if CELL_W_FIXED > 0 else max(1, (CONTENT_W_PX // max(1, ncols)))

    # If no fixed cell width is provided, use row height as a natural portrait-ish cell width
    # so the page width auto-grows with the number of columns.
    cell_w = CELL_W_FIXED if CELL_W_FIXED > 0 else max(1, ROW_H_PX)

    cell_h = ROW_H_PX
    page_w = LABEL_W_PX + ncols * cell_w + 2*MARGIN
    page_h = COL_HEADER_H + nrows * ROW_H_PX + 2*MARGIN

    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    
    font = load_font()

    # ── pre-compute TOP-K (per page) among *branched* columns only ──────────
    top_keys = set()
    if int(getattr(args, "mark_enable", 1)) and metrics_dict:
        scored = []
        for (r_idx, c_idx), img_path in percell_paths.items():
            col = COLS[c_idx]
            # skip REF column and NO-BRANCHED columns
            if str(col.get("kind","")).lower() == "ref":
                continue
            if int(col.get("use_branched_attention", 0)) == 0:
                continue
            if isinstance(img_path, Path) and img_path and img_path.exists():
                rel_key = str(img_path.relative_to(ref_dir)).replace(os.sep, "/")
                sc = metrics_dict.get(rel_key)
                if isinstance(sc, (int, float)):
                    scored.append((float(sc), rel_key))
        if scored:
            scored.sort(key=lambda t: t[0], reverse=True)
            k = max(0, int(getattr(args, "mark_topk", 3)))
            for _, rk in scored[:k]:
                top_keys.add(rk)

    # Header: column labels
    x = LABEL_W_PX
    for c in COLS:
        label = c["label"]
        tw, th = draw.textbbox((0,0), label, font=font)[2:]
        draw.text((x + (cell_w - tw)//2, MARGIN + (COL_HEADER_H - th)//2), label, font=font, fill="black")
        x += cell_w


    y = MARGIN + COL_HEADER_H
    for r_idx, row in enumerate(ROWS):
        # Left label: two separate lines
        lines = _row_label_lines(row)
        # line_h = 12
        line_h = max(10, int(round(12 * SCALE)))
        block_h = len(lines)*line_h
        ty = y + (ROW_H_PX - block_h)//2
        for i, ln in enumerate(lines):
            draw.text((10, ty + i*line_h), ln, font=font, fill="black")

        x = LABEL_W_PX

        for c_idx, col in enumerate(COLS):
            img_path = percell_paths.get((r_idx, c_idx))
            if isinstance(img_path, Path) and img_path and img_path.exists():
                im = Image.open(img_path).convert("RGB")
                im = _scale_to_fit(im, cell_w, cell_h)
            else:
                im = Image.new("RGB", (cell_w, cell_h), "gray")
            top = y + (ROW_H_PX - im.size[1])//2
            page.paste(im, (x, top))

            # ── draw thick green border if this cell is in the per-page TOP-K ──
            if int(getattr(args, "mark_enable", 1)) and isinstance(img_path, Path) and img_path and img_path.exists():
                col_kind = str(col.get("kind","")).lower()
                if col_kind != "ref":
                    rel_key = str(img_path.relative_to(ref_dir)).replace(os.sep, "/")
                    if rel_key in top_keys:
                        # Thick border around the pasted image area
                        draw.rectangle(
                            [x, top, x + im.size[0] - 1, top + im.size[1] - 1],
                            outline=(0, 180, 0),
                            width=max(4, int(6 * SCALE))
                        )


            # ── overlay id_similarity in the top-right corner (generated images only)
            if metrics_dict is not None and isinstance(img_path, Path) and img_path and img_path.exists():
                if str(col.get("kind","")).lower() != "ref":
                    # key = relative path from ref_dir using forward slashes, e.g.
                    # "r00_cid_new/mix0_pose00/elon_p0_0.jpg" or "col_nobr/elon_p0_0.jpg"
                    rel_key = str(img_path.relative_to(ref_dir)).replace(os.sep, "/")
                    score = metrics_dict.get(rel_key)

                    if score is not None:
                        txt = f"{score:.3f}"
                        # bx0, by0, bx1, by1 = draw.textbbox((0,0), txt, font=font)
                        # tw, th = (bx1 - bx0), (by1 - by0)
                        # padx, pady = 4, 2
                        bx0, by0, bx1, by1 = draw.textbbox((0,0), txt, font=font)
                        # add slight height fudge to avoid vertical clipping
                        tw, th = (bx1 - bx0), (by1 - by0) + 2
                        # a touch more padding
                        padx, pady = 6, 4
                        bx = x + im.size[0] - (tw + 2*padx) - 3
                        by = top + 3
                        draw.rectangle([bx, by, bx + tw + 2*padx, by + th + 2*pady], fill="white", outline="black", width=1)
                        draw.text((bx + padx, by + pady), txt, font=font, fill="black")


            x += cell_w
        
        y += ROW_H_PX

    # Title band at top-left with the reference filename
    draw.text((MARGIN, 2), f"{ref_base}", font=font, fill="black")
    pdf_pages.append(page)

# ─────────────────────────────────────────────────────────────────────────────


root = Path(args.root)
pdf_pages = []

# Walk per-reference subfolders
for ref_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    ref_base = ref_dir.name

    # optional: load per-ref metrics for overlays
    metrics_dict = None
    if args.metrics_root:
        csv_path = Path(args.metrics_root) / ref_base / "_metrics.csv"
        if csv_path.exists():
            d = {}
            with open(csv_path, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    # 'generated_file' and 'id_similarity' are columns from eval_NS2.py
                    try:
                        d[row["generated_file"]] = float(row["id_similarity"])
                    except Exception:
                        pass
            metrics_dict = d if d else None


    # Prefer explicit reference folder by stem match: <ref_base>.(jpg|jpeg|png|webp)
    ref_img = None
    if args.ref_folder:
        from pathlib import Path as _P
        _rf = _P(args.ref_folder)
        for ext in ("jpg","jpeg","png","webp"):
            cand = _rf / f"{ref_base}.{ext}"
            if cand.exists():
                ref_img = cand
                break
    # Fallback to prior heuristic if not found
    if ref_img is None:
        for pat in ("*ref*.jpg","*ref*.png","*reference*.jpg","*reference*.png"):
            cand = next(iter(ref_dir.rglob(pat)), None)
            if cand: ref_img = cand; break
    if ref_img is None:
        ref_img = next(iter(ref_dir.rglob("*.jpg")), None) or next(iter(ref_dir.rglob("*.png")), None)


    # Try to find a NO-BRANCHED image (folder names we used earlier)
    nobr_img = None
    for cand_dir in [ref_dir/"col_nobr"] + [d for d in ref_dir.glob("*") if d.is_dir() and "nobr" in d.name]:
        if cand_dir.exists():
            nobr_img = _pick_by_refbase(cand_dir, ref_base) or _pick_first_image(cand_dir)
            if nobr_img: break

    # Build per-cell map by looking up files rather than running anything
    cell_map = {}
    for r_idx, row in enumerate(ROWS):
        row_suffix = _extract_row_suffix(row)  # mixX_poseYY
        for c_idx, col in enumerate(COLS):
            if col.get("kind") == "ref":
                # column to show the reference image (if we found one)
                cell_map[(r_idx, c_idx)] = ref_img
                continue
            if col.get("use_branched_attention", 0) == 0:
                cell_map[(r_idx, c_idx)] = nobr_img or ref_img
                continue
            # else: branched column; search folders like rXX_c{slug}/mixX_poseYY
            slug = col.get("slug","c")
            # Prefer exact structure if present:
            sub = ref_dir / f"r{r_idx:02d}_c{slug}" / row_suffix
            img_path = None
            if sub.exists():
                img_path = _pick_by_refbase(sub, ref_base) or _pick_first_image(sub)
            # Fallback: search by slug and row suffix anywhere
            if img_path is None:
                for cand in ref_dir.rglob(f"*{slug}*{row_suffix}*"):
                    if cand.is_dir():
                        img_path = _pick_by_refbase(cand, ref_base) or _pick_first_image(cand)
                        if img_path: break
            cell_map[(r_idx, c_idx)] = img_path

    build_page_for_ref(ref_dir, ref_base, cell_map, pdf_pages, metrics_dict)

# Save PDF (portrait pages)
if pdf_pages:
    out = Path(args.pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save with higher DPI metadata; pages are already larger in pixels due to SCALE / CELL_W_FIXED.
    pdf_pages[0].save(out, save_all=True, append_images=pdf_pages[1:], resolution=float(args.dpi))
    print(f"Saved PDF to {out} with {len(pdf_pages)} pages")
else:
    print("No pages to save")