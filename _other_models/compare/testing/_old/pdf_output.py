#!/usr/bin/env python3
"""
Create a landscape A-4, multi-page PDF that shows
    ┌──────────┬──────── prompt 0 ┬ prompt 1 … prompt N
    │ reference│ 3 generated imgs│ …
    │   …      │      id/text sim│ …
from:

--image_folder   (reference images, one per person)
--new_images     (generated images)
--prompt_file    (TXT with one prompt per line)
--metrics_file   (CSV produced by eval_NS.py)
--output_pdf     (destination)

Images are scaled keeping aspect ratio; table headers repeat on every page.
"""

import argparse, math, textwrap
from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ───────────────────────── helpers ────────────────────────────────────────── #
def load_image(path, max_side=512):
    """Return a PIL.Image resized to fit into max_side×max_side keeping ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def add_header(fig, prompts, n_cols):
    """Write column headers: 'Reference', then the prompts."""
    fig.text(0.005, 0.97, "Reference", weight="bold", va="top")
    for c, prompt in enumerate(prompts, start=1):
        x = (c + 0.05) / n_cols         # rough centre
        wrapped = "\n".join(textwrap.wrap(prompt, 30))
        fig.text(x, 0.97, wrapped, ha="center", va="top", weight="bold", fontsize=8)


def cell_axes(fig, row, col, n_rows, n_cols, top_pad=0.05, h_pad=0.02, v_pad=0.02):
    """Return [left, bottom, width, height] for subplot at (row,col) in 0..n-1."""
    cell_w = (1 - h_pad) / n_cols
    cell_h = (1 - top_pad - v_pad) / n_rows
    left   = h_pad / 2 + col * cell_w
    bottom = 1 - top_pad - (row + 1) * cell_h
    return [left, bottom, cell_w * 0.95, cell_h * 0.9]


# ───────────────────────── main routine ───────────────────────────────────── #
def build_pdf(ref_dir, gen_dir, prompt_file, metrics_csv, out_pdf,
              rows_per_page=4, thumb_side=384):
    prompts = [ln.rstrip("\n") for ln in open(prompt_file, encoding="utf-8")]
    df      = pd.read_csv(metrics_csv)

    persons = sorted(df.person_id.unique())
    n_prompts = len(prompts)
    n_cols = n_prompts + 1

    # Pre-load reference images to avoid disk churn
    ref_cache = {p: load_image(Path(ref_dir) / f"{p}.jpg", thumb_side)
                 if (Path(ref_dir) / f"{p}.jpg").exists() else
                 load_image(Path(ref_dir) / f"{p}.png", thumb_side)
                 for p in persons}

    pdf = PdfPages(out_pdf)
    inches = (11.69, 8.27)  # landscape A4

    for person in persons:
        sub = df[df.person_id == person]

        # Determine how many rows (max #generated per prompt)
        max_rows = sub.groupby("prompt_idx").size().max()

        # Pad rows to multiple of rows_per_page
        total_rows = max_rows
        pages = math.ceil(total_rows / rows_per_page)

        # # Build matrix: rows × prompts with lists of file/metrics or None
        # cell = [[[] for _ in range(n_prompts)] for _ in range(max_rows)]
        # for _, r in sub.iterrows():
        #     cell[r["generated_file"].split("_")[-1].split(".")[0]  # img_id
        #          if isinstance(r["generated_file"], str) else 0]   # guard
        #     cell[r["generated_file"].split("_")[-1].split(".")[0]]
        #     # Actually simpler: use groupby later, but stay minimal here
        # # Simpler: create generator list per prompt
        gen_lists = {i: sub[sub.prompt_idx == i] for i in range(n_prompts)}

        for pg in range(pages):
            fig = plt.figure(figsize=inches)
            add_header(fig, prompts, n_cols)

            for r in range(rows_per_page):
                g_row = pg * rows_per_page + r
                if g_row >= max_rows:
                    break

                # first col = reference
                ax_pos = cell_axes(fig, r, 0, rows_per_page, n_cols)
                ax = fig.add_axes(ax_pos)
                ax.imshow(ref_cache[person])
                ax.axis("off")

                # generated columns
                for c in range(n_prompts):
                    prompt_df = gen_lists[c]
                    if g_row >= len(prompt_df):
                        continue
                    row_data = prompt_df.iloc[g_row]

                    ### DEBUG: show which file we’re about to load -------------
                    print(f"[DBG] person={person}  prompt_idx={c}  row={g_row}  "
                          f"file={row_data.generated_file}")

                    img_p   = Path(gen_dir) / row_data.generated_file
                    if not img_p.exists():
                        print(f"[DBG]   → file NOT FOUND on disk: {img_p}")
                        continue
                    
                    ### DEBUG: confirm path is good before loading -------------
                    print(f"[DBG]   → loading {img_p}")

                    g_img   = load_image(img_p, thumb_side)
                    ax = fig.add_axes(cell_axes(fig, r, c + 1,
                                                rows_per_page, n_cols))
                    ax.imshow(g_img)
                    ax.axis("off")
                    # metrics text
                    txt = f"{row_data.id_similarity:.2f}, {row_data.text_similarity:.2f}"
                    ax.text(0.5, -0.05, txt, ha="center", va="top",
                            transform=ax.transAxes, fontsize=6)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pdf.close()
    print("Saved", out_pdf)


# ───────────────────────── CLI ────────────────────────────────────────────── #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image_folder", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--new_images", required=True)     # kept for symmetry (path used for loading)
    p.add_argument("--metrics_file", required=True)
    p.add_argument("--output_pdf", required=True)
    args = p.parse_args()

    build_pdf(args.image_folder, args.new_images,
              args.prompt_file, args.metrics_file, args.output_pdf)
