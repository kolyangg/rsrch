#!/usr/bin/env python3
"""
Portrait-A4 report:
    rows  : prompts   (max 20 per page)
    cols  : N generated images for the same person (3 now)
    first column = prompt text (20 % width)
    column headers = reference image (one per generation column)
"""

import argparse, math, textwrap
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ───────────────────────── helpers ──────────────────────────────────────────
def load_image(path, max_side=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img

def prompt_axes(fig, row, prompts_area, cell_h):
    left, width = prompts_area
    bottom = 1 - 0.15 - (row+1)*cell_h        # 0.15 = header height
    return [left, bottom, width, cell_h*0.9]

def gen_axes(fig, row, col_idx, n_gen, cell_h, prompt_w):
    free = 1 - prompt_w - 0.02               # 0.02 margin
    gw   = free / n_gen
    left = prompt_w + 0.02 + col_idx*gw
    bottom = 1 - 0.15 - (row+1)*cell_h
    return [left, bottom, gw*0.95, cell_h*0.9]

def header_axes(col_idx, n_gen, prompt_w):
    free = 1 - prompt_w - 0.02
    gw   = free / n_gen
    left = prompt_w + 0.02 + col_idx*gw
    bottom = 1 - 0.13                        # little down from top
    return [left, bottom, gw*0.95, 0.11]

# ───────────────────────── main routine ─────────────────────────────────────
def build_pdf(ref_dir, gen_dir, prompt_file, metrics_csv, out_pdf,
              prompts_per_page=10, prompt_w_frac=0.20):

    prompts = [ln.rstrip("\n") for ln in open(prompt_file, encoding="utf-8")]
    df = pd.read_csv(metrics_csv)

    persons = sorted(df.person_id.unique())
    inches = (8.27, 11.69)  # portrait A4
    pdf = PdfPages(out_pdf)

    for person in persons:
        sub = df[df.person_id == person]

        # 3 generations per prompt assumed; derive dynamically
        n_gen = sub.groupby(["prompt_idx"]).size().max()

        # list-of-lists: generators[prompt_idx] → list rows sorted by img_id
        generators = {i: sub[sub.prompt_idx == i].sort_values("generated_file")
                      for i in range(len(prompts))}

        pages = math.ceil(len(prompts) / prompts_per_page)
        prompt_w = prompt_w_frac

        # reference thumbnail
        for ref_ext in (".jpg", ".png", ".jpeg", ".bmp"):
            rp = Path(ref_dir)/f"{person}{ref_ext}"
            if rp.exists():
                ref_thumb = load_image(rp, 256)
                ref_label = rp.stem              # <-- NEW (filename w/o ext)
                break
        else:
            ref_thumb = Image.new("RGB", (256,256), "gray")

        for pg in range(pages):
            fig = plt.figure(figsize=inches)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # header: prompt column title
            fig.text(0.01, 0.985, ref_label, weight="bold", va="top")

            # reference thumb over each generation column
            for c in range(n_gen):
                ax = fig.add_axes(header_axes(c, n_gen, prompt_w))
                ax.imshow(ref_thumb)
                ax.axis("off")

            # grid
            cell_h = (1 - 0.15 - 0.02) / prompts_per_page  # header+margin
            for r in range(prompts_per_page):
                global_r = pg * prompts_per_page + r
                if global_r >= len(prompts):
                    break
                prompt_txt = prompts[global_r]

                ax = fig.add_axes(prompt_axes(fig, r,
                                              (0.01, prompt_w-0.015), cell_h))
                ax.axis("off")
                ax.text(0, 0.5,
                        "\n".join(textwrap.wrap(prompt_txt, 40)),
                        va="center", fontsize=6)

                for c in range(n_gen):
                    g_list = generators.get(global_r)
                    if g_list is None or c >= len(g_list):
                        continue
                    row_data = g_list.iloc[c]
                    img_p = Path(gen_dir)/row_data.generated_file
                    if not img_p.exists():
                        continue
                    g_img = load_image(img_p, 384)

                    axg = fig.add_axes(gen_axes(fig, r, c, n_gen,
                                                cell_h, prompt_w))
                    axg.imshow(g_img)
                    axg.axis("off")
                    # metrics to right
                    axg.text(1.02, 0.5,
                             f"{row_data.id_similarity:.2f}\n{row_data.text_similarity:.2f}",
                             va="center", ha="left", transform=axg.transAxes,
                             fontsize=6)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pdf.close()
    print("Saved", out_pdf)

# ───────────────────────── CLI ──────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image_folder", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--new_images", required=True)
    p.add_argument("--metrics_file", required=True)
    p.add_argument("--output_pdf", required=True)
    args = p.parse_args()

    build_pdf(args.image_folder, args.new_images,
              args.prompt_file, args.metrics_file, args.output_pdf)
