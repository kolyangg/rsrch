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

# def gen_axes(fig, row, col_idx, n_gen, cell_h, prompt_w):
#     free = 1 - prompt_w - 0.02               # 0.02 margin
def gen_axes(fig, row, col_idx, n_gen, cell_h, prompt_w, avg_w):
    free = 1 - prompt_w - avg_w - 0.03       # 0.03 total margins
    gw   = free / n_gen
    left = prompt_w + 0.02 + col_idx*gw
    bottom = 1 - 0.15 - (row+1)*cell_h
    return [left, bottom, gw*0.95, cell_h*0.9]

# def header_axes(col_idx, n_gen, prompt_w):
#     free = 1 - prompt_w - 0.02

def header_axes(col_idx, n_gen, prompt_w, avg_w):
    free = 1 - prompt_w - avg_w - 0.03
    gw   = free / n_gen
    left = prompt_w + 0.02 + col_idx*gw
    bottom = 1 - 0.13                        # little down from top
    return [left, bottom, gw*0.95, 0.11]

def avg_axes(fig, row, cell_h, prompt_w, avg_w):
    left   = 1 - avg_w - 0.01             # 0.01 right margin
    bottom = 1 - 0.15 - (row+1)*cell_h
    return [left, bottom, avg_w*0.95, cell_h*0.9]

# ───────────────────────── main routine ─────────────────────────────────────
def build_pdf(ref_dir, gen_dir, prompt_file, metrics_csv, out_pdf,
              prompts_per_page=10, prompt_w_frac=0.20,
              thumb_side=768, dpi=300):

    prompts = [ln.rstrip("\n") for ln in open(prompt_file, encoding="utf-8")]
    df = pd.read_csv(metrics_csv)

    persons = sorted(df.person_id.unique())
    inches = (8.27, 11.69)  # portrait A4
    pdf = PdfPages(out_pdf)

    for person in persons:
        sub = df[df.person_id == person]
        
        # overall means across *all* prompts / generations for this person
        overall_id   = sub.id_similarity.mean()
        overall_txt  = sub.text_similarity.mean()

        # 3 generations per prompt assumed; derive dynamically
        n_gen = sub.groupby(["prompt_idx"]).size().max()

        # list-of-lists: generators[prompt_idx] → list rows sorted by img_id
        generators = {i: sub[sub.prompt_idx == i].sort_values("generated_file")
                      for i in range(len(prompts))}

        pages = math.ceil(len(prompts) / prompts_per_page)
        prompt_w = prompt_w_frac
        avg_w    = 0.06                  # <─ NEW: 6 % for the “AVG” column


        # reference thumbnail
        for ref_ext in (".jpg", ".png", ".jpeg", ".bmp", ".webp"):
            rp = Path(ref_dir)/f"{person}{ref_ext}"
            if rp.exists():
                ref_thumb = load_image(rp, thumb_side//3)
                ref_label = rp.stem              # <-- NEW (filename w/o ext)
                break
        else:
            ref_thumb = Image.new("RGB", (256,256), "gray")

        for pg in range(pages):
            fig = plt.figure(figsize=inches)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # header: prompt column title
            fig.text(0.01, 0.985, ref_label, weight="bold", va="top")
            fig.text(0.01, 0.965,
                     f"id {overall_id:.2f}\ntext {overall_txt:.2f}",
                     va="top", ha="left", weight="bold", fontsize=7)

            # reference thumb over each generation column
            for c in range(n_gen):
                # ax = fig.add_axes(header_axes(c, n_gen, prompt_w))
                ax = fig.add_axes(header_axes(c, n_gen, prompt_w, avg_w))
                ax.imshow(ref_thumb)
                ax.axis("off")
            # header for AVG column
            fig.text(1 - avg_w + 0.005, 0.985, "AVG",
            weight="bold", va="top", ha="left", fontsize=8)
            
            fig.text(1-avg_w+0.005, 0.965, "id",  va="top",
                     ha="left", fontsize=6)
            fig.text(1-avg_w+0.005, 0.950, "txt", va="top",
                     ha="left", fontsize=6)



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
                    # g_img = load_image(img_p, 384)
                    g_img = load_image(img_p, thumb_side)

                    axg = fig.add_axes(gen_axes(fig, r, c, n_gen,
                                                cell_h, prompt_w, avg_w))
                    axg.imshow(g_img)
                    axg.axis("off")
                    # metrics to right
                    # axg.text(1.02, 0.5,
                    #          f"{row_data.id_similarity:.2f}\n{row_data.text_similarity:.2f}",
        
                    met_str = (f"{row_data.id_similarity:.2f}\n"
                               f"{row_data.text_similarity:.2f}")

                    axg.text(1.02, 0.5, met_str,
                             va="center", ha="left", transform=axg.transAxes,
                             fontsize=6)
                
                # ── average metrics column ───────────────────────────────
                g_list = generators.get(global_r)
                if g_list is not None and len(g_list):
                    id_avg   = g_list.id_similarity.mean()
                    text_avg = g_list.text_similarity.mean()

                    axa = fig.add_axes(avg_axes(fig, r, cell_h,
                                                prompt_w, avg_w))
                    axa.axis("off")
                    # axa.text(0, 0.5,
                    #          f"{id_avg:.2f}\n{text_avg:.2f}",

                    avg_str = f"{id_avg:.2f}\n{text_avg:.2f}"

                    axa.text(0, 0.5, avg_str,
                             va="center", ha="left", fontsize=6)

            # pdf.savefig(fig, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)

            plt.close(fig)
    
    # ── SUMMARY PAGE ───────────────────────────────────────────────────────
    piv = (df.groupby(["prompt_idx", "person_id"])
              .agg(id_mean=("id_similarity", "mean"),
                   txt_mean=("text_similarity", "mean"))
              .reset_index())

    persons = sorted(df.person_id.unique())
    cell_txt = []
    for pidx in range(len(prompts)):
        # wrap prompt to max 2 lines (≈20 % page width)
        wrapped = "\n".join(textwrap.wrap(prompts[pidx], 28)[:2])
        row = [wrapped]
        for person in persons:
            r = piv[(piv.prompt_idx == pidx) & (piv.person_id == person)]
            if r.empty:
                row.append("")
            else:
                row.append(f"{r.id_mean.values[0]:.2f}\n"
                           f"{r.txt_mean.values[0]:.2f}")

        # overall across *all* persons for this prompt
        row_vals = df[df.prompt_idx == pidx]
        row.append(f"{row_vals.id_similarity.mean():.2f}\n"
                   f"{row_vals.text_similarity.mean():.2f}")
        cell_txt.append(row)
    # column-wise averages
    last = ["ALL-prompts"]
    for person in persons:
        sub = df[df.person_id == person]
        s = f"{sub.id_similarity.mean():.2f}\n{sub.text_similarity.mean():.2f}"

        last.append(s)
    
    # grand average over everything
    grand_id  = df.id_similarity.mean()
    grand_txt = df.text_similarity.mean()
    last.append(f"{grand_id:.2f}\n{grand_txt:.2f}")
    cell_txt.append(last)

    fig = plt.figure(figsize=inches)
    # cols = ["Prompt"] + persons
    # cw   = [0.20] + [0.80/len(persons)]*len(persons)          # 20 % + equal rest
    # cols = ["Prompt"] + persons + ["Overall"]
    # cw   = [0.23] + [0.73/(len(persons)+1)]*len(persons) + [0.04]  # wider prompt

    # cols = ["Prompt"] + persons + ["Overall"]

    # keep headers horizontal: wrap long person IDs to max-two 10-char lines
    wrap_hdr = lambda s: "\n".join(textwrap.wrap(str(s), 10)[:2])
    cols = ["Prompt"] + [wrap_hdr(p) for p in persons] + ["AVG"]

    # ensure every row in cell_txt has exactly len(cols) cells
    need = len(cols)
    for row in cell_txt:
        if len(row) < need:
            row.extend([""] * (need - len(row)))

    # cw   = [0.23] + [0.73/(len(persons)+1)]*len(persons) + [0.04]  # wider prompt
    cw   = [0.23] + [0.73/len(persons)]*len(persons) + [0.04]       # wider prompt

    table = plt.table(cellText=cell_txt, colLabels=cols,
                      colWidths=cw, loc="center", cellLoc="center",
                      fontsize=6)
    
    # # rotate reference headers 
    # for c in range(1, len(cols)-1):             # skip “Prompt” header
    #     cell = table[(0, c)]
    #     cell.get_text().set_rotation(90)
    #     cell.get_text().set_verticalalignment("center")
    #     cell.get_text().set_horizontalalignment("right")
    
        
    # headers are already wrapped → just center them
    for c in range(1, len(cols)-1):             # skip “Prompt” header
        cell = table[(0, c)]
        cell.get_text().set_horizontalalignment("center")
    
    # bold grand-average cell (bottom-right)
    br = table[(len(cell_txt), len(cols)-1)]
    br.get_text().set_weight("bold")

    table.auto_set_font_size(False)
    table.scale(1, 1.5)
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
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
    p.add_argument("--thumb_side", type=int, default=768,
                   help="Max pixel size of thumbnails in PDF")
    args = p.parse_args()

    build_pdf(args.image_folder, args.new_images,
              args.prompt_file, args.metrics_file, args.output_pdf, thumb_side=args.thumb_side)