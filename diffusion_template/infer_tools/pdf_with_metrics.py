#!/usr/bin/env python3
"""
Build a multi-page PDF comparing multiple generation scenarios side by side.

Config YAML format:

ref_dir: path/to/references
prompts: dataset_full/val_dataset/prompts_10.txt
classes: dataset_full/val_dataset/classes_ref.json
num_columns: 3                    # scenarios per page
max_per_row: 10                   # max generated images per scenario per prompt row
prompts_per_page: 10              # rows per page
thumb_side: 768                   # max image side for thumbnails
dpi: 300                          # PDF DPI
out_pdf: outputs/report.pdf
scenarios:
  - name: baseline
    gen_dir: path/to/gen1        # or "NA"
    metrics: path/to/metrics1.json   # or "NA"
  - name: tuned
    gen_dir: path/to/gen2
    metrics: path/to/metrics2.json

Notes:
- Layout, spacing and summary logic follow compare/testing/pdf_output4.py closely.
- Where metrics or images are NA, cells are left blank and metrics shown as NA.
"""

import argparse

import sys
from pathlib import Path as _P
# Ensure we can import diffusion_template/src as 'src'
sys.path.append(str(_P(__file__).resolve().parents[1]))
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Any

import yaml
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ───────────────────────── helpers ──────────────────────────────────────────
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def load_image(path: Path, max_side=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img


def prompt_axes(fig, row, prompts_area, cell_h, header_total):
    left, width = prompts_area
    bottom = 1 - header_total - (row+1)*cell_h        # 0.15 = header height
    return [left, bottom, width, cell_h*0.9]


def gen_axes(fig, row, col_idx, n_cols, cell_h, prompt_w, avg_w, header_total):
    # identical split as pdf_output4 but 'n_cols' is #scenario-columns on page
    free = 1 - prompt_w - avg_w - 0.03       # 0.03 total margins
    gw   = free / n_cols
    left = prompt_w + 0.02 + col_idx*gw
    bottom = 1 - header_total - (row+1)*cell_h
    return [left, bottom, gw*0.95, cell_h*0.9]


def header_axes(col_idx, n_cols, prompt_w, avg_w, top_margin, header_height, ref_same_cell_width=False, max_per_row=1):
    free = 1 - prompt_w - avg_w - 0.03
    gw   = free / n_cols
    col_w = gw*0.95
    bottom = 1 - (top_margin + header_height)
    if ref_same_cell_width:
        slot_w = col_w / max(1, max_per_row)
        ref_w = slot_w * 0.95
        left = prompt_w + 0.02 + col_idx*gw + (col_w - ref_w) / 2
        width = ref_w
    else:
        left = prompt_w + 0.02 + col_idx*gw
        width = col_w
    return [left, bottom, width, header_height]


def avg_axes(fig, row, cell_h, prompt_w, avg_w, header_total, avg_right_margin):
    left   = 1 - avg_w - avg_right_margin
    bottom = 1 - header_total - (row+1)*cell_h
    return [left, bottom, avg_w*0.95, cell_h*0.9]


def load_prompts(p: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in open(p, encoding="utf-8")]


def load_classes(p: Path) -> Dict[str, str]:
    return json.load(open(p, "r", encoding="utf-8"))


# ───────────────────────── data prep ────────────────────────────────────────

def load_metrics_json(path: Path) -> pd.DataFrame:
    if path is None or str(path).upper() == "NA" or not Path(path).exists():
        return pd.DataFrame(columns=[
            "file", "prompt", "prompt_idx", "ref", "class", "id_sim", "text_sim"
        ])
    data = json.load(open(path, "r", encoding="utf-8"))
    df = pd.DataFrame(data)
    # normalize types
    if "prompt_idx" in df.columns:
        df["prompt_idx"] = pd.to_numeric(df["prompt_idx"], errors="coerce").fillna(-1).astype(int)
    return df


def group_by_ref_and_prompt(df: pd.DataFrame) -> Dict[str, Dict[int, List[dict]]]:
    idx: Dict[str, Dict[int, List[dict]]] = {}
    if df.empty:
        return idx
    for _, row in df.iterrows():
        rid = row.get("ref")
        pidx = int(row.get("prompt_idx", -1))
        if rid is None or pidx < 0:
            continue
        idx.setdefault(rid, {}).setdefault(pidx, []).append(row.to_dict())
    return idx


# ───────────────────────── PDF builder ─────────────────────────────────────

def build_pdf_from_config(cfg: Dict[str, Any]):
    ref_dir      = Path(cfg["ref_dir"]).expanduser()
    prompts_file = Path(cfg["prompts"]).expanduser()
    classes_file = Path(cfg["classes"]).expanduser()
    out_pdf      = Path(cfg["out_pdf"]).expanduser()

    num_columns       = int(cfg.get("num_columns", 3))
    max_per_row       = int(cfg.get("max_per_row", 10))
    prompts_per_page  = int(cfg.get("prompts_per_page", 10))
    thumb_side        = int(cfg.get("thumb_side", 768))
    dpi               = int(cfg.get("dpi", 300))
    prompt_w          = float(cfg.get("prompt_w_frac", 0.12))
    ref_same_cell_w   = bool(cfg.get("ref_same_cell_width", False))
    ref_max_height    = int(cfg.get("ref_max_height", 0))
    top_margin        = float(cfg.get("header_top_margin", 0.015))
    header_height     = float(cfg.get("header_height", 0.10))
    header_to_grid_gap= float(cfg.get("header_to_grid_gap", 0.005))
    prompt_wrap_chars = cfg.get("prompt_wrap_chars")

    # New tuning knobs
    metrics_font_size = int(cfg.get("metrics_font_size", 5))
    avg_right_margin  = float(cfg.get("avg_right_margin", 0.001))

    scenarios_cfg = cfg.get("scenarios", [])

    prompts = load_prompts(prompts_file)
    classes = load_classes(classes_file)

    # Load per-scenario metrics and index by (ref, prompt_idx)
    scenarios = []
    for sc in scenarios_cfg:
        name = sc.get("name", "Unnamed")
        gen_dir = sc.get("gen_dir")
        metrics_p = sc.get("metrics")
        df = load_metrics_json(Path(metrics_p) if metrics_p else None)
        idx = group_by_ref_and_prompt(df)
        scenarios.append({
            "name": name,
            "gen_dir": Path(gen_dir) if gen_dir and str(gen_dir).upper() != "NA" else None,
            "metrics": df,
            "index": idx,
        })

    persons = sorted(classes.keys())
    inches = (8.27, 11.69)  # portrait A4
    pdf = PdfPages(out_pdf)

    # for each reference (person) create as many pages as needed to cover scenarios
    for person in persons:
        # reference thumbnail and label
        ref_thumb = None
        ref_label = str(person)
        for ext in (".jpg", ".png", ".jpeg", ".bmp", ".webp"):
            rp = ref_dir / f"{person}{ext}"
            if rp.exists():
                ref_thumb = load_image(rp, thumb_side//3)
                break
        if ref_thumb is None:
            ref_thumb = Image.new("RGB", (256, 256), "gray")

        # Compute overall averages for this reference across all scenarios
        all_rows = []
        for sc in scenarios:
            sub = sc["metrics"]
            if sub is not None and not sub.empty:
                all_rows.append(sub[sub["ref"] == person])
        if all_rows:
            cat = pd.concat(all_rows, axis=0)
            overall_id = float(cat["id_sim"].mean()) if not cat.empty else float("nan")
            overall_txt = float(cat["text_sim"].mean()) if not cat.empty else float("nan")
        else:
            overall_id = float("nan")
            overall_txt = float("nan")

        # paginate scenarios
        pages = math.ceil(max(1, len(scenarios)) / max(1, num_columns))
        for pg in range(pages):
            sc_start = pg * num_columns
            sc_end = min(len(scenarios), sc_start + num_columns)
            visible = scenarios[sc_start:sc_end]
            n_cols = max(1, len(visible))

            fig = plt.figure(figsize=inches)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # header: person label and overall metrics
            fig.text(0.01, 0.985, ref_label, weight="bold", va="top")
            if overall_id == overall_id and overall_txt == overall_txt:  # not NaN
                fig.text(0.01, 0.965, f"id {overall_id:.2f}\ntext {overall_txt:.2f}",
                         va="top", ha="left", weight="bold", fontsize=7)
            else:
                fig.text(0.01, 0.965, f"id NA\ntext NA",
                         va="top", ha="left", weight="bold", fontsize=7)

            # headers per scenario column: show scenario name and reference thumb
            avg_w = 0.06
            header_total = top_margin + header_height + header_to_grid_gap

            for c, sc in enumerate(visible):
                # scenario name centered over column
                box = header_axes(c, n_cols, prompt_w, avg_w, top_margin, header_height, ref_same_cell_width=ref_same_cell_w, max_per_row=max_per_row)
                axh = fig.add_axes(box)
                ref_disp = ref_thumb
                if ref_disp is not None and ref_max_height:
                    w, h = ref_disp.size
                    if h > ref_max_height:
                        top = max(0, (h - ref_max_height) // 2)
                        ref_disp = ref_disp.crop((0, top, w, top + ref_max_height))
                axh.imshow(ref_disp)
                axh.axis("off")
                # scenario name — up to three lines, above reference image
                center_x = box[0] + box[2] / 2
                wrap = "\n".join(textwrap.wrap(str(sc["name"]), 14)[:3])
                name_y = min(0.995, box[1] + box[3] + 0.004)
                fig.text(center_x, name_y, wrap, va="top", ha="center", fontsize=8, weight="bold")
                # per-scenario averages for this reference (across all prompts/images)
                sub = sc["metrics"]
                if sub is not None and not sub.empty:
                    subp = sub[sub["ref"] == person]
                    if not subp.empty:
                        sid = float(subp["id_sim"].mean())
                        stx = float(subp["text_sim"].mean())
                        grid_top_y = 1 - header_total
                        fig.text(
                            box[0] + 0.005,
                            grid_top_y,
                            f"id {sid:.2f}\ntext {stx:.2f}",
                            va="bottom",
                            ha="left",
                            fontsize=6,
                        )

            # header for AVG column
            header_left = 1 - avg_w - avg_right_margin
            fig.text(header_left + 0.005, 0.985, "AVG",
                     weight="bold", va="top", ha="left", fontsize=8)
            fig.text(header_left+0.005, 0.965, "id",  va="top", ha="left", fontsize=6)
            fig.text(header_left+0.005, 0.950, "txt", va="top", ha="left", fontsize=6)

            # grid
            cell_h = (1 - header_total - 0.02) / prompts_per_page  # header+margin
            for r in range(prompts_per_page):
                global_r = pg * prompts_per_page + r
                if global_r >= len(prompts):
                    break
                prompt_txt = prompts[global_r]

                axp = fig.add_axes(prompt_axes(fig, r, (0.01, prompt_w-0.015), cell_h, header_total))
                axp.axis("off")
                axp.text(0, 0.5, "\n".join(textwrap.wrap(prompt_txt, int(prompt_wrap_chars) if prompt_wrap_chars else max(12, round(200 * prompt_w)))), va="center", fontsize=6)

                # per-row accumulation for AVG column across all visible scenarios
                row_id_vals = []
                row_txt_vals = []

                for c, sc in enumerate(visible):
                    box = gen_axes(fig, r, c, n_cols, cell_h, prompt_w, avg_w, header_total)
                    # draw multiple images (up to max_per_row) horizontally in this box
                    rows_for_cell = sc["index"].get(person, {}).get(global_r, [])
                    if not rows_for_cell:
                        continue

                    # Limit to max_per_row, keep stable order by file name
                    try:
                        rows_for_cell = sorted(rows_for_cell, key=lambda d: d.get("file", ""))[:max_per_row]
                    except Exception:
                        rows_for_cell = rows_for_cell[:max_per_row]

                    k = max(1, len(rows_for_cell))
                    sub_w = box[2] / k

                    for j, row_data in enumerate(rows_for_cell):
                        left = box[0] + j * sub_w
                        # keep a small inner margin inside each sub-cell
                        sub = [left, box[1], sub_w*0.95, box[3]]
                        axg = fig.add_axes(sub)
                        gen_dir = sc.get("gen_dir")
                        img_p = None
                        if gen_dir is not None:
                            cand = Path(gen_dir) / str(row_data.get("file", ""))
                            if cand.exists():
                                img_p = cand
                        if img_p is None:
                            # skip rendering if missing image
                            axg.axis("off")
                            continue
                        g_img = load_image(img_p, thumb_side)
                        axg.imshow(g_img)
                        axg.axis("off")
                        # metrics to right of each sub-image
                        try:
                            id_v = float(row_data.get("id_sim"))
                            tx_v = float(row_data.get("text_sim"))
                            row_id_vals.append(id_v)
                            row_txt_vals.append(tx_v)
                            met = f"{id_v:.2f}\n{tx_v:.2f}"
                        except Exception:
                            met = "NA\nNA"
                        axg.text(1.02, 0.5, met,
                                 va="center", ha="left", transform=axg.transAxes, fontsize=metrics_font_size)

                # ── average metrics column (row-wise) ─────────────────
                if row_id_vals and row_txt_vals:
                    id_avg = sum(row_id_vals)/len(row_id_vals)
                    tx_avg = sum(row_txt_vals)/len(row_txt_vals)
                    axa = fig.add_axes(avg_axes(fig, r, cell_h, prompt_w, avg_w, header_total, avg_right_margin))
                    axa.axis("off")
                    axa.text(0, 0.5, f"{id_avg:.2f}\n{tx_avg:.2f}", va="center", ha="left", fontsize=6)

            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)

    # ── SUMMARY PAGE (prompts x scenarios) ─────────────────────────────────
    # Build a single table with rows = (prompt, person), columns = scenario names + AVG
    # Collect all records
    recs = []
    for sc in scenarios:
        name = sc["name"]
        df = sc["metrics"]
        if df is None or df.empty:
            continue
        # for each (ref, prompt_idx) pair, take mean over all images
        grp = df.groupby(["ref", "prompt_idx"]).agg(id_mean=("id_sim", "mean"), txt_mean=("text_sim", "mean")).reset_index()
        for _, row in grp.iterrows():
            recs.append({
                "scenario": name,
                "ref": row["ref"],
                "prompt_idx": int(row["prompt_idx"]),
                "id_mean": float(row["id_mean"]),
                "txt_mean": float(row["txt_mean"]),
            })

    if recs:
        df_all = pd.DataFrame(recs)
        persons = sorted(df_all["ref"].unique())
        fig = plt.figure(figsize=(8.27, 11.69))
        # Build cell text matrix: one row per (prompt, person)
        cw_prompt = 0.27
        cw_scin = 0.65 / max(1, len(scenarios))
        prompt_chars = int(cfg.get("summary_prompt_wrap_chars", max(12, round(90 * cw_prompt))))
        header_chars = int(cfg.get("summary_header_wrap_chars", max(8, round(60 * cw_scin))))
        cell_txt = []
        for pidx in range(len(prompts)):
            for person in persons:
                wrapped = "\n".join(textwrap.wrap(f"{prompts[pidx]}  [{person}]", prompt_chars)[:3])
                row = [wrapped]
                row_vals = []
                for sc in scenarios:
                    name = sc["name"]
                    r = df_all[(df_all["prompt_idx"] == pidx) & (df_all["ref"] == person) & (df_all["scenario"] == name)]
                    if r.empty:
                        row.append("")
                    else:
                        row.append(f"{r['id_mean'].values[0]:.2f}\n{r['txt_mean'].values[0]:.2f}")
                        row_vals.append((float(r['id_mean'].values[0]), float(r['txt_mean'].values[0])))
                # AVG across scenarios for this (prompt, person)
                if row_vals:
                    id_avg = sum(v[0] for v in row_vals)/len(row_vals)
                    tx_avg = sum(v[1] for v in row_vals)/len(row_vals)
                    row.append(f"{id_avg:.2f}\n{tx_avg:.2f}")
                else:
                    row.append("")
                cell_txt.append(row)

        # Append TOTAL AVG row across all rows for each scenario and global AVG
        total = ["TOTAL AVG"]
        for sc in scenarios:
            name = sc["name"]
            r = df_all[df_all["scenario"] == name]
            if r.empty:
                total.append("")
            else:
                total.append(f"{r['id_mean'].mean():.2f}\n{r['txt_mean'].mean():.2f}")
        g_id = df_all["id_mean"].mean() if not df_all.empty else float('nan')
        g_tx = df_all["txt_mean"].mean() if not df_all.empty else float('nan')
        total.append(f"{g_id:.2f}\n{g_tx:.2f}")
        cell_txt.append(total)

        # headers: scenario names + AVG
        wrap_hdr = lambda s: "\n".join(textwrap.wrap(str(s), header_chars)[:3])
        cols = ["Prompt"] + [wrap_hdr(sc["name"]) for sc in scenarios] + ["AVG"]
        need = len(cols)
        for row in cell_txt:
            if len(row) < need:
                row.extend([""] * (need - len(row)))

        cw = [0.13] + [0.83/len(scenarios)] * len(scenarios) + [0.04]
        table = plt.table(cellText=cell_txt, colLabels=cols,
                          colWidths=cw, loc="center", cellLoc="center", fontsize=6)
        table.auto_set_font_size(False)
        table.scale(1, 1.7)
        for c in range(len(cols)):
            hdr = table[(0, c)]
            hdr.get_text().set_fontsize(6)
            try:
                hdr.PAD = 0.01
            except Exception:
                pass
        # Bold TOTAL AVG row
        last_idx = len(cell_txt)
        for c in range(len(cols)):
            try:
                table[(last_idx, c)].get_text().set_weight("bold")
            except Exception:
                pass
        n_rows = len(cell_txt) + 1
        for r in range(1, n_rows):
            cell = table[(r, 0)]
            try:
                cell.PAD = 0.003
            except Exception:
                pass
            txt = cell.get_text()
            txt.set_ha("left")
            txt.set_x(0.01)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    pdf.close()
    print("Saved", out_pdf)


def main():
    ap = argparse.ArgumentParser(description="Create a PDF with side-by-side results and metrics from YAML config.")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    build_pdf_from_config(cfg)


if __name__ == "__main__":
    main()

# python3 infer_tools/pdf_with_metrics.py --config src/configs/pdf_output/pdf_config_04Nov.json