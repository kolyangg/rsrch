#!/usr/bin/env python3
"""
Build a multi-page PDF comparing the full-validation (96-image) results of several runs.

Tailored to full_validation_results/ + full_val_metrics.py's metrics.json (per-run:
mean_id_sim, per_identity_id_sim, per_image_id_sim{filename->sim}). No InsightFace needed —
id-sim is read straight from metrics.json.

Layout:
  * Page 1: SUMMARY table — each run's overall mean id-sim + per-identity means (+ detection rate).
  * One page PER IDENTITY: a grid with the reference in the header, RUNS as columns, PROMPTS as
    rows; every cell shows the generated image with its id-sim printed on it, and each run column
    header shows that run's mean id-sim for this identity.

Config YAML (see infer_tools/full_val_report.yaml):
  results_dir, metrics_json, refs_dir, prompts, classes, out_pdf
  runs:   optional ordered list of subfolder names; if omitted, auto-detected and ordered by
          overall mean id-sim (best first).
  labels: optional {run_dir: "short label"} overrides.
  cell_px, label_px, header_px, dpi, font_scale: layout knobs.

Usage:
  python infer_tools/pdf_full_val.py --config infer_tools/full_val_report.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def resolve_prompt(prompt: str, cls: str | None) -> str:
    if "<class>" in prompt and cls:
        return prompt.replace("<class>", cls)
    return prompt


def sim_color(s):
    if s is None:
        return (255, 140, 140)
    if s >= 0.40:
        return (120, 255, 120)
    if s >= 0.30:
        return (230, 230, 120)
    return (255, 170, 120)


def load_thumb(path: Path, side: int):
    im = Image.open(path).convert("RGB")
    return im.resize((side, side), Image.LANCZOS)


def missing_cell(side: int, text="missing"):
    im = Image.new("RGB", (side, side), (35, 35, 35))
    ImageDraw.Draw(im).text((8, side // 2 - 8), text, fill=(200, 90, 90), font=font(max(12, side // 18)))
    return im


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build(cfg):
    results_dir = Path(cfg["results_dir"])
    refs_dir = Path(cfg["refs_dir"])
    metrics = json.loads(Path(cfg["metrics_json"]).read_text())
    prompts = [ln.rstrip("\n") for ln in open(cfg["prompts"], encoding="utf-8") if ln.strip()]
    classes = json.loads(Path(cfg["classes"]).read_text())
    out_pdf = Path(cfg["out_pdf"])

    cell = int(cfg.get("cell_px", 240))
    label_px = int(cfg.get("label_px", 210))
    header_px = int(cfg.get("header_px", cell + 70))
    dpi = int(cfg.get("dpi", 150))
    fs = float(cfg.get("font_scale", 1.0))

    # runs: explicit order, else auto by overall mean id-sim (best first)
    runs = cfg.get("runs")
    if not runs:
        runs = [r for r in metrics.keys()]
        runs.sort(key=lambda r: -(metrics[r].get("mean_id_sim") or -1))
    runs = [r for r in runs if r in metrics]
    labels = cfg.get("labels", {})

    def run_label(r):
        return labels.get(r, r.replace("ba_", ""))

    # identities present in the metrics (union), ordered by classes.json order then alpha
    ids = sorted({i for r in runs for i in metrics[r].get("per_identity_id_sim", {})})

    def ref_thumb(ident, side):
        for e in IMG_EXTS:
            p = refs_dir / f"{ident}{e}"
            if p.exists():
                return load_thumb(p, side)
        return missing_cell(side, ident)

    pages = []

    # ---------- Page 1: summary table ----------
    n_id = len(ids)
    col0 = 260
    stepcol = 70
    meancol = 90
    idcol = max(64, int((1600) / max(1, n_id)))
    W = col0 + stepcol + meancol + idcol * n_id + 40
    rowh = 46
    H = 120 + rowh * (len(runs) + 1)
    pg = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(pg)
    d.text((20, 24), "Full validation (96 images) — mean id-sim per run", fill=(255, 255, 120), font=font(int(30 * fs)))
    y0 = 90
    hf = font(int(19 * fs))
    d.text((20, y0), "run", fill=(180, 210, 255), font=hf)
    d.text((col0, y0), "step", fill=(180, 210, 255), font=hf)
    d.text((col0 + stepcol, y0), "MEAN", fill=(255, 255, 160), font=hf)
    for k, ident in enumerate(ids):
        d.text((col0 + stepcol + meancol + k * idcol, y0), ident[:8], fill=(180, 210, 255), font=font(int(16 * fs)))
    # best per column highlight
    best_mean = max((metrics[r].get("mean_id_sim") or -1) for r in runs)
    y = y0 + rowh
    cf = font(int(18 * fs))
    for r in runs:
        m = metrics[r]
        mean = m.get("mean_id_sim")
        d.text((20, y), run_label(r)[:26], fill=(235, 235, 235), font=cf)
        d.text((col0, y), str(m.get("step", "")), fill=(200, 200, 200), font=cf)
        col = (120, 255, 120) if (mean is not None and mean >= best_mean - 1e-9) else sim_color(mean)
        d.text((col0 + stepcol, y), f"{mean:.3f}" if mean is not None else "NA", fill=col, font=cf)
        for k, ident in enumerate(ids):
            v = m.get("per_identity_id_sim", {}).get(ident)
            d.text((col0 + stepcol + meancol + k * idcol, y), f"{v:.3f}" if v is not None else "-", fill=sim_color(v), font=font(int(16 * fs)))
        y += rowh
    pages.append(pg)

    # ---------- One page per identity ----------
    n_runs = len(runs)
    W = label_px + n_runs * cell + 20
    H = header_px + len(prompts) * cell + 20
    for ident in ids:
        cls = classes.get(ident)
        pg = Image.new("RGB", (W, H), (14, 14, 14))
        d = ImageDraw.Draw(pg)
        # header: reference + identity + per-run means
        pg.paste(ref_thumb(ident, header_px - 40), (6, 30))
        d.text((6, 4), f"{ident}  ({cls})", fill=(255, 255, 120), font=font(int(22 * fs)))
        lf = font(int(16 * fs))
        for j, r in enumerate(runs):
            x = label_px + j * cell
            for li, line in enumerate(wrap(d, run_label(r), lf, cell - 8)):
                d.text((x + 4, 6 + li * 18), line, fill=(255, 255, 150), font=lf)
            v = metrics[r].get("per_identity_id_sim", {}).get(ident)
            d.text((x + 4, header_px - 26), f"mean {v:.3f}" if v is not None else "mean -", fill=sim_color(v), font=font(int(15 * fs)))
        # grid rows = prompts
        rf = font(int(15 * fs))
        sf = font(int(16 * fs))
        for pi, prompt in enumerate(prompts):
            ry = header_px + pi * cell
            for li, line in enumerate(wrap(d, prompt, rf, label_px - 10)):
                d.text((6, ry + 6 + li * 17), line, fill=(200, 225, 255), font=rf)
            resolved = resolve_prompt(prompt, cls)
            fname = f"{resolved[:10]}_{ident}.png"
            for j, r in enumerate(runs):
                x = label_px + j * cell
                imgp = results_dir / r / fname
                if imgp.exists():
                    pg.paste(load_thumb(imgp, cell - 4), (x + 2, ry + 2))
                else:
                    pg.paste(missing_cell(cell - 4), (x + 2, ry + 2))
                sim = metrics[r].get("per_image_id_sim", {}).get(fname)
                d.rectangle([x + 2, ry + 2, x + 78, ry + 24], fill=(0, 0, 0))
                d.text((x + 5, ry + 4), f"{sim:.3f}" if isinstance(sim, (int, float)) else "no-face", fill=sim_color(sim if isinstance(sim, (int, float)) else None), font=sf)
        pages.append(pg)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:], resolution=float(dpi))
    print(f"[pdf] wrote {out_pdf}  ({len(pages)} pages: 1 summary + {len(ids)} identities, {n_runs} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    build(yaml.safe_load(open(args.config)))


if __name__ == "__main__":
    main()
