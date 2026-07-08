#!/usr/bin/env python3
"""
Build a multi-page PDF comparing the full-validation (96-image) results of several runs.

Tailored to full_validation_results/ + full_val_metrics.py's metrics.json (per-run:
mean_id_sim, per_identity_id_sim, per_image_id_sim{filename->sim}). No InsightFace needed —
id-sim is read straight from metrics.json.

Layout:
  * Page 1: SUMMARY table — each run's overall mean id-sim + per-identity means (+ detection rate).
  * Page 2: CONFIG table — key training/inference differences for the same runs, with runs as
    columns and criteria as rows.
  * One page PER IDENTITY: a grid with the reference in the header, RUNS as columns, PROMPTS as
    rows; every cell shows the generated image with its id-sim printed on it, and each run column
    header shows that run's mean id-sim for this identity.

Config YAML (see infer_tools/full_val_report.yaml):
  results_dir, metrics_json, refs_dir, prompts, classes, out_pdf
  runs:   optional ordered list of subfolder names; if omitted, auto-detected and ordered by
          overall mean id-sim (best first).
  labels: optional {run_dir: "short label"} overrides.
  saved_dir: optional directory containing saved/<run>/config.yaml for the config table.
  config_criteria: optional list of {label, path, default} rows for the config table. Paths use
          dot notation into config.yaml, plus metric.* and a few computed.* fields.
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

DEFAULT_CONFIG_CRITERIA = [
    {"label": "96-val mean id-sim", "path": "metric.mean_id_sim"},
    {"label": "scored checkpoint step", "path": "metric.step"},
    {"label": "configured train schedule", "path": "computed.train_schedule"},
    {"label": "train batch size", "path": "dataloaders.train.batch_size"},
    {"label": "loss kind", "path": "loss_kind"},
    {"label": "face loss lambda", "path": "lambda_face"},
    {"label": "BA weight mode", "path": "branched_attn_weight_mode"},
    {"label": "BA weight kind", "path": "branched_attn_new_weight_kind"},
    {"label": "train BA self-attn", "path": "computed.train_ba_sa"},
    {"label": "train BA cross-attn", "path": "computed.train_ba_ca"},
    {"label": "train non-BA/base LoRA", "path": "non_ba_train", "default": False},
    {"label": "ID loss", "path": "computed.id_loss"},
    {"label": "ID embedding conditioning", "path": "model.use_id_embeds", "default": False},
    {"label": "face embed strategy", "path": "computed.face_embed_strategy"},
    {"label": "LoRA rank", "path": "model.rank"},
    {"label": "LoRA LR", "path": "lr_for_lora"},
    {"label": "BA noise LR scale", "path": "ba_noise_lr_scale"},
    {"label": "weight decay", "path": "optimizer.weight_decay"},
    {"label": "grad clip", "path": "trainer.max_grad_norm"},
    {"label": "warmup steps", "path": "lr_scheduler.warmup_steps"},
    {"label": "face prompt mode", "path": "model.ba_face_prompt_mode"},
    {"label": "uncond face fix", "path": "model.ba_uncond_face_fix", "default": False},
    {"label": "validation base", "path": "pretrained_model_for_validation_name_or_path"},
    {"label": "auto bboxes", "path": "computed.auto_bboxes"},
    {"label": "ref/input crop setup", "path": "computed.crop_setup"},
]


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


def wrap_lines(draw, text, fnt, max_w, max_lines=None):
    """Wrap text for table cells, including long underscore/slash-heavy tokens."""
    text = str(text)
    words = []
    for raw in text.replace("/", "/ ").replace("_", "_ ").split():
        words.append(raw)
    lines, cur = [], ""
    for word in words:
        candidate = (cur + " " + word).strip()
        if draw.textlength(candidate, font=fnt) <= max_w:
            cur = candidate
            continue
        if cur:
            lines.append(cur)
            cur = ""
        # Hard-wrap very long individual tokens.
        piece = ""
        for ch in word:
            candidate = piece + ch
            if draw.textlength(candidate, font=fnt) <= max_w:
                piece = candidate
            else:
                if piece:
                    lines.append(piece)
                piece = ch
        cur = piece
    if cur:
        lines.append(cur)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1].rstrip(".") + "..."
    return lines


def get_path(data, dotted, default=None):
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def fmt_bool(value):
    return "yes" if bool(value) else "no"


def fmt_value(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return fmt_bool(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001:
            return f"{value:.1e}"
        return f"{value:.4g}"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    replacements = {
        "SG161222/RealVisXL_V4.0": "RealVisXL_V4.0",
        "noise_and_ref": "noise+ref",
        "blended_masked": "blended",
        "masked_alternating": "alternating",
    }
    return replacements.get(text, text)


def load_run_config(saved_dir: Path, run: str):
    config_path = saved_dir / run / "config.yaml"
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text()) or {}
    return loaded if isinstance(loaded, dict) else {}


def config_value(path: str, run_cfg: dict, run_metrics: dict):
    if path.startswith("metric."):
        return get_path(run_metrics, path[len("metric.") :])
    if path == "computed.train_schedule":
        epoch_len = get_path(run_cfg, "trainer.epoch_len")
        n_epochs = get_path(run_cfg, "trainer.n_epochs")
        if epoch_len is None or n_epochs is None:
            return None
        total = epoch_len * n_epochs if isinstance(epoch_len, int) and isinstance(n_epochs, int) else None
        if total is not None and total < 100000:
            return f"{epoch_len} x {n_epochs} = {total}"
        return f"{epoch_len} x {n_epochs}"
    if path == "computed.train_ba_sa":
        return not bool(get_path(run_cfg, "disable_branched_sa", False))
    if path == "computed.train_ba_ca":
        return bool(get_path(run_cfg, "train_branched_ca_lora", False)) and not bool(
            get_path(run_cfg, "disable_branched_ca", False)
        )
    if path == "computed.id_loss":
        if not bool(get_path(run_cfg, "model.use_id_loss", False)):
            return "off"
        weight = fmt_value(get_path(run_cfg, "model.id_loss_weight"))
        gate = get_path(run_cfg, "model.id_loss_max_timestep")
        return f"w {weight}, t<={gate}" if gate is not None else f"w {weight}"
    if path == "computed.face_embed_strategy":
        value = get_path(run_cfg, "model.face_embed_strategy")
        if value in (None, "${pipeline.face_embed_strategy}"):
            value = get_path(run_cfg, "pipeline.face_embed_strategy")
        return value
    if path == "computed.auto_bboxes":
        auto = get_path(run_cfg, "automatic_bboxes")
        every_val = get_path(run_cfg, "automatic_bboxes_every_val")
        if auto is None:
            return None
        return f"{fmt_bool(auto)}; every_val={fmt_bool(every_val)}"
    if path == "computed.crop_setup":
        ref_crop = get_path(run_cfg, "train_dataset_crop_ref")
        nonface_min = get_path(run_cfg, "train_dataset_crop_nonface_min")
        nonface_max = get_path(run_cfg, "train_dataset_crop_nonface_max")
        const_ref = get_path(run_cfg, "train_dataset_const_ref")
        return f"ref_crop={fmt_bool(ref_crop)}; nonface={nonface_min}-{nonface_max}; const_ref={fmt_bool(const_ref)}"
    return get_path(run_cfg, path)


def build_config_table_page(cfg, runs, labels, metrics, run_label, fs):
    saved_dir = Path(cfg.get("saved_dir", "saved"))
    config_aliases = cfg.get("config_aliases", {})
    criteria = cfg.get("config_criteria") or DEFAULT_CONFIG_CRITERIA
    run_configs = {
        run: load_run_config(saved_dir, config_aliases.get(run, run))
        for run in runs
    }

    n_runs = len(runs)
    row_h = int(cfg.get("config_row_px", 58))
    col0 = int(cfg.get("config_label_px", 250))
    col_w = int(cfg.get("config_col_px", 150 if n_runs >= 9 else 175))
    margin = 24
    title_h = 116
    W = margin * 2 + col0 + col_w * n_runs
    H = title_h + row_h * (len(criteria) + 1) + margin
    pg = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(pg)

    title_font = font(int(30 * fs))
    small_font = font(int(15 * fs))
    header_font = font(int(15 * fs))
    cell_font = font(int(14 * fs))
    label_font = font(int(15 * fs))

    d.text((margin, 24), "Key config differences by run", fill=(255, 255, 120), font=title_font)
    d.text(
        (margin, 64),
        "Runs are columns in the same order as the result grids. Values are read from saved/<run>/config.yaml plus metrics.json.",
        fill=(205, 215, 225),
        font=small_font,
    )

    x0 = margin
    y0 = title_h
    header_fill = (42, 52, 66)
    label_fill = (34, 40, 48)
    line = (78, 86, 96)
    alt_a = (24, 24, 24)
    alt_b = (30, 30, 30)

    d.rectangle([x0, y0, x0 + col0 + col_w * n_runs, y0 + row_h], fill=header_fill, outline=line)
    d.text((x0 + 8, y0 + 10), "criterion", fill=(190, 220, 255), font=header_font)
    for j, run in enumerate(runs):
        x = x0 + col0 + j * col_w
        d.line([x, y0, x, y0 + row_h * (len(criteria) + 1)], fill=line)
        for li, txt in enumerate(wrap_lines(d, run_label(run), header_font, col_w - 12, max_lines=3)):
            d.text((x + 6, y0 + 6 + li * 16), txt, fill=(255, 255, 160), font=header_font)

    for i, criterion in enumerate(criteria):
        if isinstance(criterion, str):
            label, path, default = criterion, criterion, None
        else:
            label = criterion.get("label") or criterion.get("path", "")
            path = criterion.get("path", label)
            default = criterion.get("default")
        y = y0 + row_h * (i + 1)
        d.rectangle([x0, y, x0 + col0 + col_w * n_runs, y + row_h], fill=alt_a if i % 2 == 0 else alt_b, outline=line)
        d.rectangle([x0, y, x0 + col0, y + row_h], fill=label_fill, outline=line)
        for li, txt in enumerate(wrap_lines(d, label, label_font, col0 - 14, max_lines=2)):
            d.text((x0 + 8, y + 8 + li * 17), txt, fill=(235, 235, 235), font=label_font)
        for j, run in enumerate(runs):
            x = x0 + col0 + j * col_w
            value = config_value(path, run_configs.get(run, {}), metrics.get(run, {}))
            if value is None:
                value = default
            text = fmt_value(value)
            for li, txt in enumerate(wrap_lines(d, text, cell_font, col_w - 12, max_lines=3)):
                d.text((x + 6, y + 7 + li * 16), txt, fill=(225, 225, 225), font=cell_font)

    return pg


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

    # ---------- Page 2: key config differences ----------
    pages.append(build_config_table_page(cfg, runs, labels, metrics, run_label, fs))

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
    print(f"[pdf] wrote {out_pdf}  ({len(pages)} pages: 1 summary + 1 config + {len(ids)} identities, {n_runs} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    build(yaml.safe_load(open(args.config)))


if __name__ == "__main__":
    main()
