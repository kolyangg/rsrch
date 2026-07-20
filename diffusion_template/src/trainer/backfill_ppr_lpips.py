"""Backfill LPIPS into an existing PPR reference/noise result directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (256, 256),
            Image.Resampling.BICUBIC,
        )
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


@torch.no_grad()
def _calculate_pairs(model, pairs, *, device, batch_size: int) -> list[float]:
    output = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        left = torch.stack([_tensor(item[0]) for item in batch]).to(device)
        right = torch.stack([_tensor(item[1]) for item in batch]).to(device)
        values = model(left, right).flatten().detach().float().cpu().tolist()
        output.extend(float(value) for value in values)
    return output


def _bootstrap_ci(values: list[float]) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = np.asarray(
        [
            rng.choice(finite, size=finite.size, replace=True).mean()
            for _ in range(2000)
        ]
    )
    return (
        float(finite.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def backfill(root: Path, *, device: str, batch_size: int) -> None:
    try:
        import lpips
    except ModuleNotFoundError as error:
        raise SystemExit(
            "The active environment has no lpips package. Install it with:\n"
            "  python -m pip install lpips"
        ) from error

    root = root.expanduser().resolve()
    metrics_path = root / "metrics_per_image.csv"
    pairs_path = root / "paired_effects.csv"
    crops = root / "face_crops"
    for path in (metrics_path, pairs_path, crops):
        if not path.exists():
            raise FileNotFoundError(f"Missing required existing output: {path}")

    resolved_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = lpips.LPIPS(net="alex").to(resolved_device).eval()

    metric_rows = _read_csv(metrics_path)
    metric_pairs = []
    for row in metric_rows:
        filename = row["filename"]
        metric_pairs.append(
            (
                crops / f"{row['variant']}_{filename}",
                crops / f"PM0_{filename}",
            )
        )
    missing = [
        str(path)
        for pair in metric_pairs
        for path in pair
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing face crop: {missing[0]}")
    metric_values = _calculate_pairs(
        model,
        metric_pairs,
        device=resolved_device,
        batch_size=batch_size,
    )
    for row, value in zip(metric_rows, metric_values):
        row["lpips_core_vs_PM0"] = value
    _write_csv(metrics_path, metric_rows)

    pair_rows = _read_csv(pairs_path)
    image_pairs = [
        (
            crops / f"{row['left_variant']}_{row['filename']}",
            crops / f"{row['right_variant']}_{row['filename']}",
        )
        for row in pair_rows
    ]
    pair_values = _calculate_pairs(
        model,
        image_pairs,
        device=resolved_device,
        batch_size=batch_size,
    )
    for row, value in zip(pair_rows, pair_values):
        row["lpips_core"] = value
    _write_csv(pairs_path, pair_rows)

    summary_path = root / "metrics_summary.csv"
    summary_rows = _read_csv(summary_path)
    for effect in ("reference_image_effect", "reference_noise_effect"):
        values = [
            float(row["lpips_core"])
            for row in pair_rows
            if row["effect"] == effect
        ]
        mean, low, high = _bootstrap_ci(values)
        matches = [
            row
            for row in summary_rows
            if row["effect"] == effect and row["metric"] == "lpips_core"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one LPIPS summary row for {effect}, got {len(matches)}"
            )
        matches[0].update(
            {
                "mean": mean,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "pair_count": len(values),
            }
        )
    _write_csv(summary_path, summary_rows)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lpips_status"] = "available (backfilled from saved face crops)"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    conclusion_path = root / "conclusion.md"
    if conclusion_path.exists():
        text = conclusion_path.read_text(encoding="utf-8")
        lines = [
            (
                "- LPIPS status: available (backfilled from saved face crops)"
                if line.startswith("- LPIPS status:")
                else line
            )
            for line in text.splitlines()
        ]
        conclusion_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"LPIPS backfill complete: {root} "
        f"images={len(metric_rows)} pairs={len(pair_rows)} device={resolved_device}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Existing ppr_8k_reference_vs_noise result directory",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    backfill(args.output_dir, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
