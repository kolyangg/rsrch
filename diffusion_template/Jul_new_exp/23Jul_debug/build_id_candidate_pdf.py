#!/usr/bin/env python3
"""Build the five-candidate one-ID target/reference review PDF."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
DATASET_ROOT = Path("/home/niko/datasets")
DATA_JSON = DATASET_ROOT / "gathered_data_cosmic_large_filtered.json"
OUTPUT = HERE / "one_id_candidate_reference_target_review.pdf"
MANIFEST = HERE / "one_id_candidate_manifest.json"

# Curated after metadata filtering, image-quality inspection, and CPU-only
# InsightFace consistency measurement over all ten reference images.
CANDIDATES = [
    {
        "rank": 1,
        "candidate_id": "id_00081_1017318003459",
        "record_path": "LAION-5B-Filtered-Large/laion1B-nolang/00081/1017318003459.jpg",
        "target_ref_cosine_mean": 0.8452,
        "target_ref_cosine_min": 0.8233,
        "ref_pair_cosine_mean": 0.8948,
        "ref_pair_cosine_min": 0.8405,
        "target_face_side": 432,
        "reference_detector_score_mean": 0.847,
        "target_sharpness_laplacian": 186.0,
        "selection_note": "Best identity-consistency proxy; clean, near-frontal target and ten stable references.",
    },
    {
        "rank": 2,
        "candidate_id": "id_00125_1150962006461",
        "record_path": "LAION-5B-Filtered-Large/laion1B-nolang/00125/1150962006461.webp",
        "target_ref_cosine_mean": 0.8436,
        "target_ref_cosine_min": 0.8058,
        "ref_pair_cosine_mean": 0.8901,
        "ref_pair_cosine_min": 0.8511,
        "target_face_side": 428,
        "reference_detector_score_mean": 0.856,
        "target_sharpness_laplacian": 57.2,
        "selection_note": "Very stable identity across references; distinctive age, hairline, and facial-hair cues.",
    },
    {
        "rank": 3,
        "candidate_id": "id_00020_4037853000645",
        "record_path": "LAION-5B-Filtered-Large/laion1B-nolang/00020/4037853000645.jpg",
        "target_ref_cosine_mean": 0.8037,
        "target_ref_cosine_min": 0.7624,
        "ref_pair_cosine_mean": 0.8684,
        "ref_pair_cosine_min": 0.8017,
        "target_face_side": 498,
        "reference_detector_score_mean": 0.845,
        "target_sharpness_laplacian": 817.6,
        "selection_note": "Sharpest high-resolution target; distinctive brows, moustache, beard, and face shape.",
    },
    {
        "rank": 4,
        "candidate_id": "id_00096_3540969004221",
        "record_path": "LAION-5B-Filtered-Large/laion1B-nolang/00096/3540969004221.jpg",
        "target_ref_cosine_mean": 0.8076,
        "target_ref_cosine_min": 0.7898,
        "ref_pair_cosine_mean": 0.8569,
        "ref_pair_cosine_min": 0.8074,
        "target_face_side": 432,
        "reference_detector_score_mean": 0.842,
        "target_sharpness_laplacian": 304.4,
        "selection_note": "Consistent ten-reference set with useful expression and lighting variation.",
    },
    {
        "rank": 5,
        "candidate_id": "id_00119_1313266018184",
        "record_path": "LAION-5B-Filtered-Large/laion1B-nolang/00119/1313266018184.jpg",
        "target_ref_cosine_mean": 0.8080,
        "target_ref_cosine_min": 0.7789,
        "ref_pair_cosine_mean": 0.8705,
        "ref_pair_cosine_min": 0.8056,
        "target_face_side": 468,
        "reference_detector_score_mean": 0.844,
        "target_sharpness_laplacian": 130.5,
        "selection_note": "Distinctive mature identity with stable hair, eye, mouth, and face-shape cues.",
    },
]


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def target_with_bbox(path: Path, bbox: list[float]) -> Image.Image:
    image = load_rgb(path)
    draw = ImageDraw.Draw(image)
    width = max(3, round(min(image.size) / 220))
    draw.rectangle(tuple(float(v) for v in bbox), outline=(0, 255, 80), width=width)
    return image


def target_face(path: Path, bbox: list[float], margin: float = 0.18) -> Image.Image:
    image = load_rgb(path)
    x0, y0, x1, y1 = [float(value) for value in bbox]
    dx = (x1 - x0) * margin
    dy = (y1 - y0) * margin
    return image.crop(
        (
            max(0, round(x0 - dx)),
            max(0, round(y0 - dy)),
            min(image.width, round(x1 + dx)),
            min(image.height, round(y1 + dy)),
        )
    )


def main() -> int:
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT} or {MANIFEST}")

    records = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    manifest_candidates = []
    with PdfPages(OUTPUT) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "NN3a_new1 one-ID training candidate review"
        metadata["Author"] = "PhotoMaker branched-attention research"
        metadata["Subject"] = "Five target identities and all available reference images"

        for candidate in CANDIDATES:
            record = records[candidate["record_path"]]
            reference_paths = list(record["face_paths"])
            if len(reference_paths) != 10:
                raise ValueError(
                    f"{candidate['candidate_id']} expected 10 refs, found {len(reference_paths)}"
                )

            panels = [
                (
                    "Target image + face bbox",
                    target_with_bbox(
                        DATASET_ROOT / candidate["record_path"],
                        record["face_crop_new"],
                    ),
                ),
                (
                    "Target face",
                    target_face(
                        DATASET_ROOT / candidate["record_path"],
                        record["face_crop_new"],
                    ),
                ),
            ]
            panels.extend(
                (
                    f"Reference {index + 1}",
                    load_rgb(DATASET_ROOT / reference_path),
                )
                for index, reference_path in enumerate(reference_paths)
            )

            fig, axes = plt.subplots(3, 4, figsize=(15.5, 12.2))
            fig.suptitle(
                f"Candidate {candidate['rank']}: {candidate['candidate_id']}",
                fontsize=18,
                fontweight="bold",
                y=0.986,
            )
            metric_line = (
                f"10 refs | target↔ref cosine mean/min "
                f"{candidate['target_ref_cosine_mean']:.4f}/"
                f"{candidate['target_ref_cosine_min']:.4f} | "
                f"ref↔ref cosine mean/min {candidate['ref_pair_cosine_mean']:.4f}/"
                f"{candidate['ref_pair_cosine_min']:.4f} | "
                f"detector mean {candidate['reference_detector_score_mean']:.3f} | "
                f"target face side {candidate['target_face_side']} px"
            )
            fig.text(0.5, 0.955, metric_line, ha="center", va="top", fontsize=9.2)
            fig.text(
                0.5,
                0.936,
                candidate["selection_note"],
                ha="center",
                va="top",
                fontsize=9.2,
                color="#334155",
            )
            fig.text(
                0.5,
                0.917,
                "Cosines are CPU InsightFace dataset-consistency proxies; generated-image "
                "identity similarity will be measured separately during training.",
                ha="center",
                va="top",
                fontsize=8.3,
                color="#6b7280",
            )
            for ax, (title, image) in zip(axes.flat, panels):
                ax.imshow(image)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            fig.subplots_adjust(
                top=0.885,
                bottom=0.035,
                left=0.025,
                right=0.975,
                hspace=0.19,
                wspace=0.055,
            )
            pdf.savefig(fig, bbox_inches="tight", dpi=130)
            plt.close(fig)

            item = dict(candidate)
            item.update(
                {
                    "target_image": str(DATASET_ROOT / candidate["record_path"]),
                    "target_face_bbox": record["face_crop_new"],
                    "reference_count": len(reference_paths),
                    "reference_images": [
                        str(DATASET_ROOT / path) for path in reference_paths
                    ],
                    "facial_caption": record.get("facial_caption", ""),
                    "pose_caption": record.get("pose_caption", ""),
                    "background_caption": record.get("background_caption", ""),
                }
            )
            manifest_candidates.append(item)

    manifest = {
        "purpose": "Candidate identities for 500-step NN3a_new1 one-ID training screens",
        "ranking_method": textwrap.dedent(
            """
            Required 10 reference images and at least a 230-pixel target face.
            Filtered obvious occlusions and poor detector-confidence tails.
            Ranked metadata/image quality, then measured CPU InsightFace target-to-reference
            and pairwise reference consistency. The final five are curated for both score
            and varied facial structure. These are dataset consistency proxies, not
            post-generation model scores.
            """
        ).strip(),
        "candidates": manifest_candidates,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pdf": str(OUTPUT),
                "manifest": str(MANIFEST),
                "pages": len(CANDIDATES),
                "candidates": len(CANDIDATES),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
