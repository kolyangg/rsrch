import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.aligner import Aligner

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def collect_image_paths(images_dir: Path):
    return sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def main():
    parser = argparse.ArgumentParser(description="Create ID embeddings for manual validation set.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("../dataset_full/val_dataset/references"),
        help="Directory with reference images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../dataset_full/val_dataset/id_embeds_manual_val.pth"),
        help="Path to save the generated embeddings (.pth).",
    )
    args = parser.parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    image_paths = collect_image_paths(args.images_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in {args.images_dir}")

    aligner = Aligner()
    id_embeds = {}
    missing_ids = []

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        _, face_embeds = aligner([img])
        stem = img_path.stem

        if not face_embeds or not face_embeds[0]:
            missing_ids.append(stem)
            continue

        # take the first detected face embedding
        id_embeds[stem] = torch.tensor(face_embeds[0][0])

    if missing_ids:
        print(f"Warning: no faces detected for {len(missing_ids)} images: {missing_ids}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(id_embeds, args.output)
    print(f"Saved {len(id_embeds)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
