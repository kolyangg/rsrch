"""
Utility to generate bounding boxes for a folder of reference images.

The script runs a face detector (MTCNN by default, YOLO optional) together with a YOLOv8
person detector to extract head and body crops.
It saves a JSON mapping from image path (relative to the input directory by default)
to a dictionary with `face_crop_old`, `face_crop_new`, and `body_crop` bounding boxes.
All detections run on GPU when available unless `--device` overrides it.

Example:
    python bbox_utils/generate_bboxes.py \
        --input-dir ../dataset_full/val_dataset/references \
        --output ../dataset_full/val_dataset/ref_bboxes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from ultralytics import YOLO  # noqa: F401


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bounding boxes for images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../dataset_full/val_dataset/references"),
        help="Folder containing reference images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../dataset_full/val_dataset/ref_bboxes.json"),
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="Optional root used to compute keys in the JSON. "
        "Defaults to the input directory.",
    )
    parser.add_argument(
        "--face-detector",
        choices=["mtcnn", "yolo"],
        default="mtcnn",
        help="Face detector backend to use. "
        "'mtcnn' relies on facenet-pytorch (GPU/CPU) and is the default. "
        "'yolo' loads a YOLO checkpoint specified by --face-model.",
    )
    parser.add_argument(
        "--face-model",
        type=str,
        default="yolov8n-face.pt",
        help="Ultralytics model checkpoint for face detection (only used with --face-detector yolo).",
    )
    parser.add_argument(
        "--body-model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model checkpoint for full-body/person detection.",
    )
    parser.add_argument(
        "--face-conf",
        type=float,
        default=0.3,
        help="Confidence threshold for face detection.",
    )
    parser.add_argument(
        "--body-conf",
        type=float,
        default=0.3,
        help="Confidence threshold for body detection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on number of images to process.",
    )
    parser.add_argument(
        "--face-padding",
        type=float,
        default=0.08,
        help="Padding ratio applied to create `face_crop_old` around the detected face.",
    )
    parser.add_argument(
        "--body-padding",
        type=float,
        default=0.02,
        help="Padding ratio applied to expand the person bounding box.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unknown device specifier: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def clamp_bbox(box: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, width, height]
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def enlarge_box(box: List[float], width: int, height: int, padding_ratio: float) -> List[int]:
    if padding_ratio <= 0:
        return clamp_bbox(box, width, height)
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = box_w * padding_ratio
    pad_h = box_h * padding_ratio
    return clamp_bbox(
        [x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h],
        width,
        height,
    )


def detect_primary_box(
    model: Any,
    source: Any,
    conf: float,
    device: str,
    person_only: bool = False,
) -> Optional[List[float]]:
    if isinstance(source, Path):
        source = str(source)
    results = model(
        source,
        conf=conf,
        device=device,
        verbose=False,
    )
    if not results:
        return None
    result = results[0]
    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes
    scores = boxes.conf.detach().cpu()
    xyxy = boxes.xyxy.detach().cpu()

    best_idx = None
    best_score = -1.0
    for idx, (score, cls_id) in enumerate(zip(scores, boxes.cls.detach().cpu())):
        if person_only and int(cls_id.item()) != 0:
            continue
        if score.item() > best_score:
            best_score = score.item()
            best_idx = idx

    if best_idx is None:
        return None
    return xyxy[best_idx].tolist()


def load_face_detector(
    backend: str, model_name: str, device: str
) -> Tuple[Any, str]:
    if backend == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency missing at runtime
            raise ImportError(
                "Ultralytics package is required for --face-detector yolo. Install with `pip install ultralytics`."
            ) from exc
        try:
            detector = YOLO(model_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"YOLO face model '{model_name}' not found. "
                "Download the checkpoint and provide its path via --face-model, "
                "or switch to --face-detector mtcnn."
            ) from exc
        detector.to(device)
        return detector, "yolo"
    if backend == "mtcnn":
        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:  # pragma: no cover - dependency missing at runtime
            raise ImportError(
                "facenet-pytorch package is required for --face-detector mtcnn. "
                "Install with `pip install facenet-pytorch`."
            ) from exc
        mtcnn_device = device if device == "cpu" else "cuda:0"
        detector = MTCNN(keep_all=True, device=mtcnn_device)
        return detector, "mtcnn"
    raise ValueError(f"Unsupported face detector backend: {backend}")


def detect_face_box(
    detector: Any,
    backend: str,
    image_path: Optional[Path],
    pil_image: Image.Image,
    conf: float,
    device: str,
) -> Optional[List[float]]:
    if backend == "yolo":
        source = image_path if image_path is not None else pil_image
        return detect_primary_box(detector, source, conf, device)

    if backend == "mtcnn":
        boxes, probs = detector.detect(pil_image, landmarks=False)
        if boxes is None or len(boxes) == 0 or probs is None:
            return None
        best_idx = int(max(range(len(probs)), key=lambda idx: probs[idx] or 0.0))
        if probs[best_idx] is None or probs[best_idx] < conf:
            return None
        return boxes[best_idx].tolist()

    raise ValueError(f"Unsupported face detector backend: {backend}")

def face_record_from_pil(
    pil_image: Image.Image,
    *,
    detector: Any,
    backend: str,
    conf: float = 0.3,
    device: str = "cpu",
    face_padding: float = 0.08,
) -> Dict[str, List[int]]:
    """
    Generate a face bbox record from an in-memory PIL image.
    Returns the same dict shape as JSON entries:
      {face_crop_old, face_crop_new, body_crop}
    """
    width, height = pil_image.size
    face_box = detect_face_box(detector, backend, None, pil_image, conf, device)
    if face_box is None:
        cx, cy = width / 2, height / 2
        size = min(width, height) * 0.45
        face_box = [cx - size, cy - size, cx + size, cy + size]
    face_crop_new = clamp_bbox(face_box, width, height)
    face_crop_old = enlarge_box(face_box, width, height, face_padding)
    return {
        "face_crop_old": face_crop_old,
        "face_crop_new": face_crop_new,
        "body_crop": [0, 0, width, height],
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    relative_root = args.relative_to.resolve() if args.relative_to else input_dir
    device = resolve_device(args.device)

    face_detector, face_backend = load_face_detector(args.face_detector, args.face_model, device)
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise ImportError(
            "Ultralytics package is required for body detection. Install with `pip install ultralytics`."
        ) from exc

    body_model = YOLO(args.body_model)
    body_model.to(device)

    data: Dict[str, Dict[str, List[int]]] = {}
    images = list(iter_images(input_dir))
    if args.max_images:
        images = images[: args.max_images]

    for image_path in images:
        with Image.open(image_path).convert("RGB") as img:
            width, height = img.size
            face_box = detect_face_box(face_detector, face_backend, image_path, img, args.face_conf, device)

        body_box = detect_primary_box(body_model, image_path, args.body_conf, device, person_only=True)

        if face_box is None:
            # fall back to central crop if detection fails
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.45
            face_box = [cx - size, cy - size, cx + size, cy + size]

        if body_box is None:
            body_box = [0.0, 0.0, float(width), float(height)]

        face_crop_new = clamp_bbox(face_box, width, height)
        face_crop_old = enlarge_box(face_box, width, height, args.face_padding)
        body_crop = enlarge_box(body_box, width, height, args.body_padding)

        try:
            key_path = image_path.resolve().relative_to(relative_root.resolve())
        except ValueError:
            key_path = image_path.name

        key = str(key_path).replace("\\", "/")

        data[key] = {
            "face_crop_old": face_crop_old,
            "face_crop_new": face_crop_new,
            "body_crop": body_crop,
        }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} entries to {output_path}")


if __name__ == "__main__":
    main()
