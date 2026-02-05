from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


class AutoGenBboxStore:
    """
    Validation-time face bbox generation for *generated* images (gen-mask for BA merge).

    - JSON format matches existing bbox JSONs: key -> {face_crop_old, face_crop_new, body_crop}
    - Keys are expected to be output filenames like "<prompt[:10]>_<id>.png"
    - Designed to be CPU-only to avoid VRAM/ORT/InsightFace init-time OOMs.
    """

    def __init__(
        self,
        json_path: Path,
        *,
        face_detector: str = "mtcnn",
        face_model: str = "yolov8n-face.pt",
        face_conf: float = 0.3,
        face_padding: float = 0.08,
        device: str = "cpu",
        line_width: int = 4,
    ):
        self.json_path = Path(json_path)
        self.face_detector = str(face_detector)
        self.face_model = str(face_model)
        self.face_conf = float(face_conf)
        self.face_padding = float(face_padding)
        self.device = str(device)
        self.line_width = int(line_width)

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as fh:
                self.data: Dict[str, Dict[str, Any]] = json.load(fh) or {}
            print(f"[AutoBboxGen] using existing: {self.json_path} ({len(self.data)} entries)")
        else:
            self.data = {}
            print(f"[AutoBboxGen] will create: {self.json_path}")

        # Lazy import so training can run even if bbox deps are not installed.
        from bbox_utils.generate_bboxes import load_face_detector

        self._detector, self._backend = load_face_detector(
            backend=self.face_detector,
            model_name=self.face_model,
            device=self.device,
        )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        v = self.data.get(key)
        return v if isinstance(v, dict) else None

    def ensure(
        self,
        key: str,
        *,
        photomaker_image: Image.Image,
        meta: Optional[Dict[str, Any]] = None,
        overlay_path: Optional[Path] = None,
        force_overlay: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        existing = self.get(key)
        if existing is not None and not force_recompute:
            # Optional safety: if caller provides meta and existing meta mismatches, rebuild.
            if meta is not None:
                ex_meta = existing.get("_meta")
                if isinstance(ex_meta, dict):
                    for k, v in meta.items():
                        if k in ex_meta and ex_meta.get(k) != v:
                            force_recompute = True
                            break
            if not force_recompute:
                if overlay_path is not None and (force_overlay or not overlay_path.exists()):
                    from bbox_utils.visualize_bboxes import save_annotated_pil

                    save_annotated_pil(
                        photomaker_image,
                        {"face_crop_new": existing.get("face_crop_new") or existing.get("face_crop_old")},
                        overlay_path,
                        line_width=self.line_width,
                    )
                return existing

        from bbox_utils.generate_bboxes import face_record_from_pil
        from bbox_utils.visualize_bboxes import save_annotated_pil

        rec = face_record_from_pil(
            photomaker_image,
            detector=self._detector,
            backend=self._backend,
            conf=self.face_conf,
            device=self.device,
            face_padding=self.face_padding,
        )

        if meta is not None:
            rec["_meta"] = dict(meta)

        self.data[key] = rec
        tmp = self.json_path.with_suffix(self.json_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)
        tmp.replace(self.json_path)

        if overlay_path is not None:
            save_annotated_pil(
                photomaker_image,
                {"face_crop_new": rec["face_crop_new"]},
                overlay_path,
                line_width=self.line_width,
            )

        print(f"[AutoBboxGen] computed {key}: {rec['face_crop_new']}")
        return rec
