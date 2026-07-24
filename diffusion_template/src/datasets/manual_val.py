from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ManualPhotoMakerValDataset(Dataset):
    """
    Simple dataset that enumerates (reference image, prompt, seed) tuples
    for PhotoMaker validation.
    """

    def __init__(
        self,
        images_dir: str,
        prompts_path: str,
        classes_json_path: str | None = None,
        bbox_mask_ref: str | None = None,
        bbox_mask_gen: str | None = None,
        seeds: Sequence[int] = (0, 1, 2),
        limit: int | None = None,
        instance_transforms=None,
    ):
        self.images_dir = Path(images_dir)
        self.prompts_path = Path(prompts_path)
        self.seeds = list(seeds)
        self.instance_transforms = instance_transforms  # unused; kept for config compatibility

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")

        if classes_json_path is not None:
            classes_json_path = Path(classes_json_path)
            if not classes_json_path.exists():
                raise FileNotFoundError(f"classes_json_path not found: {classes_json_path}")
            import json

            with open(classes_json_path) as fh:
                self.classes_map = json.load(fh)
        else:
            self.classes_map = {}

        self.images = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not self.images:
            raise ValueError(f"No supported images found in {self.images_dir}")

        with open(self.prompts_path) as fh:
            raw_prompts = [line.strip() for line in fh if line.strip()]
        if not raw_prompts:
            raise ValueError(f"No prompts found in {self.prompts_path}")

        # Optional bbox maps (by image stem) for reference/gen masks
        def _load_bbox_map(path_str: str | None):
            if not path_str:
                return {}
            p = Path(path_str)
            if not p.exists():
                raise FileNotFoundError(f"bbox mask JSON not found: {p}")
            import json
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Normalize to stem -> bbox (prefer face_crop_new, fallback face_crop_old)
            m = {}
            for k, v in data.items():
                stem = Path(k).stem
                bbox = (
                    v.get("face_crop_new")
                    if isinstance(v, dict)
                    else None
                )
                if bbox is None and isinstance(v, dict):
                    bbox = v.get("face_crop_old")
                if bbox is None:
                    continue
                m[stem] = bbox
            return m

        self._bbox_map_ref = _load_bbox_map(bbox_mask_ref)
        # For generation masks, keep raw JSON to support index-based keys like "00.png"
        if bbox_mask_gen:
            import json
            p = Path(bbox_mask_gen)
            if not p.exists():
                raise FileNotFoundError(f"bbox mask JSON not found: {p}")
            with open(p, "r", encoding="utf-8") as fh:
                self._bbox_gen_json = json.load(fh)
        else:
            self._bbox_gen_json = None

        self.samples = []
        for image_path in self.images:
            img_id = image_path.stem
            class_value = self.classes_map.get(img_id)
            for prompt in raw_prompts:
                resolved_prompt = self._resolve_prompt(prompt, class_value)
                for seed in self.seeds:
                    self.samples.append(
                        {
                            "image_path": image_path,
                            "prompt": resolved_prompt,
                            "seed": seed,
                            "id": img_id,
                            # Optional per-image face bbox for reference (by stem)
                            "face_bbox_ref": self._bbox_map_ref.get(img_id),
                        }
                    )
                    if limit is not None and len(self.samples) >= limit:
                        return

    def _resolve_prompt(self, prompt: str, class_value: str | None) -> str:
        if "<class>" not in prompt:
            return prompt
        replacement = "img" if class_value is None else f"{class_value} img"
        return prompt.replace("<class>", replacement)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ref_img = Image.open(sample["image_path"]).convert("RGB")
        # Derive generation bbox by validation ordering if JSON provided
        face_bbox_gen = None
        if self._bbox_gen_json is not None:
            key = f"{idx:02d}.png"
            record = self._bbox_gen_json.get(key)
            if isinstance(record, dict):
                face_bbox_gen = record.get("face_crop_new") or record.get("face_crop_old")
        return {
            "ref_images": [ref_img],
            "prompt": sample["prompt"],
            "seed": sample["seed"],
            "id": sample["id"],
            "face_bbox_ref": sample.get("face_bbox_ref"),
            "face_bbox_gen": face_bbox_gen,
        }
