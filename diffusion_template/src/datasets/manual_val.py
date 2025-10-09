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
        return {
            "ref_images": [ref_img],
            "prompt": sample["prompt"],
            "seed": sample["seed"],
            "id": sample["id"],
        }
