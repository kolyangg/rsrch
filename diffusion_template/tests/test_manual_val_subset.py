from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.datasets.manual_val import ManualPhotoMakerValDataset


class ManualValidationSubsetTests(unittest.TestCase):
    def test_seeded_subset_is_stable_and_keeps_original_bbox_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            for name in ("a", "b"):
                Image.new("RGB", (16, 16), "white").save(images / f"{name}.png")
            prompts = root / "prompts.txt"
            prompts.write_text("portrait one\nportrait two\nportrait three\n")
            boxes = {
                f"{index:02d}.png": {"face_crop_new": [index, 1, index + 2, 3]}
                for index in range(6)
            }
            bbox_path = root / "boxes.json"
            bbox_path.write_text(json.dumps(boxes))

            kwargs = dict(
                images_dir=str(images),
                prompts_path=str(prompts),
                bbox_mask_gen=str(bbox_path),
                seeds=[0],
                limit=6,
                subset_size=3,
                subset_seed=20260722,
            )
            first = ManualPhotoMakerValDataset(**kwargs)
            second = ManualPhotoMakerValDataset(**kwargs)
            first_indices = [sample["validation_index"] for sample in first.samples]
            second_indices = [sample["validation_index"] for sample in second.samples]
            self.assertEqual(first_indices, second_indices)
            self.assertEqual(len(first_indices), 3)
            for local_index, original_index in enumerate(first_indices):
                self.assertEqual(
                    first[local_index]["face_bbox_gen"],
                    [original_index, 1, original_index + 2, 3],
                )


if __name__ == "__main__":
    unittest.main()
