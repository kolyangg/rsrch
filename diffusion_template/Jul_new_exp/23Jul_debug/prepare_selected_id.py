#!/usr/bin/env python3
"""Create the fixed 8-train/2-holdout one-ID bundle under 23Jul_debug."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_JSON = Path("/home/niko/datasets/gathered_data_cosmic_large_filtered.json")
FACES_ROOT = Path(
    "/home/niko/datasets/LAION-5B-Filtered-Large-Faces/laion1B-nolang"
)
TARGET_KEY = (
    "LAION-5B-Filtered-Large/laion1B-nolang/00081/1017318003459.jpg"
)
OUT = HERE / "data" / "id_00081_1017318003459"
PROMPTS = [
    "Reading paper <class>, park bench, calm face, grey overcoat",
    "Rushing <class> portrait, subway platform, anxious face, swinging briefcase",
    "Kickboxing <class>, gym ring, fierce roar face, sweatband",
    "Dancing <class>, neon club, euphoric face, silver jumpsuit",
]


def face_source(path_value: str) -> Path:
    marker = "LAION-5B-Filtered-Large-Faces/laion1B-nolang/"
    if not path_value.startswith(marker):
        raise ValueError(f"Unexpected reference prefix: {path_value}")
    return FACES_ROOT / path_value[len(marker) :]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    with SOURCE_JSON.open(encoding="utf-8") as handle:
        source = json.load(handle)
    original = source[TARGET_KEY]

    all_refs = list(original["face_paths"])
    if len(all_refs) != 10:
        raise RuntimeError(f"Expected 10 reference images, found {len(all_refs)}")
    train_refs = all_refs[:8]
    holdout_refs = all_refs[8:]

    train_record = copy.deepcopy(original)
    train_record["face_paths"] = train_refs
    train_record["face_scores"] = list(original["face_scores"])[:8]
    train_record["face_bboxes"] = {
        ref: original["face_bboxes"][ref] for ref in train_refs
    }
    dump(OUT / "train_8refs.json", {TARGET_KEY: train_record})

    ref_dir = OUT / "validation_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    holdout_manifest = []
    ref_bbox_json = {}
    class_json = {}
    for label, ref in zip(("holdout_A", "holdout_B"), holdout_refs):
        suffix = face_source(ref).suffix.lower()
        filename = f"{label}{suffix}"
        destination = ref_dir / filename
        shutil.copy2(face_source(ref), destination)
        bbox = list(original["face_bboxes"][ref])
        ref_bbox_json[filename] = {"face_crop_new": bbox}
        class_json[Path(filename).stem] = "woman"
        holdout_manifest.append(
            {
                "label": label,
                "source_path": ref,
                "local_path": str(destination),
                "face_bbox": bbox,
                "face_score": original["face_scores"][all_refs.index(ref)],
            }
        )

    (OUT / "validation_prompts_4.txt").write_text(
        "\n".join(PROMPTS) + "\n", encoding="utf-8"
    )
    dump(OUT / "classes_ref.json", class_json)
    dump(OUT / "ref_bboxes.json", ref_bbox_json)
    dump(OUT / "gen_bboxes_seed.json", {})
    dump(
        OUT / "split_manifest.json",
        {
            "selected_id": "id_00081_1017318003459",
            "target_key": TARGET_KEY,
            "source_json": str(SOURCE_JSON),
            "class": "woman",
            "train_reference_paths": train_refs,
            "holdouts": holdout_manifest,
            "recurring_validation_reference": "holdout_A",
            "final_only_validation_reference": "holdout_B",
            "validation_prompts": PROMPTS,
            "seed": 0,
        },
    )
    print(f"Prepared selected-ID bundle: {OUT}")


if __name__ == "__main__":
    main()

