#!/usr/bin/env python3
"""Static parity gate for the sealed CL14 generation path."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "src/pipelines/br_pipeline_helpers.py": "4c1516d3536a85c028580f601b61773df55c49d6b16dfac9d93c997102be5c95",
    "src/pipelines/photomaker_branched_clean.py": "85e1b3a2da90ba4a007f8bda895c722c7a21c8e5a519b86881626ed665e9071c",
}
VALIDATION_INPUT_EXPECTED = {
    "../dataset_full/val_dataset/protocols/cl14/pm96_bboxes_new.json": (
        "a39645e22b68027175946a028e185b7c5393a7514f5d68c94cd74e7cc9f5e614"
    ),
    "../dataset_full/val_dataset/protocols/cl14/pm96_bboxes_new_auto.json": (
        "b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d"
    ),
    "../dataset_full/val_dataset/prompts_10.txt": "e8fb3290e6da6eacc70c6cc67f2affa0c923c1ca605efc35ddca95ee48f1ebaf",
    "../dataset_full/val_dataset/classes_ref.json": "d1f53322d6964c2d30d28ef2cc765366a42776117e3982909d6fdfc1ae99872b",
    "../dataset_full/val_dataset/ref_bboxes.json": "eadb9411b9d0b98238714bb263db708e56a30abee91c67c4df0c7e1e5c4a268f",
    "../dataset_full/val_dataset/id_embeds_manual_val.pth": "23ae97075e967f2bcb790c5094ef350b316249c7023df67a68f735bfebb747c6",
    "../dataset_full/val_dataset/references/eddie.webp": "488c1ba267c3bada5aed1d72bf5b569b5be6ce7fb9050554559f307155cdcb8e",
    "../dataset_full/val_dataset/references/elon.jpg": "6e68491ee0f393df834ff9570dd15eaa01fb5f8805f6fce3f075818a7ea02381",
    "../dataset_full/val_dataset/references/jennie.webp": "ce286f8242cb1f702b0289ceaa20d67cd4ac1ffd8b8a909658ff6648a0129c81",
    "../dataset_full/val_dataset/references/jensen.png": "2f540b82ece53e4f3f4862a72fb2fbefd67854dbb9aa2d33b8183d322a50831a",
    "../dataset_full/val_dataset/references/jisoo.webp": "62c380c9b5ec08ec8b1fe613a390ff18b0f16497a23ebbfd1459ff887988e806",
    "../dataset_full/val_dataset/references/keanu.jpg": "750d34d29d14fc8875bbebecff56c1fbd32fa642e3a1a6454fd6f79c489531c3",
    "../dataset_full/val_dataset/references/lex.jpeg": "cb0fc3ea4ffad8973b5e5eef8ffac84b84f19b467d786f9cadc0b0aeb7254d15",
    "../dataset_full/val_dataset/references/marion.jpg": "3884de5c8ca4c97840512c4976daa3cc79bb9e33eef4369c9b6ec93aed3f5a22",
    "../dataset_full/val_dataset/references/michael.jpg": "aebeb74d7df036204ad077fea58b01e89d488d5005bcc3afc8dd673568b7d0e3",
    "../dataset_full/val_dataset/references/robert.webp": "1496154a4d55749521b9e09b4b14c9294af0a555218a017a97264b574976ca5d",
    "../dataset_full/val_dataset/references/sydney.jpg": "114f74d1728558b2488cb30c3e2a7b13ded13885285f95efe9b902706f145402",
    "../dataset_full/val_dataset/references/tom.jpg": "dff3797d55eccaf1e9b72289a4f0d126ff3aee2cc79442dcb4f15124000bd5a6",
}
TWO_BRANCH_SHA256 = "9145856534abe92a6f48e9328dad5a1692ff65f27a51ac3554f9b4db82ad3689"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_pipeline(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# 10 Aug 2026 - E13C-"):
        lines = lines[1:]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    function_source = ast.get_source_segment(source, node)
    # Git hygiene removes historical trailing spaces; ignore only that
    # non-semantic difference while retaining the complete function source.
    return "\n".join(line.rstrip() for line in function_source.splitlines())


def main() -> None:
    # 10 Aug 2026 - E13C-PIPE-02: The hashes are derived from source revision
    # c04970f...+cl12-cl14-snapshot-v1-20260809. This catches any drift in
    # reference setup, RNG use, denoising schedule or prompt batching before a
    # costly fixed-panel replay is attempted.
    failures = []
    for relative_path, expected in EXPECTED.items():
        actual = _sha256(_canonical_pipeline(ROOT / relative_path))
        if actual != expected:
            failures.append(f"{relative_path}: expected {expected}, got {actual}")
    # 10 Aug 2026 - E13C-PIPE-03: Fixed prompts, identities, references and
    # bboxes are generation inputs. Seal the complete CL14 validation panel
    # alongside pipeline source rather than only checking its item count.
    for relative_path, expected in VALIDATION_INPUT_EXPECTED.items():
        path = ROOT / relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"{relative_path}: expected {expected}, got {actual}")
    runtime = ROOT / "src/model/photomaker_branched/branched_runtime.py"
    actual_two_branch = _sha256(_function_source(runtime, "two_branch_predict"))
    if actual_two_branch != TWO_BRANCH_SHA256:
        failures.append(
            "two_branch_predict: expected "
            f"{TWO_BRANCH_SHA256}, got {actual_two_branch}"
        )
    if failures:
        raise SystemExit("CL14 generation parity failed:\n" + "\n".join(failures))
    print(
        "CL14 generation parity: sealed pipeline, denoising path and "
        "fixed-96 inputs match"
    )


if __name__ == "__main__":
    main()
