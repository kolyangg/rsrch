#!/usr/bin/env python3
"""Generate the canonical PhotoMaker baseline, face boxes, and inspection PDF."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import torch
from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = REPO_ROOT / "diffusion_template"
sys.path.insert(0, str(TEMPLATE_ROOT))

from bbox_utils.generate_bboxes import clamp_bbox, enlarge_box, load_face_detector
from src.pipelines.photomaker_branched_clean import (
    PhotoMakerStableDiffusionXLPipeline,
)


DEFAULT_DATASET = REPO_ROOT / "dataset_full" / "cosmic_large_one_id"
DEFAULT_BASE_MODEL = "SG161222/RealVisXL_V4.0"
DEFAULT_PHOTOMAKER = Path("/home/niko/models/PhotoMaker-V2")
NEGATIVE_PROMPT = (
    "lowres, text, error, cropped, worst quality, low quality, "
    "jpeg artifacts, signature, watermark, blurry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--photomaker-dir", type=Path, default=DEFAULT_PHOTOMAKER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts in {path}")
    return [prompt.replace("<class>", "woman img") for prompt in prompts]


def patch_transformers_loader_compat() -> None:
    """Bridge Diffusers 0.35 and Transformers 4.57 loader arguments."""
    for model_class in (CLIPTextModel, CLIPTextModelWithProjection):
        original = model_class.from_pretrained

        def compatible_from_pretrained(
            cls, *args, _original=original, **kwargs
        ):
            kwargs.pop("offload_state_dict", None)
            return _original(*args, **kwargs)

        model_class.from_pretrained = classmethod(compatible_from_pretrained)


def generate_images(args: argparse.Namespace, prompts: list[str]) -> list[Path]:
    output_dir = args.dataset_dir / "photomaker_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / f"{idx:02d}.png" for idx in range(len(prompts))]
    missing = [
        (idx, prompt, path)
        for idx, (prompt, path) in enumerate(zip(prompts, output_paths))
        if args.force or not path.exists()
    ]
    if not missing:
        print("All PhotoMaker outputs already exist; skipping GPU generation.")
        return output_paths

    reference = Image.open(
        args.dataset_dir / "validation_refs" / "holdout_A.jpg"
    ).convert("RGB")
    patch_transformers_loader_compat()
    scheduler = DDIMScheduler.from_pretrained(
        args.base_model,
        subfolder="scheduler",
    )
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    adapter_state = torch.load(
        args.photomaker_dir / "photomaker-v2.bin",
        map_location="cpu",
        weights_only=False,
    )
    # The repository loader prints the complete state dict when passed a dict.
    # Suppress that diagnostic while retaining the useful warnings on stderr.
    with redirect_stdout(io.StringIO()):
        pipe.load_photomaker_adapter(
            adapter_state,
            subfolder="",
            weight_name="photomaker-v2.bin",
            trigger_word="img",
            pm_version="v2",
        )
    del adapter_state
    # Match the RHCA validation pipeline: its "default" adapter contains the
    # stock PhotoMaker weights.  This standalone pipeline stores those same
    # weights under the name "photomaker".
    pipe.unet.set_adapter("photomaker")
    pipe._branched_active_adapters = ["photomaker"]
    pipe.photomaker_use_lora_adapter = True
    pipe.pose_adapt_ratio = 0.0
    pipe.ca_mixing_for_face = False
    pipe.face_embed_strategy = "id"
    pipe.use_id_embeds = False
    pipe.id_alpha = 0.3
    pipe.strict_face_routing = False
    pipe.set_progress_bar_config(disable=False)

    # RHCA validates all 12 prompts in one batch and constructs one CUDA
    # generator per sample.  A CPU generator with the same integer seed does
    # not produce the same latent noise stream.
    batch_prompts = [item[1] for item in missing]
    batch_refs = [[reference.copy()] for _ in missing]
    batch_generators = [
        torch.Generator(device="cuda").manual_seed(args.seed)
        for _ in missing
    ]
    batch_ref_bboxes = [[59, 42, 203, 236] for _ in missing]
    print(
        f"Generating {len(missing)} PhotoMaker images as one RHCA-compatible "
        f"CUDA batch (DDIM, seed={args.seed})."
    )
    images = pipe(
        prompt=batch_prompts if len(batch_prompts) > 1 else batch_prompts[0],
        negative_prompt=NEGATIVE_PROMPT,
        input_id_images=batch_refs if len(batch_refs) > 1 else batch_refs[0],
        face_bbox_ref=(
            batch_ref_bboxes if len(batch_ref_bboxes) > 1 else batch_ref_bboxes[0]
        ),
        generator=(
            batch_generators if len(batch_generators) > 1 else batch_generators[0]
        ),
        num_images_per_prompt=1,
        num_inference_steps=args.steps,
        guidance_scale=5.0,
        height=args.height,
        width=args.width,
        target_size=(args.height, args.width),
        original_size=(args.height, args.width),
        crops_coords_top_left=(0, 0),
        start_merge_step=10,
        photomaker_start_step=10,
        merge_start_step=10,
        branched_attn_start_step=15,
        use_branched_attention=False,
        use_bbox_mask_ref=True,
        use_bbox_mask_gen=False,
        auto_mask_ref=False,
        use_dynamic_mask=False,
        photomaker_use_lora_adapter=True,
        val_debug=False,
    ).images
    for (idx, prompt, output_path), image in zip(missing, images):
        image.save(output_path)
        print(f"[{idx + 1:02d}/{len(prompts):02d}] saved {prompt}")

    del pipe
    torch.cuda.empty_cache()
    return output_paths


def detect_boxes(
    args: argparse.Namespace,
    prompts: list[str],
    output_paths: list[Path],
) -> dict[str, dict]:
    detector, backend = load_face_detector(
        backend="mtcnn",
        model_name="yolov8n-face.pt",
        device="cpu",
    )
    records = {}
    overlay_dir = args.dataset_dir / "bbox_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for idx, (prompt, path) in enumerate(zip(prompts, output_paths)):
        image = Image.open(path).convert("RGB")
        boxes, probabilities = detector.detect(image, landmarks=False)
        candidates = []
        if boxes is not None and probabilities is not None:
            for box, probability in zip(boxes, probabilities):
                if probability is None or probability < 0.3:
                    continue
                area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                center_y = (box[1] + box[3]) / 2.0
                # Prefer a large, high-confidence face near the upper part of
                # the image.  Area alone incorrectly selects the printed face
                # on the drummer's T-shirt; confidence alone selects a small
                # background chef instead of the foreground subject.
                vertical_penalty = 1.0 + 2.0 * (center_y / image.height) ** 2
                primary_score = area * float(probability) ** 4 / vertical_penalty
                candidates.append((primary_score, box.tolist()))
        if candidates:
            # Validation prompts describe one primary subject. The composite
            # score is robust to both background extras and face-like prints.
            face_box = max(candidates, key=lambda item: item[0])[1]
        else:
            width, height = image.size
            size = min(width, height) * 0.45
            face_box = [
                width / 2 - size,
                height / 2 - size,
                width / 2 + size,
                height / 2 + size,
            ]
        width, height = image.size
        record = {
            "face_crop_old": enlarge_box(face_box, width, height, 0.08),
            "face_crop_new": clamp_bbox(face_box, width, height),
            "body_crop": [0, 0, width, height],
        }
        record["_meta"] = {
            "prompt": prompt,
            "reference": "validation_refs/holdout_A.jpg",
            "seed": args.seed,
            "photomaker_start_step": 10,
            "branched_attn_start_step": 15,
            "num_inference_steps": args.steps,
            "base_model": args.base_model,
            "scheduler": "DDIMScheduler",
            "generator_device": "cuda",
            "generation_batch_size": len(prompts),
            "face_selection": "area_x_confidence4_over_vertical_penalty",
        }
        records[f"{idx:02d}.png"] = record

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(record["face_crop_old"], outline="red", width=6)
        draw.rectangle(record["face_crop_new"], outline="lime", width=4)
        overlay.save(overlay_dir / f"{idx:02d}.png")

    bbox_path = args.dataset_dir / "photomaker_generated_bboxes.json"
    with bbox_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    print(f"Saved {len(records)} bbox records to {bbox_path}")
    return records


def save_pdf(
    args: argparse.Namespace,
    prompts: list[str],
    output_paths: list[Path],
    records: dict[str, dict],
) -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(16, 13), constrained_layout=True)
    for idx, axis in enumerate(axes.flat):
        image = Image.open(output_paths[idx]).convert("RGB")
        record = records[f"{idx:02d}.png"]
        axis.imshow(image)
        for field, color, line_width in (
            ("face_crop_old", "red", 2.0),
            ("face_crop_new", "lime", 1.5),
        ):
            x0, y0, x1, y1 = record[field]
            axis.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=color,
                    linewidth=line_width,
                )
            )
        short_prompt = prompts[idx].replace("woman img", "woman")
        axis.set_title(f"{idx:02d} · {short_prompt}", fontsize=8)
        axis.axis("off")

    fig.suptitle(
        "Cosmic Large one-ID · PhotoMaker V2 seed 0\n"
        "green = detected face_crop_new · red = padded face_crop_old",
        fontsize=14,
    )
    pdf_path = (
        args.dataset_dir
        / "cosmic_large_one_id_photomaker_bboxes_rhca_seed0.pdf"
    )
    fig.savefig(pdf_path, format="pdf", dpi=150)
    plt.close(fig)
    print(f"Saved bbox inspection PDF to {pdf_path}")
    return pdf_path


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.dataset_dir / "validation_prompts.txt")
    output_paths = generate_images(args, prompts)
    records = detect_boxes(args, prompts, output_paths)
    save_pdf(args, prompts, output_paths, records)


if __name__ == "__main__":
    main()
