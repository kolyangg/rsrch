"""Checkpoint-only A-E generation matrix for NN2-PPR inference diagnosis."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from src.model.photomaker_branched.packed_residual_attn_processor import (
    make_inner_core_mask,
)


OPTIONS = {
    "A": "A_exact_pm",
    "B": "B_current_ppr",
    "C": "C_no_anchor",
    "D": "D_ppr_x4",
    "E": "E_reference_swap",
}

METRIC_FIELDS = [
    "sample_index",
    "filename",
    "option",
    "identity",
    "spatial_swap_identity",
    "seed",
    "prompt",
    "sha256",
    "whole_image_mae_vs_A",
    "face_core_mae_vs_A",
    "id_similarity",
    "text_similarity",
]


def _diagnostic_options(config) -> tuple[str, ...]:
    raw = getattr(config, "ppr_diagnostic_options", tuple(OPTIONS))
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.strip("[]").split(",") if part.strip()]
    selected = tuple(str(option).upper() for option in raw)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"Invalid ppr_diagnostic_options: {selected}")
    unknown = set(selected) - set(OPTIONS)
    if unknown:
        raise ValueError(f"Unknown PPR diagnostic options: {sorted(unknown)}")
    if selected != tuple(OPTIONS) and selected != ("E",):
        raise ValueError(
            "PPR diagnostics currently support either [A,B,C,D,E] or E-only"
        )
    return selected


def _per_sample(value: Any, batch_size: int) -> list[Any]:
    if batch_size == 1:
        return [value]
    if isinstance(value, list) and len(value) == batch_size:
        return value
    return [value] * batch_size


def _normalize_refs(value: Any, batch_size: int) -> list[list[Image.Image]]:
    refs = _per_sample(value, batch_size)
    normalized = []
    for sample_refs in refs:
        if isinstance(sample_refs, (list, tuple)):
            sample_refs = list(sample_refs)
        else:
            sample_refs = [sample_refs]
        if not sample_refs:
            raise RuntimeError("PPR diagnostic sample has no reference image")
        normalized.append(sample_refs)
    return normalized


def _select_spatial_swap_indices(dataset, count: int) -> set[int]:
    entries = []
    for index in range(len(dataset)):
        sample = dataset[index]
        bbox = sample.get("face_bbox_gen")
        if bbox is None or len(bbox) != 4:
            continue
        width = max(float(bbox[2]) - float(bbox[0]), 0.0)
        height = max(float(bbox[3]) - float(bbox[1]), 0.0)
        entries.append((width * height, index))
    if not entries:
        return set(range(min(count, len(dataset))))
    entries.sort()
    count = min(max(int(count), 1), len(entries))
    groups = np.array_split(np.arange(len(entries)), 3)
    selected = []
    base = count // 3
    remainder = count % 3
    for group_index, group in enumerate(groups):
        take = base + (1 if group_index < remainder else 0)
        if take <= 0 or len(group) == 0:
            continue
        positions = np.linspace(0, len(group) - 1, num=take, dtype=int)
        selected.extend(entries[int(group[position])][1] for position in range(take))
    return set(selected[:count])


def _initialize_state(trainer) -> dict[str, Any]:
    root = Path(str(trainer.config.ppr_diagnostic_output_dir)).expanduser().resolve()
    overwrite = bool(getattr(trainer.config, "ppr_diagnostic_overwrite", False))
    options = _diagnostic_options(trainer.config)
    reuse_output = bool(
        getattr(trainer.config, "ppr_diagnostic_reuse_output", False)
    )
    e_only = options == ("E",)
    if reuse_output and not e_only:
        raise ValueError("ppr_diagnostic_reuse_output is supported only for E-only mode")
    if e_only and not reuse_output:
        raise ValueError(
            "E-only mode requires ppr_diagnostic_reuse_output=true and an existing matrix"
        )
    if reuse_output and overwrite:
        raise ValueError(
            "E-only reuse and ppr_diagnostic_overwrite=true are mutually exclusive"
        )

    if root.exists() and any(root.iterdir()) and not reuse_output:
        if not overwrite:
            raise FileExistsError(
                f"PPR diagnostic output already exists: {root}. "
                "Set ppr_diagnostic_overwrite=true to replace it."
            )
        shutil.rmtree(root)

    if len(trainer.evaluation_dataloaders) != 1:
        raise RuntimeError("PPR diagnostic matrix requires exactly one validation dataset")
    dataloader = next(iter(trainer.evaluation_dataloaders.values()))
    dataset = dataloader.dataset
    swap_count = int(getattr(trainer.config, "ppr_diagnostic_swap_count", 12))
    swap_indices = _select_spatial_swap_indices(dataset, swap_count)

    rows = []
    epsilon = []
    if reuse_output:
        manifest_path = root / "manifest.json"
        metrics_path = root / "metrics.csv"
        epsilon_path = root / "epsilon_diagnostics.jsonl"
        baseline_dir = root / OPTIONS["A"]
        required = (manifest_path, metrics_path, epsilon_path, baseline_dir)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "E-only reuse requires the completed existing matrix; missing: "
                + ", ".join(missing)
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("sample_count", -1)) != len(dataset):
            raise RuntimeError(
                "Existing diagnostic sample count does not match current dataset: "
                f"{manifest.get('sample_count')} vs {len(dataset)}"
            )
        existing_checkpoint_value = str(manifest.get("checkpoint", "")).strip()
        active_checkpoint_value = str(
            getattr(trainer.config, "saved_checkpoint", "")
        ).strip()
        if (
            not existing_checkpoint_value
            or not active_checkpoint_value
            or Path(existing_checkpoint_value).expanduser().resolve()
            != Path(active_checkpoint_value).expanduser().resolve()
        ):
            raise RuntimeError(
                "E-only reuse checkpoint mismatch: "
                f"existing={existing_checkpoint_value}, active={active_checkpoint_value}"
            )
        existing_base = str(manifest.get("validation_base", ""))
        active_base = str(
            getattr(
                trainer.config,
                "pretrained_model_for_validation_name_or_path",
                "",
            )
        )
        if existing_base != active_base:
            raise RuntimeError(
                "E-only reuse validation-base mismatch: "
                f"existing={existing_base}, active={active_base}"
            )
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if str(row.get("option", "")).upper() != "E"
            ]
        with epsilon_path.open("r", encoding="utf-8") as handle:
            epsilon = [
                record
                for line in handle
                if line.strip()
                for record in (json.loads(line),)
                if str(record.get("variant", "")).upper() != "E"
            ]
        if len(list(baseline_dir.glob("*.png"))) < len(dataset):
            raise RuntimeError(
                "E-only reuse requires an A_exact_pm PNG for every validation sample"
            )

    root.mkdir(parents=True, exist_ok=True)
    for option, directory in OPTIONS.items():
        if option in options or not reuse_output:
            (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "contact_sheets").mkdir(parents=True, exist_ok=True)

    identity_sources = []
    for image_path in getattr(dataset, "images", ()):
        identity = image_path.stem
        bbox = getattr(dataset, "_bbox_map_ref", {}).get(identity)
        identity_sources.append((identity, Path(image_path), bbox))
    if len({identity for identity, _, _ in identity_sources}) < 2:
        raise RuntimeError("Reference-swap diagnostic requires at least two identities")

    state = {
        "root": root,
        "rows": rows,
        "epsilon": epsilon,
        "next_index": 0,
        "swap_indices": swap_indices,
        "identity_sources": identity_sources,
        "filenames": [],
        "options": options,
        "reuse_output": reuse_output,
    }
    trainer._ppr_diagnostic_state = state
    print(
        "[PPR diagnostic matrix] "
        f"output={root} options={list(options)} reuse={reuse_output} "
        f"samples={len(dataset)} swap_indices={sorted(swap_indices)}"
    )
    return state


def _swap_source(state: dict[str, Any], original_identity: str):
    sources = state["identity_sources"]
    for identity, path, bbox in sources:
        if identity != original_identity:
            return identity, Image.open(path).convert("RGB"), bbox
    raise RuntimeError(f"No alternate identity is available for {original_identity}")


def _metric_values(trainer, *, image, prompt: str, identity: str) -> dict[str, float]:
    values = {"id_similarity": float("nan"), "text_similarity": float("nan")}
    sample = {"generated": [image], "prompt": prompt, "id": identity}
    for metric in trainer.metrics:
        class_name = metric.__class__.__name__
        if class_name not in {"IDSimBest", "IDSimMax", "TextSimMetric"}:
            continue
        result = metric(**sample)
        if "id_sim" in result:
            values["id_similarity"] = float(result["id_sim"])
        if "text_sim" in result:
            values["text_similarity"] = float(result["text_sim"])
    return values


def _face_core_mask(image: Image.Image, bbox) -> np.ndarray | None:
    if bbox is None or len(bbox) != 4:
        return None
    width, height = image.size
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = torch.zeros(1, 1, height, width)
    mask[:, :, y0:y1, x0:x1] = 1
    return make_inner_core_mask(mask, erode_frac=0.10)[0, 0].numpy()


def _pixel_mae(image: Image.Image, baseline: Image.Image, bbox) -> tuple[float, float]:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    baseline_array = np.asarray(baseline.convert("RGB"), dtype=np.float32) / 255.0
    if image_array.shape != baseline_array.shape:
        raise RuntimeError(
            f"Diagnostic image shape mismatch: {image_array.shape} vs {baseline_array.shape}"
        )
    absolute = np.abs(image_array - baseline_array)
    whole = float(absolute.mean())
    core = _face_core_mask(image, bbox)
    if core is None or float(core.sum()) <= 0:
        return whole, float("nan")
    face = float((absolute * core[:, :, None]).sum() / (core.sum() * 3.0))
    return whole, face


def _save_image(root: Path, option: str, filename: str, image: Image.Image) -> str:
    path = root / OPTIONS[option] / filename
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(
    trainer,
    *,
    option: str,
    prompts: list[str],
    seeds: list[int],
    identities: list[str],
    references: list[list[Image.Image]],
    reference_bboxes: list[Any],
    generation_bboxes: list[Any],
    sample_keys: list[str],
    ppr_reference_image=None,
    ppr_face_bbox_ref=None,
    runtime_settings: tuple[bool, str, float] | None = None,
    diagnostic_variant: str | None = None,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    pipeline = trainer.pipe
    settings = {
        "A": (True, "base_outside_core", 1.0),
        "B": (False, "base_outside_core", 1.0),
        "C": (False, "none", 1.0),
        "D": (False, "base_outside_core", 4.0),
        "E": (False, "base_outside_core", 1.0),
    }
    force_base, anchor_mode, runtime_scale = (
        runtime_settings if runtime_settings is not None else settings[option]
    )
    previous = {
        name: getattr(pipeline, name, None)
        for name in (
            "ba_ppr_force_base_output",
            "ba_output_anchor_mode",
            "ba_ppr_runtime_scale",
            "ba_ppr_collect_diagnostics",
            "ba_ppr_diagnostic_variant",
            "ba_ppr_diagnostic_sample_keys",
        )
    }
    processor_records: list[dict[str, Any]] = []
    epsilon_records: list[dict[str, Any]] = []
    pipeline.ba_ppr_force_base_output = force_base
    pipeline.ba_output_anchor_mode = anchor_mode
    pipeline.ba_ppr_runtime_scale = runtime_scale
    pipeline.ba_ppr_collect_diagnostics = True
    pipeline.ba_ppr_diagnostic_variant = diagnostic_variant or option
    pipeline.ba_ppr_diagnostic_sample_keys = tuple(sample_keys)
    pipeline._ba_ppr_processor_diagnostics = processor_records
    pipeline._ba_ppr_epsilon_diagnostics = epsilon_records

    kwargs = OmegaConf.to_container(
        trainer.config.validation_args,
        resolve=True,
    )
    kwargs["debug_dir"] = None
    kwargs["val_debug"] = False
    kwargs["use_branched_attention"] = True
    generators = [
        torch.Generator(device=trainer.device).manual_seed(int(seed))
        for seed in seeds
    ]
    batch_size = len(prompts)
    try:
        result = pipeline(
            prompt=prompts if batch_size > 1 else prompts[0],
            generator=generators if batch_size > 1 else generators[0],
            input_id_images=references if batch_size > 1 else references[0],
            face_bbox_ref=reference_bboxes if batch_size > 1 else reference_bboxes[0],
            face_bbox_gen=generation_bboxes if batch_size > 1 else generation_bboxes[0],
            ppr_reference_image=ppr_reference_image,
            ppr_face_bbox_ref=ppr_face_bbox_ref,
            **kwargs,
        )
        images = result.images
        if not isinstance(images, list):
            images = [images]
        if len(images) != batch_size:
            raise RuntimeError(
                f"PPR diagnostic {option} returned {len(images)} images for batch {batch_size}"
            )
        fingerprints = dict(
            getattr(pipeline, "_ba_ppr_randomness_fingerprints", {})
        )
    finally:
        for name, value in previous.items():
            setattr(pipeline, name, value)
        pipeline._ba_ppr_processor_diagnostics = None
        pipeline._ba_ppr_epsilon_diagnostics = None

    for record in processor_records:
        record["samples"] = list(sample_keys)
    randomness_records = []
    latent_hashes = fingerprints.get("initial_latents_sha256", ())
    reference_hashes = fingerprints.get("reference_noise_sha256", ())
    if len(latent_hashes) != batch_size or len(reference_hashes) != batch_size:
        raise RuntimeError(
            f"PPR diagnostic {option} did not record per-sample randomness fingerprints"
        )
    for sample_key, latent_hash, reference_hash in zip(
        sample_keys,
        latent_hashes,
        reference_hashes,
    ):
        randomness_records.append(
            {
                "record_type": "generation_randomness",
                "variant": option,
                "sample": sample_key,
                "initial_latents_sha256": latent_hash,
                "reference_noise_sha256": reference_hash,
            }
        )
    return images, epsilon_records + processor_records + randomness_records


@torch.no_grad()
def run_ppr_diagnostic_batch(trainer, batch, eval_metrics):
    state = getattr(trainer, "_ppr_diagnostic_state", None)
    if state is None:
        state = _initialize_state(trainer)

    prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
    batch_size = len(prompts)
    identities = [
        str(value)
        for value in _per_sample(batch.get("id"), batch_size)
    ]
    seeds = [
        int(value)
        for value in _per_sample(
            batch.get("seed", trainer.config.validation_args.get("seed", 0)),
            batch_size,
        )
    ]
    references = _normalize_refs(batch.get("ref_images"), batch_size)
    reference_bboxes = _per_sample(batch.get("face_bbox_ref"), batch_size)
    generation_bboxes = _per_sample(batch.get("face_bbox_gen"), batch_size)
    if any(bbox is None for bbox in generation_bboxes):
        raise RuntimeError("PPR diagnostic matrix requires fixed generation bboxes")

    start_index = int(state["next_index"])
    global_indices = list(range(start_index, start_index + batch_size))
    state["next_index"] += batch_size
    filenames = [
        f"{global_index:03d}_{identity}_seed{seed}.png"
        for global_index, identity, seed in zip(global_indices, identities, seeds)
    ]
    state["filenames"].extend(filenames)

    if state["reuse_output"]:
        missing_baselines = [
            str(state["root"] / OPTIONS["A"] / filename)
            for filename in filenames
            if not (state["root"] / OPTIONS["A"] / filename).exists()
        ]
        if missing_baselines:
            raise FileNotFoundError(
                "E-only baseline filenames do not match this validation ordering; "
                f"first missing: {missing_baselines[0]}"
            )

    generated = {}
    if not state["reuse_output"]:
        for option in ("A", "B", "C", "D"):
            images, diagnostics = _generate(
                trainer,
                option=option,
                prompts=prompts,
                seeds=seeds,
                identities=identities,
                references=references,
                reference_bboxes=reference_bboxes,
                generation_bboxes=generation_bboxes,
                sample_keys=filenames,
            )
            generated[option] = images
            state["epsilon"].extend(diagnostics)

    if "E" in state["options"]:
        for local_index, global_index in enumerate(global_indices):
            if global_index not in state["swap_indices"]:
                continue
            swap_identity, swap_image, swap_bbox = _swap_source(
                state,
                identities[local_index],
            )
            images, diagnostics = _generate(
                trainer,
                option="E",
                prompts=[prompts[local_index]],
                seeds=[seeds[local_index]],
                identities=[identities[local_index]],
                references=[references[local_index]],
                reference_bboxes=[reference_bboxes[local_index]],
                generation_bboxes=[generation_bboxes[local_index]],
                sample_keys=[filenames[local_index]],
                ppr_reference_image=[swap_image],
                ppr_face_bbox_ref=swap_bbox,
            )
            generated.setdefault("E", {})[local_index] = (images[0], swap_identity)
            state["epsilon"].extend(diagnostics)

    display_images = []
    for local_index, filename in enumerate(filenames):
        if state["reuse_output"]:
            baseline_path = state["root"] / OPTIONS["A"] / filename
            if not baseline_path.exists():
                raise FileNotFoundError(f"Missing E-only baseline: {baseline_path}")
            with Image.open(baseline_path) as baseline_file:
                baseline = baseline_file.convert("RGB")
        else:
            baseline = generated["A"][local_index]
        for option in state["options"]:
            swap_identity = ""
            if option == "E":
                record = generated.get("E", {}).get(local_index)
                if record is None:
                    continue
                image, swap_identity = record
            else:
                image = generated[option][local_index]
            sha256 = _save_image(state["root"], option, filename, image)
            whole_mae, face_mae = _pixel_mae(
                image,
                baseline,
                generation_bboxes[local_index],
            )
            metric_values = _metric_values(
                trainer,
                image=image,
                prompt=prompts[local_index],
                identity=identities[local_index],
            )
            for metric_name, metric_value in metric_values.items():
                if np.isfinite(metric_value):
                    eval_metrics.update(f"{option}/{metric_name}", metric_value)
            state["rows"].append(
                {
                    "sample_index": global_indices[local_index],
                    "filename": filename,
                    "option": option,
                    "identity": identities[local_index],
                    "spatial_swap_identity": swap_identity,
                    "seed": seeds[local_index],
                    "prompt": prompts[local_index],
                    "sha256": sha256,
                    "whole_image_mae_vs_A": whole_mae,
                    "face_core_mae_vs_A": face_mae,
                    **metric_values,
                }
            )
        e_record = generated.get("E", {}).get(local_index)
        display_images.append(e_record[0] if e_record is not None else baseline)

    if state["reuse_output"]:
        batch["generated"] = display_images
    else:
        batch["generated"] = (
            generated["B"] if batch_size > 1 else [generated["B"][0]]
        )
    batch["generated_masks"] = [None] * batch_size
    return batch


def _create_contact_sheets(state: dict[str, Any], rows_per_page: int = 6) -> None:
    root = state["root"]
    filenames = list(dict.fromkeys(state["filenames"]))
    columns = list(OPTIONS.items())
    cell_width, cell_height, label_height = 256, 256, 24
    for page_start in range(0, len(filenames), rows_per_page):
        page_files = filenames[page_start : page_start + rows_per_page]
        sheet = Image.new(
            "RGB",
            (cell_width * len(columns), label_height + (cell_height + label_height) * len(page_files)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for column_index, (option, label) in enumerate(columns):
            draw.text(
                (column_index * cell_width + 4, 4),
                f"{option}: {label}",
                fill="black",
            )
        for row_index, filename in enumerate(page_files):
            y = label_height + row_index * (cell_height + label_height)
            for column_index, (option, directory) in enumerate(columns):
                image_path = root / directory / filename
                if image_path.exists():
                    image = Image.open(image_path).convert("RGB")
                    image.thumbnail((cell_width, cell_height))
                    x = column_index * cell_width + (cell_width - image.width) // 2
                    sheet.paste(image, (x, y))
                else:
                    draw.rectangle(
                        (
                            column_index * cell_width,
                            y,
                            (column_index + 1) * cell_width - 1,
                            y + cell_height - 1,
                        ),
                        fill=(225, 225, 225),
                    )
            draw.text((4, y + cell_height + 3), filename, fill="black")
        page_end = page_start + len(page_files) - 1
        sheet.save(
            root / "contact_sheets" / f"samples_{page_start:03d}_{page_end:03d}.jpg",
            quality=92,
        )


def finalize_ppr_diagnostic_matrix(trainer) -> None:
    state = getattr(trainer, "_ppr_diagnostic_state", None)
    if state is None:
        raise RuntimeError("PPR diagnostic matrix produced no batches")
    root = state["root"]
    rows = sorted(
        state["rows"],
        key=lambda row: (int(row["sample_index"]), str(row["option"])),
    )
    with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "epsilon_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for record in state["epsilon"]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    randomness = {}
    for record in state["epsilon"]:
        if record.get("record_type") != "generation_randomness":
            continue
        randomness.setdefault(record["sample"], {})[record["variant"]] = (
            record["initial_latents_sha256"],
            record["reference_noise_sha256"],
        )
    for sample, variants in randomness.items():
        missing = set(("A", "B", "C", "D")) - set(variants)
        if missing:
            raise RuntimeError(
                f"Missing randomness fingerprints for {sample}: {sorted(missing)}"
            )
        baseline = variants["A"]
        for option in ("B", "C", "D", "E"):
            if option in variants and variants[option] != baseline:
                raise RuntimeError(
                    f"Generation randomness mismatch for sample={sample}, option={option}"
                )
    manifest = {
        "checkpoint": str(getattr(trainer.config, "saved_checkpoint", "")),
        "validation_base": str(
            getattr(
                trainer.config,
                "pretrained_model_for_validation_name_or_path",
                "",
            )
        ),
        "sample_count": int(state["next_index"]),
        "spatial_swap_indices": sorted(state["swap_indices"]),
        "options": OPTIONS,
        "generated_options": list(state["options"]),
        "reused_existing_output": bool(state["reuse_output"]),
        "diagnostic_steps": [15, 25, 35, 49],
        "randomness_fingerprints_verified": True,
        "validation_args": OmegaConf.to_container(
            trainer.config.validation_args,
            resolve=True,
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if state["reuse_output"]:
        expected_filenames = set(state["filenames"])
        for image_path in (root / OPTIONS["E"]).glob("*.png"):
            if image_path.name not in expected_filenames:
                image_path.unlink()
        shutil.rmtree(root / "contact_sheets", ignore_errors=True)
        (root / "contact_sheets").mkdir(parents=True, exist_ok=True)
    _create_contact_sheets(state)
    print(
        "[PPR diagnostic matrix complete] "
        f"output={root} rows={len(rows)} epsilon_records={len(state['epsilon'])}"
    )
