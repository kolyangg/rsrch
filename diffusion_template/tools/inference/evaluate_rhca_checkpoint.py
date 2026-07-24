#!/usr/bin/env python3
"""Evaluate an RHCA checkpoint on a fixed validation set without training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_ROOT = PROJECT_ROOT / "src" / "configs"


class _NoAccelerator:
    is_main_process = True
    num_processes = 1

    @staticmethod
    def unwrap_model(model, keep_fp32_wrapper=False):
        del keep_fp32_wrapper
        return model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_config(config_arg: str) -> tuple[DictConfig, str]:
    config_path = Path(config_arg)
    if config_path.is_file():
        config = OmegaConf.load(config_path)
        source = str(config_path.resolve())
    else:
        name = config_arg
        if name.endswith(".yaml"):
            name = name[:-5]
        with hydra.initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIG_ROOT),
        ):
            config = hydra.compose(config_name=name)
        source = f"hydra:{name}"
    OmegaConf.set_struct(config, False)
    return config, source


def checkpoint_state(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a mapping, got {type(checkpoint)}")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or "lora_weights" not in state:
        raise ValueError(
            "Checkpoint does not contain the RHCA lora_weights state structure"
        )
    metadata = {
        "kind": "full" if "state_dict" in checkpoint else "weights_only",
        "epoch": checkpoint.get("epoch"),
        "top_level_keys": sorted(str(key) for key in checkpoint),
        "lora_tensor_count": len(state.get("lora_weights", {})),
        "processor_count": len(state.get("attn_processors", {})),
    }
    return state, metadata


def configure_model_before_install(
    model,
    config: DictConfig,
    *,
    disable_branched_ca: bool,
) -> None:
    # 24 Jul 2026 - Processor type is decided at first installation; setting
    # this flag afterwards does not remove an already-installed branched CA.
    setattr(
        model,
        "disable_branched_sa",
        bool(getattr(config, "disable_branched_sa", False)),
    )
    setattr(model, "disable_branched_ca", bool(disable_branched_ca))
    setattr(
        model,
        "strict_face_routing",
        bool(getattr(config, "strict_face_routing", False)),
    )


def processor_base_buffers(model) -> dict[str, torch.Tensor]:
    return {
        name: buffer.detach().cpu().clone()
        for name, buffer in model.unet.named_buffers()
        if ".processor." in name and name.endswith("base_weight")
    }


def processor_type_audit(model) -> dict:
    by_type: dict[str, int] = {}
    branched_self = []
    branched_cross = []
    for name, processor in model.unet.attn_processors.items():
        type_name = type(processor).__name__
        by_type[type_name] = by_type.get(type_name, 0) + 1
        if type_name == "BranchedAttnProcessor":
            branched_self.append(name)
        elif type_name == "BranchedCrossAttnProcessor":
            branched_cross.append(name)
    return {
        "by_type": by_type,
        "branched_self_attention_count": len(branched_self),
        "branched_cross_attention_count": len(branched_cross),
        "branched_self_attention_names": branched_self,
        "branched_cross_attention_names": branched_cross,
    }


def compare_base_buffers(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict:
    changed = [
        name
        for name in sorted(set(before) | set(after))
        if name not in before
        or name not in after
        or not torch.equal(before[name], after[name])
    ]
    return {
        "buffer_count_before": len(before),
        "buffer_count_after": len(after),
        "changed_count": len(changed),
        "changed_examples": changed[:20],
    }


def instantiate_prepared_model(
    config: DictConfig,
    *,
    base_model: str,
    device: torch.device,
    disable_branched_ca: bool,
):
    previous_base = config.model.pretrained_model_name_or_path
    try:
        config.model.pretrained_model_name_or_path = base_model
        model = instantiate(config.model, device=device)
    finally:
        config.model.pretrained_model_name_or_path = previous_base
    configure_model_before_install(
        model,
        config,
        disable_branched_ca=disable_branched_ca,
    )
    model.prepare_for_training()
    return model


def clone_processor_states_to_cpu(model) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for name, processor in model.unet.attn_processors.items():
        if not isinstance(processor, torch.nn.Module):
            continue
        result[name] = {
            key: value.detach().cpu().clone()
            for key, value in processor.state_dict().items()
        }
    return result


def load_evaluation_model(
    config: DictConfig,
    state: dict,
    *,
    validation_base: str,
    processor_base_mode: str,
    device: torch.device,
    disable_branched_ca: bool,
) -> tuple[Any, dict]:
    training_base = str(config.model.pretrained_model_name_or_path)
    audit = {
        "mode": processor_base_mode,
        "training_base": training_base,
        "validation_base": validation_base,
        "disable_branched_ca": disable_branched_ca,
    }

    legacy_processor_states = None
    if processor_base_mode == "legacy_full_copy":
        training_model = instantiate_prepared_model(
            config,
            base_model=training_base,
            device=device,
            disable_branched_ca=disable_branched_ca,
        )
        training_model.load_state_dict_(state)
        legacy_processor_states = clone_processor_states_to_cpu(training_model)
        audit["legacy_source_processor_count"] = len(legacy_processor_states)
        training_model.to("cpu")
        del training_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validation_model = instantiate_prepared_model(
        config,
        base_model=validation_base,
        device=device,
        disable_branched_ca=disable_branched_ca,
    )
    installed_processors = processor_type_audit(validation_model)
    audit["installed_processors"] = installed_processors
    if (
        disable_branched_ca
        and installed_processors["branched_cross_attention_count"] != 0
    ):
        raise RuntimeError(
            "disable_branched_ca was set before installation, but branched "
            "cross-attention processors are still present"
        )
    if installed_processors["branched_self_attention_count"] == 0:
        raise RuntimeError("No branched self-attention processors were installed")
    base_before_delta = processor_base_buffers(validation_model)
    validation_model.load_state_dict_(state)
    base_after_delta = processor_base_buffers(validation_model)
    delta_audit = compare_base_buffers(base_before_delta, base_after_delta)
    audit["delta_load_base_buffer_audit"] = delta_audit
    if delta_audit["changed_count"]:
        raise RuntimeError(
            "Trainable-delta loading unexpectedly changed validation-native "
            f"processor base buffers: {delta_audit['changed_examples']}"
        )

    if legacy_processor_states is not None:
        copied = 0
        for name, source_state in legacy_processor_states.items():
            destination = validation_model.unet.attn_processors.get(name)
            if destination is None or not hasattr(destination, "load_state_dict"):
                continue
            destination.load_state_dict(source_state, strict=False)
            copied += 1
        audit["legacy_full_processor_copy_count"] = copied
        audit["legacy_copy_base_buffer_audit"] = compare_base_buffers(
            base_after_delta,
            processor_base_buffers(validation_model),
        )
    else:
        audit["validation_native_invariant"] = (
            "validation-base processor buffers unchanged by checkpoint delta loading"
        )

    validation_model.to(device)
    return validation_model, audit


def build_pipeline(
    config: DictConfig,
    model,
    *,
    validation_base: str,
    device: torch.device,
    disable_branched_ca: bool,
):
    previous_base = config.pipeline.pretrained_model_name_or_path
    try:
        config.pipeline.pretrained_model_name_or_path = validation_base
        pipeline = instantiate(
            config.pipeline,
            model=model,
            accelerator=_NoAccelerator(),
            _recursive_=False,
        )
    finally:
        config.pipeline.pretrained_model_name_or_path = previous_base

    setattr(
        pipeline,
        "disable_branched_sa",
        bool(getattr(config, "disable_branched_sa", False)),
    )
    setattr(pipeline, "disable_branched_ca", bool(disable_branched_ca))
    pipeline.to(device)
    if hasattr(pipeline, "id_encoder"):
        pipeline.id_encoder.to(device=device, dtype=pipeline.unet.dtype)
    return pipeline


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def instantiate_metrics(config: DictConfig, device: torch.device, *, skip: bool):
    if skip:
        return []
    metrics = []
    for metric_name in config.inference_metrics:
        metric = instantiate(
            config.metrics[metric_name],
            name=metric_name,
            device=device,
        )
        metric.to_cuda()
        metrics.append(metric)
    return metrics


def output_filename(prompt: str, person_id: str) -> str:
    label = f"{prompt[:10]}_{person_id}"
    return f"{label.replace(' ', '_')[:80]}.png"


def find_distinct_dataset_reference(samples: list[dict]) -> tuple[Image.Image, Any] | None:
    first_bytes = None
    for sample in samples:
        refs = sample["ref_images"]
        ref = refs[0] if isinstance(refs, (list, tuple)) else refs
        payload = ref.convert("RGB").tobytes()
        digest = hashlib.sha256(payload).digest()
        if first_bytes is None:
            first_bytes = digest
        elif digest != first_bytes:
            return ref, sample.get("face_bbox_ref")
    return None


def apply_reference_condition(
    args,
    samples: list[dict],
) -> tuple[list[list[Image.Image]], list[Any], torch.Tensor | None, dict]:
    matched_refs = [
        list(sample["ref_images"])
        if isinstance(sample["ref_images"], (list, tuple))
        else [sample["ref_images"]]
        for sample in samples
    ]
    matched_bboxes = [sample.get("face_bbox_ref") for sample in samples]

    if args.reference_condition == "matched":
        return matched_refs, matched_bboxes, None, {
            "kind": "matched",
            "scope": "PhotoMaker and spatial BA reference",
        }

    if args.reference_condition == "wrong":
        if args.wrong_reference is not None:
            wrong_image = Image.open(args.wrong_reference).convert("RGB")
            wrong_bbox = args.wrong_reference_bbox
            if wrong_bbox is None:
                raise ValueError(
                    "--wrong-reference-bbox is required with --wrong-reference"
                )
            source = str(args.wrong_reference.resolve())
        else:
            distinct = find_distinct_dataset_reference(samples)
            if distinct is None:
                raise ValueError(
                    "Validation dataset has no distinct wrong identity; provide "
                    "--wrong-reference and --wrong-reference-bbox."
                )
            wrong_image, wrong_bbox = distinct
            source = "distinct validation-dataset reference"
        if wrong_bbox is None:
            raise ValueError("Wrong-reference intervention requires an explicit bbox")
        return (
            [[wrong_image.copy()] for _ in samples],
            [list(wrong_bbox) for _ in samples],
            None,
            {
                "kind": "wrong",
                "scope": "end-to-end PhotoMaker and spatial BA reference replacement",
                "source": source,
            },
        )

    neutral = Image.new("RGB", (256, 256), color=(127, 127, 127))
    zero_id_embeds = torch.zeros(len(samples), 512, dtype=torch.float32)
    return (
        [[neutral.copy()] for _ in samples],
        [[0.0, 0.0, 0.0, 0.0] for _ in samples],
        zero_id_embeds,
        {
            "kind": "null",
            "scope": (
                "end-to-end intervention: zero PhotoMaker identity embedding, "
                "neutral spatial reference, and zero reference face mask"
            ),
        },
    )


def run(args) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(path.is_file() for path in output_dir.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    config, config_source = load_config(args.config)
    if args.validation_dataset not in config.datasets.val:
        raise KeyError(
            f"Validation dataset {args.validation_dataset!r} is not registered"
        )
    config.val_datasets_names = [args.validation_dataset]
    config.datasets.val[args.validation_dataset].limit = int(args.limit)
    config.validation_args.guidance_scale = float(args.guidance_scale)
    config.validation_args.num_images_per_prompt = 1
    config.validation_args.val_debug = False
    config.val_debug = False

    validation_base = (
        args.validation_base
        or str(
            getattr(
                config,
                "pretrained_model_for_validation_name_or_path",
                config.model.pretrained_model_name_or_path,
            )
        )
    )
    config.pretrained_model_for_validation_name_or_path = validation_base
    config.disable_branched_ca = bool(args.disable_branched_ca)
    config.model.train_branched_ca_lora = not bool(args.disable_branched_ca)

    checkpoint_path = args.checkpoint.resolve()
    state, checkpoint_metadata = checkpoint_state(checkpoint_path)
    processor_names = list((state.get("attn_processors") or {}).keys())
    has_saved_ca = any(name.endswith("attn2.processor") for name in processor_names)
    if (
        not args.disable_branched_ca
        and processor_names
        and not has_saved_ca
        and not args.allow_untrained_ca
    ):
        raise ValueError(
            "Checkpoint has saved branched self-attention processors but no "
            "branched CA state. Keep --disable-branched-ca enabled; using CA-on "
            "would invent untrained CA weights."
        )

    device = torch.device(args.device)
    model, processor_audit = load_evaluation_model(
        config,
        state,
        validation_base=validation_base,
        processor_base_mode=args.processor_base_mode,
        device=device,
        disable_branched_ca=bool(args.disable_branched_ca),
    )
    pipeline = build_pipeline(
        config,
        model,
        validation_base=validation_base,
        device=device,
        disable_branched_ca=bool(args.disable_branched_ca),
    )

    dataset = instantiate(config.datasets.val[args.validation_dataset])
    samples = [dataset[index] for index in range(min(len(dataset), args.limit))]
    if len(samples) != args.limit:
        raise ValueError(f"Requested {args.limit} samples, dataset returned {len(samples)}")

    conditioned_refs, conditioned_bboxes, id_embeds, intervention = (
        apply_reference_condition(args, samples)
    )
    prompts = [str(sample["prompt"]) for sample in samples]
    person_ids = [str(sample["id"]) for sample in samples]
    seeds = [int(sample.get("seed", 0)) for sample in samples]
    generation_bboxes = [sample.get("face_bbox_gen") for sample in samples]
    if bool(config.validation_args.get("use_bbox_mask_gen", False)) and any(
        bbox is None for bbox in generation_bboxes
    ):
        raise ValueError(
            "Fixed validation requires face_bbox_gen for every sample; the "
            "configured generation-bbox JSON is incomplete."
        )

    generators = [
        torch.Generator(device=device).manual_seed(seed) for seed in seeds
    ]
    validation_kwargs = OmegaConf.to_container(
        config.validation_args,
        resolve=True,
    )
    validation_kwargs["debug_dir"] = None
    validation_kwargs["debug_idx"] = 0
    validation_kwargs["debug_total"] = len(samples)
    validation_kwargs["val_debug"] = False

    with torch.no_grad():
        generated = pipeline(
            prompt=prompts if len(prompts) > 1 else prompts[0],
            input_id_images=(
                conditioned_refs if len(prompts) > 1 else conditioned_refs[0]
            ),
            face_bbox_ref=(
                conditioned_bboxes if len(prompts) > 1 else conditioned_bboxes[0]
            ),
            face_bbox_gen=(
                generation_bboxes if len(prompts) > 1 else generation_bboxes[0]
            ),
            generator=generators if len(prompts) > 1 else generators[0],
            id_embeds=(id_embeds.to(device) if id_embeds is not None else None),
            **validation_kwargs,
        ).images
    if not isinstance(generated, list):
        generated = [generated]
    if len(generated) != len(samples):
        raise RuntimeError(
            f"Expected {len(samples)} generated images, got {len(generated)}"
        )

    metrics = instantiate_metrics(config, device, skip=args.skip_metrics)
    per_image = []
    seen_filenames = set()
    for index, (sample, image) in enumerate(zip(samples, generated)):
        filename = output_filename(prompts[index], person_ids[index])
        if filename in seen_filenames:
            raise RuntimeError(f"Duplicate validation output filename: {filename}")
        seen_filenames.add(filename)
        image_path = images_dir / filename
        image.save(image_path)

        metric_values = {}
        metric_batch = dict(sample)
        metric_batch.update(
            {
                "prompt": prompts[index],
                "id": person_ids[index],
                # Metrics remain anchored to the true held-out identity even
                # for wrong/null causal interventions.
                "ref_images": sample["ref_images"],
                "generated": [image],
            }
        )
        for metric in metrics:
            metric_values.update(to_jsonable(metric(**metric_batch)))
        per_image.append(
            {
                "index": index,
                "filename": filename,
                "prompt": prompts[index],
                "identity_id": person_ids[index],
                "seed": seeds[index],
                "reference_condition": args.reference_condition,
                "face_bbox_ref": conditioned_bboxes[index],
                "face_bbox_gen": generation_bboxes[index],
                "image_sha256": sha256_file(image_path),
                "metrics": metric_values,
            }
        )

    for metric in metrics:
        metric.to_cpu()

    OmegaConf.save(config, output_dir / "resolved_config.yaml")
    write_json(output_dir / "per_image.json", per_image)
    manifest = {
        "git_commit": git_commit(),
        "config_source": config_source,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "validation_dataset": args.validation_dataset,
        "validation_base": validation_base,
        "guidance_scale": args.guidance_scale,
        "processor_base_mode": args.processor_base_mode,
        "disable_branched_ca": bool(args.disable_branched_ca),
        "reference_intervention": intervention,
        "image_count": len(per_image),
        "processor_load_audit": processor_audit,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(
        output_dir / "command_manifest.json",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "git_commit": manifest["git_commit"],
        },
    )
    print(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed validation on an RHCA checkpoint without training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-dataset",
        choices=["cosmic_large_one_id_val", "one_id_val"],
        required=True,
    )
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--disable-branched-ca",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--validation-base")
    parser.add_argument(
        "--processor-base-mode",
        choices=["validation_native", "legacy_full_copy"],
        default="validation_native",
    )
    parser.add_argument(
        "--reference-condition",
        choices=["matched", "wrong", "null"],
        default="matched",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--wrong-reference", type=Path)
    parser.add_argument(
        "--wrong-reference-bbox",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument(
        "--allow-untrained-ca",
        action="store_true",
        help="Diagnostic escape hatch; never use for fair CA-off checkpoint comparisons.",
    )
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
