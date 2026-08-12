#!/usr/bin/env python3
"""Evaluate an RHCA checkpoint on a fixed validation set without training."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torch


# 09 Aug 2026 - AICODE-NOTE: Serv diagnostic sidecars may execute this patched
# evaluator outside an immutable experiment runtime. The explicit override lets
# the evaluator import that runtime's exact model/config code without editing it.
PROJECT_ROOT = Path(
    os.environ.get("PM_EVAL_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_ROOT = PROJECT_ROOT / "src" / "configs"


# 09 Aug 2026 - AICODE-NOTE: Fixed-checkpoint replay must copy the same
# versioned processor runtime attributes as BaseTrainer's alternate-base
# validation path. Keep this list aligned with src/trainer/base_trainer.py.
VALIDATION_PIPELINE_RUNTIME_ATTRIBUTES = (
    "ba_architecture_version",
    "branched_trainable_dtype",
    "ba_ref_kv_rank",
    "ba_output_rank",
    "ba_branch_q_rank",
    "ba_face_fusion_mode",
    "ba_face_branch_scale",
    "ba_gate_init",
    "ba_gate_max",
    "ba_gate_timestep",
    "ba_gate_face_area",
    "ba_mix_init",
    "ba_mix_floor",
    "ba_mix_max",
    "ba_mix_timestep",
    "ba_mix_face_area",
    "ba_reference_rms_match",
    "ba_reference_rms_clip_min",
    "ba_reference_rms_clip_max",
    "ba_mix_override",
    "ba_telemetry_enabled",
    "ba_telemetry_interval",
    "ba_require_denoise_progress",
    "ba_self_attention_groups",
    "ba_reference_loss_mode",
    "ba_enforce_reference_only_hard_route",
    "ba_hard_v1_true_reference_key_mask",
    "ba_hard_v1_branch_output_rank",
    "ba_hard_v1_reference_roi_warp",
    "ba_hard_v1_lora_rank",
    "ba_identity_ca_v2_enabled",
    "ba_identity_ca_v2_groups",
    "ba_identity_ca_v2_rank",
    "ba_residual_identity_ca_v3_enabled",
    "ba_residual_identity_ca_v3_groups",
    "ba_residual_identity_ca_v3_rank",
    "ba_residual_identity_ca_v3_gate_init",
    "ba_residual_identity_ca_v3_gate_max",
)


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
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state must contain a mapping")
    schema_version = int(state.get("schema_version", 1))
    is_schema_v1 = "lora_weights" in state
    is_schema_v2 = (
        schema_version == 2
        and state.get("state_format") == "trainable_unet_v2"
        and isinstance(state.get("trainable_unet"), dict)
    )
    if not (is_schema_v1 or is_schema_v2):
        raise ValueError(
            "Checkpoint contains neither schema-v1 RHCA lora_weights nor "
            "schema-v2 trainable_unet state"
        )
    metadata = {
        "kind": "full" if "state_dict" in checkpoint else "weights_only",
        "epoch": checkpoint.get("epoch"),
        "schema_version": schema_version,
        "state_format": state.get("state_format", "legacy_lora_v1"),
        "top_level_keys": sorted(str(key) for key in checkpoint),
        "lora_tensor_count": len(state.get("lora_weights", {})),
        "trainable_unet_tensor_count": len(state.get("trainable_unet", {})),
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
        if type_name in {
            "BranchedAttnProcessor",
            "ResidualBranchedSelfAttnProcessorV2",
            "AnchoredMixBranchedSelfAttnProcessorV3",
        }:
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


def configured_processor_base_mode(config: DictConfig) -> str:
    explicit = getattr(config, "validation_processor_base_mode", None)
    if explicit is None:
        mode = (
            "legacy_full_copy"
            if bool(getattr(config, "update_proc_weights_val", False))
            else "no_processor_update"
        )
    else:
        mode = str(explicit).lower()
    allowed = {"legacy_full_copy", "validation_native", "no_processor_update"}
    if mode not in allowed:
        raise ValueError(
            f"Unknown validation_processor_base_mode={mode!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return mode


def snapshot_adapter_parameters(
    unet,
    adapter_marker: str,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in unet.named_parameters()
        if adapter_marker in name and parameter.requires_grad
    }


def restore_adapter_parameters(
    unet,
    snapshot: dict[str, torch.Tensor],
    *,
    adapter_marker: str,
) -> int:
    named_parameters = dict(unet.named_parameters())
    current_names = {
        name
        for name, parameter in named_parameters.items()
        if adapter_marker in name and parameter.requires_grad
    }
    if current_names != set(snapshot):
        raise RuntimeError(
            "Validation shadow-adapter parameter map changed: "
            f"missing={sorted(set(snapshot) - current_names)[:3]}, "
            f"unexpected={sorted(current_names - set(snapshot))[:3]}"
        )
    with torch.no_grad():
        for name, value in snapshot.items():
            parameter = named_parameters[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return len(snapshot)


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
        # 2 Aug 2026 - Mirror train.py's top-level architecture overrides so a
        # schema-v2 checkpoint is compared against the architecture that wrote
        # its exact trainable-name manifest.
        model = instantiate(
            config.model,
            device=device,
            train_ba_only=bool(getattr(config, "train_ba_only", False)),
            ba_train_top_k=float(getattr(config, "ba_train_top_k", 1.0)),
            ba_patch_top_k=float(getattr(config, "ba_patch_top_k", 1.0)),
            non_ba_train=bool(getattr(config, "non_ba_train", False)),
            train_ba_all_steps=bool(
                getattr(config, "train_ba_all_steps", False)
            ),
            ba_weights_split=bool(getattr(config, "ba_weights_split", False)),
            use_attn_v2=bool(getattr(config, "use_attn_v2", False)),
        )
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
        processor_state = {
            key: value.detach().cpu().clone()
            for key, value in processor.state_dict().items()
        }
        if processor_state:
            result[name] = processor_state
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

    # 09 Aug 2026 - E13-family training validation snapshots the pretrained
    # PhotoMaker default adapter on the RealVis model, loads the live/checkpoint
    # state, then restores that snapshot. Omitting this turns the replay into a
    # different global-conditioning model and can change the entire image.
    shadow_default = bool(
        getattr(config, "validation_shadow_photomaker_default", False)
    )
    default_snapshot = (
        snapshot_adapter_parameters(validation_model.unet, ".default.")
        if shadow_default
        else None
    )
    audit["validation_shadow_photomaker_default"] = shadow_default
    audit["shadow_default_snapshot_tensor_count"] = (
        0 if default_snapshot is None else len(default_snapshot)
    )
    if shadow_default and not default_snapshot:
        raise RuntimeError(
            "Requested PhotoMaker-default shadow validation but the alternate "
            "validation model has no trainable default adapter tensors"
        )

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

    if default_snapshot is not None:
        audit["shadow_default_restored_tensor_count"] = restore_adapter_parameters(
            validation_model.unet,
            default_snapshot,
            adapter_marker=".default.",
        )

    if legacy_processor_states is not None:
        strict_copy = bool(
            getattr(config, "strict_validation_processor_copy", False)
        )
        copied = 0
        for name, source_state in legacy_processor_states.items():
            destination = validation_model.unet.attn_processors.get(name)
            if destination is None or not hasattr(destination, "load_state_dict"):
                if strict_copy:
                    raise RuntimeError(
                        f"Validation U-Net is missing stateful processor {name!r}"
                    )
                continue
            try:
                destination.load_state_dict(source_state, strict=strict_copy)
            except Exception:
                if strict_copy:
                    raise
                continue
            copied += 1
        if strict_copy and copied == 0:
            raise RuntimeError(
                "Strict legacy processor copy found no stateful processors"
            )
        audit["strict_validation_processor_copy"] = strict_copy
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
    for attribute in VALIDATION_PIPELINE_RUNTIME_ATTRIBUTES:
        if hasattr(model, attribute):
            setattr(pipeline, attribute, getattr(model, attribute))
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


def generation_bbox_key(prompt: str, person_id: str) -> str:
    # 2 Aug 2026 - Trainer bbox stores intentionally retain spaces in the
    # infer.py-style key; filesystem-safe output names are a separate concern.
    return f"{prompt[:10]}_{person_id}.png"


def _bbox_from_record(record: Any) -> list[float] | None:
    if not isinstance(record, dict):
        return None
    value = record.get("face_crop_new") or record.get("face_crop_old")
    if value is None:
        return None
    return [float(item) for item in value]


def resolve_generation_bboxes(
    config: DictConfig,
    samples: list[dict],
    prompts: list[str],
    person_ids: list[str],
    *,
    validation_dataset: str,
) -> tuple[list[list[float] | None], dict]:
    """Mirror the trainer's fixed cached-auto/manual bbox resolution."""
    dataset_config = config.datasets.val[validation_dataset]
    configured_path = getattr(dataset_config, "bbox_mask_gen", None)
    if not configured_path:
        return [sample.get("face_bbox_gen") for sample in samples], {
            "kind": "dataset_values",
            "configured_path": None,
        }

    manual_path = Path(str(configured_path)).resolve()
    manual_payload = json.loads(manual_path.read_text(encoding="utf-8"))
    automatic = bool(getattr(config, "automatic_bboxes", False))
    if automatic:
        active_path = manual_path.with_name(f"{manual_path.stem}_auto.json")
        active_payload = json.loads(active_path.read_text(encoding="utf-8"))
    else:
        active_path = manual_path
        active_payload = manual_payload

    bboxes = []
    sources = []
    for index, (sample, prompt, person_id) in enumerate(
        zip(samples, prompts, person_ids)
    ):
        key = generation_bbox_key(prompt, person_id)
        manual_record = manual_payload.get(key)
        force_manual = bool(
            isinstance(manual_record, dict)
            and manual_record.get("force_manual", False)
        )
        record = manual_record if force_manual else active_payload.get(key)
        bbox = _bbox_from_record(record)
        source = "manual_force" if force_manual else (
            "cached_auto" if automatic else "configured_map"
        )
        if bbox is None:
            # Index-keyed historical maps remain supported as a fallback.
            index_record = active_payload.get(f"{index:02d}.png")
            bbox = _bbox_from_record(index_record) or sample.get("face_bbox_gen")
            source = "index_or_dataset_fallback"
        bboxes.append(bbox)
        sources.append({"index": index, "key": key, "source": source})

    return bboxes, {
        "kind": "trainer_equivalent_cached_bbox_resolution",
        "automatic_bboxes": automatic,
        "automatic_bboxes_every_val": bool(
            getattr(config, "automatic_bboxes_every_val", True)
        ),
        "configured_manual_path": str(manual_path),
        "configured_manual_sha256": sha256_file(manual_path),
        "active_path": str(active_path),
        "active_sha256": sha256_file(active_path),
        "sources": sources,
    }


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


def first_reference(sample: dict) -> Image.Image:
    refs = sample["ref_images"]
    return refs[0] if isinstance(refs, (list, tuple)) else refs


def shuffled_spatial_reference(
    samples: list[dict],
    batch_indices: list[int],
) -> tuple[Image.Image, Any, dict]:
    batch_ids = {str(samples[index]["id"]) for index in batch_indices}
    if len(batch_ids) != 1:
        raise ValueError(
            "Spatial-reference shuffle requires identity-homogeneous batches; "
            f"indices {batch_indices[0]}:{batch_indices[-1] + 1} contain "
            f"identities {sorted(batch_ids)}"
        )
    for source_index, sample in enumerate(samples):
        source_id = str(sample["id"])
        if source_id in batch_ids:
            continue
        bbox = sample.get("face_bbox_ref")
        if bbox is None:
            raise ValueError(
                f"Shuffle source {source_index} ({source_id}) has no face_bbox_ref"
            )
        return first_reference(sample), bbox, {
            "target_identity": next(iter(batch_ids)),
            "source_identity": source_id,
            "source_index": source_index,
            "target_indices": batch_indices,
        }
    raise ValueError("Validation dataset has no different identity for spatial shuffle")


@contextmanager
def spatial_reference_override(
    pipeline,
    *,
    condition: str,
    image: Image.Image | None = None,
    bbox: Any = None,
):
    """Override only BA spatial setup; PhotoMaker inputs stay unchanged."""
    if condition == "matched":
        yield
        return

    pipeline_module = importlib.import_module(pipeline.__class__.__module__)
    original_setup = pipeline_module.run_branched_setup_helper

    def wrapped_setup(*setup_args, **setup_kwargs):
        setup_kwargs = dict(setup_kwargs)
        if condition == "shuffle":
            if image is None or bbox is None:
                raise RuntimeError("Shuffle override requires an image and bbox")
            setup_kwargs["input_id_images"] = [image]
            setup_kwargs["face_bbox_ref"] = bbox
        original_setup(*setup_args, **setup_kwargs)
        if condition == "zero":
            # Keep the matched PhotoMaker embedding and matched spatial mask,
            # but remove all identity-bearing signal at the spatial branch
            # input, including its otherwise independently sampled noise.
            pipeline._ref_latents_all.zero_()
            pipeline._reference_latents = pipeline._ref_latents_all
            pipeline._ref_noise.zero_()

    pipeline_module.run_branched_setup_helper = wrapped_setup
    try:
        yield
    finally:
        pipeline_module.run_branched_setup_helper = original_setup


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
    allow_contract_override = bool(
        getattr(args, "allow_validation_contract_override", False)
    )
    if args.photomaker_path is not None:
        config.model.photomaker_path = str(args.photomaker_path.resolve())
    if args.ba_mix_override is not None:
        # 2 Aug 2026 - V3 fixed-checkpoint causal arms must record and
        # propagate alpha explicitly; a run-name convention is not auditable.
        if not 0.0 <= float(args.ba_mix_override) <= 1.0:
            raise ValueError("--ba-mix-override must be in [0, 1]")
        architecture = str(
            getattr(config.model, "ba_architecture_version", "hard_replace_v1")
        ).lower()
        if architecture != "anchored_mix_sa_v3":
            raise ValueError(
                "--ba-mix-override is restricted to anchored_mix_sa_v3, "
                f"got {architecture!r}"
            )
        config.model.ba_mix_override = float(args.ba_mix_override)
    if args.validation_dataset not in config.datasets.val:
        raise KeyError(
            f"Validation dataset {args.validation_dataset!r} is not registered"
        )
    config.val_datasets_names = [args.validation_dataset]

    configured_validation_base = str(
        getattr(
            config,
            "pretrained_model_for_validation_name_or_path",
            config.model.pretrained_model_name_or_path,
        )
    )
    requested_validation_base = getattr(args, "validation_base", None)
    if (
        requested_validation_base is not None
        and str(requested_validation_base) != configured_validation_base
        and not allow_contract_override
    ):
        raise ValueError(
            "--validation-base differs from the training-validation contract: "
            f"requested={requested_validation_base!r}, "
            f"configured={configured_validation_base!r}. Pass "
            "--allow-validation-contract-override only for a labeled ablation."
        )
    validation_base = str(
        requested_validation_base or configured_validation_base
    )

    configured_processor_mode = configured_processor_base_mode(config)
    requested_processor_mode = getattr(args, "processor_base_mode", None)
    if (
        requested_processor_mode is not None
        and str(requested_processor_mode) != configured_processor_mode
        and not allow_contract_override
    ):
        raise ValueError(
            "--processor-base-mode differs from the training-validation "
            f"contract: requested={requested_processor_mode!r}, "
            f"configured={configured_processor_mode!r}. Pass "
            "--allow-validation-contract-override only for a labeled ablation."
        )
    processor_base_mode = str(
        requested_processor_mode or configured_processor_mode
    )
    if processor_base_mode == "no_processor_update":
        raise ValueError(
            "Standalone schema-v2 evaluation does not support "
            "no_processor_update; use the experiment's validation_native or "
            "legacy_full_copy contract."
        )

    configured_guidance_scale = float(config.validation_args.guidance_scale)
    requested_guidance_scale = getattr(args, "guidance_scale", None)
    if (
        requested_guidance_scale is not None
        and float(requested_guidance_scale) != configured_guidance_scale
        and not allow_contract_override
    ):
        raise ValueError(
            "--guidance-scale differs from the training-validation contract: "
            f"requested={requested_guidance_scale}, "
            f"configured={configured_guidance_scale}."
        )
    guidance_scale = float(
        configured_guidance_scale
        if requested_guidance_scale is None
        else requested_guidance_scale
    )

    dataloader_config = config.dataloaders.get(args.validation_dataset)
    if dataloader_config is None:
        dataloader_config = config.dataloaders.get("val_default")
    if dataloader_config is None:
        raise KeyError(
            f"No dataloader config for validation dataset {args.validation_dataset!r}"
        )
    configured_batch_size = int(dataloader_config.batch_size)
    requested_batch_size = getattr(args, "batch_size", None)
    if (
        requested_batch_size is not None
        and int(requested_batch_size) != configured_batch_size
        and not allow_contract_override
    ):
        raise ValueError(
            "--batch-size differs from the training-validation contract: "
            f"requested={requested_batch_size}, configured={configured_batch_size}."
        )
    batch_size = int(
        configured_batch_size if requested_batch_size is None else requested_batch_size
    )

    configured_disable_branched_ca = bool(
        getattr(config, "disable_branched_ca", False)
    )
    requested_disable_branched_ca = getattr(args, "disable_branched_ca", None)
    if (
        requested_disable_branched_ca is not None
        and bool(requested_disable_branched_ca) != configured_disable_branched_ca
        and not allow_contract_override
    ):
        raise ValueError(
            "--disable-branched-ca/--no-disable-branched-ca differs from the "
            "training-validation contract."
        )
    disable_branched_ca = (
        configured_disable_branched_ca
        if requested_disable_branched_ca is None
        else bool(requested_disable_branched_ca)
    )

    config.validation_args.guidance_scale = guidance_scale
    config.validation_args.num_images_per_prompt = 1
    config.validation_args.val_debug = False
    config.val_debug = False
    config.pretrained_model_for_validation_name_or_path = validation_base
    config.disable_branched_ca = disable_branched_ca
    config.model.train_branched_ca_lora = not disable_branched_ca

    checkpoint_path = args.checkpoint.resolve()
    state, checkpoint_metadata = checkpoint_state(checkpoint_path)
    processor_names = list((state.get("attn_processors") or {}).keys())
    if not processor_names:
        processor_names = list(
            (state.get("architecture") or {}).get("patched_processor_names", ())
        )
    has_saved_ca = any(name.endswith("attn2.processor") for name in processor_names)
    if (
        not disable_branched_ca
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
        processor_base_mode=processor_base_mode,
        device=device,
        disable_branched_ca=disable_branched_ca,
    )
    pipeline = build_pipeline(
        config,
        model,
        validation_base=validation_base,
        device=device,
        disable_branched_ca=disable_branched_ca,
    )
    setattr(pipeline, "ba_mix_override", args.ba_mix_override)

    # Keep the configured full validation dataset intact. Limiting its Hydra
    # node to 12 changes the replay context recorded in resolved_config.yaml and
    # made the prior Eddie sidecar unlike the fixed-96 training event.
    dataset = instantiate(config.datasets.val[args.validation_dataset])
    dataset_size = len(dataset)
    samples = [dataset[index] for index in range(min(len(dataset), args.limit))]
    if len(samples) != args.limit:
        raise ValueError(f"Requested {args.limit} samples, dataset returned {len(samples)}")

    conditioned_refs, conditioned_bboxes, id_embeds, intervention = (
        apply_reference_condition(args, samples)
    )
    if (
        args.spatial_reference_condition != "matched"
        and args.reference_condition != "matched"
    ):
        raise ValueError(
            "A spatial-only intervention requires --reference-condition matched "
            "so PhotoMaker identity tokens remain fixed"
        )
    prompts = [str(sample["prompt"]) for sample in samples]
    person_ids = [str(sample["id"]) for sample in samples]
    seeds = [int(sample.get("seed", 0)) for sample in samples]
    generation_bboxes, generation_bbox_protocol = resolve_generation_bboxes(
        config,
        samples,
        prompts,
        person_ids,
        validation_dataset=args.validation_dataset,
    )
    if bool(config.validation_args.get("use_bbox_mask_gen", False)) and any(
        bbox is None for bbox in generation_bboxes
    ):
        raise ValueError(
            "Fixed validation requires face_bbox_gen for every sample; the "
            "configured generation-bbox JSON is incomplete."
        )

    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    validation_kwargs = OmegaConf.to_container(
        config.validation_args,
        resolve=True,
    )
    validation_kwargs["debug_dir"] = None
    validation_kwargs["debug_idx"] = 0
    validation_kwargs["debug_total"] = dataset_size
    validation_kwargs["val_debug"] = False

    generated = []
    spatial_batches = []
    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        batch_indices = list(range(start, end))
        batch_prompts = prompts[start:end]
        batch_refs = conditioned_refs[start:end]
        batch_ref_bboxes = conditioned_bboxes[start:end]
        batch_gen_bboxes = generation_bboxes[start:end]
        batch_generators = generators[start:end]
        override_image = None
        override_bbox = None
        batch_intervention = {
            "condition": args.spatial_reference_condition,
            "target_indices": batch_indices,
        }
        if args.spatial_reference_condition == "shuffle":
            override_image, override_bbox, shuffle_metadata = (
                shuffled_spatial_reference(samples, batch_indices)
            )
            batch_intervention.update(shuffle_metadata)
        elif args.spatial_reference_condition == "zero":
            batch_intervention["meaning"] = (
                "matched PhotoMaker identity and matched spatial mask; zero "
                "reference latent and zero reference noise at BA input"
            )
        else:
            batch_intervention["meaning"] = (
                "matched PhotoMaker identity and matched BA spatial reference"
            )

        batch_kwargs = dict(validation_kwargs)
        batch_kwargs["debug_idx"] = start
        with spatial_reference_override(
            pipeline,
            condition=args.spatial_reference_condition,
            image=override_image,
            bbox=override_bbox,
        ):
            with torch.no_grad():
                batch_generated = pipeline(
                    prompt=(
                        batch_prompts if len(batch_prompts) > 1 else batch_prompts[0]
                    ),
                    input_id_images=(
                        batch_refs if len(batch_prompts) > 1 else batch_refs[0]
                    ),
                    face_bbox_ref=(
                        batch_ref_bboxes
                        if len(batch_prompts) > 1
                        else batch_ref_bboxes[0]
                    ),
                    face_bbox_gen=(
                        batch_gen_bboxes
                        if len(batch_prompts) > 1
                        else batch_gen_bboxes[0]
                    ),
                    generator=(
                        batch_generators
                        if len(batch_prompts) > 1
                        else batch_generators[0]
                    ),
                    id_embeds=(
                        id_embeds[start:end].to(device)
                        if id_embeds is not None
                        else None
                    ),
                    **batch_kwargs,
                ).images
        if not isinstance(batch_generated, list):
            batch_generated = [batch_generated]
        if len(batch_generated) != len(batch_indices):
            raise RuntimeError(
                f"Batch {start}:{end} expected {len(batch_indices)} images, "
                f"got {len(batch_generated)}"
            )
        generated.extend(batch_generated)
        spatial_batches.append(batch_intervention)

    intervention = {
        "photomaker_and_default_spatial_reference": intervention,
        "spatial_ba_only": {
            "kind": args.spatial_reference_condition,
            "photomaker_identity_inputs_unchanged_by_spatial_override": True,
            "batches": spatial_batches,
        },
    }
    if len(generated) != len(samples):
        raise RuntimeError(
            f"Expected {len(samples)} generated images, got {len(generated)}"
        )

    metrics = instantiate_metrics(config, device, skip=args.skip_metrics)
    per_image = []
    seen_filenames = set()
    for index, (sample, image) in enumerate(zip(samples, generated)):
        filename = f"{index:03d}_{output_filename(prompts[index], person_ids[index])}"
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
                "spatial_reference_condition": args.spatial_reference_condition,
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
    write_json(
        output_dir / "face_quality_input_manifest.json",
        {
            "schema_version": 1,
            "kind": "local_fixed_checkpoint_validation_images",
            "experiment_key": None,
            "project_name": "rhca_fixed_checkpoint_diagnostics",
            "steps": {
                str(args.checkpoint_step): [
                    {
                        "asset_id": f"local-{row['index']:03d}",
                        "file_name": row["filename"],
                        "local_path": str((images_dir / row["filename"]).resolve()),
                    }
                    for row in per_image
                ]
            },
        },
    )
    manifest = {
        "git_commit": git_commit(),
        "config_source": config_source,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_step": args.checkpoint_step,
        "validation_dataset": args.validation_dataset,
        "validation_base": validation_base,
        "guidance_scale": guidance_scale,
        "batch_size": batch_size,
        "processor_base_mode": processor_base_mode,
        "disable_branched_ca": disable_branched_ca,
        "configured_validation_dataset_size": dataset_size,
        "validation_contract": {
            "override_allowed": allow_contract_override,
            "validation_base": configured_validation_base,
            "processor_base_mode": configured_processor_mode,
            "guidance_scale": configured_guidance_scale,
            "batch_size": configured_batch_size,
            "disable_branched_ca": configured_disable_branched_ca,
            "validation_shadow_photomaker_default": bool(
                getattr(config, "validation_shadow_photomaker_default", False)
            ),
            "strict_validation_processor_copy": bool(
                getattr(config, "strict_validation_processor_copy", False)
            ),
            "num_inference_steps": int(config.validation_args.num_inference_steps),
            "pose_adapt_ratio": float(config.pipeline.pose_adapt_ratio),
            "ca_mixing_for_face": bool(config.pipeline.ca_mixing_for_face),
        },
        "ba_mix_override": args.ba_mix_override,
        "reference_intervention": intervention,
        "generation_bbox_protocol": generation_bbox_protocol,
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
        choices=["manual_val", "cosmic_large_one_id_val", "one_id_val"],
        required=True,
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Defaults to the composed training-validation value.",
    )
    parser.add_argument(
        "--disable-branched-ca",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--validation-base")
    parser.add_argument("--photomaker-path", type=Path)
    parser.add_argument(
        "--processor-base-mode",
        choices=["validation_native", "legacy_full_copy"],
        help="Defaults to validation_processor_base_mode from the run config.",
    )
    parser.add_argument(
        "--reference-condition",
        choices=["matched", "wrong", "null"],
        default="matched",
    )
    parser.add_argument(
        "--spatial-reference-condition",
        choices=["matched", "zero", "shuffle"],
        default="matched",
        help=(
            "Evaluation-only BA spatial intervention. PhotoMaker reference "
            "images/tokens remain matched; use identity-homogeneous batches "
            "for shuffle."
        ),
    )
    parser.add_argument(
        "--ba-mix-override",
        type=float,
        help=(
            "Anchored-v3 diagnostic alpha override in [0, 1]. Omit to use "
            "the checkpoint's learned bounded mix."
        ),
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Defaults to the configured validation dataloader batch size.",
    )
    parser.add_argument("--checkpoint-step", type=int, default=0)
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
    parser.add_argument(
        "--allow-validation-contract-override",
        action="store_true",
        help=(
            "Permit an explicitly labeled base/processor/CFG/batch/CA ablation. "
            "Never use for replaying an in-training validation event."
        ),
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.checkpoint_step < 0:
        parser.error("--checkpoint-step cannot be negative")
    try:
        run(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
