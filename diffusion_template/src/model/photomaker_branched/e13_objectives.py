"""Training-only objectives and diagnostics for selected E13-family leaves."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .lora2_helpers import (
    bbox_to_reference_mask,
    encode_reference_latents,
    run_branched_forward_pass,
)

# 22 Aug 2026 - Leaf-only CL18/27 objectives and telemetry stay out of the
# shared model orchestration so E13's forward remains directly comparable.

def prepare_frequency_surface_mask(
    model,
    occluder_masks: Sequence[torch.Tensor] | None,
    face_mask: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Expose CL27's deterministic dataset mask to its selected processors."""
    model._ba_ownership_target_mask = None
    if not model.ba_frequency_surface_loss_enabled:
        return
    if occluder_masks is None:
        raise RuntimeError("CL27 requires deterministic occluder masks")

    prepared = []
    for value in occluder_masks:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError("ba_occluder_mask items must have shape HxW or 1xHxW")
        prepared.append(tensor)
    ownership_mask = torch.stack(prepared).to(model.device)
    if ownership_mask.shape[-2:] != face_mask.shape[-2:]:
        ownership_mask = F.interpolate(
            ownership_mask, size=face_mask.shape[-2:], mode="nearest"
        )
    model._ba_ownership_target_mask = ownership_mask.to(dtype=dtype)


def _collect_identity_ca_telemetry(model) -> dict[str, torch.Tensor]:
    processor_names = tuple(
        getattr(model, "_ba_identity_ca_processor_names", ())
    )
    if not processor_names:
        return {}

    # AICODE-NOTE: Diffusers rebuilds this map on property access. Resolve it
    # once, after the disabled-collector check, for every collector below.
    processors = model.unet.attn_processors
    grouped: dict[str, list[dict[str, torch.Tensor]]] = {}
    for name in processor_names:
        processor = processors.get(name)
        getter = getattr(processor, "latest_ba_telemetry", None)
        values = getter() if getter is not None else {}
        if not values:
            continue
        prefix = name.split(".", 2)[:2]
        group = f"up{prefix[1]}" if prefix[0] == "up_blocks" else "other"
        grouped.setdefault(group, []).append(values)
    if not grouped:
        return {}

    grouped["all"] = [entry for entries in grouped.values() for entry in entries]
    output = {}
    for group, entries in grouped.items():
        metric_names = set(entries[0])
        if any(set(entry) != metric_names for entry in entries):
            raise RuntimeError(f"Inconsistent CL14_CA telemetry in {group}")
        for metric_name in sorted(metric_names):
            output[f"ba/{metric_name}/{group}"] = torch.stack([
                entry[metric_name].detach().float() for entry in entries
            ]).mean()
    return output


def _collect_frequency_surface_loss(model):
    if not model.ba_frequency_surface_loss_enabled:
        return None, {}

    processors = model.unet.attn_processors
    grouped: dict[str, list[dict[str, torch.Tensor]]] = {}
    top_losses, floor_losses, applied = [], [], []
    for name in model._ba_patched_processor_names:
        processor = processors.get(name)
        if not bool(getattr(processor, "frequency_surface_loss_enabled", False)):
            continue
        values = processor.frequency_surface_aux_loss()
        telemetry = processor.latest_ba_telemetry() or {}
        if values is not None:
            top_loss, floor_loss = values
            top_losses.append(top_loss.float())
            floor_losses.append(floor_loss.float())
        if telemetry:
            prefix = name.split(".", 2)[:2]
            group = f"up{prefix[1]}" if prefix[0] == "up_blocks" else "other"
            grouped.setdefault(group, []).append(telemetry)
            applied.append(
                telemetry["frequency_surface_applied_fraction"].float()
            )
    if not top_losses:
        return None, {}

    output = {}
    for group, entries in grouped.items():
        for metric_name in (
            "frequency_surface_top_high_rms",
            "frequency_surface_top_low_rms",
            "frequency_surface_visible_ratio",
        ):
            output[f"ba/{metric_name}/{group}"] = torch.stack([
                entry[metric_name].detach().float() for entry in entries
            ]).mean()
        for metric_name in (
            "null_key/null_mass",
            "null_key/reference_fraction",
            "null_key/object_minus_visible_mass",
        ):
            if all(metric_name in entry for entry in entries):
                output[f"ba/{metric_name}/{group}"] = torch.stack([
                    entry[metric_name].detach().float() for entry in entries
                ]).mean()

    all_entries = [entry for entries in grouped.values() for entry in entries]
    for metric_name in (
        "null_key/null_mass",
        "null_key/reference_fraction",
        "null_key/object_minus_visible_mass",
    ):
        if all_entries and all(metric_name in entry for entry in all_entries):
            output[f"ba/{metric_name}/all"] = torch.stack([
                entry[metric_name].detach().float() for entry in all_entries
            ]).mean()
    output["ba/frequency_surface_applied_fraction"] = (
        torch.stack(applied).mean()
        if applied
        else top_losses[0].new_tensor(0.0)
    )
    return (
        torch.stack(top_losses).mean(),
        torch.stack(floor_losses).mean(),
    ), output


def _crossview_consistency_loss(
    model,
    *,
    noise_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: dict,
    mask4: torch.Tensor,
    face_prompt_embeds: torch.Tensor,
    class_tokens_mask: torch.Tensor,
    spatial_ref_images_alt: Sequence[Sequence[Image.Image]] | None,
    face_bbox_ref_alt: Sequence[Sequence[float]] | None,
) -> torch.Tensor:
    zero = noise_pred.float().new_tensor(0.0)
    if not (
        model.training
        and model.ba_crossview_consistency_enabled
        and spatial_ref_images_alt is not None
        and face_bbox_ref_alt is not None
        and torch.rand((), device=noisy_latents.device).item()
        < model.ba_crossview_consistency_probability
    ):
        return zero

    alternate_refs = []
    alternate_masks = []
    for refs, bbox in zip(spatial_ref_images_alt, face_bbox_ref_alt):
        refs = refs if isinstance(refs, (list, tuple)) else [refs]
        if not refs:
            raise RuntimeError("Cross-view consistency received no alternate ref")
        ref = refs[0]
        alternate_refs.append(ref)
        ref_size = (
            tuple(ref.shape[-2:])
            if isinstance(ref, torch.Tensor)
            else (ref.height, ref.width)
        )
        alternate_masks.append(
            bbox_to_reference_mask(
                model,
                bbox,
                latent_shape=noisy_latents.shape[-2:],
                image_shape=ref_size,
            )
        )

    alternate_latents = encode_reference_latents(
        model,
        alternate_refs,
        target_shape=noisy_latents.shape[-2:],
    ).to(dtype=noisy_latents.dtype)
    alternate_mask4 = torch.cat(alternate_masks).to(
        device=model.device, dtype=noisy_latents.dtype
    )
    paired_reference_noise = getattr(model, "_ref_noise", None)
    if paired_reference_noise is None:
        raise RuntimeError("Cross-view consistency lost paired reference noise")
    student_pred = run_branched_forward_pass(
        model,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        mask4=mask4,
        mask4_ref=alternate_mask4,
        reference_latents=alternate_latents,
        face_prompt_embeds=face_prompt_embeds,
        class_tokens_mask=class_tokens_mask,
        reference_noise=paired_reference_noise,
    )

    face = mask4.float()
    if face.shape[-2:] != noise_pred.shape[-2:]:
        face = F.interpolate(face, size=noise_pred.shape[-2:], mode="nearest")
    teacher_face = noise_pred.detach().float() * face
    student_face = student_pred.float() * face
    smooth_map = F.smooth_l1_loss(
        student_face, teacher_face, reduction="none"
    )
    smooth = (smooth_map * face).sum() / (
        face.sum() * student_face.shape[1]
    ).clamp_min(1.0)
    cosine = F.cosine_similarity(
        student_face.flatten(1), teacher_face.flatten(1), dim=1
    ).mean()
    return smooth + 0.10 * (1.0 - cosine)


def compute_e13_objectives(model, **inputs) -> dict[str, object]:
    """Compute only the auxiliary objectives enabled by the selected leaf."""
    noise_pred = inputs["noise_pred"]
    crossview_loss = _crossview_consistency_loss(model, **inputs)
    surface_loss = noise_pred.float().new_tensor(0.0)
    surface_telemetry = {}
    if model.ba_frequency_surface_loss_enabled:
        surface, surface_telemetry = _collect_frequency_surface_loss(model)
        if surface is None:
            raise RuntimeError("CL27 selected no live frequency-surface losses")
        top_loss, floor_loss = surface
        surface_loss = (
            model.ba_frequency_surface_top_weight * top_loss
            + model.ba_frequency_surface_visible_floor_weight * floor_loss
        )
        surface_telemetry["loss_ba_frequency_surface"] = surface_loss.detach()

    telemetry = _collect_identity_ca_telemetry(model)
    telemetry.update(surface_telemetry)
    return {
        "ba_telemetry": telemetry,
        "ba_aux_loss": (
            model.ba_crossview_consistency_weight * crossview_loss
            + surface_loss
        ),
        "ba_crossview_loss": crossview_loss.detach(),
    }
