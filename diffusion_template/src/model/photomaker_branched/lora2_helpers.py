from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .branched_runtime import patch_unet_attention_processors, select_branched_processor_names, two_branch_predict
from .insightface_package import analyze_faces
from .e13_contract import (
    assert_trainable_contract as assert_e13_trainable_contract,
    configure_trainables as configure_e13_trainables,
)

from copy import deepcopy


# 13 Aug 2026 - CL14_CA-OBS-01: report route health without adding a loss edge.
def collect_identity_ca_telemetry(model) -> dict[str, torch.Tensor]:
    """Aggregate detached CL14_CA diagnostics by selected U-Net group."""
    processor_names = tuple(
        getattr(model, "_ba_identity_ca_processor_names", ())
    )
    if not processor_names:
        return {}
    grouped: dict[str, list[dict[str, torch.Tensor]]] = {}
    # 18 Aug 2026 - AICODE-NOTE: Diffusers rebuilds this complete map on each
    # property access. Resolve once before any per-layer collector loop.
    processors = model.unet.attn_processors
    for name in processor_names:
        processor = processors.get(name)
        getter = getattr(processor, "latest_ba_telemetry", None)
        values = getter() if getter is not None else {}
        if not values:
            continue
        group = name.split(".", 2)[:2]
        group = f"up{group[1]}" if group[0] == "up_blocks" else "other"
        grouped.setdefault(group, []).append(values)
    if not grouped:
        return {}

    grouped["all"] = [entry for entries in grouped.values() for entry in entries]
    output: dict[str, torch.Tensor] = {}
    for group, entries in grouped.items():
        metric_names = set(entries[0])
        if any(set(entry) != metric_names for entry in entries):
            raise RuntimeError(f"Inconsistent CL14_CA telemetry in {group}")
        for metric_name in sorted(metric_names):
            output[f"ba/{metric_name}/{group}"] = torch.stack([
                entry[metric_name].detach().float() for entry in entries
            ]).mean()
    return output


def collect_frequency_surface_aux_loss(model):
    """Return CL27's live loss graph and already-required detached metrics."""
    if not bool(getattr(model, "ba_frequency_surface_loss_enabled", False)):
        return None, {}
    # 18 Aug 2026 - The fixed pipeline resolves Diffusers' recursive processor
    # property once, never once per selected attention layer.
    processors = model.unet.attn_processors
    grouped: dict[str, list[dict[str, torch.Tensor]]] = {}
    top_losses, floor_losses, applied = [], [], []
    for name in getattr(model, "_ba_patched_processor_names", ()):
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
            group = name.split(".", 2)[:2]
            group = f"up{group[1]}" if group[0] == "up_blocks" else "other"
            grouped.setdefault(group, []).append(telemetry)
            applied.append(telemetry["frequency_surface_applied_fraction"].float())
    if not top_losses:
        return None, {}
    telemetry_out: dict[str, torch.Tensor] = {}
    for group, entries in grouped.items():
        for metric_name in (
            "frequency_surface_top_high_rms",
            "frequency_surface_top_low_rms",
            "frequency_surface_visible_ratio",
        ):
            telemetry_out[f"ba/{metric_name}/{group}"] = torch.stack([
                entry[metric_name].detach().float() for entry in entries
            ]).mean()
        for metric_name in (
            "null_key/null_mass",
            "null_key/reference_fraction",
            "null_key/object_minus_visible_mass",
        ):
            if all(metric_name in entry for entry in entries):
                telemetry_out[f"ba/{metric_name}/{group}"] = torch.stack([
                    entry[metric_name].detach().float() for entry in entries
                ]).mean()
    all_entries = [entry for entries in grouped.values() for entry in entries]
    for metric_name in (
        "null_key/null_mass",
        "null_key/reference_fraction",
        "null_key/object_minus_visible_mass",
    ):
        if all_entries and all(metric_name in entry for entry in all_entries):
            telemetry_out[f"ba/{metric_name}/all"] = torch.stack([
                entry[metric_name].detach().float() for entry in all_entries
            ]).mean()
    telemetry_out["ba/frequency_surface_applied_fraction"] = (
        torch.stack(applied).mean()
        if applied
        else top_losses[0].new_tensor(0.0)
    )
    return (
        torch.stack(top_losses).mean(),
        torch.stack(floor_losses).mean(),
    ), telemetry_out


def configure_branched_trainables(model) -> None:
    if not getattr(model, "train_ba_only", False):
        return

    mode = (getattr(model, "branched_attn_weight_mode", "shared") or "shared").lower()
    new_weight_kind = (getattr(model, "branched_attn_new_weight_kind", "full") or "full").lower()
    train_ca = bool(getattr(model, "train_branched_ca_lora", True))
    ba_train_top_k = float(getattr(model, "ba_train_top_k", 1.0))
    non_ba_train = bool(getattr(model, "non_ba_train", False))
    if mode not in {"shared", "ref_only", "noise_and_ref"}:
        raise ValueError(f"Unknown branched_attn_weight_mode: {mode}")
    if new_weight_kind not in {"full", "lora"}:
        raise ValueError(f"Unknown branched_attn_new_weight_kind: {new_weight_kind}")

    patched_proc_names = tuple(getattr(model, "_ba_patched_processor_names", ()))
    candidate_proc_names = list(patched_proc_names or model.unet.attn_processors.keys())
    selected_proc_names = select_branched_processor_names(
        candidate_proc_names,
        include_self_attention=True,
        include_cross_attention=train_ca,
        top_k=ba_train_top_k,
        param_name="ba_train_top_k",
    )
    setattr(model, "_ba_trainable_processor_names", tuple(selected_proc_names))
    selected_proc_prefixes = tuple(f"{name}." for name in selected_proc_names)
    selected_attn_prefixes = tuple(f"{name.rsplit('.processor', 1)[0]}." for name in selected_proc_names)
    patched_proc_name_set = set(patched_proc_names)
    non_ba_attn_prefixes = tuple(
        f"{name.rsplit('.processor', 1)[0]}."
        for name in model.unet.attn_processors.keys()
        if name.endswith("attn1.processor") and name not in patched_proc_name_set
    )

    for _, p in model.unet.named_parameters():
        p.requires_grad_(False)

    for name, p in model.unet.named_parameters():
        is_non_ba_attn = bool(non_ba_attn_prefixes) and name.startswith(non_ba_attn_prefixes)
        if mode == "shared":
            is_selected_attn = bool(selected_attn_prefixes) and name.startswith(selected_attn_prefixes)
            if is_selected_attn and ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name and ".attn1." in name:
                p.requires_grad_(True)
        else:
            is_selected_proc = bool(selected_proc_prefixes) and name.startswith(selected_proc_prefixes)
            if is_selected_proc and ".attn1.processor.ref_to_" in name and (
                new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
            ):
                p.requires_grad_(True)
            elif is_selected_proc and mode == "noise_and_ref" and ".attn1.processor.noise_to_" in name and (
                new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
            ):
                p.requires_grad_(True)

        if non_ba_train and is_non_ba_attn and ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name:
            p.requires_grad_(True)

        if train_ca:
            if mode == "shared":
                is_selected_attn = bool(selected_attn_prefixes) and name.startswith(selected_attn_prefixes)
                if is_selected_attn and ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name and ".attn2." in name:
                    p.requires_grad_(True)
            else:
                is_selected_proc = bool(selected_proc_prefixes) and name.startswith(selected_proc_prefixes)
                if is_selected_proc and ".attn2.processor.ref_to_" in name and (
                    new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
                ):
                    p.requires_grad_(True)
                elif is_selected_proc and mode == "noise_and_ref" and ".attn2.processor.noise_to_" in name and (
                    new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
                ):
                    p.requires_grad_(True)


def install_branched_processors_for_training(model) -> None:
    """Install branched attention processors once before optimizer creation."""
    if bool(getattr(model, "e13_family_contract", False)):
        # 10 Aug 2026 - E13C-CORE-01: Strict installation must propagate any
        # processor/ownership failure. The historical warning-and-continue path
        # could silently leave the base U-Net or the wrong adapters trainable.
        h = model.target_size // int(model.vae_scale_factor)
        w = model.target_size // int(model.vae_scale_factor)
        zero_ctx = torch.zeros(
            1, 1, h, w, device=model.unet.device, dtype=model.unet.dtype
        )
        patch_unet_attention_processors(
            pipeline=model,
            mask=zero_ctx,
            mask_ref=zero_ctx,
            scale=1.0,
            id_embeds=None,
            class_tokens_mask=None,
        )
        configure_e13_trainables(model)
        assert_e13_trainable_contract(model)
        return
    try:
        h = model.target_size // int(model.vae_scale_factor)
        w = model.target_size // int(model.vae_scale_factor)
        zero_ctx = torch.zeros(1, 1, h, w, device=model.unet.device, dtype=model.unet.dtype)

        patch_unet_attention_processors(
            pipeline=model,
            mask=zero_ctx,
            mask_ref=zero_ctx,
            scale=1.0,
            id_embeds=None,
            class_tokens_mask=None,
        )

        if hasattr(model.unet, "attn_processors"):
            for proc in model.unet.attn_processors.values():
                for p in proc.parameters():
                    p.requires_grad_(True)

        if model.face_embed_strategy == "id_embeds" and not model.use_attn_v2:
            for name, proc in model.unet.attn_processors.items():
                if not name.endswith("attn1.processor"):
                    continue
                if getattr(proc, "id_to_hidden", None) is None and hasattr(proc, "hidden_size"):
                    proc.id_to_hidden = torch.nn.Linear(2048, proc.hidden_size, bias=False).to(
                        model.unet.device, dtype=model.unet.dtype
                    )
                    with torch.no_grad():
                        proc.id_to_hidden.weight.mul_(0.1)

        configure_branched_trainables(model)
    except Exception as e:
        print(f"[PhotomakerBranchedLora] exception while installing branched processors: {e}")


def prepare_branched_training_inputs(
    model,
    *,
    prompts: Sequence[str],
    ref_images: Sequence[Sequence],
    face_bbox: Sequence[Sequence[float]],
    face_bbox_ref: Sequence[Sequence[float]] | None = None,
    pixel_values: torch.Tensor,
    noisy_latents: torch.Tensor,
):
    """
    Build all branched-training tensors from prompts/references/bboxes.
    Returns prompt embeddings, pooled embeddings, class-token mask, face-branch embeds,
    optional ID features, masks, and reference latents.
    """
    if bool(getattr(model, "batched_conditioning_preparation", False)):
        return _prepare_branched_training_inputs_batched(
            model,
            prompts=prompts,
            ref_images=ref_images,
            face_bbox=face_bbox,
            face_bbox_ref=face_bbox_ref,
            pixel_values=pixel_values,
            noisy_latents=noisy_latents,
        )
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    class_tokens_mask_list = []
    mask_list = []
    ref_mask_list = []
    ref_latents_list = []
    pm_feature_list = []

    image_h, image_w = pixel_values.shape[-2:]
    latent_h, latent_w = noisy_latents.shape[-2:]

    for i, (prompt, refs, bbox) in enumerate(zip(prompts, ref_images, face_bbox)):
        refs = refs if isinstance(refs, (list, tuple)) else [refs]
        if len(refs) != 1:
            raise ValueError("Training batch must contain exactly one reference image per sample")
        if face_bbox_ref is None:
            raise ValueError("Training batch is missing reference bboxes")
        ref_bbox = face_bbox_ref[i]

        prompt_embeds, pooled_prompt_embeds, class_tokens_mask = model.encode_prompt_with_trigger_word(
            prompt=prompt,
            num_id_images=1,
            do_cfg=False,
        )

        with torch.no_grad():
            # id_pixel_values = model.id_image_processor(refs, return_tensors="pt").pixel_values.unsqueeze(0)
            id_pixel_values = model.id_image_processor(deepcopy(refs), return_tensors="pt").pixel_values.unsqueeze(0) # DONE 01 JUN replaced refs with deepcopy of refs to avoid potential issues
            id_pixel_values = id_pixel_values.to(model.device, dtype=model.id_encoder.dtype)

            prompt_for_id = prompt_embeds.to(dtype=model.id_encoder.dtype)
            id_embed_list = []
            for ref in refs:
                img_np = np.array(ref.convert("RGB"))[:, :, ::-1]
                faces = analyze_faces(model.face_analyzer, img_np)
                if faces:
                    embedding = torch.from_numpy(faces[0]["embedding"]).float()
                else:
                    embedding = torch.zeros(512, dtype=torch.float32)
                id_embed_list.append(embedding)

            id_embeds = torch.stack(id_embed_list, dim=0).unsqueeze(0)
            id_embeds = id_embeds.to(device=model.device, dtype=model.id_encoder.dtype)

            prompt_embeds = model.id_encoder(
                id_pixel_values,
                prompt_for_id,
                class_tokens_mask,
                id_embeds,
            )

            ref = refs[0]
            reference_latent = model._encode_reference_latent(ref, target_shape=(latent_h, latent_w))
            if isinstance(ref, torch.Tensor):
                ref_h, ref_w = ref.shape[-2:]
            else:
                ref_w, ref_h = ref.size
            ref_mask = model._bbox_to_ref_mask(
                ref_bbox,
                latent_shape=(latent_h, latent_w),
                image_shape=(ref_h, ref_w),
            )

            if model.face_embed_strategy == "id_embeds":
                pm_features = model.id_encoder.extract_id_features(
                    id_pixel_values.to(device=model.device, dtype=model.id_encoder.dtype),
                    id_embeds=id_embeds,
                    class_tokens_mask=class_tokens_mask,
                )
                pm_feature_list.append(pm_features.to(device=model.device, dtype=model.unet.dtype))

        class_tokens_mask_list.append(class_tokens_mask)
        ref_latents_list.append(reference_latent)
        ref_mask_list.append(ref_mask)
        mask_list.append(
            model._bbox_to_mask(
                bbox,
                latent_shape=(latent_h, latent_w),
                image_shape=(image_h, image_w),
            )
        )
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=0).to(device=model.device, dtype=model.unet.dtype)
    pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0).to(device=model.device, dtype=model.unet.dtype)
    class_tokens_mask = torch.cat(class_tokens_mask_list, dim=0).to(device=model.device)

    id_features = None
    if model.face_embed_strategy == "face":
        face_prompt_text = ["a close-up human face laughing hard"] * prompt_embeds.shape[0]
        face_prompt_embeds, _ = model.encode_prompt(face_prompt_text, do_cfg=False)
        face_prompt_embeds = face_prompt_embeds.to(device=model.device, dtype=model.unet.dtype)
    elif model.face_embed_strategy == "id_embeds":
        if not pm_feature_list:
            raise ValueError("id_embeds strategy requires PM features in training forward.")
        id_features = torch.cat(pm_feature_list, dim=0)
        seq_len = prompt_embeds.shape[1]
        dim = prompt_embeds.shape[2]
        face_prompt_embeds = id_features.unsqueeze(1).expand(-1, seq_len, dim).contiguous()
    else:
        face_prompt_embeds = prompt_embeds

    mask4 = torch.cat(mask_list, dim=0).to(device=model.device, dtype=noisy_latents.dtype)
    mask4_ref = torch.cat(ref_mask_list, dim=0).to(device=model.device, dtype=noisy_latents.dtype)
    reference_latents = torch.cat(ref_latents_list, dim=0).to(device=model.device, dtype=noisy_latents.dtype)

    model._ref_latents_all = reference_latents
    model._face_prompt_embeds = prompt_embeds
    model.do_classifier_free_guidance = False
    if hasattr(model, "_ref_noise"):
        delattr(model, "_ref_noise")

    return (
        prompt_embeds,
        pooled_prompt_embeds,
        class_tokens_mask,
        face_prompt_embeds,
        id_features,
        mask4,
        mask4_ref,
        reference_latents,
    )


# 10 Aug 2026 - E13C-PERF-01: Batch the frozen text, PhotoMaker and VAE
# calls without changing per-sample prompts, boxes, masks or references.
def _prepare_branched_training_inputs_batched(
    model,
    *,
    prompts: Sequence[str],
    ref_images: Sequence[Sequence],
    face_bbox: Sequence[Sequence[float]],
    face_bbox_ref: Sequence[Sequence[float]] | None,
    pixel_values: torch.Tensor,
    noisy_latents: torch.Tensor,
):
    """Batch frozen conditioning work for large datasets with unique samples."""
    if face_bbox_ref is None:
        raise ValueError("Training batch is missing face_bbox_ref")

    refs_per_sample = [
        refs if isinstance(refs, (list, tuple)) else [refs]
        for refs in ref_images
    ]
    if any(len(refs) != 1 for refs in refs_per_sample):
        raise ValueError(
            "batched_conditioning_preparation currently requires one reference "
            "image per training sample"
        )
    flat_refs = [refs[0] for refs in refs_per_sample]
    batch_size = len(flat_refs)
    if not (
        len(prompts)
        == len(face_bbox)
        == len(face_bbox_ref)
        == batch_size
        == pixel_values.shape[0]
    ):
        raise ValueError("Batched conditioning inputs have inconsistent batch sizes")

    image_h, image_w = pixel_values.shape[-2:]
    latent_h, latent_w = noisy_latents.shape[-2:]
    prompt_embeds, pooled_prompt_embeds, class_tokens_mask = (
        model.encode_prompts_with_trigger_word(prompts, num_id_images=1)
    )

    # 26 Jul 2026 - Full Cosmic supplies effectively unique target/reference
    # pairs. Batch all frozen encoders so throughput does not depend on cache
    # reuse; legacy per-sample preparation remains the default.
    # AICODE-NOTE: Batching changes only execution grouping. References,
    # supplied boxes, PhotoMaker features, and target masks remain per sample.
    with torch.no_grad():
        id_pixel_values = model.id_image_processor(
            flat_refs, return_tensors="pt"
        ).pixel_values.unsqueeze(1)
        id_pixel_values = id_pixel_values.to(
            model.device, dtype=model.id_encoder.dtype
        )

        id_embed_list = []
        for ref in flat_refs:
            img_np = np.array(ref.convert("RGB"))[:, :, ::-1]
            faces = analyze_faces(model.face_analyzer, img_np)
            if faces:
                embedding = torch.from_numpy(faces[0]["embedding"]).float()
            else:
                embedding = torch.zeros(512, dtype=torch.float32)
            id_embed_list.append(embedding)
        id_embeds = torch.stack(id_embed_list, dim=0).unsqueeze(1).to(
            device=model.device, dtype=model.id_encoder.dtype
        )

        prompt_embeds = model.id_encoder(
            id_pixel_values,
            prompt_embeds.to(dtype=model.id_encoder.dtype),
            class_tokens_mask,
            id_embeds,
        )
        reference_latents = model._encode_reference_latents(
            flat_refs, target_shape=(latent_h, latent_w)
        )

        id_features = None
        if model.face_embed_strategy == "id_embeds":
            id_features = model.id_encoder.extract_id_features(
                id_pixel_values,
                id_embeds=id_embeds,
                class_tokens_mask=class_tokens_mask,
            ).to(device=model.device, dtype=model.unet.dtype)

    target_masks = []
    ref_masks = []
    for bbox, ref_bbox, ref in zip(face_bbox, face_bbox_ref, flat_refs):
        ref_w, ref_h = ref.size
        target_masks.append(
            model._bbox_to_mask(
                bbox,
                latent_shape=(latent_h, latent_w),
                image_shape=(image_h, image_w),
            )
        )
        ref_masks.append(
            model._bbox_to_ref_mask(
                ref_bbox,
                latent_shape=(latent_h, latent_w),
                image_shape=(ref_h, ref_w),
            )
        )

    prompt_embeds = prompt_embeds.to(device=model.device, dtype=model.unet.dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(
        device=model.device, dtype=model.unet.dtype
    )
    class_tokens_mask = class_tokens_mask.to(device=model.device)

    if model.face_embed_strategy == "face":
        face_prompt_text = ["a close-up human face laughing hard"] * batch_size
        face_prompt_embeds, _ = model.encode_prompt(
            face_prompt_text, do_cfg=False
        )
        face_prompt_embeds = face_prompt_embeds.to(
            device=model.device, dtype=model.unet.dtype
        )
    elif model.face_embed_strategy == "id_embeds":
        seq_len = prompt_embeds.shape[1]
        dim = prompt_embeds.shape[2]
        face_prompt_embeds = id_features.unsqueeze(1).expand(
            -1, seq_len, dim
        ).contiguous()
    else:
        face_prompt_embeds = prompt_embeds

    mask4 = torch.cat(target_masks, dim=0).to(
        device=model.device, dtype=noisy_latents.dtype
    )
    mask4_ref = torch.cat(ref_masks, dim=0).to(
        device=model.device, dtype=noisy_latents.dtype
    )
    reference_latents = reference_latents.to(
        device=model.device, dtype=noisy_latents.dtype
    )

    model._ref_latents_all = reference_latents
    model._face_prompt_embeds = prompt_embeds
    model.do_classifier_free_guidance = False
    if hasattr(model, "_ref_noise"):
        delattr(model, "_ref_noise")

    return (
        prompt_embeds,
        pooled_prompt_embeds,
        class_tokens_mask,
        face_prompt_embeds,
        id_features,
        mask4,
        mask4_ref,
        reference_latents,
    )


def run_branched_forward_pass(
    model,
    *,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: dict,
    mask4: torch.Tensor,
    mask4_ref: torch.Tensor,
    reference_latents: torch.Tensor,
    face_prompt_embeds: torch.Tensor,
    class_tokens_mask: torch.Tensor,
    id_features: torch.Tensor | None,
    reference_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run branched two-branch prediction and return merged noise prediction."""
    noise_pred, _, _ = two_branch_predict(
        pipeline=model,
        latent_model_input=noisy_latents,
        t=timesteps,
        prompt_embeds=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        mask4=mask4,
        mask4_ref=mask4_ref,
        reference_latents=reference_latents,
        reference_noise=reference_noise,
        face_prompt_embeds=face_prompt_embeds,
        class_tokens_mask=class_tokens_mask,
        face_embed_strategy=model.face_embed_strategy,
        id_embeds=id_features if model.face_embed_strategy == "id_embeds" else None,
        step_idx=0,
        scale=1.0,
        timestep_cond=None,
    )
    return noise_pred


def ensure_branched_after_eval(model) -> None:
    """Re-install branched processors after validation when needed."""
    dev = getattr(model, "device", None) or model.unet.device
    if not hasattr(model, "device"):
        model.device = dev
    dt = model.unet.dtype

    z = torch.zeros(1, 1, 1, 1, device=dev, dtype=dt)
    idem = torch.zeros(1, 2048, device=dev, dtype=dt)
    patch_unet_attention_processors(
        model,
        z,
        z,
        scale=1.0,
        id_embeds=idem,
        class_tokens_mask=None,
    )
