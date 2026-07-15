from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .branched_runtime import (
    hard_epsilon_merge,
    patch_unet_attention_processors,
    select_branched_processor_names,
    target_face_predict,
    two_branch_predict,
)
from .insightface_package import analyze_faces
from .identity_memory import bbox_normalized_reference

from copy import deepcopy


def configure_branched_trainables(model) -> None:
    if not getattr(model, "train_ba_only", False):
        return

    mode = (getattr(model, "branched_attn_weight_mode", "shared") or "shared").lower()
    new_weight_kind = (getattr(model, "branched_attn_new_weight_kind", "full") or "full").lower()
    train_ca = bool(getattr(model, "train_branched_ca_lora", True))
    ca_train_mode = str(getattr(model, "ba_ca_train_mode", "all") or "all").lower()
    ba_train_top_k = float(getattr(model, "ba_train_top_k", 1.0))
    non_ba_train = bool(getattr(model, "non_ba_train", False))
    train_sa_id_embed_proj = bool(getattr(model, "ba_train_sa_id_embed_proj", False))
    sa_mode = str(getattr(model, "ba_sa_mode", "legacy") or "legacy").lower()
    ca_mode = str(getattr(model, "ba_ca_mode", "legacy_ref_branch") or "legacy_ref_branch").lower()
    if mode not in {"shared", "ref_only", "noise_and_ref"}:
        raise ValueError(f"Unknown branched_attn_weight_mode: {mode}")
    if new_weight_kind not in {"full", "lora"}:
        raise ValueError(f"Unknown branched_attn_new_weight_kind: {new_weight_kind}")
    if ca_train_mode not in {"all", "ref_only", "noise_only", "target_face"}:
        raise ValueError(f"Unknown ba_ca_train_mode: {ca_train_mode}")

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
        is_selected_proc = bool(selected_proc_prefixes) and name.startswith(selected_proc_prefixes)
        if sa_mode == "pm_face_residual":
            if is_selected_proc and (
                ".attn1.processor.ref_to_k." in name
                or ".attn1.processor.ref_to_v." in name
                or ".attn1.processor.face_delta_out." in name
                or name.endswith(".attn1.processor.face_residual_gate")
            ) and (new_weight_kind == "full" or "lora_" in name or "face_delta_out." in name or name.endswith("face_residual_gate")):
                p.requires_grad_(True)
        elif mode == "shared":
            is_selected_attn = bool(selected_attn_prefixes) and name.startswith(selected_attn_prefixes)
            if is_selected_attn and ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name and ".attn1." in name:
                p.requires_grad_(True)
        else:
            if is_selected_proc and ".attn1.processor.ref_to_" in name and (
                new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
            ):
                p.requires_grad_(True)
            elif is_selected_proc and mode == "noise_and_ref" and ".attn1.processor.noise_to_" in name and (
                new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
            ):
                p.requires_grad_(True)
            elif is_selected_proc and train_sa_id_embed_proj and ".attn1.processor.id_to_hidden." in name:
                p.requires_grad_(True)
            elif is_selected_proc and ".attn1.processor.face_fusion_logit" in name:
                p.requires_grad_(True)

        if non_ba_train and is_non_ba_attn and ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name:
            p.requires_grad_(True)

        if train_ca:
            if ca_mode == "target_face_residual":
                if is_selected_proc and (
                    ".attn2.processor.target_id_to_k." in name
                    or ".attn2.processor.target_id_to_v." in name
                    or ".attn2.processor.face_delta_out." in name
                    or name.endswith(".attn2.processor.face_residual_gate")
                    or name.endswith(".attn2.processor.id_token_basis")
                ) and (
                    new_weight_kind == "full"
                    or "lora_" in name
                    or ".face_delta_out." in name
                    or name.endswith("face_residual_gate")
                    or name.endswith("id_token_basis")
                ):
                    p.requires_grad_(True)
            elif mode == "shared":
                is_selected_attn = bool(selected_attn_prefixes) and name.startswith(selected_attn_prefixes)
                if is_selected_attn and ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name and ".attn2." in name:
                    p.requires_grad_(True)
            else:
                if is_selected_proc and ca_train_mode in {"all", "ref_only"} and ".attn2.processor.ref_to_" in name and (
                    new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
                ):
                    p.requires_grad_(True)
                elif is_selected_proc and ca_train_mode in {"all", "noise_only"} and mode == "noise_and_ref" and ".attn2.processor.noise_to_" in name and (
                    new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
                ):
                    p.requires_grad_(True)


def install_branched_processors_for_training(model) -> None:
    """Install branched attention processors once before optimizer creation."""
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
                if not isinstance(proc, torch.nn.Module):
                    continue
                for p in proc.parameters():
                    p.requires_grad_(True)

        # Keep a handle on the freshly installed branched processors so that
        # ensure_branched_after_eval() can re-attach these exact instances
        # (with their trained weights, still referenced by the optimizer)
        # instead of rebuilding new ones from the base attention weights.
        model._branched_attn_processors_train = dict(model.unet.attn_processors)

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
        raise RuntimeError("Failed to install branched attention processors") from e


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
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    class_tokens_mask_list = []
    mask_list = []
    ref_mask_list = []
    ref_latents_list = []
    pm_feature_list = []

    image_h, image_w = pixel_values.shape[-2:]
    latent_h, latent_w = noisy_latents.shape[-2:]
    needs_spatial_reference = not bool(getattr(model, "disable_reference_spatial_branch", False))
    needs_id_features = (
        model.face_embed_strategy == "id_embeds"
        or str(getattr(model, "ba_ca_mode", "legacy_ref_branch")) == "target_face_residual"
    )

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

            if needs_spatial_reference:
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

            if needs_id_features:
                feature_pixels = id_pixel_values
                if getattr(model, "ba_identity_image_mode", "full_reference") == "bbox_normalized":
                    cropped_refs = [
                        bbox_normalized_reference(
                            ref,
                            ref_bbox,
                            padding=getattr(model, "ba_identity_crop_padding", 0.10),
                        )
                        for ref in refs
                    ]
                    feature_pixels = model.id_image_processor(
                        cropped_refs, return_tensors="pt"
                    ).pixel_values.unsqueeze(0).to(model.device, dtype=model.id_encoder.dtype)
                feature_reduce = (
                    "tokens"
                    if getattr(model, "ba_identity_memory_mode", "mean_plus_basis") == "qformer_tokens"
                    else "mean"
                )
                pm_features = model.id_encoder.extract_id_features(
                    feature_pixels.to(device=model.device, dtype=model.id_encoder.dtype),
                    id_embeds=id_embeds,
                    class_tokens_mask=class_tokens_mask,
                    reduce=feature_reduce,
                )
                pm_feature_list.append(pm_features.to(device=model.device, dtype=model.unet.dtype))

        class_tokens_mask_list.append(class_tokens_mask)
        if needs_spatial_reference:
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
        prompt_id_features = id_features.mean(dim=1) if id_features.ndim == 3 else id_features
        face_prompt_embeds = prompt_id_features.unsqueeze(1).expand(-1, seq_len, dim).contiguous()
    else:
        face_prompt_embeds = prompt_embeds

    mask4 = torch.cat(mask_list, dim=0).to(device=model.device, dtype=noisy_latents.dtype)
    if needs_spatial_reference:
        mask4_ref = torch.cat(ref_mask_list, dim=0).to(device=model.device, dtype=noisy_latents.dtype)
        reference_latents = torch.cat(ref_latents_list, dim=0).to(device=model.device, dtype=noisy_latents.dtype)
    else:
        mask4_ref = mask4
        reference_latents = None

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
    reference_latents: torch.Tensor | None,
    face_prompt_embeds: torch.Tensor,
    class_tokens_mask: torch.Tensor,
    id_features: torch.Tensor | None,
) -> torch.Tensor:
    """Run the selected BA architecture and optionally hard-merge it with PhotoMaker."""
    preservation_mode = str(getattr(model, "ba_pm_preservation_mode", "none") or "none").lower()
    photomaker_pred = None
    if preservation_mode == "hard_epsilon_merge":
        set_branched_training_mode(model, branched_active=False)
        try:
            with torch.no_grad():
                photomaker_pred = model.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
        finally:
            set_branched_training_mode(model, branched_active=True)

    set_branched_training_mode(model, branched_active=True)
    if str(getattr(model, "ba_ca_mode", "legacy_ref_branch")) == "target_face_residual":
        if id_features is None:
            raise ValueError("target_face_residual training requires PhotoMaker identity features")
        noise_pred = target_face_predict(
            model,
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            mask4,
            id_features,
            class_tokens_mask=class_tokens_mask,
        )
    else:
        if reference_latents is None:
            raise ValueError("Spatial branched attention requires reference latents")
        noise_pred, _, _ = two_branch_predict(
            pipeline=model,
            latent_model_input=noisy_latents,
            t=timesteps,
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            mask4=mask4,
            mask4_ref=mask4_ref,
            reference_latents=reference_latents,
            face_prompt_embeds=face_prompt_embeds,
            class_tokens_mask=class_tokens_mask,
            face_embed_strategy=model.face_embed_strategy,
            id_embeds=id_features if model.face_embed_strategy == "id_embeds" else None,
            step_idx=0,
            scale=1.0,
            timestep_cond=None,
        )
    if photomaker_pred is not None:
        noise_pred = hard_epsilon_merge(photomaker_pred, noise_pred, mask4)
    model._ba_active_this_batch = True
    return noise_pred


def set_branched_training_mode(model, *, branched_active: bool) -> None:
    """Swap processor sets without rebuilding optimizer-owned BA modules."""
    attr = "_branched_attn_processors_train" if branched_active else "_original_attn_processors"
    target = getattr(model, attr, None)
    if not target:
        raise RuntimeError(f"Cannot select training attention mode: {attr} is unavailable")

    current = model.unet.attn_processors
    if all(current.get(name) is proc for name, proc in target.items()):
        return
    model.unet.set_attn_processor(dict(target))


def attach_inactive_branched_params(model, output: torch.Tensor) -> torch.Tensor:
    """Keep intentionally inactive BA params in the graph with exactly zero gradients."""
    processors = getattr(model, "_branched_attn_processors_train", {}).values()
    params = []
    seen = set()
    for proc in processors:
        if not isinstance(proc, torch.nn.Module):
            continue
        for param in proc.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    if not params:
        return output

    anchor = sum((param.reshape(-1)[0].float() * 0.0 for param in params), output.new_zeros(()).float())
    return output + anchor.to(device=output.device, dtype=output.dtype)


def ensure_branched_after_eval(model) -> None:
    """Re-install branched processors after validation when needed."""
    dev = getattr(model, "device", None) or model.unet.device
    if not hasattr(model, "device"):
        model.device = dev
    dt = model.unet.dtype

    # If validation swapped the shared UNet back to the original processors
    # (set_validation_unet_mode(branched_active=False)), re-attach the SAME
    # trained processor instances. Rebuilding via patch_unet_attention_processors
    # would create fresh clones (zero LoRA deltas) and silently detach training:
    # the optimizer would keep updating orphaned modules.
    trained_procs = getattr(model, "_branched_attn_processors_train", None)
    if trained_procs:
        current = model.unet.attn_processors
        if any(current.get(name) is not proc for name, proc in trained_procs.items()):
            model.unet.set_attn_processor(dict(trained_procs))

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
