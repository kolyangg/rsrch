from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .branched_runtime import patch_unet_attention_processors, two_branch_predict
from .insightface_package import analyze_faces


def select_branched_trainable_processor_names(
    attn_processor_names: Sequence[str],
    *,
    train_ca: bool,
    ba_train_top_k: float,
) -> list[str]:
    top_k = float(ba_train_top_k)
    if not 0.0 <= top_k <= 1.0:
        raise ValueError(f"ba_train_top_k must be in [0.0, 1.0], got {top_k}")

    candidate_names: list[str] = []
    for name in attn_processor_names:
        if name.endswith("attn1.processor"):
            candidate_names.append(name)
        elif train_ca and name.endswith("attn2.processor"):
            candidate_names.append(name)

    if not candidate_names or top_k >= 1.0:
        return candidate_names
    if top_k <= 0.0:
        return []

    keep_count = max(1, math.ceil(len(candidate_names) * top_k))
    return candidate_names[:keep_count]


def configure_branched_trainables(model) -> None:
    if not getattr(model, "train_ba_only", False):
        return

    mode = (getattr(model, "branched_attn_weight_mode", "shared") or "shared").lower()
    new_weight_kind = (getattr(model, "branched_attn_new_weight_kind", "full") or "full").lower()
    train_ca = bool(getattr(model, "train_branched_ca_lora", True))
    ba_train_top_k = float(getattr(model, "ba_train_top_k", 1.0))
    if mode not in {"shared", "ref_only", "noise_and_ref"}:
        raise ValueError(f"Unknown branched_attn_weight_mode: {mode}")
    if new_weight_kind not in {"full", "lora"}:
        raise ValueError(f"Unknown branched_attn_new_weight_kind: {new_weight_kind}")

    selected_proc_prefixes: tuple[str, ...] = ()
    if mode != "shared":
        selected_proc_names = select_branched_trainable_processor_names(
            list(model.unet.attn_processors.keys()),
            train_ca=train_ca,
            ba_train_top_k=ba_train_top_k,
        )
        setattr(model, "_ba_trainable_processor_names", tuple(selected_proc_names))
        selected_proc_prefixes = tuple(f"{name}." for name in selected_proc_names)

    for _, p in model.unet.named_parameters():
        p.requires_grad_(False)

    for name, p in model.unet.named_parameters():
        if mode == "shared":
            if ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name and ".attn1." in name:
                p.requires_grad_(True)
        else:
            is_selected_proc = any(name.startswith(prefix) for prefix in selected_proc_prefixes)
            if is_selected_proc and ".attn1.processor.ref_to_" in name and (
                new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
            ):
                p.requires_grad_(True)
            elif is_selected_proc and mode == "noise_and_ref" and ".attn1.processor.noise_to_" in name and (
                new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
            ):
                p.requires_grad_(True)

        if train_ca:
            if mode == "shared":
                if ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name and ".attn2." in name:
                    p.requires_grad_(True)
            else:
                is_selected_proc = any(name.startswith(prefix) for prefix in selected_proc_prefixes)
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
        ref0 = refs[0]

        prompt_embeds, pooled_prompt_embeds, class_tokens_mask = model.encode_prompt_with_trigger_word(
            prompt=prompt,
            num_id_images=len(refs),
            do_cfg=False,
        )

        with torch.no_grad():
            id_pixel_values = model.id_image_processor(refs, return_tensors="pt").pixel_values.unsqueeze(0)
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

            reference_latent = model._encode_reference_latent(ref0, target_shape=(latent_h, latent_w))

            if model.face_embed_strategy == "id_embeds":
                pm_features = model.id_encoder.extract_id_features(
                    id_pixel_values.to(device=model.device, dtype=model.id_encoder.dtype),
                    id_embeds=id_embeds,
                    class_tokens_mask=class_tokens_mask,
                )
                pm_feature_list.append(pm_features.to(device=model.device, dtype=model.unet.dtype))

        class_tokens_mask_list.append(class_tokens_mask)
        ref_latents_list.append(reference_latent)
        ref_bbox = None if face_bbox_ref is None else face_bbox_ref[i]
        if ref_bbox is None:
            raise ValueError("Training batch is missing face_bbox_ref for reference masking")

        if isinstance(ref0, torch.Tensor):
            ref_h, ref_w = ref0.shape[-2:]
        else:
            ref_w, ref_h = ref0.size

        ref_mask_list.append(
            model._bbox_to_ref_mask(
                ref_bbox,
                latent_shape=(latent_h, latent_w),
                image_shape=(ref_h, ref_w),
            )
        )
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
