from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .branched_runtime import patch_unet_attention_processors, two_branch_predict
from .e13_contract import (
    assert_trainable_contract,
    configure_trainables,
)
from .insightface_package import analyze_faces


def install_branched_processors_for_training(model) -> None:
    """Install the fixed E13-family processors before optimizer creation."""
    height = model.target_size // int(model.vae_scale_factor)
    context = torch.zeros(
        1,
        1,
        height,
        height,
        device=model.unet.device,
        dtype=model.unet.dtype,
    )
    patch_unet_attention_processors(
        pipeline=model,
        mask=context,
        mask_ref=context,
        scale=1.0,
        id_embeds=None,
        class_tokens_mask=None,
    )
    configure_trainables(model)
    assert_trainable_contract(model)


def _encode_prompts_with_trigger_word(
    model,
    prompts: Sequence[str],
    *,
    num_id_images: int = 1,
):
    """Encode the one-trigger PhotoMaker prompt batch in one frozen pass."""
    prompts = list(prompts)
    if not prompts:
        raise ValueError("prompts must not be empty")

    image_token_id = model.tokenizer_2.convert_tokens_to_ids(model.trigger_word)
    tokenizers = (
        [model.tokenizer, model.tokenizer_2]
        if model.tokenizer is not None
        else [model.tokenizer_2]
    )
    text_encoders = (
        [model.text_encoder, model.text_encoder_2]
        if model.text_encoder is not None
        else [model.text_encoder_2]
    )

    prompt_embeds_list = []
    class_tokens_mask = None
    pooled_prompt_embeds = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        cleaned_rows = []
        mask_rows = []
        for prompt, token_ids in zip(prompts, text_inputs.input_ids.tolist()):
            clean_input_ids = []
            class_token_indices = []
            for token_id in token_ids:
                if token_id == image_token_id:
                    class_token_indices.append(len(clean_input_ids) - 1)
                else:
                    clean_input_ids.append(token_id)

            if len(class_token_indices) != 1:
                raise ValueError(
                    "PhotoMaker requires exactly one trigger word per prompt. "
                    f"Trigger word: {model.trigger_word}, prompt: {prompt}."
                )
            class_token_index = class_token_indices[0]
            class_token = clean_input_ids[class_token_index]
            clean_input_ids = (
                clean_input_ids[:class_token_index]
                + [class_token] * num_id_images * model.num_tokens
                + clean_input_ids[class_token_index + 1 :]
            )

            max_len = tokenizer.model_max_length
            clean_input_ids = clean_input_ids[:max_len]
            clean_input_ids += [tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )
            cleaned_rows.append(clean_input_ids)
            mask_rows.append([
                class_token_index
                <= index
                < class_token_index + num_id_images * model.num_tokens
                for index in range(max_len)
            ])

        text_input_ids = torch.tensor(
            cleaned_rows, dtype=torch.long, device=model.device
        )
        class_tokens_mask = torch.tensor(
            mask_rows, dtype=torch.bool, device=model.device
        )
        encoded = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = encoded[0]
        prompt_embeds_list.append(encoded.hidden_states[-2])

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1).to(model.device)
    pooled_prompt_embeds = pooled_prompt_embeds.view(len(prompts), -1)
    return prompt_embeds, pooled_prompt_embeds, class_tokens_mask


def bbox_to_reference_mask(
    model,
    bbox: Sequence[float] | None,
    latent_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> torch.Tensor:
    mask = torch.zeros(
        1, 1, model.target_size, model.target_size, device=model.device
    )
    if bbox is None or len(bbox) < 4:
        mask.fill_(1.0)
    else:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if x1 <= x0 or y1 <= y0:
            mask.fill_(1.0)
        else:
            image_h, image_w = image_shape
            scale = min(
                model.target_size / max(image_w, 1),
                model.target_size / max(image_h, 1),
            )
            resized_w = max(8, int(round(image_w * scale)) // 8 * 8)
            resized_h = max(8, int(round(image_h * scale)) // 8 * 8)
            pad_left = (model.target_size - resized_w) // 2
            pad_top = (model.target_size - resized_h) // 2
            scale_w = resized_w / max(image_w, 1)
            scale_h = resized_h / max(image_h, 1)
            x_start = max(0, min(model.target_size, int(round(x0 * scale_w + pad_left))))
            x_end = max(0, min(model.target_size, int(round(x1 * scale_w + pad_left))))
            y_start = max(0, min(model.target_size, int(round(y0 * scale_h + pad_top))))
            y_end = max(0, min(model.target_size, int(round(y1 * scale_h + pad_top))))
            if x_end <= x_start or y_end <= y_start:
                mask.fill_(1.0)
            else:
                mask[:, :, y_start:y_end, x_start:x_end] = 1.0

    if mask.shape[-2:] != latent_shape:
        mask = F.interpolate(mask, size=latent_shape, mode="nearest")
    return mask


def _bbox_to_target_mask(
    model,
    bbox: Sequence[float] | None,
    latent_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> torch.Tensor:
    mask = torch.zeros(1, 1, *latent_shape, device=model.device)
    if bbox is None or len(bbox) < 4:
        return mask.fill_(1.0)

    x0, y0, x1, y1 = [float(value) for value in bbox]
    if x1 <= x0 or y1 <= y0:
        return mask.fill_(1.0)

    scale_w = latent_shape[1] / max(image_shape[1], 1)
    scale_h = latent_shape[0] / max(image_shape[0], 1)
    x_start = max(0, min(latent_shape[1], int(round(x0 * scale_w))))
    x_end = max(0, min(latent_shape[1], int(round(x1 * scale_w))))
    y_start = max(0, min(latent_shape[0], int(round(y0 * scale_h))))
    y_end = max(0, min(latent_shape[0], int(round(y1 * scale_h))))
    if x_end <= x_start or y_end <= y_start:
        return mask.fill_(1.0)

    mask[:, :, y_start:y_end, x_start:x_end] = 1.0
    feather = int(model.ba_training_mask_feather)
    for step in range(1, feather + 1):
        weight = step / float(feather + 1)
        ys, ye = y_start + step - 1, y_end - step + 1
        xs, xe = x_start + step - 1, x_end - step + 1
        if ye <= ys or xe <= xs:
            break
        mask[:, :, ys, xs:xe] = weight
        mask[:, :, ye - 1, xs:xe] = weight
        mask[:, :, ys:ye, xs] = weight
        mask[:, :, ys:ye, xe - 1] = weight
    return mask


def encode_reference_latents(
    model,
    ref_images: Sequence[Image.Image],
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """Encode equal-sized references in one frozen VAE pass."""
    ref_tensors = []
    for ref_image in ref_images:
        if not isinstance(ref_image, Image.Image):
            raise TypeError(
                "Batched conditioning requires PIL references, "
                f"got {type(ref_image)}"
            )
        original_w, original_h = ref_image.size
        scale = min(
            model.target_size / original_w,
            model.target_size / original_h,
        )
        resized_w = max(8, int(round(original_w * scale)) // 8 * 8)
        resized_h = max(8, int(round(original_h * scale)) // 8 * 8)
        pad_left = (model.target_size - resized_w) // 2
        pad_right = model.target_size - resized_w - pad_left
        pad_top = (model.target_size - resized_h) // 2
        pad_bottom = model.target_size - resized_h - pad_top
        resized = ref_image.resize((resized_w, resized_h), Image.BILINEAR)
        array = np.array(resized).astype(np.float32) / 255.0
        tensor = (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5
        ref_tensors.append(
            F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        )

    ref_batch = torch.stack(ref_tensors).to(
        device=model.device, dtype=model.vae.dtype
    )
    with torch.no_grad():
        latents = model.vae.encode(ref_batch).latent_dist.mode()
    latents = latents * model.vae.config.scaling_factor
    if latents.shape[-2:] != target_shape:
        latents = F.interpolate(
            latents,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )
    return latents


# AICODE-NOTE: The selected recipes batch frozen text, PhotoMaker and VAE
# conditioning. This is the optimized execution path, not a second algorithm.
def prepare_branched_training_inputs(
    model,
    *,
    prompts: Sequence[str],
    ref_images: Sequence[Sequence],
    face_bbox: Sequence[Sequence[float]],
    face_bbox_ref: Sequence[Sequence[float]] | None,
    pixel_values: torch.Tensor,
    noisy_latents: torch.Tensor,
):
    if face_bbox_ref is None:
        raise ValueError("Training batch is missing face_bbox_ref")

    refs_per_sample = [
        refs if isinstance(refs, (list, tuple)) else [refs]
        for refs in ref_images
    ]
    if any(len(refs) != 1 for refs in refs_per_sample):
        raise ValueError("Training requires one reference image per sample")
    flat_refs = [refs[0] for refs in refs_per_sample]
    batch_size = len(flat_refs)
    if not (
        len(prompts)
        == len(face_bbox)
        == len(face_bbox_ref)
        == batch_size
        == pixel_values.shape[0]
    ):
        raise ValueError("Conditioning inputs have inconsistent batch sizes")

    image_h, image_w = pixel_values.shape[-2:]
    latent_h, latent_w = noisy_latents.shape[-2:]
    prompt_embeds, pooled_prompt_embeds, class_tokens_mask = (
        _encode_prompts_with_trigger_word(model, prompts, num_id_images=1)
    )

    with torch.no_grad():
        id_pixel_values = model.id_image_processor(
            flat_refs, return_tensors="pt"
        ).pixel_values.unsqueeze(1)
        id_pixel_values = id_pixel_values.to(
            model.device, dtype=model.id_encoder.dtype
        )

        id_embed_list = []
        for ref in flat_refs:
            image = np.array(ref.convert("RGB"))[:, :, ::-1]
            faces = analyze_faces(model.face_analyzer, image)
            embedding = (
                torch.from_numpy(faces[0]["embedding"]).float()
                if faces
                else torch.zeros(512, dtype=torch.float32)
            )
            id_embed_list.append(embedding)
        id_embeds = torch.stack(id_embed_list).unsqueeze(1).to(
            device=model.device, dtype=model.id_encoder.dtype
        )

        prompt_embeds = model.id_encoder(
            id_pixel_values,
            prompt_embeds.to(dtype=model.id_encoder.dtype),
            class_tokens_mask,
            id_embeds,
        )
        reference_latents = encode_reference_latents(
            model, flat_refs, target_shape=(latent_h, latent_w)
        )

    target_masks = [
        _bbox_to_target_mask(
            model,
            bbox,
            latent_shape=(latent_h, latent_w),
            image_shape=(image_h, image_w),
        )
        for bbox in face_bbox
    ]
    ref_masks = [
        bbox_to_reference_mask(
            model,
            bbox,
            latent_shape=(latent_h, latent_w),
            image_shape=(ref.height, ref.width),
        )
        for bbox, ref in zip(face_bbox_ref, flat_refs)
    ]

    prompt_embeds = prompt_embeds.to(device=model.device, dtype=model.unet.dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(
        device=model.device, dtype=model.unet.dtype
    )
    class_tokens_mask = class_tokens_mask.to(device=model.device)
    face_prompt_embeds = prompt_embeds

    mask4 = torch.cat(target_masks).to(
        device=model.device, dtype=noisy_latents.dtype
    )
    mask4_ref = torch.cat(ref_masks).to(
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
    reference_noise: torch.Tensor | None = None,
) -> torch.Tensor:
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
        face_embed_strategy="id",
        id_embeds=None,
        step_idx=0,
        scale=1.0,
        timestep_cond=None,
    )
    return noise_pred


def ensure_branched_after_eval(model) -> None:
    device = getattr(model, "device", None) or model.unet.device
    model.device = device
    context = torch.zeros(1, 1, 1, 1, device=device, dtype=model.unet.dtype)
    patch_unet_attention_processors(
        model,
        context,
        context,
        scale=1.0,
        id_embeds=torch.zeros(1, 2048, device=device, dtype=model.unet.dtype),
        class_tokens_mask=None,
    )
