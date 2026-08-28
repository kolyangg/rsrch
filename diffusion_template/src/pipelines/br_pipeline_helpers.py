from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from transformers import CLIPImageProcessor

from src.model.photomaker_branched.branched_runtime import (
    encode_face_prompt,
    two_branch_predict,
)
from src.model.photomaker_branched.branch_helpers import prepare_mask4
from src.model.photomaker_branched.debug_helpers import (
    debug_reference_latents_once,
    log_debug_image,
    save_branch_previews,
    save_debug_ref_latents,
    save_debug_ref_mask_overlay,
)
from src.model.photomaker_branched.insightface_package import analyze_faces, create_face_analyzer
from src.face_subject_selector import LEGACY_FIRST, select_subject_face


def _val_debug_enabled(pipeline) -> bool:
    return bool(getattr(pipeline, "_val_debug", True))


def expand_bbox_xyxy(
    bbox: Optional[Sequence[float]],
    *,
    expansion_ratio: float,
    width: int,
    height: int,
) -> Optional[List[float]]:
    if bbox is None:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    x0, y0, x1, y1 = [float(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return [x0, y0, x1, y1]

    grow = float(expansion_ratio) - 1.0
    dx = (x1 - x0) * grow
    dy = (y1 - y0) * grow

    return [
        max(0.0, min(float(width), x0 - dx)),
        max(0.0, min(float(height), y0 - dy)),
        max(0.0, min(float(width), x1 + dx)),
        max(0.0, min(float(height), y1 + dy)),
    ]


def annotate_original_and_expanded_bbox(
    img: PIL.Image.Image,
    *,
    original_bbox: Optional[Sequence[float]],
    expanded_bbox: Optional[Sequence[float]],
    line_width: int = 4,
) -> PIL.Image.Image:
    from bbox_utils.visualize_bboxes import annotate_pil

    boxes = {}
    if original_bbox is not None:
        boxes["orig"] = original_bbox
    if expanded_bbox is not None:
        orig_rounded = tuple(int(round(v)) for v in original_bbox) if original_bbox is not None else None
        exp_rounded = tuple(int(round(v)) for v in expanded_bbox)
        if exp_rounded != orig_rounded:
            boxes["expanded"] = expanded_bbox
    return annotate_pil(img, boxes, line_width=line_width)


def _bbox_mask_from_original_and_expanded(
    *,
    original_bbox: Optional[Sequence[float]],
    expanded_bbox: Optional[Sequence[float]],
    height: int,
    width: int,
    mask_softness: float,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    if original_bbox is None and expanded_bbox is None:
        return mask

    base_bbox = expanded_bbox if expanded_bbox is not None else original_bbox
    core_bbox = original_bbox if original_bbox is not None else base_bbox
    if base_bbox is None or core_bbox is None:
        return mask

    ex0, ey0, ex1, ey1 = [int(round(v)) for v in base_bbox]
    ox0, oy0, ox1, oy1 = [int(round(v)) for v in core_bbox]

    ex0 = max(0, min(width, ex0))
    ex1 = max(0, min(width, ex1))
    ey0 = max(0, min(height, ey0))
    ey1 = max(0, min(height, ey1))
    ox0 = max(0, min(width, ox0))
    ox1 = max(0, min(width, ox1))
    oy0 = max(0, min(height, oy0))
    oy1 = max(0, min(height, oy1))

    if ex1 <= ex0 or ey1 <= ey0 or ox1 <= ox0 or oy1 <= oy0:
        return mask

    hard = np.zeros((height, width), dtype=np.float32)
    hard[ey0:ey1, ex0:ex1] = 1.0

    softness = float(np.clip(mask_softness, 0.0, 1.0))
    if softness <= 0.0:
        return hard

    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    wx = np.zeros(width, dtype=np.float32)
    wy = np.zeros(height, dtype=np.float32)

    if ox0 > ex0:
        sel = (xs >= ex0) & (xs < ox0)
        wx[sel] = (xs[sel] - ex0) / max(float(ox0 - ex0), 1.0)
    wx[(xs >= ox0) & (xs < ox1)] = 1.0
    if ex1 > ox1:
        sel = (xs >= ox1) & (xs < ex1)
        wx[sel] = (ex1 - xs[sel]) / max(float(ex1 - ox1), 1.0)

    if oy0 > ey0:
        sel = (ys >= ey0) & (ys < oy0)
        wy[sel] = (ys[sel] - ey0) / max(float(oy0 - ey0), 1.0)
    wy[(ys >= oy0) & (ys < oy1)] = 1.0
    if ey1 > oy1:
        sel = (ys >= oy1) & (ys < ey1)
        wy[sel] = (ey1 - ys[sel]) / max(float(ey1 - oy1), 1.0)

    soft = np.minimum(wy[:, None], wx[None, :]).astype(np.float32)
    return ((1.0 - softness) * hard + softness * soft).astype(np.float32)


def ensure_face_analyzer(pipeline) -> None:
    if hasattr(pipeline, "_face_analyzer"):
        return
    pipeline._face_analyzer = create_face_analyzer(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
        ctx_id=0,
        det_size=(640, 640),
        fallback_ctx_id=-1,
        quiet=True,
    )


def ensure_id_embeds(
    pipeline,
    *,
    id_embeds: Optional[torch.FloatTensor],
    input_id_images: Sequence[Any],
    face_bbox_ref: Optional[Sequence[float] | Sequence[Sequence[float]]],
    face_subject_selection_policy: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.FloatTensor:
    #### 08 MAR - FIX BATCHED VALIDATION ####
    def _normalize_id_embeds(x: torch.FloatTensor) -> torch.FloatTensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Unsupported id_embeds shape: {tuple(x.shape)}")
        return x.to(device=device, dtype=dtype)

    if id_embeds is not None:
        return _normalize_id_embeds(id_embeds)

    ensure_face_analyzer(pipeline)

    is_per_prompt = (
        isinstance(input_id_images, (list, tuple))
        and len(input_id_images) > 0
        and isinstance(input_id_images[0], (list, tuple))
    )
    if is_per_prompt:
        refs = []
        for refs_for_prompt in input_id_images:
            if isinstance(refs_for_prompt, (list, tuple)) and len(refs_for_prompt) > 0:
                refs.append(refs_for_prompt[0])
            else:
                refs.append(refs_for_prompt)
    else:
        refs = list(input_id_images)

    if is_per_prompt:
        if (
            isinstance(face_bbox_ref, (list, tuple))
            and len(face_bbox_ref) == len(refs)
            and all(
                bbox is None
                or (isinstance(bbox, (list, tuple)) and len(bbox) == 4)
                for bbox in face_bbox_ref
            )
        ):
            declared_bboxes = list(face_bbox_ref)
        else:
            declared_bboxes = [face_bbox_ref] * len(refs)
    else:
        declared_bboxes = [None] * len(refs)
        if declared_bboxes:
            declared_bboxes[0] = face_bbox_ref

    embeddings = []
    selections = []
    for ref, declared_bbox in zip(refs, declared_bboxes):
        if isinstance(ref, torch.Tensor):
            ref_img = ref.detach().cpu()
            if ref_img.dim() == 3:
                ref_img = ref_img.unsqueeze(0)
            ref_img = ref_img[0]
            ref_img = (ref_img * 0.5 + 0.5).clamp(0, 1)
            ref_img = (ref_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_np = ref_img[:, :, ::-1]
        else:
            img_np = np.array(ref.convert("RGB"))[:, :, ::-1]

        faces = analyze_faces(pipeline._face_analyzer, img_np)
        if faces:
            selected, audit = select_subject_face(
                faces,
                declared_bbox=declared_bbox,
                policy=face_subject_selection_policy,
            )
            embedding = torch.from_numpy(selected["embedding"]).float()
            selections.append(audit.to_dict())
        else:
            if str(face_subject_selection_policy).lower() != LEGACY_FIRST:
                raise RuntimeError(
                    "Subject-v2 identity conditioning found no face in a reference image"
                )
            embedding = torch.zeros(512, dtype=torch.float32)
            selections.append(
                {
                    "selection_reason": "legacy_no_face_zero_vector",
                    "face_count": 0,
                }
            )
        embeddings.append(embedding)

    pipeline._face_subject_selections = selections

    stacked = torch.stack(embeddings, dim=0)
    if is_per_prompt:
        stacked = stacked.unsqueeze(1)
    else:
        stacked = stacked.unsqueeze(0)
    #### 08 MAR - FIX BATCHED VALIDATION ####
    return stacked.to(device=device, dtype=dtype)


def prepare_ref_latents(
    pipeline,
    *,
    pil: PIL.Image.Image,
    height: int,
    width: int,
    latents_dtype: torch.dtype,
) -> torch.Tensor:
    ow, oh = pil.size
    pipeline._ref_orig_size = (oh, ow)

    s = min(width / ow, height / oh)
    rw = max(8, int(round(ow * s)) // 8 * 8)
    rh = max(8, int(round(oh * s)) // 8 * 8)
    pl = (width - rw) // 2
    pr = width - rw - pl
    pt = (height - rh) // 2
    pb = height - rh - pt
    pipeline._ref_pad = (pl, pr, pt, pb)
    pipeline._ref_scaled_size = (rh, rw)

    with torch.no_grad():
        ref_pixels = pipeline.image_processor.preprocess(pil, height=rh, width=rw)
        ref_pixels = F.pad(ref_pixels, (pl, pr, pt, pb), value=0.0)
        ref_pixels = ref_pixels.to(device=pipeline._execution_device, dtype=latents_dtype)
        ref_latents = pipeline.vae.encode(ref_pixels).latent_dist.mode() * pipeline.vae.config.scaling_factor
    return ref_latents


def prepare_ref_mask(
    pipeline,
    *,
    pil: PIL.Image.Image,
    auto_mask_ref: bool,
    use_bbox_mask_ref: bool,
    face_bbox_ref: Optional[List[float]],
    mask_expansion_ratio: float,
    mask_softness: float,
    import_mask_ref: Optional[str],
    debug_dir: Optional[str],
    height: int,
    width: int,
) -> Optional[str]:
    if auto_mask_ref:
        from src.model.photomaker_branched.create_mask_ref import compute_face_mask_from_pil

        os.makedirs(debug_dir, exist_ok=True)
        auto_ref_path = os.path.join(debug_dir, "auto_ref_mask.png")
        mask_array = compute_face_mask_from_pil(pil)
        PIL.Image.fromarray(mask_array).save(auto_ref_path)
        log_debug_image(f"[DebugImage] auto_ref_mask → {auto_ref_path}")
        import_mask_ref = auto_ref_path
        print(f"[AutoMaskRef] Generated ref mask → {auto_ref_path}")
    else:
        print(f"[AutoMaskRef] Using existing ref mask at {import_mask_ref}")

    if (not auto_mask_ref) and use_bbox_mask_ref and face_bbox_ref is None:
        raise RuntimeError(
            "use_bbox_mask_ref=True but no face_bbox_ref provided for reference image;"
            " ensure ref_bboxes.json contains an entry for this reference."
        )

    if (not auto_mask_ref) and use_bbox_mask_ref and face_bbox_ref is not None:
        ref_mask = np.zeros((height, width), dtype=bool)
        ow, oh = pil.size
        pl, _, pt, _ = pipeline._ref_pad
        s = min(width / float(ow), height / float(oh))
        sx, sy = s, s
        expanded_bbox_ref = expand_bbox_xyxy(
            face_bbox_ref,
            expansion_ratio=mask_expansion_ratio,
            width=ow,
            height=oh,
        )
        x0, y0, x1, y1 = [float(v) for v in expanded_bbox_ref]
        x0s = int(round(x0 * sx + pl))
        x1s = int(round(x1 * sx + pl))
        y0s = int(round(y0 * sy + pt))
        y1s = int(round(y1 * sy + pt))
        x0s = max(0, min(width, x0s))
        x1s = max(0, min(width, x1s))
        y0s = max(0, min(height, y0s))
        y1s = max(0, min(height, y1s))
        orig_x0, orig_y0, orig_x1, orig_y1 = [float(v) for v in face_bbox_ref]
        ox0s = int(round(orig_x0 * sx + pl))
        ox1s = int(round(orig_x1 * sx + pl))
        oy0s = int(round(orig_y0 * sy + pt))
        oy1s = int(round(orig_y1 * sy + pt))
        ox0s = max(0, min(width, ox0s))
        ox1s = max(0, min(width, ox1s))
        oy0s = max(0, min(height, oy0s))
        oy1s = max(0, min(height, oy1s))
        ref_mask = _bbox_mask_from_original_and_expanded(
            original_bbox=[ox0s, oy0s, ox1s, oy1s],
            expanded_bbox=[x0s, y0s, x1s, y1s],
            height=height,
            width=width,
            mask_softness=mask_softness,
        )
        pipeline._face_bbox_ref_original = list(face_bbox_ref)
        pipeline._face_bbox_ref_expanded = list(expanded_bbox_ref)
        pipeline._face_mask_ref = ref_mask
        pipeline._face_mask_t_ref = torch.from_numpy(ref_mask.astype(np.float32))[None, None]

    return import_mask_ref


def prepare_gen_mask(
    pipeline,
    *,
    use_dynamic_mask: bool,
    use_bbox_mask_gen: bool,
    face_bbox_gen: Optional[List[float]],
    ba_target_visibility_mask: Optional[Any] = None,
    mask_expansion_ratio: float,
    mask_softness: float,
    height: int,
    width: int,
    batch_size: int = 1,
) -> None:
    automask = getattr(pipeline, "_ba_automask_target_mask", None)
    if automask is not None:
        value = torch.as_tensor(automask, dtype=torch.float32)
        if value.ndim == 3:
            value = value[:, None]
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError("AutoMask-OS target router must be [B,1,H,W]")
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1, -1, -1)
        if value.shape[0] != batch_size:
            raise ValueError("AutoMask-OS validation target batch mismatch")
        value = F.interpolate(value, (height, width), mode="bilinear", align_corners=False).clamp(0, 1)
        pipeline._face_mask_t = value
        pipeline._face_mask = value[:, 0].numpy()
        return

    def apply_visibility_mask(gen_mask: np.ndarray) -> np.ndarray:
        if ba_target_visibility_mask is None:
            pipeline._ba_target_visibility_mask = None
            return gen_mask

        value = ba_target_visibility_mask
        if isinstance(value, torch.Tensor):
            visibility = value.detach().float().cpu()
        else:
            if isinstance(value, PIL.Image.Image):
                value = np.asarray(value.convert("L"), dtype=np.float32) / 255.0
            elif isinstance(value, (list, tuple)) and value and isinstance(
                value[0], PIL.Image.Image
            ):
                value = np.stack(
                    [
                        np.asarray(item.convert("L"), dtype=np.float32) / 255.0
                        for item in value
                    ],
                    axis=0,
                )
            visibility = torch.as_tensor(np.asarray(value), dtype=torch.float32)

        if visibility.ndim == 2:
            visibility = visibility[None, None]
        elif visibility.ndim == 3:
            visibility = visibility[:, None]
        elif visibility.ndim == 4 and visibility.shape[1] == 1:
            pass
        else:
            raise ValueError(
                "ba_target_visibility_mask must have shape HxW, BxHxW, or Bx1xHxW"
            )
        if visibility.shape[0] == 1 and batch_size > 1:
            visibility = visibility.expand(batch_size, -1, -1, -1)
        if visibility.shape[0] != batch_size:
            raise ValueError(
                "ba_target_visibility_mask batch mismatch: "
                f"got {visibility.shape[0]}, expected {batch_size}"
            )
        if visibility.shape[-2:] != (height, width):
            visibility = F.interpolate(
                visibility,
                size=(height, width),
                mode="nearest",
            )
        if not bool(torch.isfinite(visibility).all()):
            raise ValueError("ba_target_visibility_mask contains non-finite values")
        if bool(torch.any((visibility < 0.0) | (visibility > 1.0))):
            raise ValueError("ba_target_visibility_mask values must be in [0, 1]")

        visibility_np = visibility[:, 0].numpy().astype(np.float32)
        base = np.asarray(gen_mask, dtype=np.float32)
        if base.ndim == 2:
            base = base[None]
        # 10 Aug 2026 - AICODE-NOTE: Visibility can only remove target queries
        # from the existing reference-owned face mask. Excluded pixels fall
        # through to the unchanged native/background lane; reference K/V and
        # the reference mask are deliberately untouched.
        result = base * visibility_np
        pipeline._ba_target_visibility_mask = visibility_np
        return result if np.asarray(gen_mask).ndim == 3 else result[0]

    #### 08 MAR - FIX BATCHED VALIDATION ####
    if (not use_dynamic_mask) and use_bbox_mask_gen and face_bbox_gen is not None:
        per_sample_boxes = (
            isinstance(face_bbox_gen, (list, tuple))
            and len(face_bbox_gen) > 0
            and isinstance(face_bbox_gen[0], (list, tuple))
        )
        if per_sample_boxes:
            boxes = list(face_bbox_gen)
            if len(boxes) != batch_size:
                raise RuntimeError(
                    f"use_bbox_mask_gen batch mismatch: got {len(boxes)} bboxes for batch_size={batch_size}"
                )
            gen_mask = np.zeros((batch_size, height, width), dtype=bool)
            expanded_boxes = []
            for bi, box in enumerate(boxes):
                expanded_box = expand_bbox_xyxy(
                    box,
                    expansion_ratio=mask_expansion_ratio,
                    width=width,
                    height=height,
                )
                expanded_boxes.append(list(expanded_box))
                gen_mask[bi] = _bbox_mask_from_original_and_expanded(
                    original_bbox=box,
                    expanded_bbox=expanded_box,
                    height=height,
                    width=width,
                    mask_softness=mask_softness,
                )
            pipeline._face_bbox_gen_original = [list(box) for box in boxes]
            pipeline._face_bbox_gen_expanded = expanded_boxes
            gen_mask = apply_visibility_mask(gen_mask)
            pipeline._face_mask = gen_mask
            pipeline._face_mask_t = torch.from_numpy(gen_mask.astype(np.float32))[:, None]
            return

        gen_mask = np.zeros((height, width), dtype=np.float32)
        expanded_bbox_gen = expand_bbox_xyxy(
            face_bbox_gen,
            expansion_ratio=mask_expansion_ratio,
            width=width,
            height=height,
        )
        gen_mask = _bbox_mask_from_original_and_expanded(
            original_bbox=face_bbox_gen,
            expanded_bbox=expanded_bbox_gen,
            height=height,
            width=width,
            mask_softness=mask_softness,
        )
        pipeline._face_bbox_gen_original = list(face_bbox_gen)
        pipeline._face_bbox_gen_expanded = list(expanded_bbox_gen)
        gen_mask = apply_visibility_mask(gen_mask)
        pipeline._face_mask = gen_mask
        pipeline._face_mask_t = torch.from_numpy(gen_mask.astype(np.float32))[None, None]
    elif (not use_dynamic_mask) and use_bbox_mask_gen and face_bbox_gen is None:
        raise RuntimeError(
            "use_bbox_mask_gen=True but no face_bbox_gen provided for generated image;"
            " ensure pm20_bboxes.json contains an entry for the current validation index"
            " (e.g., '00.png', '01.png', ...)."
        )
    #### 08 MAR - FIX BATCHED VALIDATION ####


def prepare_id_features(
    pipeline,
    *,
    id_pixel_values: Optional[torch.Tensor],
    prompt_embeds: torch.Tensor,
    id_embeds: Optional[torch.Tensor],
    class_tokens_mask: torch.LongTensor,
) -> None:
    if id_pixel_values is not None and hasattr(pipeline, "id_encoder"):
        pm_feats = pipeline.id_encoder.extract_id_features(
            id_pixel_values.to(device=pipeline.device, dtype=prompt_embeds.dtype),
            id_embeds=id_embeds,
            class_tokens_mask=class_tokens_mask,
        )
        pipeline._pm_id_embeds_2048 = pm_feats.to(device=pipeline.device, dtype=pipeline.unet.dtype)


def _set_unet_adapters(unet, adapter_names) -> None:
    if not hasattr(unet, "set_adapter"):
        return
    if isinstance(adapter_names, (list, tuple)):
        if len(adapter_names) == 1:
            unet.set_adapter(adapter_names[0])
        else:
            unet.set_adapter(list(adapter_names))
    else:
        unet.set_adapter(adapter_names)


def _get_unet_active_adapters(unet):
    adapters = getattr(unet, "active_adapters", None)
    if callable(adapters):
        adapters = adapters()
    if adapters is None:
        return None
    if isinstance(adapters, str):
        return [adapters]
    return list(adapters)


def set_validation_unet_mode(pipeline, *, branched_active: bool) -> None:
    if getattr(pipeline, "_runtime_uses_branched_unet", None) == branched_active:
        return

    if branched_active:
        if hasattr(pipeline, "_branched_attn_processors"):
            pipeline.unet.set_attn_processor(dict(pipeline._branched_attn_processors))
        adapters = getattr(pipeline, "_branched_active_adapters", None)
        if adapters:
            _set_unet_adapters(pipeline.unet, adapters)
    else:
        if hasattr(pipeline, "_original_attn_processors"):
            pipeline.unet.set_attn_processor(dict(pipeline._original_attn_processors))
        if bool(getattr(pipeline, "photomaker_use_lora_adapter", False)):
            adapters = getattr(pipeline, "_branched_active_adapters", None)
            if adapters:
                _set_unet_adapters(pipeline.unet, adapters)
            else:
                _set_unet_adapters(pipeline.unet, "default")
        else:
            _set_unet_adapters(pipeline.unet, "default")

    pipeline._runtime_uses_branched_unet = branched_active


def run_branched_setup(
    pipeline,
    *,
    use_branched_attention: bool,
    input_id_images: Sequence[Any],
    height: int,
    width: int,
    latents: torch.Tensor,
    id_pixel_values: torch.Tensor,
    auto_mask_ref: bool,
    use_bbox_mask_ref: bool,
    face_bbox_ref: Optional[List[float]],
    mask_expansion_ratio: float,
    mask_softness: float,
    import_mask_ref: Optional[str],
    debug_dir: Optional[str],
    use_dynamic_mask: bool,
    use_bbox_mask_gen: bool,
    face_bbox_gen: Optional[List[float]],
    ba_target_visibility_mask: Optional[Any] = None,
    generator: Optional[torch.Generator],
    device: torch.device,
    face_embed_strategy: str,
    batch_size: int,
    prompt_embeds: torch.Tensor,
    id_embeds: Optional[torch.Tensor],
    class_tokens_mask: torch.LongTensor,
) -> None:
    if use_branched_attention and input_id_images:
        pil = input_id_images[0] if isinstance(input_id_images, (list, tuple)) else input_id_images
        pipeline._ref_latents_all = prepare_ref_latents(
            pipeline,
            pil=pil,
            height=height,
            width=width,
            latents_dtype=latents.dtype,
        )
        prepare_ref_mask(
            pipeline,
            pil=pil,
            auto_mask_ref=auto_mask_ref,
            use_bbox_mask_ref=use_bbox_mask_ref,
            face_bbox_ref=face_bbox_ref,
            mask_expansion_ratio=mask_expansion_ratio,
            mask_softness=mask_softness,
            import_mask_ref=import_mask_ref,
            debug_dir=debug_dir,
            height=height,
            width=width,
        )

    prepare_gen_mask(
        pipeline,
        use_dynamic_mask=use_dynamic_mask,
        use_bbox_mask_gen=use_bbox_mask_gen,
        face_bbox_gen=face_bbox_gen,
        ba_target_visibility_mask=ba_target_visibility_mask,
        mask_expansion_ratio=mask_expansion_ratio,
        mask_softness=mask_softness,
        height=height,
        width=width,
        batch_size=batch_size,
    )

    if use_branched_attention and hasattr(pipeline, "_ref_latents_all"):
        pipeline._reference_latents = pipeline._ref_latents_all

    pipeline._ref_img = id_pixel_values[0] if id_pixel_values.dim() == 5 else id_pixel_values

    if use_branched_attention and hasattr(pipeline, "_ref_latents_all") and not hasattr(pipeline, "_ref_noise"):
        gen = None
        if generator is not None:
            cand = generator[0] if isinstance(generator, (list, tuple)) and len(generator) > 0 else generator
            if isinstance(cand, torch.Generator):
                if hasattr(cand, "device") and cand.device.type == device.type:
                    gen = cand
                else:
                    try:
                        gen = torch.Generator(device=device)
                        gen.set_state(cand.get_state())
                    except Exception:
                        gen = None
        pipeline._ref_noise = torch.randn(
            pipeline._ref_latents_all.shape,
            generator=gen,
            device=device,
            dtype=pipeline._ref_latents_all.dtype,
        )

    fes = (face_embed_strategy or "face").lower()
    if fes in {"faceanalysis"}:
        fes = "face"
    pipeline.face_embed_strategy = fes

    if pipeline.face_embed_strategy == "face":
        cfg_mult = 2 if pipeline.do_classifier_free_guidance else 1
        pipeline._face_prompt_embeds = encode_face_prompt(
            pipeline,
            device=device,
            batch_size=batch_size * cfg_mult,
            do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
        ).to(device)

    prepare_id_features(
        pipeline,
        id_pixel_values=id_pixel_values,
        prompt_embeds=prompt_embeds,
        id_embeds=id_embeds,
        class_tokens_mask=class_tokens_mask,
    )


def ensure_ref_latents_ready(
    pipeline,
    *,
    use_branched_attention: bool,
    id_pixel_values: Optional[torch.Tensor],
) -> None:
    if not use_branched_attention:
        return
    if hasattr(pipeline, "_ref_latents_all"):
        return

    has_id_pixels = id_pixel_values is not None
    msg = (
        "[BranchedAttention] Missing _ref_latents_all before denoising loop. "
        f"use_branched_attention={use_branched_attention}, has_id_pixel_values={has_id_pixels}. "
        "Reference latents must be prepared in the earlier branched setup block."
    )
    print(msg)
    raise RuntimeError(msg)


def select_mode_and_prompts(
    pipeline,
    *,
    i: int,
    photomaker_start_step: int,
    branched_attn_start_step: int,
    prompt_embeds_text_only: torch.Tensor,
    pooled_prompt_embeds_text_only: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    force_par_before_pm: bool,
    pose_forced_logged: bool,
    pose_relaxed_logged: bool,
) -> Tuple[str, torch.Tensor, torch.Tensor, bool, bool]:
    use_text_only = i <= photomaker_start_step
    base_prompt = prompt_embeds_text_only if use_text_only else prompt_embeds
    base_pooled = pooled_prompt_embeds_text_only if use_text_only else pooled_prompt_embeds

    desired_par = 1.0 if (force_par_before_pm and use_text_only) else pipeline._pose_user_ratio
    if getattr(pipeline, "pose_adapt_ratio", None) != desired_par:
        pipeline.pose_adapt_ratio = desired_par
        if desired_par == 1.0 and not pose_forced_logged:
            print(
                f"[PoseAdapt] Forcing POSE_ADAPT_RATIO=1.0 until "
                f"photomaker_start_step={photomaker_start_step}"
            )
            pose_forced_logged = True
        elif desired_par != 1.0 and not pose_relaxed_logged:
            print(
                f"[PoseAdapt] Relaxing POSE_ADAPT_RATIO to user value "
                f"{pipeline._pose_user_ratio:.2f} at step {i}"
            )
            pose_relaxed_logged = True

    sm = photomaker_start_step
    bs = branched_attn_start_step
    a = min(sm, bs)
    b = max(sm, bs)
    bsm = getattr(pipeline, "branched_start_mode", "both").lower()
    if i < a:
        mode = "NO_ID"
    elif sm < bs:
        mode = "PHOTOMAKER" if i < b else ("BOTH" if bsm == "both" else "BRANCHED")
    else:
        mode = ("BOTH" if bsm == "both" else "BRANCHED") if i < b else "PHOTOMAKER"

    if mode in ("PHOTOMAKER", "BOTH"):
        base_prompt = prompt_embeds
        base_pooled = pooled_prompt_embeds

    return mode, base_prompt, base_pooled, pose_forced_logged, pose_relaxed_logged


def run_branched_step(
    pipeline,
    *,
    i: int,
    t: torch.Tensor,
    mode: str,
    latent_model_input: torch.Tensor,
    current_prompt_embeds: torch.Tensor,
    added_cond_kwargs: Dict[str, Any],
    class_tokens_mask: Optional[torch.LongTensor],
    timestep_cond: Optional[torch.Tensor],
    photomaker_scale: float,
    merge_start_step: int,
    photomaker_start_step: int,
    branched_attn_start_step: int,
    debug_dir: Optional[str],
    num_outputs: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    mask4 = prepare_mask4(pipeline, latent_model_input, suffix="")
    mask4_ref = prepare_mask4(pipeline, latent_model_input, suffix="_ref")
    if mask4 is None or mask4_ref is None:
        raise RuntimeError("Branched attention requires both mask4 and mask4_ref.")

    if _val_debug_enabled(pipeline) and (i == branched_attn_start_step or i % 10 == 0):
        print(
            f"[PL] step={i}  mask_gen>0.5={(mask4 > 0.5).float().mean().item():.4f}  "
            f"mask_ref>0.5={(mask4_ref > 0.5).float().mean().item():.4f}"
        )
        md = (mask4 - mask4_ref).abs().mean().item()
        if md < 0.01:
            print(f"[Warning] Noise and ref masks are nearly identical (diff={md:.4f})")

    if _val_debug_enabled(pipeline) and debug_dir is not None:
        debug_reference_latents_once(pipeline, mask4_ref, debug_dir)
    if _val_debug_enabled(pipeline) and i == branched_attn_start_step:
        base_debug_dir = Path(debug_dir) if debug_dir is not None else None
        if base_debug_dir is not None:
            ref_masks = mask4_ref
            if ref_masks.dim() == 4 and ref_masks.shape[0] == num_outputs:
                ref_masks_iter = [ref_masks[idx:idx + 1] for idx in range(num_outputs)]
            else:
                ref_masks_iter = [ref_masks] * num_outputs
            for idx, mask_ref_single in enumerate(ref_masks_iter):
                per_image_dir = base_debug_dir if num_outputs == 1 else base_debug_dir / f"{idx:02d}"
                per_image_dir.mkdir(parents=True, exist_ok=True)
                save_debug_ref_latents(pipeline, str(per_image_dir))
                save_debug_ref_mask_overlay(pipeline, mask_ref_single, str(per_image_dir))
        else:
            save_debug_ref_latents(pipeline, debug_dir)
            save_debug_ref_mask_overlay(pipeline, mask4_ref, debug_dir)
        print(f"[Debug] Step {i}: Ref mask overlay saved.")

    fes_step = pipeline.face_embed_strategy
    if fes_step in {"id", "id_embeds"} and i < photomaker_start_step and mode != "BOTH":
        fes_step = "face"

    id_face_ehs = None
    proc_id_embeds = None
    if fes_step == "id_embeds":
        pm = getattr(pipeline, "_pm_id_embeds_2048", None)
        if pm is None:
            raise ValueError("id_embeds strategy requires cached _pm_id_embeds_2048.")

        seq_len = current_prompt_embeds.shape[1]
        dim = current_prompt_embeds.shape[2]
        b_pos = current_prompt_embeds.shape[0] // (2 if pipeline.do_classifier_free_guidance else 1)
        if pm.shape[0] == b_pos:
            pm_b = pm
        elif pm.shape[0] == 1:
            pm_b = pm.expand(b_pos, -1)
        else:
            pm_b = pm.mean(dim=0, keepdim=True).expand(b_pos, -1)

        pos = pm_b.unsqueeze(1).expand(b_pos, seq_len, dim)
        if pipeline.do_classifier_free_guidance:
            neg = torch.zeros_like(pos)
            id_face_ehs = torch.cat([neg, pos], dim=0)
            proc_id_embeds = torch.cat([torch.zeros_like(pm_b), pm_b], dim=0)
        else:
            id_face_ehs = pos
            proc_id_embeds = pm_b

        id_face_ehs = id_face_ehs.to(
            device=current_prompt_embeds.device,
            dtype=current_prompt_embeds.dtype,
        )
        proc_id_embeds = proc_id_embeds.to(
            device=current_prompt_embeds.device,
            dtype=current_prompt_embeds.dtype,
        )

    mask4_for_merge = mask4 if i >= merge_start_step else torch.zeros_like(mask4)
    mask4_ref_for_merge = mask4_ref if i >= merge_start_step else torch.zeros_like(mask4_ref)

    noise_pred, noise_face, _ = two_branch_predict(
        pipeline,
        latent_model_input,
        t=t,
        prompt_embeds=current_prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        mask4=mask4_for_merge,
        mask4_ref=mask4_ref_for_merge,
        reference_latents=pipeline._ref_latents_all,
        face_prompt_embeds=(pipeline._face_prompt_embeds if fes_step == "face" else id_face_ehs),
        class_tokens_mask=class_tokens_mask,
        face_embed_strategy=fes_step,
        id_embeds=proc_id_embeds,
        step_idx=i,
        scale=photomaker_scale,
        timestep_cond=timestep_cond,
    )

    if _val_debug_enabled(pipeline) and i < (branched_attn_start_step + 3):
        print(
            f"[Debug] Step {i}: noise_pred stats - "
            f"mean={noise_pred.mean().item():.4f}, "
            f"std={noise_pred.std().item():.4f}, "
            f"min={noise_pred.min().item():.4f}, "
            f"max={noise_pred.max().item():.4f}"
        )

    if hasattr(pipeline, "_kv_override"):
        pipeline._kv_override = None

    return noise_pred, noise_face, mask4


def save_step_previews(
    pipeline,
    *,
    i: int,
    t: torch.Tensor,
    num_inference_steps: int,
    debug_dir: Optional[str],
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    mask4: Optional[torch.Tensor],
    noise_face: Optional[torch.Tensor],
    extra_step_kwargs: Dict[str, Any],
) -> None:
    if not _val_debug_enabled(pipeline):
        return

    if not (i % 10 == 0 or i == num_inference_steps - 1):
        return

    if mask4 is not None and noise_face is not None:
        base_debug_dir = Path(debug_dir) if debug_dir is not None else None
        if base_debug_dir is not None:
            total_outputs = latents.shape[0]
            for idx, latent_sample in enumerate(latents):
                per_image_dir = base_debug_dir if total_outputs == 1 else base_debug_dir / f"{idx:02d}"
                per_image_dir.mkdir(parents=True, exist_ok=True)
                mask_slice = mask4[idx:idx + 1] if mask4.shape[0] > idx else mask4
                save_branch_previews(
                    pipeline,
                    latent_sample.unsqueeze(0),
                    noise_pred,
                    mask_slice,
                    t,
                    i,
                    str(per_image_dir),
                    extra_step_kwargs,
                )
        else:
            save_branch_previews(pipeline, latents, noise_pred, mask4, t, i, debug_dir, extra_step_kwargs)
        return

    if debug_dir is not None:
        base_debug_dir = Path(debug_dir)
        base_debug_dir.mkdir(parents=True, exist_ok=True)
        total_outputs = latents.shape[0]
        full_mask = torch.ones_like(latents[:, :1, :, :])
        for idx, latent_sample in enumerate(latents):
            per_image_dir = base_debug_dir if total_outputs == 1 else base_debug_dir / f"{idx:02d}"
            per_image_dir.mkdir(parents=True, exist_ok=True)
            mask_slice = full_mask[idx:idx + 1]
            save_branch_previews(
                pipeline,
                latent_sample.unsqueeze(0),
                noise_pred,
                mask_slice,
                t,
                i,
                str(per_image_dir),
                extra_step_kwargs,
            )


def run_denoising_step(
    pipeline,
    *,
    i: int,
    t: torch.Tensor,
    prev_mode: Optional[str],
    photomaker_start_step: int,
    branched_attn_start_step: int,
    prompt_embeds_text_only: torch.Tensor,
    pooled_prompt_embeds_text_only: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    force_par_before_pm: bool,
    pose_forced_logged: bool,
    pose_relaxed_logged: bool,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_pooled_prompt_embeds: Optional[torch.Tensor],
    add_time_ids: torch.Tensor,
    ip_adapter_image,
    ip_adapter_image_embeds,
    image_embeds: Optional[torch.Tensor],
    use_branched_attention: bool,
    latent_model_input: torch.Tensor,
    timestep_cond: Optional[torch.Tensor],
    class_tokens_mask: Optional[torch.LongTensor],
    photomaker_scale: float,
    merge_start_step: int,
    debug_dir: Optional[str],
    latents: torch.Tensor,
    extra_step_kwargs: Dict[str, Any],
    num_inference_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[str], bool, bool]:
    mode, base_prompt, base_pooled, pose_forced_logged, pose_relaxed_logged = select_mode_and_prompts(
        pipeline,
        i=i,
        photomaker_start_step=photomaker_start_step,
        branched_attn_start_step=branched_attn_start_step,
        prompt_embeds_text_only=prompt_embeds_text_only,
        pooled_prompt_embeds_text_only=pooled_prompt_embeds_text_only,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        force_par_before_pm=force_par_before_pm,
        pose_forced_logged=pose_forced_logged,
        pose_relaxed_logged=pose_relaxed_logged,
    )

    if mode != prev_mode:
        print(
            f"[Switch] step {int(i)} → {mode}  "
            f"(photomaker_start_step={int(photomaker_start_step)}, "
            f"branched_attn_start_step={int(branched_attn_start_step)})"
        )
        prev_mode = mode

    current_prompt_embeds = (
        torch.cat([negative_prompt_embeds, base_prompt], dim=0)
        if pipeline.do_classifier_free_guidance
        else base_prompt
    )
    add_text_embeds = (
        torch.cat([negative_pooled_prompt_embeds, base_pooled], dim=0)
        if pipeline.do_classifier_free_guidance
        else base_pooled
    )

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        added_cond_kwargs["image_embeds"] = image_embeds

    noise_face = None
    mask4 = None
    branched_active = use_branched_attention and (mode in ("BRANCHED", "BOTH"))
    set_validation_unet_mode(pipeline, branched_active=branched_active)
    if branched_active:
        noise_pred, noise_face, mask4 = run_branched_step(
            pipeline,
            i=i,
            t=t,
            mode=mode,
            latent_model_input=latent_model_input,
            current_prompt_embeds=current_prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            class_tokens_mask=class_tokens_mask,
            timestep_cond=timestep_cond,
            photomaker_scale=photomaker_scale,
            merge_start_step=merge_start_step,
            photomaker_start_step=photomaker_start_step,
            branched_attn_start_step=branched_attn_start_step,
            debug_dir=debug_dir,
            num_outputs=latents.shape[0],
        )
    else:
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=current_prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=pipeline.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

    save_step_previews(
        pipeline,
        i=i,
        t=t,
        num_inference_steps=num_inference_steps,
        debug_dir=debug_dir,
        latents=latents,
        noise_pred=noise_pred,
        mask4=mask4,
        noise_face=noise_face,
        extra_step_kwargs=extra_step_kwargs,
    )

    return noise_pred, add_text_embeds, prev_mode, pose_forced_logged, pose_relaxed_logged


def cleanup_branched_runtime(pipeline, *, use_branched_attention: bool) -> None:
    del use_branched_attention
    set_validation_unet_mode(pipeline, branched_active=False)
    for attr in ["_reference_latents", "_face_prompt_embeds", "_ref_latents_all", "_ref_noise"]:
        if hasattr(pipeline, attr):
            delattr(pipeline, attr)


def build_pipeline_from_pretrained(
    pipeline_cls,
    *,
    model,
    accelerator,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
):
    kwargs = dict(kwargs)
    if "torch_dtype" in kwargs:
        kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])

    unwrapped_model = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    scheduler = DDIMScheduler.from_pretrained(
        kwargs["pretrained_model_name_or_path"],
        subfolder="scheduler",
    )

    kwargs.pop("photomaker_start_step", None)
    kwargs.pop("merge_start_step", None)
    kwargs.pop("branched_attn_start_step", None)
    kwargs.pop("branched_start_mode", None)
    photomaker_use_lora_adapter_cfg = bool(
        kwargs.pop("photomaker_use_lora_adapter", False)
    )
    pose_adapt_ratio_cfg = kwargs.pop(
        "pose_adapt_ratio",
        getattr(unwrapped_model, "pose_adapt_ratio", 0.25),
    )
    ca_mixing_for_face_cfg = kwargs.pop(
        "ca_mixing_for_face",
        getattr(unwrapped_model, "ca_mixing_for_face", True),
    )
    face_embed_strategy_cfg = kwargs.pop(
        "face_embed_strategy",
        getattr(unwrapped_model, "face_embed_strategy", "face"),
    )
    use_id_embeds_cfg = kwargs.pop(
        "use_id_embeds",
        getattr(unwrapped_model, "use_id_embeds", True),
    )
    id_alpha_cfg = kwargs.pop(
        "id_alpha",
        getattr(unwrapped_model, "id_alpha", 0.3),
    )

    pipeline = pipeline_cls.from_pretrained(
        scheduler=scheduler,
        tokenizer=unwrapped_model.tokenizer,
        tokenizer_2=unwrapped_model.tokenizer_2,
        text_encoder=unwrapped_model.text_encoder,
        text_encoder_2=unwrapped_model.text_encoder_2,
        unet=unwrapped_model.unet,
        vae=unwrapped_model.vae,
        *args,
        **kwargs,
    )
    pipeline.set_progress_bar_config(disable=True)

    pipeline.num_tokens = getattr(unwrapped_model, "num_tokens", 2)
    pipeline.pm_version = "v2"
    pipeline.trigger_word = unwrapped_model.trigger_word

    pipeline.id_image_processor = CLIPImageProcessor()
    pipeline.id_encoder = unwrapped_model.id_encoder

    pipeline.pose_adapt_ratio = pose_adapt_ratio_cfg
    pipeline.ca_mixing_for_face = ca_mixing_for_face_cfg
    pipeline.face_embed_strategy = face_embed_strategy_cfg
    pipeline.use_id_embeds = bool(use_id_embeds_cfg)
    pipeline.id_alpha = float(id_alpha_cfg)
    pipeline.strict_face_routing = bool(getattr(unwrapped_model, "strict_face_routing", False))
    pipeline.branched_attn_weight_mode = getattr(unwrapped_model, "branched_attn_weight_mode", "shared")
    pipeline.branched_attn_new_weight_kind = getattr(unwrapped_model, "branched_attn_new_weight_kind", "full")
    pipeline.branched_attn_lora_rank = int(
        getattr(unwrapped_model, "branched_attn_lora_rank", getattr(unwrapped_model, "lora_rank", 16))
    )
    if hasattr(unwrapped_model, "_original_attn_processors"):
        pipeline._original_attn_processors = dict(unwrapped_model._original_attn_processors)
    if hasattr(unwrapped_model.unet, "attn_processors"):
        pipeline._branched_attn_processors = dict(unwrapped_model.unet.attn_processors)
    active_adapters = _get_unet_active_adapters(unwrapped_model.unet)
    if active_adapters:
        pipeline._branched_active_adapters = active_adapters
    pipeline.photomaker_use_lora_adapter = photomaker_use_lora_adapter_cfg
    pipeline._runtime_uses_branched_unet = None

    pipeline.tokenizer.add_tokens([pipeline.trigger_word], special_tokens=True)
    pipeline.tokenizer_2.add_tokens([pipeline.trigger_word], special_tokens=True)

    return pipeline
