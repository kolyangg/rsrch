from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .branched_runtime import (
    patch_unet_attention_processors,
    select_branched_processor_names,
    select_branched_self_attention_names,
    two_branch_predict,
)
from .packed_residual_attn_processor import make_inner_core_mask
from .insightface_package import analyze_faces

from copy import deepcopy


class InvalidBranchedSampleError(RuntimeError):
    """A data error that must reject the complete branched-attention microbatch."""

    def __init__(self, reason: str, detail: str):
        self.reason = str(reason)
        super().__init__(f"{self.reason}: {detail}")


def _reject_invalid_sample(model, reason: str, detail: str) -> None:
    policy = str(getattr(model, "ba_invalid_sample_policy", "legacy") or "legacy").lower()
    if policy == "skip_batch":
        raise InvalidBranchedSampleError(reason, detail)
    raise ValueError(f"{reason}: {detail}")


def _validated_bbox(bbox, *, image_shape, label: str) -> tuple[float, float, float, float]:
    if bbox is None:
        raise ValueError(f"{label} is missing")
    if torch.is_tensor(bbox):
        values = bbox.detach().flatten().tolist()
    else:
        try:
            values = list(bbox)
        except TypeError as exc:
            raise ValueError(f"{label} is not a sequence") from exc
    if len(values) != 4:
        raise ValueError(f"{label} has {len(values)} values; expected four")
    x0, y0, x1, y1 = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError(f"{label} contains non-finite coordinates: {values[:4]}")
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{label} is inverted or empty: {values[:4]}")
    image_h, image_w = (int(image_shape[0]), int(image_shape[1]))
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"{label} has invalid image shape {image_shape}")
    x0c, x1c = max(0.0, x0), min(float(image_w), x1)
    y0c, y1c = max(0.0, y0), min(float(image_h), y1)
    if x1c <= x0c or y1c <= y0c:
        raise ValueError(
            f"{label} is empty after clamping {values[:4]} to image {(image_h, image_w)}"
        )
    return x0, y0, x1, y1


def _processor_trainable_manifest(model) -> dict[str, dict[str, int]]:
    categories: dict[str, dict[str, int]] = {}
    for name, parameter in model.unet.named_parameters():
        if not parameter.requires_grad or ".processor." not in name:
            continue
        if ".attn1.processor.face_mix_logits" in name:
            key = "sa_face_mix"
        elif ".attn1.processor.face_residual_gain" in name:
            key = "sa_face_residual"
        elif ".attn1.processor.connector_down." in name:
            key = "sa_connector_down"
        elif ".attn1.processor.connector_up." in name:
            key = "sa_connector_up"
        elif ".attn1.processor.gate_logit" in name:
            key = "sa_gate"
        elif ".attn1.processor.null_memory" in name:
            key = "sa_null_memory"
        else:
            attention = "sa" if ".attn1.processor." in name else "ca" if ".attn2.processor." in name else "other"
            branch = "ref" if ".ref_to_" in name else "noise" if ".noise_to_" in name else "other"
            projection = "q" if "_to_q" in name else "k" if "_to_k" in name else "v" if "_to_v" in name else "other"
            key = f"{attention}_{branch}_{projection}"
        entry = categories.setdefault(key, {"tensors": 0, "parameters": 0})
        entry["tensors"] += 1
        entry["parameters"] += int(parameter.numel())
    return categories


def _assert_branched_installation(model) -> None:
    processors = model.unet.attn_processors
    all_names = list(processors)
    variant = str(getattr(model, "ba_processor_variant", "legacy") or "legacy").lower()
    if variant == "packed_residual_v1":
        expected_sa = set(
            select_branched_self_attention_names(
                all_names,
                getattr(model, "ba_site_policy", "all"),
            )
        )
    else:
        expected_sa = set(
            select_branched_processor_names(
                all_names,
                include_self_attention=True,
                include_cross_attention=False,
                top_k=float(getattr(model, "ba_patch_top_k", 1.0)),
                param_name="ba_patch_top_k",
            )
        )
    if bool(getattr(model, "disable_branched_sa", False)):
        expected_sa.clear()
    expected_ca = {
        name for name in all_names if name.endswith("attn2.processor")
    }
    if bool(getattr(model, "disable_branched_ca", False)):
        expected_ca.clear()

    actual_sa = {
        name for name, proc in processors.items()
        if name.endswith("attn1.processor")
        and bool(getattr(proc, "_is_branched_processor", False))
        and getattr(proc, "_branched_kind", None) == "self"
    }
    actual_ca = {
        name for name, proc in processors.items()
        if name.endswith("attn2.processor")
        and bool(getattr(proc, "_is_branched_processor", False))
        and getattr(proc, "_branched_kind", None) == "cross"
    }
    if actual_sa != expected_sa or actual_ca != expected_ca:
        raise RuntimeError(
            "Strict BA installation site mismatch: "
            f"missing_sa={sorted(expected_sa - actual_sa)[:5]}, "
            f"unexpected_sa={sorted(actual_sa - expected_sa)[:5]}, "
            f"missing_ca={sorted(expected_ca - actual_ca)[:5]}, "
            f"unexpected_ca={sorted(actual_ca - expected_ca)[:5]}"
        )

    patched = tuple(sorted(getattr(model, "_ba_patched_processor_names", ())))
    expected = tuple(sorted(expected_sa | expected_ca))
    if patched != expected:
        raise RuntimeError(
            "Strict BA installation processor-name mismatch: "
            f"patched={len(patched)}, expected={len(expected)}"
        )

    trainable_keys = tuple(
        sorted(name for name, parameter in model.unet.named_parameters() if parameter.requires_grad)
    )
    trainable_processor_keys = tuple(name for name in trainable_keys if ".processor." in name)
    if getattr(model, "train_ba_only", False) and not trainable_processor_keys:
        raise RuntimeError("Strict BA installation found no trainable processor parameters")
    if getattr(model, "train_ba_only", False) and any(
        ".processor." not in name for name in trainable_keys
    ):
        raise RuntimeError("train_ba_only found trainable parameters outside processors")
    if any(
        ".processor." in name
        and not any(name.startswith(f"{proc_name}.") for proc_name in expected)
        for name in trainable_keys
    ):
        raise RuntimeError("Strict BA installation found trainable parameters outside patched processors")

    train_ca = bool(getattr(model, "train_branched_ca_lora", True))
    if not train_ca and any(".attn2.processor." in name for name in trainable_processor_keys):
        raise RuntimeError("Cross-attention is configured frozen but has trainable processor parameters")

    sa_train_mode = str(getattr(model, "ba_sa_train_mode", "all") or "all").lower()
    manifest = _processor_trainable_manifest(model)
    if sa_train_mode == "packed_residual":
        allowed_fragments = [
            ".attn1.processor.ref_to_k.lora_A",
            ".attn1.processor.ref_to_k.lora_B",
            ".attn1.processor.ref_to_v.lora_A",
            ".attn1.processor.ref_to_v.lora_B",
            ".attn1.processor.connector_down.weight",
            ".attn1.processor.connector_up.weight",
            ".attn1.processor.gate_logit",
        ]
        learned_null = (
            str(
                getattr(
                    model,
                    "ba_connector_input_mode",
                    "reference_minus_target",
                )
            ).lower()
            == "reference_minus_learned_null"
        )
        if learned_null:
            allowed_fragments.append(".attn1.processor.null_memory")
        invalid = [
            name for name in trainable_processor_keys
            if not any(fragment in name for fragment in allowed_fragments)
        ]
        if invalid:
            raise RuntimeError(
                "packed_residual selected but unexpected parameters are trainable: "
                + ", ".join(invalid[:5])
            )
        required_categories = {
            "sa_ref_k",
            "sa_ref_v",
            "sa_connector_down",
            "sa_connector_up",
            "sa_gate",
        }
        expected_local = {
            "ref_to_k.lora_A",
            "ref_to_k.lora_B",
            "ref_to_v.lora_A",
            "ref_to_v.lora_B",
            "connector_down.weight",
            "connector_up.weight",
            "gate_logit",
        }
        if learned_null:
            required_categories.add("sa_null_memory")
            expected_local.add("null_memory")
        for name in sorted(expected_sa):
            local_trainable = {
                key for key, parameter in processors[name].named_parameters()
                if parameter.requires_grad
            }
            if local_trainable != expected_local:
                raise RuntimeError(
                    f"Packed residual trainability mismatch at {name}: "
                    f"missing={sorted(expected_local - local_trainable)}, "
                    f"unexpected={sorted(local_trainable - expected_local)}"
                )
    elif sa_train_mode == "ref_kv_only":
        invalid = [
            name for name in trainable_processor_keys
            if not (
                ".attn1.processor.ref_to_k." in name
                or ".attn1.processor.ref_to_v." in name
                or ".attn1.processor.face_mix_logits" in name
                or ".attn1.processor.face_residual_gain" in name
            )
        ]
        if invalid:
            raise RuntimeError(
                "ref_kv_only selected but other processor parameters are trainable: "
                + ", ".join(invalid[:5])
            )
        required_categories = {"sa_ref_k", "sa_ref_v"}
    else:
        mode = str(getattr(model, "branched_attn_weight_mode", "shared") or "shared").lower()
        required_categories = set()
        if mode in {"ref_only", "noise_and_ref"}:
            required_categories.update({"sa_ref_q", "sa_ref_k", "sa_ref_v"})
        if mode == "noise_and_ref":
            required_categories.update({"sa_noise_q", "sa_noise_k", "sa_noise_v"})
        if train_ca and mode in {"ref_only", "noise_and_ref"}:
            required_categories.update({"ca_ref_q", "ca_ref_k", "ca_ref_v"})
        if train_ca and mode == "noise_and_ref":
            required_categories.update({"ca_noise_q", "ca_noise_k", "ca_noise_v"})
    face_mode = str(getattr(model, "ba_sa_face_mode", "reference") or "reference").lower()
    if face_mode == "dual":
        required_categories.add("sa_face_mix")
    elif face_mode == "confidence_residual":
        required_categories.add("sa_face_residual")
    missing_categories = sorted(required_categories - set(manifest))
    if missing_categories:
        raise RuntimeError(
            "Strict BA installation is missing expected trainable categories: "
            + ", ".join(missing_categories)
        )

    model._ba_expected_processor_names = expected
    model._ba_expected_trainable_keys = trainable_keys
    model._ba_expected_trainable_processor_names = tuple(
        sorted(
            proc_name
            for proc_name in expected
            if any(name.startswith(f"{proc_name}.") for name in trainable_processor_keys)
        )
    )
    print(
        "[BA strict install] "
        f"variant={variant} site_policy={getattr(model, 'ba_site_policy', 'all')} "
        f"SA={len(expected_sa)} CA={len(expected_ca)} "
        f"trainable_tensors={len(trainable_processor_keys)} "
        f"trainable_parameters={sum(parameter.numel() for parameter in model.unet.parameters() if parameter.requires_grad)}"
    )
    print(
        "[BA validity] rejection counters initialized: "
        "target_bbox=0 target_core=0 reference_bbox=0 "
        "reference_recognition=0"
    )
    print(
        "[BA architecture] "
        f"variant={variant} "
        f"site_policy={getattr(model, 'ba_site_policy', 'all')} "
        f"face_mode={getattr(model, 'ba_sa_face_mode', 'reference')} "
        f"ref_tokens={getattr(model, 'ba_sa_ref_token_mode', 'full_grid')} "
        f"ref_scope={getattr(model, 'ba_sa_ref_layer_scope', 'all')} "
        f"roi_grid={getattr(model, 'ba_sa_roi_grid_size', 8)} "
        f"core_ratio={getattr(model, 'ba_sa_core_ratio', 0.7)} "
        f"connector_input={getattr(model, 'ba_connector_input_mode', 'reference_minus_target')}"
    )
    print(f"[BA strict install] processor names: {', '.join(expected)}")
    for category, counts in sorted(manifest.items()):
        print(
            f"[BA strict install] {category}: "
            f"tensors={counts['tensors']} parameters={counts['parameters']}"
        )


def configure_branched_trainables(model) -> None:
    if not getattr(model, "train_ba_only", False):
        return

    mode = (getattr(model, "branched_attn_weight_mode", "shared") or "shared").lower()
    new_weight_kind = (getattr(model, "branched_attn_new_weight_kind", "full") or "full").lower()
    train_ca = bool(getattr(model, "train_branched_ca_lora", True))
    ba_train_top_k = float(getattr(model, "ba_train_top_k", 1.0))
    non_ba_train = bool(getattr(model, "non_ba_train", False))
    sa_train_mode = str(getattr(model, "ba_sa_train_mode", "all") or "all").lower()
    if mode not in {"shared", "ref_only", "noise_and_ref"}:
        raise ValueError(f"Unknown branched_attn_weight_mode: {mode}")
    if new_weight_kind not in {"full", "lora"}:
        raise ValueError(f"Unknown branched_attn_new_weight_kind: {new_weight_kind}")
    if sa_train_mode not in {"all", "ref_kv_only", "packed_residual"}:
        raise ValueError(f"Unknown ba_sa_train_mode: {sa_train_mode}")

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
        if sa_train_mode == "packed_residual":
            is_packed_parameter = (
                ".attn1.processor.ref_to_k.lora_A" in name
                or ".attn1.processor.ref_to_k.lora_B" in name
                or ".attn1.processor.ref_to_v.lora_A" in name
                or ".attn1.processor.ref_to_v.lora_B" in name
                or ".attn1.processor.connector_down.weight" in name
                or ".attn1.processor.connector_up.weight" in name
                or ".attn1.processor.gate_logit" in name
                or ".attn1.processor.null_memory" in name
            )
            if is_selected_proc and is_packed_parameter:
                p.requires_grad_(True)
        elif mode == "shared":
            is_selected_attn = bool(selected_attn_prefixes) and name.startswith(selected_attn_prefixes)
            if is_selected_attn and ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name and ".attn1." in name:
                p.requires_grad_(True)
        else:
            is_ref_projection = ".attn1.processor.ref_to_" in name
            if sa_train_mode == "ref_kv_only":
                is_ref_projection = (
                    ".attn1.processor.ref_to_k." in name
                    or ".attn1.processor.ref_to_v." in name
                )
            if is_selected_proc and is_ref_projection and (
                new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
            ):
                p.requires_grad_(True)
            elif (
                sa_train_mode == "all"
                and is_selected_proc
                and mode == "noise_and_ref"
                and ".attn1.processor.noise_to_" in name
                and (
                    new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
                )
            ):
                p.requires_grad_(True)
            elif (
                is_selected_proc
                and (
                    ".attn1.processor.face_mix_logits" in name
                    or ".attn1.processor.face_residual_gain" in name
                )
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
        if bool(getattr(model, "ba_correctness_guards", False)):
            _assert_branched_installation(model)
    except Exception as e:
        if bool(getattr(model, "ba_correctness_guards", False)):
            raise RuntimeError("Failed to install strict branched-attention processors") from e
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
    base_prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    class_tokens_mask_list = []
    mask_list = []
    ref_mask_list = []
    ref_latents_list = []
    pm_feature_list = []

    image_h, image_w = pixel_values.shape[-2:]
    latent_h, latent_w = noisy_latents.shape[-2:]
    policy = str(getattr(model, "ba_invalid_sample_policy", "legacy") or "legacy").lower()
    batch_size = len(prompts)
    if policy != "legacy":
        if face_bbox is None or len(face_bbox) != batch_size:
            _reject_invalid_sample(
                model,
                "target_bbox",
                f"expected {batch_size} target bboxes, got {0 if face_bbox is None else len(face_bbox)}",
            )
        if face_bbox_ref is None or len(face_bbox_ref) != batch_size:
            _reject_invalid_sample(
                model,
                "reference_bbox",
                f"expected {batch_size} reference bboxes, got "
                f"{0 if face_bbox_ref is None else len(face_bbox_ref)}",
            )
        if len(ref_images) != batch_size:
            _reject_invalid_sample(
                model,
                "reference_recognition",
                f"expected {batch_size} reference-image groups, got {len(ref_images)}",
            )

    for i, (prompt, refs, bbox) in enumerate(zip(prompts, ref_images, face_bbox)):
        refs = refs if isinstance(refs, (list, tuple)) else [refs]
        if len(refs) != 1:
            raise ValueError("Training batch must contain exactly one reference image per sample")
        if face_bbox_ref is None:
            if policy != "legacy":
                _reject_invalid_sample(model, "reference_bbox", "training batch is missing reference bboxes")
            raise ValueError("Training batch is missing reference bboxes")
        ref_bbox = face_bbox_ref[i]
        ref = refs[0]
        if isinstance(ref, torch.Tensor):
            ref_h, ref_w = ref.shape[-2:]
        else:
            ref_w, ref_h = ref.size

        if policy != "legacy":
            try:
                _validated_bbox(
                    bbox,
                    image_shape=(image_h, image_w),
                    label=f"target bbox for sample {i}",
                )
            except ValueError as exc:
                _reject_invalid_sample(model, "target_bbox", str(exc))
            try:
                _validated_bbox(
                    ref_bbox,
                    image_shape=(ref_h, ref_w),
                    label=f"reference bbox for sample {i}",
                )
            except ValueError as exc:
                _reject_invalid_sample(model, "reference_bbox", str(exc))

        prompt_embeds, pooled_prompt_embeds, class_tokens_mask = model.encode_prompt_with_trigger_word(
            prompt=prompt,
            num_id_images=1,
            do_cfg=False,
        )
        if float(
            getattr(model, "ba_pm_id_attenuation_probability", 0.0)
        ) > 0.0:
            base_prompt_embeds_list.append(prompt_embeds)

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
                    try:
                        raw_embedding = faces[0]["embedding"]
                    except (KeyError, TypeError):
                        raw_embedding = getattr(faces[0], "embedding", None)
                    if raw_embedding is None:
                        if policy != "legacy":
                            _reject_invalid_sample(
                                model,
                                "reference_recognition",
                                f"InsightFace returned no recognition embedding for sample {i}",
                            )
                        embedding = torch.zeros(512, dtype=torch.float32)
                    else:
                        embedding = torch.from_numpy(raw_embedding).float()
                        if policy != "legacy" and (
                            embedding.numel() != 512
                            or not torch.isfinite(embedding).all()
                            or float(embedding.norm().item()) <= 0.0
                        ):
                            _reject_invalid_sample(
                                model,
                                "reference_recognition",
                                f"InsightFace returned an invalid embedding for sample {i}",
                            )
                else:
                    if policy != "legacy":
                        _reject_invalid_sample(
                            model,
                            "reference_recognition",
                            f"InsightFace found no reference face for sample {i}",
                        )
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

            reference_latent = model._encode_reference_latent(ref, target_shape=(latent_h, latent_w))
            try:
                ref_mask = model._bbox_to_ref_mask(
                    ref_bbox,
                    latent_shape=(latent_h, latent_w),
                    image_shape=(ref_h, ref_w),
                )
            except ValueError as exc:
                if policy != "legacy":
                    _reject_invalid_sample(model, "reference_bbox", str(exc))
                raise

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
        try:
            target_mask = model._bbox_to_mask(
                bbox,
                latent_shape=(latent_h, latent_w),
                image_shape=(image_h, image_w),
            )
        except ValueError as exc:
            if policy != "legacy":
                _reject_invalid_sample(model, "target_bbox", str(exc))
            raise
        mask_list.append(target_mask)
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=0).to(device=model.device, dtype=model.unet.dtype)
    base_prompt_embeds = (
        torch.cat(base_prompt_embeds_list, dim=0).to(
            device=model.device,
            dtype=model.unet.dtype,
        )
        if base_prompt_embeds_list
        else None
    )
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
    if bool(getattr(model, "ba_correctness_guards", False)):
        if not torch.isfinite(mask4).all() or not bool((mask4 > 0).flatten(1).any(dim=1).all()):
            _reject_invalid_sample(model, "target_bbox", "target mask is empty or non-finite after resize")
        if not torch.isfinite(mask4_ref).all() or not bool((mask4_ref > 0).flatten(1).any(dim=1).all()):
            _reject_invalid_sample(model, "reference_bbox", "reference mask is empty or non-finite after resize")
        if str(getattr(model, "ba_processor_variant", "legacy")) == "packed_residual_v1":
            target_core = make_inner_core_mask(
                mask4,
                erode_frac=float(
                    getattr(model, "ba_target_core_erode_frac", 0.10)
                ),
            )
            core_has_support = target_core.float().flatten(1).sum(dim=1) > 0
            if not bool(core_has_support.all()):
                bad_rows = (
                    (~core_has_support)
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
                _reject_invalid_sample(
                    model,
                    "target_core",
                    f"feathered target core is empty at rows {bad_rows}",
                )

    model._ref_latents_all = reference_latents
    model._face_prompt_embeds = prompt_embeds
    model.do_classifier_free_guidance = False
    for attr in ("_ref_noise", "_ref_noise_base"):
        if hasattr(model, attr):
            delattr(model, attr)

    return (
        prompt_embeds,
        base_prompt_embeds,
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

    # If validation swapped the shared UNet back to the original processors
    # (set_validation_unet_mode(branched_active=False)), re-attach the SAME
    # trained processor instances. Rebuilding via patch_unet_attention_processors
    # would create fresh clones (zero LoRA deltas) and silently detach training:
    # the optimizer would keep updating orphaned modules.
    trained_procs = getattr(model, "_branched_attn_processors_train", None)
    before_restore_ids = {
        name: id(proc)
        for name, proc in sorted((trained_procs or {}).items())
        if bool(getattr(proc, "_is_branched_processor", False))
    }
    model._ba_processor_object_ids_before_restore = before_restore_ids
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
    if bool(getattr(model, "ba_strict_processor_restore", False)):
        current = model.unet.attn_processors
        after_restore_ids = {
            name: id(proc)
            for name, proc in sorted(current.items())
            if bool(getattr(proc, "_is_branched_processor", False))
        }
        model._ba_processor_object_ids_after_restore = after_restore_ids
        missing = sorted(set(trained_procs or {}) - set(current))
        detached = sorted(
            name
            for name, proc in (trained_procs or {}).items()
            if current.get(name) is not proc
        )
        expected = set(getattr(model, "_ba_expected_processor_names", ()))
        actual = set(getattr(model, "_ba_patched_processor_names", ()))
        if missing or detached or (expected and actual != expected):
            raise RuntimeError(
                "Strict BA processor restore failed: "
                f"missing={missing[:3]}, detached={detached[:3]}, "
                f"expected_names={len(expected)}, actual_names={len(actual)}"
            )
        print(
            "[BA processor restore] "
            f"identities_preserved={before_restore_ids == after_restore_ids} "
            f"branched_processors={len(after_restore_ids)}"
        )
