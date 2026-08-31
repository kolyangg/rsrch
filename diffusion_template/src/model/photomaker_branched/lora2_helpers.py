from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .branched_runtime import patch_unet_attention_processors, select_branched_processor_names, two_branch_predict
from .insightface_package import analyze_faces
from src.face_subject_selector import LEGACY_FIRST, select_subject_face


def _subject_embedding(model, ref, declared_bbox):
    img_np = np.array(ref.convert("RGB"))[:, :, ::-1]
    faces = analyze_faces(model.face_analyzer, img_np)
    policy = str(
        getattr(model, "face_subject_selection_policy", LEGACY_FIRST)
    ).lower()
    if not faces:
        if policy != LEGACY_FIRST:
            raise RuntimeError(
                "Subject-v2 training conditioning found no face in a reference image"
            )
        return torch.zeros(512, dtype=torch.float32)
    selected, _audit = select_subject_face(
        faces,
        declared_bbox=declared_bbox,
        policy=policy,
    )
    return torch.from_numpy(selected["embedding"]).float()


def _branched_trainable_context(model) -> dict:
    mode = (getattr(model, "branched_attn_weight_mode", "shared") or "shared").lower()
    new_weight_kind = (getattr(model, "branched_attn_new_weight_kind", "full") or "full").lower()
    train_ca = bool(getattr(model, "train_branched_ca_lora", True))
    ba_train_top_k = float(getattr(model, "ba_train_top_k", 1.0))
    non_ba_train = bool(getattr(model, "non_ba_train", False))
    generic_adapter_train_scope = str(
        getattr(model, "generic_adapter_train_scope", "none") or "none"
    ).lower()
    photomaker_default_train_scope = str(
        getattr(model, "photomaker_default_train_scope", "none") or "none"
    ).lower()
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
    identity_ca_proc_names = tuple(
        name
        for name in patched_proc_names
        if bool(
            getattr(
                model.unet.attn_processors.get(name),
                "is_identity_ca_v2",
                False,
            )
            or getattr(
                model.unet.attn_processors.get(name),
                "is_residual_identity_ca_v3",
                False,
            )
            or getattr(
                model.unet.attn_processors.get(name),
                "is_intrinsic_identity_ca",
                False,
            )
        )
    )
    all_trainable_processor_names = tuple(
        dict.fromkeys((*selected_proc_names, *identity_ca_proc_names))
    )
    setattr(model, "_ba_trainable_processor_names", all_trainable_processor_names)
    selected_proc_prefixes = tuple(f"{name}." for name in selected_proc_names)
    selected_attn_prefixes = tuple(f"{name.rsplit('.processor', 1)[0]}." for name in selected_proc_names)
    patched_proc_name_set = set(patched_proc_names)
    non_ba_attn_prefixes = tuple(
        f"{name.rsplit('.processor', 1)[0]}."
        for name in model.unet.attn_processors.keys()
        if name.endswith("attn1.processor") and name not in patched_proc_name_set
    )
    return {
        "architecture_version": str(
            getattr(model, "ba_architecture_version", "hard_replace_v1")
        ).lower(),
        "mode": mode,
        "new_weight_kind": new_weight_kind,
        "train_ca": train_ca,
        "non_ba_train": non_ba_train,
        "generic_adapter_train_scope": generic_adapter_train_scope,
        "photomaker_default_train_scope": photomaker_default_train_scope,
        "selected_proc_prefixes": selected_proc_prefixes,
        "selected_attn_prefixes": selected_attn_prefixes,
        "non_ba_attn_prefixes": non_ba_attn_prefixes,
        "selected_proc_names": tuple(selected_proc_names),
        "identity_ca_proc_names": identity_ca_proc_names,
        "intrinsic_id_sidecar": bool(
            getattr(model, "ba_intrinsic_id_sidecar_enabled", False)
        ),
    }


def _is_scoped_outer_adapter_trainable(
    name: str,
    *,
    adapter_marker: str,
    scope: str,
) -> bool:
    if scope == "none":
        return False
    if scope not in {
        "effective_all",
        "cross_attention",
        "self_attention_output",
    }:
        raise ValueError(f"Unknown outer-adapter train scope: {scope!r}")
    if adapter_marker not in name or not (
        "lora_A" in name or "lora_B" in name
    ):
        return False

    is_cross_attention = ".attn2." in name
    is_self_attention_output = (
        ".attn1." in name and ".to_out.0." in name
    )
    # 4 Aug 2026 - AICODE-NOTE: hard-v1 noise_and_ref bypasses outer SA
    # Q/K/V. Historical E0 confirms their 210 LoRA-B tensors remain zero;
    # allowlist only ordinary CA and the shared post-merge SA output.
    if scope == "cross_attention":
        return is_cross_attention
    if scope == "self_attention_output":
        return is_self_attention_output
    return is_cross_attention or is_self_attention_output


def _is_expected_branched_trainable(name: str, context: dict) -> bool:
    mode = context["mode"]
    new_weight_kind = context["new_weight_kind"]
    train_ca = context["train_ca"]
    selected_proc_prefixes = context["selected_proc_prefixes"]
    selected_attn_prefixes = context["selected_attn_prefixes"]
    non_ba_attn_prefixes = context["non_ba_attn_prefixes"]
    is_non_ba_attn = bool(non_ba_attn_prefixes) and name.startswith(non_ba_attn_prefixes)

    expected = False
    if mode == "shared":
        is_selected_attn = bool(selected_attn_prefixes) and name.startswith(selected_attn_prefixes)
        expected = (
            is_selected_attn
            and ("lora_A" in name or "lora_B" in name)
            and ".lora_adapter." in name
            and ".attn1." in name
        )
    else:
        is_selected_proc = bool(selected_proc_prefixes) and name.startswith(selected_proc_prefixes)
        expected = is_selected_proc and ".attn1.processor.ref_to_" in name and (
            new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
        )
        if not expected and mode == "noise_and_ref":
            expected = is_selected_proc and ".attn1.processor.noise_to_" in name and (
                new_weight_kind == "full" or "lora_A" in name or "lora_B" in name
            )
        if not expected:
            # 3 Aug 2026 - The optional hard-v1 output capacity belongs only
            # to the explicit reference-face branch. Do not reopen generic
            # U-Net or PhotoMaker adapter ownership to train it.
            expected = (
                is_selected_proc
                and ".attn1.processor.face_to_out." in name
                and ("lora_A" in name or "lora_B" in name)
            )
        if not expected and is_selected_proc:
            hardcase_markers = (
                ".attn1.processor.roi_gate_raw",
                ".attn1.processor.memory_gate_raw",
                ".attn1.processor.ownership_scale_raw",
                ".attn1.processor.ownership_mlp.",
                ".attn1.processor.frequency_schedule_raw",
                ".attn1.processor.roi_route.gate_raw",
            )
            expected = any(marker in name for marker in hardcase_markers)
        if not expected and is_selected_proc and any(
            marker in name
            for marker in (
                ".attn1.processor.memory_to_k.",
                ".attn1.processor.memory_to_v.",
                ".attn1.processor.memory_to_out.",
            )
        ):
            expected = "lora_A" in name or "lora_B" in name

    if context["non_ba_train"] and is_non_ba_attn:
        expected = expected or (
            ("lora_A" in name or "lora_B" in name) and ".lora_adapter." in name
        )

    if train_ca:
        if mode == "shared":
            is_selected_attn = bool(selected_attn_prefixes) and name.startswith(selected_attn_prefixes)
            expected = expected or (
                is_selected_attn
                and ("lora_A" in name or "lora_B" in name)
                and ".lora_adapter." in name
                and ".attn2." in name
            )
        else:
            is_selected_proc = bool(selected_proc_prefixes) and name.startswith(selected_proc_prefixes)
            expected = expected or (
                is_selected_proc
                and ".attn2.processor.ref_to_" in name
                and (new_weight_kind == "full" or "lora_A" in name or "lora_B" in name)
            )
            if mode == "noise_and_ref":
                expected = expected or (
                    is_selected_proc
                    and ".attn2.processor.noise_to_" in name
                    and (
                        new_weight_kind == "full"
                        or "lora_A" in name
                        or "lora_B" in name
                    )
                )
    expected = expected or _is_scoped_outer_adapter_trainable(
        name,
        adapter_marker=".lora_adapter.",
        scope=context["generic_adapter_train_scope"],
    )
    expected = expected or _is_scoped_outer_adapter_trainable(
        name,
        adapter_marker=".default.",
        scope=context["photomaker_default_train_scope"],
    )
    if bool(context.get("intrinsic_id_sidecar", False)):
        expected = expected or name.startswith("ba_intrinsic_id_projector.")
    return bool(expected)


def expected_branched_trainable_names(model) -> tuple[str, ...]:
    """Return the exact U-Net allowlist implied by the active BA configuration."""
    if not getattr(model, "train_ba_only", False):
        return tuple(
            name for name, parameter in model.unet.named_parameters()
            if parameter.requires_grad
        )
    context = _branched_trainable_context(model)
    if context["architecture_version"] in {
        "residual_sa_v2",
        "anchored_mix_sa_v3",
        "query_adaptive_hard_sa_v4",
    }:
        name_by_id = {
            id(parameter): name
            for name, parameter in model.unet.named_parameters()
        }
        expected = []
        seen_ids = set()
        for processor_name in context["selected_proc_names"]:
            processor = model.unet.attn_processors[processor_name]
            declaration = getattr(processor, "named_ba_trainables", None)
            if declaration is None:
                raise RuntimeError(
                    "Versioned BA processor does not declare trainables: "
                    f"{processor_name} ({type(processor).__name__})"
                )
            for local_name, parameter, role in declaration():
                del local_name, role
                parameter_id = id(parameter)
                if parameter_id in seen_ids:
                    raise RuntimeError(
                        f"Duplicate versioned BA trainable in {processor_name}"
                    )
                seen_ids.add(parameter_id)
                global_name = name_by_id.get(parameter_id)
                if global_name is None:
                    raise RuntimeError(
                        "Versioned BA declared an unregistered parameter in "
                        f"{processor_name}"
                    )
                expected.append(global_name)
        return tuple(sorted(expected))
    if context["architecture_version"] != "hard_replace_v1":
        raise ValueError(
            "Unknown ba_architecture_version="
            f"{context['architecture_version']!r}"
        )
    expected = {
        name for name, _ in model.unet.named_parameters()
        if _is_expected_branched_trainable(name, context)
    }
    if bool(getattr(model, "ba_frequency_shared_schedule_enabled", False)):
        shared_name = "ba_frequency_shared_schedule_raw"
        if shared_name not in dict(model.unet.named_parameters()):
            raise RuntimeError("Shared frequency schedule parameter is unregistered")
        expected.add(shared_name)
    if context["identity_ca_proc_names"]:
        name_by_id = {
            id(parameter): name
            for name, parameter in model.unet.named_parameters()
        }
        seen_ids = set()
        for processor_name in context["identity_ca_proc_names"]:
            processor = model.unet.attn_processors[processor_name]
            declaration = getattr(processor, "named_ba_trainables", None)
            if declaration is None:
                raise RuntimeError(
                    "Corrected identity-CA processor does not declare trainables: "
                    f"{processor_name}"
                )
            for _, parameter, _ in declaration():
                parameter_id = id(parameter)
                if parameter_id in seen_ids:
                    raise RuntimeError(
                        f"Duplicate identity-CA trainable in {processor_name}"
                    )
                seen_ids.add(parameter_id)
                global_name = name_by_id.get(parameter_id)
                if global_name is None:
                    raise RuntimeError(
                        "Identity-CA declared an unregistered parameter in "
                        f"{processor_name}"
                    )
                expected.add(global_name)
    if context["intrinsic_id_sidecar"]:
        projector_names = {
            name for name, _ in model.unet.named_parameters()
            if name.startswith("ba_intrinsic_id_projector.")
        }
        if not projector_names:
            raise RuntimeError("Intrinsic-ID projector is not installed under the U-Net")
        expected.update(projector_names)
    return tuple(sorted(expected))


def branched_trainable_role_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    """Return processor-declared versioned BA roles with exact ownership."""
    architecture_version = str(
        getattr(model, "ba_architecture_version", "hard_replace_v1")
    ).lower()
    if architecture_version not in {
        "residual_sa_v2",
        "anchored_mix_sa_v3",
        "query_adaptive_hard_sa_v4",
    }:
        return {}
    context = _branched_trainable_context(model)
    # 2 Aug 2026 - AICODE-NOTE: Keep this allowlist outside the parameter loop.
    # Rebuilding it once per U-Net tensor made residual-v2 optimizer discovery
    # accidentally quadratic and left startup CPU-bound for several minutes.
    expected_names = set(expected_branched_trainable_names(model))
    expected_ids = {
        id(parameter)
        for name, parameter in model.unet.named_parameters()
        if name in expected_names
    }
    groups: dict[str, list[torch.nn.Parameter]] = {}
    seen_ids = set()
    for processor_name in context["selected_proc_names"]:
        processor = model.unet.attn_processors[processor_name]
        for _, parameter, role in processor.named_ba_trainables():
            parameter_id = id(parameter)
            if parameter_id not in expected_ids:
                raise RuntimeError(
                    f"Unexpected {architecture_version} optimizer parameter in {processor_name}"
                )
            if parameter_id in seen_ids:
                raise RuntimeError(
                    f"Duplicate {architecture_version} optimizer parameter in {processor_name}"
                )
            seen_ids.add(parameter_id)
            groups.setdefault(str(role), []).append(parameter)
    if seen_ids != expected_ids:
        raise RuntimeError(
            f"{architecture_version} role enumeration does not match the trainable allowlist"
        )
    return groups


def _ba_semantic_group(processor_name: str) -> str:
    if processor_name.startswith("mid_block."):
        return "mid"
    for prefix, short in (("up_blocks.", "up"), ("down_blocks.", "down")):
        if processor_name.startswith(prefix):
            block = processor_name[len(prefix) :].split(".", 1)[0]
            return f"{short}{block}"
    return "other"


def collect_branched_telemetry(model) -> dict[str, torch.Tensor]:
    """Aggregate detached versioned-BA telemetry by semantic U-Net group."""
    architecture_version = str(
        getattr(model, "ba_architecture_version", "hard_replace_v1")
    ).lower()
    identity_ca_enabled = bool(
        getattr(model, "ba_identity_ca_v2_enabled", False)
        or getattr(model, "ba_residual_identity_ca_v3_enabled", False)
        or getattr(model, "ba_intrinsic_id_sidecar_enabled", False)
    )
    hardcase_telemetry_enabled = str(
        getattr(model, "ba_hardcase_mode", "off") or "off"
    ).lower() in {"visibility_order", "temporal_frequency", "anchored_roi"}
    cl39x_arm = str(
        getattr(model, "_cl39x_manifest", {}).get("active_arm") or ""
    )
    if architecture_version not in {
        "anchored_mix_sa_v3",
        "query_adaptive_hard_sa_v4",
    } and not (
        architecture_version == "hard_replace_v1" and identity_ca_enabled
    ) and not (
        architecture_version == "hard_replace_v1" and hardcase_telemetry_enabled
    ) and not (
        architecture_version == "hard_replace_v1" and bool(cl39x_arm)
    ):
        return {}
    n9_parent_telemetry = (
        architecture_version == "hard_replace_v1"
        and identity_ca_enabled
        and str(getattr(model, "ba_intrinsic_id_confidence_source", "none"))
        == "cl39_complement_detached"
    )
    processor_names = (
        getattr(model, "_ba_identity_ca_processor_names", ())
        if architecture_version == "hard_replace_v1"
        and identity_ca_enabled
        and not n9_parent_telemetry
        else getattr(model, "_ba_patched_processor_names", ())
    )
    # 16 Aug 2026 - AICODE-NOTE: This property recursively rebuilds the full
    # Diffusers processor map; resolve it once before any per-layer collector.
    processors = model.unet.attn_processors
    grouped: dict[str, list[dict[str, torch.Tensor]]] = {}
    for processor_name in processor_names:
        processor = processors.get(processor_name)
        getter = getattr(processor, "latest_ba_telemetry", None)
        if getter is None:
            continue
        telemetry = getter()
        if telemetry:
            grouped.setdefault(_ba_semantic_group(processor_name), []).append(
                telemetry
            )
    if not grouped:
        return {}

    all_entries = [entry for entries in grouped.values() for entry in entries]
    grouped["all"] = all_entries
    aggregated: dict[str, torch.Tensor] = {}
    for group, entries in grouped.items():
        # 25 Aug 2026 - AICODE-NOTE: CL39-X extensions are intentionally active
        # only in selected U-Net groups. Aggregate each diagnostic over the
        # processors that emit it; requiring one global schema crashes the
        # first training batch without changing the model or objective.
        metric_names = set().union(*(entry.keys() for entry in entries))
        for metric_name in sorted(metric_names):
            values = [
                entry[metric_name].detach().float()
                for entry in entries
                if metric_name in entry
            ]
            aggregated[f"ba/{metric_name}/{group}"] = torch.stack(values).mean()

    if cl39x_arm == "valid_kv":
        # 31 Aug 2026 - AICODE-NOTE: CL39-X diagnostics are cadence sampled,
        # but writer.loss_names is a per-step schema. Preserve fail-closed
        # logging by emitting explicit zeros only for unsampled diagnostics.
        zero = next(iter(aggregated.values())).new_zeros(())
        valid_kv_metrics = {
            "valid_fraction", "valid_count", "all_invalid_fraction",
            "entropy_valid",
        }
        if str(
            getattr(model, "ba_valid_kv_confidence_source", "valid_only")
        ) == "legacy_masked_full":
            valid_kv_metrics.update({
                "entropy_legacy", "entropy_legacy_minus_valid",
                "confidence_legacy", "confidence_valid",
                "confidence_legacy_minus_valid",
            })
        for metric_name in valid_kv_metrics:
            aggregated.setdefault(f"ba/valid_kv/{metric_name}/all", zero)
    return aggregated


def collect_hardcase_aux_loss(model) -> torch.Tensor | None:
    """Return the live semantic-ownership supervision graph, if present."""
    if str(getattr(model, "ba_hardcase_mode", "off")).lower() not in {
        "semantic_ownership",
        "visibility_order",
    }:
        return None
    # 16 Aug 2026 - AICODE-NOTE: Diffusers rebuilds attn_processors by
    # recursively walking the U-Net. Resolve it once, never once per BA layer.
    processors = model.unet.attn_processors
    losses = []
    for processor_name in getattr(model, "_ba_patched_processor_names", ()):
        processor = processors.get(processor_name)
        getter = getattr(processor, "ownership_aux_loss", None)
        if getter is None:
            continue
        value = getter()
        if value is not None:
            losses.append(value.float())
    if not losses:
        return None
    return torch.stack(losses).mean()


def collect_frequency_surface_aux_loss(model):
    """Aggregate CL27's live top-object and visible-floor graphs."""
    processors = model.unet.attn_processors
    top_losses = []
    floor_losses = []
    applied = []
    for processor_name in getattr(model, "_ba_patched_processor_names", ()):
        processor = processors.get(processor_name)
        if not bool(getattr(processor, "frequency_surface_loss_enabled", False)):
            continue
        getter = getattr(processor, "frequency_surface_aux_loss", None)
        values = None if getter is None else getter()
        telemetry_getter = getattr(processor, "latest_ba_telemetry", None)
        telemetry = None if telemetry_getter is None else telemetry_getter()
        if telemetry is not None:
            applied.append(telemetry["frequency_surface_applied_fraction"].float())
        if values is not None:
            top, floor = values
            top_losses.append(top.float())
            floor_losses.append(floor.float())
    if not top_losses:
        return None
    return (
        torch.stack(top_losses).mean(),
        torch.stack(floor_losses).mean(),
        torch.stack(applied).mean() if applied else top_losses[0].new_tensor(0.0),
    )


def collect_frequency_schedule_anchor_loss(model) -> torch.Tensor | None:
    """Return the mean squared bounded-endpoint correction for CL28."""
    processors = model.unet.attn_processors
    losses = []
    for processor_name in getattr(model, "_ba_patched_processor_names", ()):
        processor = processors.get(processor_name)
        getter = getattr(processor, "frequency_schedule_anchor_loss", None)
        value = None if getter is None else getter()
        if value is not None:
            losses.append(value.float())
    return None if not losses else torch.stack(losses).mean()


def collect_lowband_contrastive_loss(model, *, temperature: float):
    """Compute CL29 InfoNCE from processor-local matched-query embeddings."""
    groups = tuple(getattr(model, "ba_frequency_lowband_contrastive_groups", ()))
    names = [
        name
        for name in getattr(model, "_ba_patched_processor_names", ())
        if any(name.startswith(f"{group}.") for group in groups)
    ]
    if not names:
        raise RuntimeError("Low-band contrastive groups selected zero processors")
    processors = model.unet.attn_processors
    losses = []
    positives = []
    negatives = []
    for name in names:
        processor = processors.get(name)
        getter = getattr(processor, "lowband_contrastive_embeddings", None)
        values = None if getter is None else getter()
        if values is None:
            raise RuntimeError(f"Missing low-band contrastive capture for {name}")
        anchor, positive, negative = (
            F.normalize(value.float(), dim=-1) for value in values
        )
        positive_cosine = (anchor * positive).sum(dim=-1)
        negative_cosine = (anchor * negative).sum(dim=-1)
        logits = torch.stack([positive_cosine, negative_cosine], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits / float(temperature), labels))
        positives.append(positive_cosine.detach().mean())
        negatives.append(negative_cosine.detach().mean())
    positive = torch.stack(positives).mean()
    negative = torch.stack(negatives).mean()
    return torch.stack(losses).mean(), {
        "ba/lowband_positive_cosine/all": positive,
        "ba/lowband_wrong_cosine/all": negative,
        "ba/lowband_correct_wrong_margin/all": positive - negative,
    }


def collect_lowband_positive_loss(model):
    """Cosine pull between two same-ID messages under the same detached query."""
    groups = tuple(getattr(model, "ba_frequency_positive_sameid_groups", ()))
    processors = model.unet.attn_processors
    losses, similarities = [], []
    for name in getattr(model, "_ba_patched_processor_names", ()):
        if not any(name.startswith(f"{group}.") for group in groups):
            continue
        getter = getattr(processors.get(name), "lowband_positive_embeddings", None)
        values = None if getter is None else getter()
        if values is None:
            raise RuntimeError(f"Missing positive same-ID capture for {name}")
        anchor = F.normalize(values[0].detach().float(), dim=-1)
        positive = F.normalize(values[1].float(), dim=-1)
        cosine = (anchor * positive).sum(dim=-1).mean()
        losses.append(1.0 - cosine)
        similarities.append(cosine.detach())
    if not losses:
        raise RuntimeError("Positive same-ID groups selected zero processors")
    return torch.stack(losses).mean(), torch.stack(similarities).mean()


def collect_attention_ownership_loss(model):
    """Aggregate sampled chunked Q/K ownership objectives from selected layers."""
    groups = tuple(getattr(model, "ba_attention_ownership_groups", ()))
    processors = model.unet.attn_processors
    losses, visible, top = [], [], []
    for name in getattr(model, "_ba_patched_processor_names", ()):
        if not any(name.startswith(f"{group}.") for group in groups):
            continue
        getter = getattr(processors.get(name), "attention_ownership_aux", None)
        values = None if getter is None else getter()
        if values is None:
            raise RuntimeError(f"Missing attention ownership capture for {name}")
        loss, visible_mass, top_mass = values
        losses.append(loss.float())
        visible.append(visible_mass.detach().float())
        top.append(top_mass.detach().float())
    if not losses:
        raise RuntimeError("Attention ownership groups selected zero processors")
    return (
        torch.stack(losses).mean(),
        torch.stack(visible).mean(),
        torch.stack(top).mean(),
    )


def collect_roi_teacher_loss(model):
    """Aggregate CL37 training-only high-resolution ROI teacher graphs."""
    groups = tuple(getattr(model, "ba_roi_teacher_distill_groups", ()))
    processors = model.unet.attn_processors
    losses, cosine, eligible = [], [], []
    for name in getattr(model, "_ba_patched_processor_names", ()):
        if not any(name.startswith(f"{group}.") for group in groups):
            continue
        getter = getattr(processors.get(name), "roi_teacher_aux", None)
        values = None if getter is None else getter()
        if values is None:
            raise RuntimeError(f"Missing ROI teacher capture for {name}")
        loss, similarity, fraction = values
        losses.append(loss.float())
        cosine.append(similarity.detach().float())
        eligible.append(fraction.detach().float())
    if not losses:
        raise RuntimeError("ROI teacher groups selected zero processors")
    return (
        torch.stack(losses).mean(),
        torch.stack(cosine).mean(),
        torch.stack(eligible).mean(),
    )


def clear_lowband_contrastive_state(model) -> None:
    processors = model.unet.attn_processors
    for processor_name in getattr(model, "_ba_patched_processor_names", ()):
        processor = processors.get(processor_name)
        if bool(getattr(processor, "frequency_lowband_contrastive_enabled", False)):
            processor.set_lowband_contrastive("off")


def configure_branched_trainables(model) -> None:
    if not getattr(model, "train_ba_only", False):
        return

    expected = set(expected_branched_trainable_names(model))
    if not expected:
        raise RuntimeError("BA-only configuration resolved to zero trainable parameters")
    for name, parameter in model.unet.named_parameters():
        parameter.requires_grad_(name in expected)
    setattr(model, "_ba_expected_trainable_names", tuple(sorted(expected)))


def assert_branched_trainable_contract(model, optimizer=None) -> dict:
    """Fail if requires-grad or optimizer membership differs from the BA allowlist."""
    if not getattr(model, "train_ba_only", False):
        return {}

    expected = set(expected_branched_trainable_names(model))
    actual = {
        name for name, parameter in model.unet.named_parameters()
        if parameter.requires_grad
    }
    non_unet = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("unet.")
    }
    if actual != expected or non_unet:
        raise RuntimeError(
            "BA-only trainable contract mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}, "
            f"non_unet={sorted(non_unet)}"
        )

    if optimizer is not None:
        named_parameters = dict(model.unet.named_parameters())
        name_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
        expected_ids = {id(named_parameters[name]) for name in expected}
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group.get("params", ())
        }
        missing_ids = expected_ids - optimizer_ids
        unexpected_ids = optimizer_ids - expected_ids
        if missing_ids or unexpected_ids:
            raise RuntimeError(
                "BA-only optimizer contract mismatch: "
                f"missing={sorted(name_by_id[parameter_id] for parameter_id in missing_ids)}, "
                "unexpected="
                f"{sorted(name_by_id.get(parameter_id, f'<unknown:{parameter_id}>') for parameter_id in unexpected_ids)}"
            )

    numel = sum(
        int(parameter.numel())
        for name, parameter in model.unet.named_parameters()
        if name in expected
    )
    return {"tensor_count": len(expected), "parameter_count": numel}


def _install_branched_processors(model, *, legacy_processor_enable: bool) -> None:
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

    if legacy_processor_enable and hasattr(model.unet, "attn_processors"):
        for proc in model.unet.attn_processors.values():
            for parameter in proc.parameters():
                parameter.requires_grad_(True)

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


def install_branched_processors_for_training(model) -> None:
    """Install branched processors before optimizer creation."""
    fail_closed = bool(getattr(model, "strict_branched_install", False)) or bool(
        getattr(model, "strict_trainable_contract", False)
    )

    if fail_closed:
        # 1 Aug 2026 - AICODE-NOTE: Strict BA installation deliberately skips
        # the historical processor-wide enable loop. CA-off maps contain plain
        # AttnProcessor2_0 objects, so that loop raised before the BA-only freeze.
        try:
            _install_branched_processors(model, legacy_processor_enable=False)
            if bool(getattr(model, "strict_trainable_contract", False)):
                assert_branched_trainable_contract(model)
        except Exception as exc:
            raise RuntimeError(
                "Branched processor installation/trainable ownership failed"
            ) from exc
        return

    # Historical behavior remains available for exact replay. It intentionally
    # retains the old warning-and-continue failure semantics.
    try:
        _install_branched_processors(model, legacy_processor_enable=True)
    except Exception as e:
        print(f"[PhotomakerBranchedLora] exception while installing branched processors: {e}")


def prepare_branched_training_inputs(
    model,
    *,
    prompts: Sequence[str],
    ref_images: Sequence[Sequence],
    face_bbox: Sequence[Sequence[float]],
    face_bbox_ref: Sequence[Sequence[float]] | None = None,
    reference_cache_keys: Sequence[str] | None = None,
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
    raw_id_embed_list = []

    image_h, image_w = pixel_values.shape[-2:]
    latent_h, latent_w = noisy_latents.shape[-2:]
    cache_enabled = bool(getattr(model, "conditioning_cache_enabled", False))
    cache_limit = max(0, int(getattr(model, "conditioning_cache_max_entries", 0)))
    cache = getattr(model, "_conditioning_cache", None)
    if cache_enabled and cache_limit > 0 and cache is None:
        cache = OrderedDict()
        model._conditioning_cache = cache

    def reference_key(sample_index: int, refs) -> tuple[str, ...] | None:
        # Cache only when the dataset explicitly identifies the transformed
        # reference content. A path alone is unsafe for augmented references.
        if not cache_enabled or cache_limit <= 0 or reference_cache_keys is None:
            return None
        if isinstance(reference_cache_keys, str):
            value = reference_cache_keys if sample_index == 0 else None
        elif sample_index < len(reference_cache_keys):
            value = reference_cache_keys[sample_index]
        else:
            value = None
        if value is None:
            return None
        values = value if isinstance(value, (list, tuple)) else [value]
        if len(values) != len(refs):
            return None
        return tuple(str(path) for path in values)

    for i, (prompt, refs, bbox) in enumerate(zip(prompts, ref_images, face_bbox)):
        refs = refs if isinstance(refs, (list, tuple)) else [refs]
        ref0 = refs[0]
        ref_bbox = None if face_bbox_ref is None else face_bbox_ref[i]
        if ref_bbox is None:
            raise ValueError("Training batch is missing face_bbox_ref for reference masking")

        if isinstance(ref0, torch.Tensor):
            ref_h, ref_w = ref0.shape[-2:]
        else:
            ref_w, ref_h = ref0.size

        paths = reference_key(i, refs)
        cache_key = None
        cached = None
        if paths is not None:
            cache_key = (
                str(prompt),
                paths,
                tuple(float(v) for v in bbox),
                tuple(float(v) for v in ref_bbox),
                (int(image_h), int(image_w)),
                (int(ref_h), int(ref_w)),
                (int(latent_h), int(latent_w)),
                str(model.face_embed_strategy),
                str(model.unet.dtype),
            )
            cached = cache.get(cache_key)
            if cached is not None:
                cache.move_to_end(cache_key)
                model._conditioning_cache_hits = (
                    int(getattr(model, "_conditioning_cache_hits", 0)) + 1
                )

        if cached is None:
            prompt_embeds, pooled_prompt_embeds, class_tokens_mask = (
                model.encode_prompt_with_trigger_word(
                    prompt=prompt,
                    num_id_images=len(refs),
                    do_cfg=False,
                )
            )

            with torch.no_grad():
                id_pixel_values = model.id_image_processor(
                    refs, return_tensors="pt"
                ).pixel_values.unsqueeze(0)
                id_pixel_values = id_pixel_values.to(
                    model.device, dtype=model.id_encoder.dtype
                )

                prompt_for_id = prompt_embeds.to(dtype=model.id_encoder.dtype)
                id_embed_list = []
                for ref_index, ref in enumerate(refs):
                    id_embed_list.append(
                        _subject_embedding(
                            model,
                            ref,
                            ref_bbox if ref_index == 0 else None,
                        )
                    )

                id_embeds = torch.stack(id_embed_list, dim=0).unsqueeze(0)
                id_embeds = id_embeds.to(
                    device=model.device, dtype=model.id_encoder.dtype
                )
                raw_id_embed = id_embeds[:, 0]

                prompt_embeds = model.id_encoder(
                    id_pixel_values,
                    prompt_for_id,
                    class_tokens_mask,
                    id_embeds,
                )
                reference_latent = model._encode_reference_latent(
                    ref0, target_shape=(latent_h, latent_w)
                )

                pm_features = None
                if model.face_embed_strategy == "id_embeds":
                    pm_features = model.id_encoder.extract_id_features(
                        id_pixel_values,
                        id_embeds=id_embeds,
                        class_tokens_mask=class_tokens_mask,
                    ).to(device=model.device, dtype=model.unet.dtype)

            ref_mask = model._bbox_to_ref_mask(
                ref_bbox,
                latent_shape=(latent_h, latent_w),
                image_shape=(ref_h, ref_w),
            )
            target_mask = model._bbox_to_mask(
                bbox,
                latent_shape=(latent_h, latent_w),
                image_shape=(image_h, image_w),
            )

            if cache_key is not None:
                cached = (
                    prompt_embeds.detach(),
                    pooled_prompt_embeds.detach(),
                    class_tokens_mask.detach(),
                    reference_latent.detach(),
                    None if pm_features is None else pm_features.detach(),
                    id_embeds[:, 0].detach(),
                    target_mask.detach(),
                    ref_mask.detach(),
                )
                cache[cache_key] = cached
                cache.move_to_end(cache_key)
                while len(cache) > cache_limit:
                    cache.popitem(last=False)
                model._conditioning_cache_misses = (
                    int(getattr(model, "_conditioning_cache_misses", 0)) + 1
                )
        else:
            (
                prompt_embeds,
                pooled_prompt_embeds,
                class_tokens_mask,
                reference_latent,
                pm_features,
                raw_id_embed,
                target_mask,
                ref_mask,
            ) = cached

        class_tokens_mask_list.append(class_tokens_mask)
        ref_latents_list.append(reference_latent)
        ref_mask_list.append(ref_mask)
        mask_list.append(target_mask)
        if pm_features is not None:
            pm_feature_list.append(pm_features)
        raw_id_embed_list.append(raw_id_embed)
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
    raw_id_embeds = torch.cat(raw_id_embed_list, dim=0).to(
        device=model.device, dtype=torch.float32
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
        raw_id_embeds,
    )


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
    if not refs_per_sample or any(not refs for refs in refs_per_sample):
        raise ValueError("Batched conditioning requires at least one reference")
    num_id_images = len(refs_per_sample[0])
    if any(len(refs) != num_id_images for refs in refs_per_sample):
        raise ValueError(
            "Batched conditioning requires a uniform reference count per sample"
        )
    # 17 Aug 2026 - AICODE-NOTE: Additional identity references are batched
    # only through PhotoMaker/identity losses. refs[0] remains the sole spatial
    # latent and reference K/V lane, preserving the CL5 dataset invariant.
    spatial_refs = [refs[0] for refs in refs_per_sample]
    flat_identity_refs = [ref for refs in refs_per_sample for ref in refs]
    batch_size = len(spatial_refs)
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
        model.encode_prompts_with_trigger_word(
            prompts, num_id_images=num_id_images
        )
    )

    # 26 Jul 2026 - Full Cosmic supplies effectively unique target/reference
    # pairs. Batch all frozen encoders so throughput does not depend on cache
    # reuse; legacy per-sample preparation remains the default.
    # AICODE-NOTE: Batching changes only execution grouping. References,
    # supplied boxes, PhotoMaker features, and target masks remain per sample.
    with torch.no_grad():
        id_pixel_values = model.id_image_processor(
            flat_identity_refs, return_tensors="pt"
        ).pixel_values
        id_pixel_values = id_pixel_values.reshape(
            batch_size, num_id_images, *id_pixel_values.shape[1:]
        )
        id_pixel_values = id_pixel_values.to(
            model.device, dtype=model.id_encoder.dtype
        )

        id_embed_list = []
        for refs, ref_bbox in zip(refs_per_sample, face_bbox_ref):
            id_embed_list.append(
                torch.stack(
                    [
                        _subject_embedding(
                            model,
                            ref,
                            ref_bbox if ref_index == 0 else None,
                        )
                        for ref_index, ref in enumerate(refs)
                    ],
                    dim=0,
                )
            )
        id_embeds = torch.stack(id_embed_list, dim=0).to(
            device=model.device, dtype=model.id_encoder.dtype
        )

        prompt_embeds = model.id_encoder(
            id_pixel_values,
            prompt_embeds.to(dtype=model.id_encoder.dtype),
            class_tokens_mask,
            id_embeds,
        )
        reference_latents = model._encode_reference_latents(
            spatial_refs, target_shape=(latent_h, latent_w)
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
    for bbox, ref_bbox, ref in zip(face_bbox, face_bbox_ref, spatial_refs):
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
        id_embeds[:, 0].to(device=model.device, dtype=torch.float32),
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
