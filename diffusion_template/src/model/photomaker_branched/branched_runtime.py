"""
branched_new.py - Simplified branched attention implementation with cross-attention
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Sequence
import os
from PIL import Image

from .debug_helpers import save_debug_images


def select_branched_processor_names(
    attn_processor_names: Sequence[str],
    *,
    include_self_attention: bool,
    include_cross_attention: bool,
    top_k: float,
    param_name: str,
) -> list[str]:
    top_k = float(top_k)
    if not 0.0 <= top_k <= 1.0:
        raise ValueError(f"{param_name} must be in [0.0, 1.0], got {top_k}")

    candidate_names: list[str] = []
    for name in attn_processor_names:
        if include_self_attention and name.endswith("attn1.processor"):
            candidate_names.append(name)
        elif include_cross_attention and name.endswith("attn2.processor"):
            candidate_names.append(name)

    if not candidate_names or top_k >= 1.0:
        return candidate_names
    if top_k <= 0.0:
        return []

    keep_count = max(1, math.ceil(len(candidate_names) * top_k))
    return candidate_names[:keep_count]


def patch_unet_attention_processors(
    pipeline,
    mask: torch.Tensor,
    mask_ref: torch.Tensor,
    scale: float = 1.0,
    id_embeds: Optional[torch.Tensor] = None,
    class_tokens_mask: Optional[torch.Tensor] = None,
    ba_denoise_progress: Optional[torch.Tensor] = None,
)-> None:
    """
    Patch UNet with branched attention processors for both self and cross attention.
    """
    disable_sa = bool(getattr(pipeline, "disable_branched_sa", False))
    disable_ca = bool(getattr(pipeline, "disable_branched_ca", False))

    # Historical configs keep the current hard replacement implementation.
    # The old use_attn_v2 flag never selected a different active processor and
    # is retained only for replay compatibility.
    configured_architecture_version = getattr(
        pipeline, "ba_architecture_version", None
    )
    from .attn_processor_cleanest import (
        BranchedAttnProcessor as HardReplaceBranchedAttnProcessor,
        BranchedCrossAttnProcessor,
    )
    from .anchored_mix_sa_processor_v3 import (
        AnchoredMixBranchedSelfAttnProcessorV3,
    )
    from .query_adaptive_hard_sa_processor_v4 import (
        QueryAdaptiveHardBranchedSelfAttnProcessorV4,
    )
    from .residual_sa_processor_v2 import ResidualBranchedSelfAttnProcessorV2
    from .identity_ca_processor_v2 import HardIdentityCrossAttnProcessorV2
    from .residual_identity_ca_processor_v3 import (
        ResidualIdentityCrossAttnProcessorV3,
    )

    identity_ca_v2_enabled = bool(
        getattr(pipeline, "ba_identity_ca_v2_enabled", False)
    )
    residual_identity_ca_v3_enabled = bool(
        getattr(pipeline, "ba_residual_identity_ca_v3_enabled", False)
    )
    if identity_ca_v2_enabled and residual_identity_ca_v3_enabled:
        raise RuntimeError("Hard and residual identity CA cannot both be enabled")

    hardcase_mode = str(getattr(pipeline, "ba_hardcase_mode", "off") or "off").lower()
    hardcase_groups = tuple(
        str(group) for group in (getattr(pipeline, "ba_hardcase_groups", None) or ())
    )

    if configured_architecture_version is None:
        # Validation pipelines reuse the already-installed training U-Net but
        # historically copy only a subset of model attributes. Infer the exact
        # processor version; an unpatched pipeline remains legacy.
        if any(
            type(proc) is QueryAdaptiveHardBranchedSelfAttnProcessorV4
            for proc in pipeline.unet.attn_processors.values()
        ):
            architecture_version = "query_adaptive_hard_sa_v4"
        elif any(
            type(proc) is AnchoredMixBranchedSelfAttnProcessorV3
            for proc in pipeline.unet.attn_processors.values()
        ):
            architecture_version = "anchored_mix_sa_v3"
        elif any(
            type(proc) is ResidualBranchedSelfAttnProcessorV2
            for proc in pipeline.unet.attn_processors.values()
        ):
            architecture_version = "residual_sa_v2"
        else:
            architecture_version = "hard_replace_v1"
    else:
        architecture_version = str(configured_architecture_version).lower()

    if hardcase_mode != "off":
        if architecture_version != "hard_replace_v1":
            raise RuntimeError("CL15+ hard-case routes require hard_replace_v1")
        if not hardcase_groups:
            raise RuntimeError("ba_hardcase_mode requires non-empty ba_hardcase_groups")
        invalid_hardcase_groups = [
            group
            for group in hardcase_groups
            if not (
                group == "mid_block"
                or group.startswith("down_blocks.")
                or group.startswith("up_blocks.")
            )
        ]
        if invalid_hardcase_groups:
            raise ValueError(
                f"Invalid ba_hardcase_groups={invalid_hardcase_groups!r}"
            )

    if bool(getattr(pipeline, "ba_enforce_reference_only_hard_route", False)):
        if architecture_version != "hard_replace_v1":
            raise RuntimeError(
                "The audited Large Dataset suite requires hard_replace_v1"
            )
        if not disable_ca:
            raise RuntimeError(
                "The audited Large Dataset suite requires disable_branched_ca=true"
            )
        if float(getattr(pipeline, "pose_adapt_ratio", 0.0)) != 0.0:
            raise RuntimeError(
                "The audited Large Dataset suite requires pose_adapt_ratio=0"
            )
        if bool(getattr(pipeline, "ca_mixing_for_face", False)):
            raise RuntimeError(
                "The audited Large Dataset suite requires ca_mixing_for_face=false"
            )
        if str(
            getattr(pipeline, "ba_face_fusion_mode", "hard_reference_replace")
        ).lower() != "hard_reference_replace":
            raise RuntimeError(
                "The audited Large Dataset suite forbids native/reference face mixing"
            )
        if (identity_ca_v2_enabled or residual_identity_ca_v3_enabled) and bool(
            getattr(pipeline, "train_branched_ca_lora", False)
        ):
            raise RuntimeError(
                "Corrected identity CA cannot enable the legacy branched CA trainables"
            )

    if architecture_version == "hard_replace_v1":
        BranchedAttnProcessor = HardReplaceBranchedAttnProcessor
    elif architecture_version == "residual_sa_v2":
        BranchedAttnProcessor = ResidualBranchedSelfAttnProcessorV2
    elif architecture_version == "anchored_mix_sa_v3":
        BranchedAttnProcessor = AnchoredMixBranchedSelfAttnProcessorV3
    elif architecture_version == "query_adaptive_hard_sa_v4":
        BranchedAttnProcessor = QueryAdaptiveHardBranchedSelfAttnProcessorV4
    else:
        raise ValueError(
            f"Unknown ba_architecture_version={architecture_version!r}"
        )

    if architecture_version in {
        "residual_sa_v2",
        "anchored_mix_sa_v3",
        "query_adaptive_hard_sa_v4",
    }:
        reusing_installed_version = any(
            type(proc) is BranchedAttnProcessor
            for proc in pipeline.unet.attn_processors.values()
        )
        if not disable_ca and not reusing_installed_version:
            raise RuntimeError(
                f"{architecture_version} requires disable_branched_ca=true"
            )
        if reusing_installed_version and any(
            isinstance(proc, BranchedCrossAttnProcessor)
            for proc in pipeline.unet.attn_processors.values()
        ):
            raise RuntimeError(
                f"{architecture_version} cannot reuse a U-Net with branched CA processors"
            )
        if float(getattr(pipeline, "pose_adapt_ratio", 0.0)) != 0.0:
            raise RuntimeError(f"{architecture_version} requires pose_adapt_ratio=0")
        if bool(getattr(pipeline, "ca_mixing_for_face", False)):
            raise RuntimeError(
                f"{architecture_version} requires ca_mixing_for_face=false"
            )
    known_branched_types = (
        HardReplaceBranchedAttnProcessor,
        ResidualBranchedSelfAttnProcessorV2,
        AnchoredMixBranchedSelfAttnProcessorV3,
        QueryAdaptiveHardBranchedSelfAttnProcessorV4,
        BranchedCrossAttnProcessor,
        HardIdentityCrossAttnProcessorV2,
        ResidualIdentityCrossAttnProcessorV3,
    )

    # print(f'[TEMP DEBUG] mask in patch_unet_attention_processors: {mask}')
    
    # Store original processors once
    if not hasattr(pipeline, '_original_attn_processors'):
        pipeline._original_attn_processors = {}
        for name, proc in pipeline.unet.attn_processors.items():
            pipeline._original_attn_processors[name] = proc
    
    # Check if already patched
    current_procs = pipeline.unet.attn_processors
    has_branched = any(isinstance(p, known_branched_types) for p in current_procs.values())
    incompatible_self_processors = [
        name
        for name, proc in current_procs.items()
        if name.endswith("attn1.processor")
        and isinstance(proc, known_branched_types)
        and type(proc) is not BranchedAttnProcessor
    ]
    if incompatible_self_processors:
        raise RuntimeError(
            "Installed branched processor architecture does not match "
            f"{architecture_version}: {incompatible_self_processors[:5]}"
        )
    if has_branched and architecture_version == "hard_replace_v1":
        mismatched_hardcase_routes = []
        for name, processor in current_procs.items():
            if type(processor) is not HardReplaceBranchedAttnProcessor:
                continue
            expected_mode = (
                hardcase_mode
                if any(name.startswith(f"{group}.") for group in hardcase_groups)
                else "off"
            )
            if processor.hardcase_mode != expected_mode:
                mismatched_hardcase_routes.append(
                    (name, processor.hardcase_mode, expected_mode)
                )
        if mismatched_hardcase_routes:
            # 11 Aug 2026 - AICODE-NOTE: processor reuse must never turn a YAML
            # toggle into a silent no-op; validation must use the trained route.
            raise RuntimeError(
                "Installed hard-case processor map does not match configuration: "
                f"{mismatched_hardcase_routes[:5]}"
            )

    def _resolve_attn_module(unet, proc_name):
        mod = unet
        for part in proc_name.rsplit(".processor", 1)[0].split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        return mod


    def _apply_runtime_flags(proc, pipe):
        # 26 Jul 2026 - Refresh the face K/V blend on every patch call so
        # training and validation honor the same pipeline setting. The default
        # remains 0.0, which is byte-for-byte the historical reference-only mix.
        pose_adapt_ratio = float(getattr(pipe, "pose_adapt_ratio", 0.0))
        if not 0.0 <= pose_adapt_ratio <= 1.0:
            raise ValueError(
                f"pose_adapt_ratio must be in [0, 1], got {pose_adapt_ratio}"
            )
        setattr(proc, "pose_adapt_ratio", pose_adapt_ratio)

        # Optional toggle for per-branch BA-specific adapters.
        if hasattr(pipe, "ba_weights_split"):
            setattr(proc, "ba_weights_split", getattr(pipe, "ba_weights_split"))
        if hasattr(pipe, "force_binary_masks"):
            setattr(proc, "force_binary_masks", bool(getattr(pipe, "force_binary_masks")))
        if isinstance(proc, HardReplaceBranchedAttnProcessor):
            setattr(
                proc,
                "true_reference_key_mask",
                bool(getattr(pipe, "ba_hard_v1_true_reference_key_mask", False)),
            )
            setattr(
                proc,
                "reference_roi_warp",
                bool(getattr(pipe, "ba_hard_v1_reference_roi_warp", False)),
            )
        # Explicitly reset to False on validation pipelines, which do not
        # opt in even when they reuse processors from the training U-Net.
        setattr(
            proc,
            "cache_prepared_masks",
            bool(getattr(pipe, "cache_prepared_masks", False)),
        )
        if hasattr(proc, "set_denoise_progress"):
            proc.set_denoise_progress(ba_denoise_progress)
        if hasattr(proc, "set_ownership_target_mask"):
            proc.set_ownership_target_mask(
                getattr(pipe, "_ba_ownership_target_mask", None)
            )
        if hasattr(proc, "set_mix_override"):
            proc.set_mix_override(getattr(pipe, "ba_mix_override", None))
        if hasattr(proc, "set_telemetry_enabled"):
            telemetry_enabled = bool(
                getattr(pipe, "ba_telemetry_enabled", False)
            ) and not bool(getattr(pipe, "_ba_suppress_telemetry", False))
            proc.set_telemetry_enabled(telemetry_enabled)
            
        

   
    # Build safe, consistent context (batch, id_embeds)
    # Ensure masks are non-None to avoid runtime errors
    B = (mask.shape[0] if mask is not None else mask_ref.shape[0])
    dev, dt = pipeline.device, pipeline.unet.dtype
    _mask  = mask     if mask     is not None else torch.zeros(B, 1,  mask_ref.shape[-2], mask_ref.shape[-1], device=dev, dtype=dt)
    _mref  = mask_ref if mask_ref is not None else _mask
    # Always provide id_embeds so processor-local weights participate on every rank
    _idem = id_embeds.to(dev, dt) if id_embeds is not None else torch.zeros(B, 2048, device=dev, dtype=dt)   

    ba_patch_top_k = float(getattr(pipeline, "ba_patch_top_k", 1.0))
    patchable_sa_names = select_branched_processor_names(
        list(pipeline.unet.attn_processors.keys()),
        include_self_attention=True,
        include_cross_attention=False,
        top_k=ba_patch_top_k,
        param_name="ba_patch_top_k",
    )
    semantic_groups = getattr(pipeline, "ba_self_attention_groups", None)
    if semantic_groups:
        semantic_groups = tuple(str(group) for group in semantic_groups)
        invalid_groups = [
            group
            for group in semantic_groups
            if not (
                group == "mid_block"
                or group.startswith("down_blocks.")
                or group.startswith("up_blocks.")
            )
        ]
        if invalid_groups:
            raise ValueError(
                f"Unknown ba_self_attention_groups={invalid_groups!r}"
            )
        patchable_sa_names = [
            name
            for name in patchable_sa_names
            if any(
                name.startswith(f"{group}.")
                for group in semantic_groups
            )
        ]
        if not patchable_sa_names:
            raise RuntimeError(
                "ba_self_attention_groups selected zero self-attention processors"
            )
    patchable_sa_name_set = set(patchable_sa_names)
    setattr(pipeline, "_ba_semantic_processor_names", tuple(patchable_sa_names))

    identity_ca_names: list[str] = []
    identity_ca_enabled = identity_ca_v2_enabled or residual_identity_ca_v3_enabled
    if identity_ca_enabled:
        if architecture_version != "hard_replace_v1":
            raise RuntimeError("Corrected identity CA requires hard_replace_v1")
        if not disable_ca:
            raise RuntimeError(
                "Corrected identity CA requires the legacy branched CA path disabled"
            )
        if float(getattr(pipeline, "pose_adapt_ratio", 0.0)) != 0.0:
            raise RuntimeError("Corrected identity CA requires pose_adapt_ratio=0")
        if bool(getattr(pipeline, "ca_mixing_for_face", False)):
            raise RuntimeError(
                "Corrected identity CA forbids PhotoMaker/native face-output mixing"
            )
        group_attribute = (
            "ba_identity_ca_v2_groups"
            if identity_ca_v2_enabled
            else "ba_residual_identity_ca_v3_groups"
        )
        identity_groups = tuple(
            str(group)
            for group in (
                getattr(pipeline, group_attribute, None) or ()
            )
        )
        invalid_identity_groups = [
            group
            for group in identity_groups
            if not (
                group == "mid_block"
                or group.startswith("down_blocks.")
                or group.startswith("up_blocks.")
            )
        ]
        if invalid_identity_groups or not identity_groups:
            raise ValueError(
                f"Invalid {group_attribute}="
                f"{invalid_identity_groups or identity_groups!r}"
            )
        identity_ca_names = [
            name
            for name in pipeline.unet.attn_processors.keys()
            if name.endswith("attn2.processor")
            and any(name.startswith(f"{group}.") for group in identity_groups)
        ]
        if not identity_ca_names:
            raise RuntimeError(
                f"{group_attribute} selected zero cross-attention processors"
            )
    identity_ca_name_set = set(identity_ca_names)
    setattr(pipeline, "_ba_identity_ca_processor_names", tuple(identity_ca_names))

    installed_identity_ca_names = {
        name
        for name, processor in current_procs.items()
        if isinstance(
            processor,
            (HardIdentityCrossAttnProcessorV2, ResidualIdentityCrossAttnProcessorV3),
        )
    }
    if has_branched and installed_identity_ca_names != identity_ca_name_set:
        raise RuntimeError(
            "Installed corrected identity-CA map does not match configuration: "
            f"installed={sorted(installed_identity_ca_names)}, "
            f"expected={sorted(identity_ca_name_set)}"
        )

    if not has_branched:
        # Create new processors
        new_procs = {}
        patched_proc_names: list[str] = []
        
        # Get cross-attention dimension
        cross_attention_dim = pipeline.unet.config.cross_attention_dim
        if isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = cross_attention_dim[0]
        
        for name in pipeline.unet.attn_processors.keys():
            # Get hidden size
            if "mid_block" in name:
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]
            else:
                hidden_size = pipeline.unet.config.block_out_channels[0]
            
            if name.endswith("attn1.processor"):
                if disable_sa or name not in patchable_sa_name_set:
                    # Keep original self-attn processor; no branching on attn1.
                    new_procs[name] = pipeline._original_attn_processors[name]
                else:
                    # Self-attention: use branched processor
                    if architecture_version in {
                        "residual_sa_v2",
                        "anchored_mix_sa_v3",
                        "query_adaptive_hard_sa_v4",
                    }:
                        trainable_dtype_name = str(
                            getattr(pipeline, "branched_trainable_dtype", "fp32")
                        ).lower()
                        if trainable_dtype_name not in {"fp32", "float32"}:
                            raise ValueError(
                                f"{architecture_version} currently requires "
                                "branched_trainable_dtype=fp32"
                            )
                        configured_ref_rank = getattr(
                            pipeline, "ba_ref_kv_rank", None
                        )
                        configured_output_rank = getattr(
                            pipeline, "ba_output_rank", None
                        )
                        fallback_rank = int(
                            getattr(
                                pipeline,
                                "branched_attn_lora_rank",
                                getattr(pipeline, "lora_rank", 32),
                            )
                        )
                        if architecture_version == "residual_sa_v2":
                            proc = BranchedAttnProcessor(
                                hidden_size=hidden_size,
                                cross_attention_dim=hidden_size,
                                scale=scale,
                                ref_kv_rank=int(configured_ref_rank or fallback_rank),
                                output_rank=int(configured_output_rank or fallback_rank),
                                gate_init=float(
                                    getattr(pipeline, "ba_gate_init", 0.10)
                                ),
                                gate_max=float(
                                    getattr(pipeline, "ba_gate_max", 1.0)
                                ),
                                gate_timestep=bool(
                                    getattr(pipeline, "ba_gate_timestep", True)
                                ),
                                gate_face_area=bool(
                                    getattr(pipeline, "ba_gate_face_area", True)
                                ),
                                trainable_dtype=torch.float32,
                                require_denoise_progress=bool(
                                    getattr(
                                        pipeline,
                                        "ba_require_denoise_progress",
                                        True,
                                    )
                                ),
                            )
                        elif architecture_version == "anchored_mix_sa_v3":
                            proc = BranchedAttnProcessor(
                                hidden_size=hidden_size,
                                cross_attention_dim=hidden_size,
                                scale=scale,
                                ref_kv_rank=int(configured_ref_rank or fallback_rank),
                                output_rank=int(configured_output_rank or fallback_rank),
                                mix_init=float(
                                    getattr(pipeline, "ba_mix_init", 0.50)
                                ),
                                mix_floor=float(
                                    getattr(pipeline, "ba_mix_floor", 0.25)
                                ),
                                mix_max=float(
                                    getattr(pipeline, "ba_mix_max", 0.90)
                                ),
                                mix_timestep=bool(
                                    getattr(pipeline, "ba_mix_timestep", True)
                                ),
                                mix_face_area=bool(
                                    getattr(pipeline, "ba_mix_face_area", True)
                                ),
                                reference_rms_match=bool(
                                    getattr(
                                        pipeline,
                                        "ba_reference_rms_match",
                                        True,
                                    )
                                ),
                                reference_rms_clip_min=float(
                                    getattr(
                                        pipeline,
                                        "ba_reference_rms_clip_min",
                                        0.50,
                                    )
                                ),
                                reference_rms_clip_max=float(
                                    getattr(
                                        pipeline,
                                        "ba_reference_rms_clip_max",
                                        2.00,
                                    )
                                ),
                                trainable_dtype=torch.float32,
                                require_denoise_progress=bool(
                                    getattr(
                                        pipeline,
                                        "ba_require_denoise_progress",
                                        True,
                                    )
                                ),
                                telemetry_enabled=bool(
                                    getattr(
                                        pipeline,
                                        "ba_telemetry_enabled",
                                        False,
                                    )
                                ),
                                telemetry_interval=int(
                                    getattr(
                                        pipeline,
                                        "ba_telemetry_interval",
                                        50,
                                    )
                                ),
                                mix_override=getattr(
                                    pipeline, "ba_mix_override", None
                                ),
                            )
                        else:
                            proc = BranchedAttnProcessor(
                                hidden_size=hidden_size,
                                cross_attention_dim=hidden_size,
                                scale=float(
                                    getattr(
                                        pipeline,
                                        "ba_face_branch_scale",
                                        scale,
                                    )
                                ),
                                branch_q_rank=int(
                                    getattr(
                                        pipeline,
                                        "ba_branch_q_rank",
                                        16,
                                    )
                                ),
                                ref_kv_rank=int(configured_ref_rank or fallback_rank),
                                output_rank=int(configured_output_rank or fallback_rank),
                                trainable_dtype=torch.float32,
                                telemetry_enabled=bool(
                                    getattr(
                                        pipeline,
                                        "ba_telemetry_enabled",
                                        False,
                                    )
                                ),
                                telemetry_interval=int(
                                    getattr(
                                        pipeline,
                                        "ba_telemetry_interval",
                                        50,
                                    )
                                ),
                            )
                    else:
                        trainable_dtype_name = str(
                            getattr(pipeline, "branched_trainable_dtype", "inherit")
                        ).lower()
                        hard_trainable_dtype = (
                            torch.float32
                            if trainable_dtype_name in {"fp32", "float32"}
                            else None
                        )
                        proc = BranchedAttnProcessor(
                            hidden_size=hidden_size,
                            cross_attention_dim=hidden_size,
                            scale=scale,
                            branched_attn_weight_mode=getattr(pipeline, "branched_attn_weight_mode", "shared"),
                            branched_attn_new_weight_kind=getattr(pipeline, "branched_attn_new_weight_kind", "full"),
                            branched_attn_lora_rank=int(
                                getattr(pipeline, "ba_hard_v1_lora_rank", None)
                                or getattr(
                                    pipeline,
                                    "branched_attn_lora_rank",
                                    getattr(pipeline, "lora_rank", 16),
                                )
                            ),
                            trainable_dtype=hard_trainable_dtype,
                            true_reference_key_mask=bool(
                                getattr(
                                    pipeline,
                                    "ba_hard_v1_true_reference_key_mask",
                                    False,
                                )
                            ),
                            branch_output_rank=getattr(
                                pipeline,
                                "ba_hard_v1_branch_output_rank",
                                None,
                            ),
                            reference_roi_warp=bool(
                                getattr(
                                    pipeline,
                                    "ba_hard_v1_reference_roi_warp",
                                    False,
                                )
                            ),
                            hardcase_mode=(
                                hardcase_mode
                                if any(
                                    name.startswith(f"{group}.")
                                    for group in hardcase_groups
                                )
                                else "off"
                            ),
                            hardcase_rank=int(
                                getattr(pipeline, "ba_hardcase_rank", 64)
                            ),
                            hardcase_gate_max=float(
                                getattr(pipeline, "ba_hardcase_gate_max", 0.20)
                            ),
                            hardcase_roi_size=int(
                                getattr(pipeline, "ba_hardcase_roi_size", 32)
                            ),
                            hardcase_face_threshold_px=int(
                                getattr(
                                    pipeline,
                                    "ba_hardcase_face_threshold_px",
                                    256,
                                )
                            ),
                            hardcase_transition_cells=int(
                                getattr(
                                    pipeline,
                                    "ba_hardcase_transition_cells",
                                    2,
                                )
                            ),
                            hardcase_ownership_hidden_dim=int(
                                getattr(
                                    pipeline,
                                    "ba_hardcase_ownership_hidden_dim",
                                    128,
                                )
                            ),
                            hardcase_visible_face_floor=float(
                                getattr(
                                    pipeline,
                                    "ba_hardcase_visible_face_floor",
                                    0.20,
                                )
                            ),
                        )
                    proc.init_from_attention(_resolve_attn_module(pipeline.unet, name))
                    if architecture_version in {
                        "residual_sa_v2",
                        "anchored_mix_sa_v3",
                        "query_adaptive_hard_sa_v4",
                    }:
                        proc = proc.to(pipeline.device)
                    elif hard_trainable_dtype is not None:
                        # 3 Aug 2026 - Keep only hard-v1 BA parameters in FP32;
                        # cloned effective base weights remain frozen BF16 buffers.
                        proc = proc.to(pipeline.device)
                    else:
                        proc = proc.to(pipeline.device, dtype=pipeline.unet.dtype)
                    proc.set_masks(_mask, _mref)
                    setattr(proc, "strict_face_routing", bool(getattr(pipeline, "strict_face_routing", False)))
                    _apply_runtime_flags(proc, pipeline)

                    # Wire id_embeds (zeros if missing); whether they are used is controlled by use_id_embeds
                    proc.id_embeds = _idem

                    new_procs[name] = proc
                    patched_proc_names.append(name)
                
            elif name.endswith("attn2.processor"):
                if name in identity_ca_name_set:
                    trainable_dtype_name = str(
                        getattr(pipeline, "branched_trainable_dtype", "inherit")
                    ).lower()
                    identity_trainable_dtype = (
                        torch.float32
                        if (
                            residual_identity_ca_v3_enabled
                            or trainable_dtype_name in {"fp32", "float32"}
                        )
                        else None
                    )
                    if identity_ca_v2_enabled:
                        proc = HardIdentityCrossAttnProcessorV2(
                            hidden_size=hidden_size,
                            cross_attention_dim=int(cross_attention_dim),
                            rank=int(
                                getattr(pipeline, "ba_identity_ca_v2_rank", 16)
                            ),
                            trainable_dtype=identity_trainable_dtype,
                        )
                    else:
                        proc = ResidualIdentityCrossAttnProcessorV3(
                            hidden_size=hidden_size,
                            cross_attention_dim=int(cross_attention_dim),
                            rank=int(
                                getattr(
                                    pipeline,
                                    "ba_residual_identity_ca_v3_rank",
                                    64,
                                )
                            ),
                            gate_init=float(
                                getattr(
                                    pipeline,
                                    "ba_residual_identity_ca_v3_gate_init",
                                    0.02,
                                )
                            ),
                            gate_max=float(
                                getattr(
                                    pipeline,
                                    "ba_residual_identity_ca_v3_gate_max",
                                    0.20,
                                )
                            ),
                            trainable_dtype=(
                                identity_trainable_dtype or torch.float32
                            ),
                        )
                    proc.init_from_attention(
                        _resolve_attn_module(pipeline.unet, name)
                    )
                    if identity_trainable_dtype is not None:
                        proc = proc.to(pipeline.device)
                    else:
                        proc = proc.to(
                            pipeline.device,
                            dtype=pipeline.unet.dtype,
                        )
                    proc.set_masks(_mask, _mref)
                    proc.set_class_tokens_mask(class_tokens_mask)
                    _apply_runtime_flags(proc, pipeline)
                    new_procs[name] = proc
                    patched_proc_names.append(name)
                elif disable_ca:
                    # Keep original cross-attn processor; no branched CA.
                    new_procs[name] = pipeline._original_attn_processors[name]
                else:
                    # Cross-attention: use branched cross-attention processor
                    num_tokens = 77  # Standard CLIP token count
                    if hasattr(pipeline, 'tokenizer_2'):
                        num_tokens = pipeline.tokenizer_2.model_max_length

                    proc = BranchedCrossAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=scale,
                        num_tokens=num_tokens,
                        branched_attn_weight_mode=getattr(pipeline, "branched_attn_weight_mode", "shared"),
                        branched_attn_new_weight_kind=getattr(pipeline, "branched_attn_new_weight_kind", "full"),
                        branched_attn_lora_rank=int(
                            getattr(pipeline, "branched_attn_lora_rank", getattr(pipeline, "lora_rank", 16))
                        ),
                    ).to(pipeline.device, dtype=pipeline.unet.dtype)
                    proc.init_from_attention(_resolve_attn_module(pipeline.unet, name))
                    # enable KV equalizer for face branch
                    setattr(proc, "equalize_face_kv", True)
                    setattr(proc, "equalize_clip", (1/3, 8.0))
                    setattr(proc, "strict_face_routing", bool(getattr(pipeline, "strict_face_routing", False)))
                    proc.set_masks(_mask, _mref)
                    # Keep CA path consistent too (even if CA doesn’t always consume id_embeds)
                    proc.id_embeds = _idem
                    proc.class_tokens_mask = class_tokens_mask

                    new_procs[name] = proc
                    patched_proc_names.append(name)
                
            else:
                # Keep original for other processors
                new_procs[name] = pipeline._original_attn_processors[name]
        
        pipeline.unet.set_attn_processor(new_procs)
        setattr(pipeline, "_ba_patched_processor_names", tuple(patched_proc_names))
    else:
        patched_proc_names: list[str] = []
        # Update masks on existing processors
        for name, proc in pipeline.unet.attn_processors.items():
            if isinstance(
                proc,
                (
                    BranchedAttnProcessor,
                    BranchedCrossAttnProcessor,
                    HardIdentityCrossAttnProcessorV2,
                    ResidualIdentityCrossAttnProcessorV3,
                ),
            ):
                patched_proc_names.append(name)
                # proc.set_masks(mask, mask_ref)
                proc.set_masks(_mask, _mref)
                _apply_runtime_flags(proc, pipeline)

                # (Re)apply id_embeds (zeros if missing); actual usage is gated by use_id_embeds
                if hasattr(proc, "id_embeds"):
                    proc.id_embeds = _idem
                if isinstance(
                    proc,
                    (
                        HardIdentityCrossAttnProcessorV2,
                        ResidualIdentityCrossAttnProcessorV3,
                    ),
                ):
                    # 4 Aug 2026 - ID-token membership changes with the current
                    # prompt/CFG batch and must be refreshed on every forward.
                    proc.set_class_tokens_mask(class_tokens_mask)
        setattr(pipeline, "_ba_patched_processor_names", tuple(patched_proc_names))

    pose_adapt_ratio = float(getattr(pipeline, "pose_adapt_ratio", 0.0))
    if (
        pose_adapt_ratio != 0.0
        and getattr(pipeline, "_logged_pose_adapt_ratio", None)
        != pose_adapt_ratio
    ):
        print(
            "POSE_ADAPT_RUNTIME "
            f"ratio={pose_adapt_ratio:.4f} "
            f"processors={len(getattr(pipeline, '_ba_patched_processor_names', ()))}"
        )
        pipeline._logged_pose_adapt_ratio = pose_adapt_ratio

def encode_face_prompt(
    pipeline,
    device: torch.device,
    batch_size: int,
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """
    Encode "face" text prompt for face branch cross-attention.
    """
    # Simple "face" prompt
    face_text = "a close-up human face laughing hard"
    
    # Use the pipeline's text encoder
    if hasattr(pipeline, 'encode_prompt'):
        # face_embeds, neg_embeds, _, _ = pipeline.encode_prompt(
        # Get the full prompt embeddings with correct sequence length
        face_embeds, neg_face_embeds, _, _ = pipeline.encode_prompt(
            face_text,
            face_text,  # prompt_2
            device,
            1,  # num_images_per_prompt
            do_classifier_free_guidance,
            negative_prompt="" if do_classifier_free_guidance else None,
            negative_prompt_2="" if do_classifier_free_guidance else None,
        )
        
        # Expand to batch size
        if do_classifier_free_guidance:
            # Build [neg(B), pos(B)] to match CFG prompt layout.
            if batch_size % 2 == 0:
                half = batch_size // 2
                neg = neg_face_embeds.expand(half, -1, -1)
                pos = face_embeds.expand(half, -1, -1)
                return torch.cat([neg, pos], dim=0)
            # Fallback if caller passed non-CFG batch size while CFG is on.
            return face_embeds.expand(batch_size, -1, -1)
        else:
            return face_embeds.expand(batch_size, -1, -1)
    
    return None


def two_branch_predict(
    pipeline,
    latent_model_input: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: Dict[str, Any],
    mask4: torch.Tensor,
    mask4_ref: torch.Tensor,
    reference_latents: torch.Tensor,
    reference_noise: Optional[torch.Tensor] = None,
    face_prompt_embeds: Optional[torch.Tensor] = None,
    class_tokens_mask: Optional[torch.Tensor] = None,
    face_embed_strategy: str = "face",
    id_embeds: Optional[torch.Tensor] = None, 
    step_idx: int = 0,
    scale: float = 1.0,
    timestep_cond: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute two-branch prediction with doubled batch for both latents and prompts.
    """

    full_debug = False

    # --- quick shape + CFG sanity ---
    if full_debug:
        if step_idx in (0, 1) or step_idx % 10 == 0:
            def stat(x): 
                x = x.float()
                return f"shape={tuple(x.shape)} μ={x.mean().item():.4f} σ={x.std().item():.4f}"
            print(f"[2BP] step={step_idx}  CFG={pipeline.do_classifier_free_guidance}")
        print(f"[2BP]   latent_in:   {stat(latent_model_input)}")
        print(f"[2BP]   ref_latents: {stat(reference_latents)}")


        # --- quick mask stats ---
        if step_idx in (0, 1) or step_idx % 10 == 0:
            m = mask4.detach().float()
            mr = mask4_ref.detach().float()
            def mstat(m):
                return f"{tuple(m.shape)}  mean={m.mean().item():.4f}  ones={(m>0.5).float().mean().item():.4f}"
            print(f"[2BP]   mask gen: {mstat(m)}   mask ref: {mstat(mr)}   |diff|={(m-mr).abs().mean().item():.4f}")



    device = latent_model_input.device
    dtype = latent_model_input.dtype
    batch_size = latent_model_input.shape[0]
    
    
    REF_NOISE_ONCE = True  # keep same ref noise across steps within one generation
    if reference_noise is None and not hasattr(pipeline, "_ref_noise"):
        gen = getattr(pipeline, "generator", None)
        if isinstance(gen, (list, tuple)):
            gen = gen[0] if gen else None

        if isinstance(gen, torch.Generator):
            ref_gen = gen
            if ref_gen.device.type != device.type:
                ref_gen2 = torch.Generator(device=device)
                ref_gen2.set_state(ref_gen.get_state())
                ref_gen = ref_gen2
            try:
                pipeline._ref_noise = torch.randn_like(reference_latents, generator=ref_gen)
            except TypeError:
                pipeline._ref_noise = torch.randn(
                    reference_latents.shape,
                    generator=ref_gen,
                    device=reference_latents.device,
                   dtype=reference_latents.dtype,
                )
        else:
            # IMPORTANT: don't use a fresh unseeded torch.Generator() (it’s deterministic); use global RNG instead.
            pipeline._ref_noise = torch.randn_like(reference_latents)



    
    t_ref = t if torch.is_tensor(t) else torch.tensor([t], device=device, dtype=torch.long)
    if t_ref.ndim == 0:
        t_ref = t_ref.unsqueeze(0)
    expected_ref = reference_latents.shape[0]
    current_ref = t_ref.shape[0]
    if current_ref != expected_ref:
        reps = (expected_ref + current_ref - 1) // current_ref
        t_ref = t_ref.repeat(reps)[:expected_ref]
    
    if reference_noise is None:
        reference_noise = pipeline._ref_noise
    if reference_noise.shape != reference_latents.shape:
        raise RuntimeError(
            "Reference-noise shape mismatch: "
            f"noise={tuple(reference_noise.shape)}, "
            f"latents={tuple(reference_latents.shape)}"
        )
    reference_noise = reference_noise.to(
        device=reference_latents.device,
        dtype=reference_latents.dtype,
    )
    ref_noised = pipeline.scheduler.add_noise(
        reference_latents,
        reference_noise,
        t_ref
    )

    
    ref_noised = pipeline.scheduler.scale_model_input(ref_noised, t_ref).to(latent_model_input.dtype) # critical: match UNet’s expected scaling at this timestep

    if full_debug:
        if step_idx in (0, 1) or step_idx % 10 == 0:
            print(f"[2BP]   ref_noised:  {stat(ref_noised)}  Δ(noise,ref)σ={(latent_model_input.std()-ref_noised.std()).item():.4f}")

    
    # Ensure same batch size
    if ref_noised.shape[0] < batch_size:
        ref_noised = ref_noised.expand(batch_size, -1, -1, -1)
    
    # Create doubled batch: [noise, reference]
    batched_latents = torch.cat([latent_model_input, ref_noised], dim=0)
    
    timestep_for_progress = t if torch.is_tensor(t) else torch.tensor([t], device=device)
    if timestep_for_progress.ndim == 0:
        timestep_for_progress = timestep_for_progress.unsqueeze(0)
    num_train_timesteps = int(pipeline.scheduler.config.num_train_timesteps)
    if num_train_timesteps <= 1:
        raise RuntimeError(
            f"Invalid scheduler num_train_timesteps={num_train_timesteps}"
        )
    ba_denoise_progress = 1.0 - (
        timestep_for_progress.to(device=device, dtype=torch.float32)
        / float(num_train_timesteps - 1)
    )

    # Patch processors with masks and the real scheduler timestep. Training
    # historically passes step_idx=0, so architecture gates must not use it.
    patch_unet_attention_processors(
        pipeline, mask4, mask4_ref, scale,
        id_embeds=id_embeds if face_embed_strategy == "id_embeds" else None,
        class_tokens_mask=class_tokens_mask,
        ba_denoise_progress=ba_denoise_progress,
    )

    # --- quick patch check
    if full_debug:
        if step_idx == 0:
            procs = pipeline.unet.attn_processors
            n_sa = sum("attn1.processor" in k for k in procs)  # self-attn slots
            n_ca = sum("attn2.processor" in k for k in procs)  # cross-attn slots
            any_branched = any(p.__class__.__name__.startswith("Branched") for p in procs.values())
            sample_k = next(iter(procs))
            print(f"[2BP]   processors patched? {any_branched}  (SA={n_sa}, CA={n_ca})  sample={procs[sample_k].__class__.__name__}")

        
    # Prepare timesteps for doubled batch
    t_batched = t if torch.is_tensor(t) else torch.tensor([t], device=device)
    if t_batched.ndim == 0:
        t_batched = t_batched.unsqueeze(0)
    expected = batched_latents.shape[0]
    current = t_batched.shape[0]
    if current != expected:
        reps = (expected + current - 1) // current
        t_batched = t_batched.repeat(reps)[:expected]
    
    # Prepare face prompt if not provided
    if face_prompt_embeds is None:
        face_prompt_embeds = encode_face_prompt(
            pipeline, 
            device, 
            batch_size,
            pipeline.do_classifier_free_guidance
        )

    
    # Only mirror the main text into the face branch for legacy "id".
    # For "id_embeds" we keep actual "face" text and use the 2048-D ID features.
    if (face_embed_strategy or "face") in {"id"}:    
        # keep dtype/device aligned with text encoder / UNet
        d, dev = prompt_embeds.dtype, prompt_embeds.device
        face_prompt_embeds = prompt_embeds.clone()
        if class_tokens_mask is not None:
            m = class_tokens_mask.to(dev)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            if m.shape[0] != face_prompt_embeds.shape[0]:
                if m.shape[0] == 1:
                    m = m.expand(face_prompt_embeds.shape[0], -1)
                elif face_prompt_embeds.shape[0] % m.shape[0] == 0:
                    reps = face_prompt_embeds.shape[0] // m.shape[0]
                    m = m.repeat(reps, 1)
                else:
                    raise RuntimeError(
                        f"class_tokens_mask batch mismatch: mask={tuple(m.shape)} "
                        f"vs face_prompt_embeds={tuple(face_prompt_embeds.shape)}"
                    )
            m = m.unsqueeze(-1).to(dtype=d)                # [B,L,1]
            one = torch.tensor(1.0, device=dev, dtype=d)
            id_scale = torch.tensor(getattr(pipeline, "id_token_scale", 2.5),
                                   device=dev, dtype=d)
            # face_prompt_embeds = face_prompt_embeds * (one - m) + face_prompt_embeds * m * id_scale
            
            # Use only ID tokens for the face branch (no leakage from other words)
            face_prompt_embeds = face_prompt_embeds * m * id_scale
           
        else:
         print(f"[2BP]   WARNING: class_tokens_mask is None, falling back to face text")
         # Fallback to face text encoding
         face_prompt_embeds = encode_face_prompt(
             pipeline, device, batch_size, pipeline.do_classifier_free_guidance
         ).to(prompt_embeds.device, prompt_embeds.dtype)
                  
        # per-token std match: bring face tokenwise std ~ gen tokenwise std
        eps = 1e-6
        std_gen  = prompt_embeds.float().std(dim=-1, keepdim=True).clamp_min(eps)
        std_face = face_prompt_embeds.float().std(dim=-1, keepdim=True).clamp_min(eps)
        # face_prompt_embeds = (face_prompt_embeds / std_face) * std_gen
        face_prompt_embeds = ((face_prompt_embeds.float() / std_face) * std_gen).to(d)
    
    if full_debug:
        # ---quick prompt stats---
        if step_idx in (0, 1) or step_idx % 10 == 0:
            pe = prompt_embeds.detach().float()
            fe = face_prompt_embeds.detach().float()
            same_shape = pe.shape == fe.shape
            # frac of zeros in face prompt (detect padding/truncation artefacts)
            frac_zero = (fe.abs() < 1e-8).float().mean().item()
            diff_mean = (pe - fe).abs().mean().item() if same_shape else float('nan')
            print(f"[2BP]   prompts: gen={tuple(pe.shape)}  face={tuple(fe.shape)}  zero_frac(face)={frac_zero:.3f}  Δμ={diff_mean:.4f}")


    
    # --- Build face-branch text properly and concat ------------------------

    if (face_embed_strategy or "face") == "id_embeds":
        if face_prompt_embeds is None or face_prompt_embeds.shape != prompt_embeds.shape:
            raise ValueError("id_embeds mode requires face_prompt_embeds.shape == prompt_embeds.shape")
    elif face_prompt_embeds is None or face_prompt_embeds.shape != prompt_embeds.shape:
        face_prompt_embeds = encode_face_prompt(
            pipeline, device, batch_size, pipeline.do_classifier_free_guidance
        )    

        
        
    face_prompt_embeds = face_prompt_embeds.to(prompt_embeds.device, prompt_embeds.dtype)

    # Double-stack encoder states for branched CA:
    #   first half → generation prompt
    #   second half → face prompt
    encoder_hidden_states = torch.cat([prompt_embeds, face_prompt_embeds], dim=0)

    if full_debug:
        # quick sanity – these should *not* be identical
        if (step_idx in (0, 1)) or (step_idx % 10 == 0):
            diff_mu = (prompt_embeds.detach().float() - face_prompt_embeds.detach().float()).abs().mean().item()
            print(f"[2BP]   encoder_hidden_states Δ(gen,face)μ={diff_mu:.4f}")


    # Double added_cond_kwargs
    doubled_kwargs = {}
    for k, v in added_cond_kwargs.items():
        if torch.is_tensor(v):
            # Double the tensor
            doubled_kwargs[k] = torch.cat([v, v], dim=0)
        else:
            doubled_kwargs[k] = v
    
    # Double timestep_cond if present
    if timestep_cond is not None:
        timestep_cond_doubled = torch.cat([timestep_cond, timestep_cond], dim=0)
    else:
        timestep_cond_doubled = None

    if str(getattr(pipeline, "ba_hardcase_mode", "off")).lower() == "clean_memory":
        memory_source = getattr(pipeline, "_ba_clean_memory_source_tensor", None)
        if memory_source is not reference_latents:
            memory_processors = [
                processor
                for processor in pipeline.unet.attn_processors.values()
                if getattr(processor, "hardcase_mode", "off") == "clean_memory"
            ]
            if not memory_processors:
                raise RuntimeError("Clean-memory mode installed no memory processors")
            for processor in memory_processors:
                processor.clear_clean_memory()
                processor.set_clean_memory_capture(True)
            clean_timestep = torch.ones(
                batched_latents.shape[0], device=device, dtype=t_batched.dtype
            )
            clean_reference = reference_latents.to(dtype=latent_model_input.dtype)
            if clean_reference.shape[0] < batch_size:
                clean_reference = clean_reference.expand(
                    batch_size, -1, -1, -1
                )
            clean_batched = torch.cat([clean_reference, clean_reference], dim=0)
            null_encoder = torch.zeros_like(encoder_hidden_states)
            null_kwargs = {
                key: (torch.zeros_like(value) if torch.is_tensor(value) else value)
                for key, value in doubled_kwargs.items()
            }
            try:
                # 11 Aug 2026 - The cache is an identity source, not a second
                # trainable U-Net trajectory. Detaching it also bounds memory.
                with torch.no_grad():
                    pipeline.unet(
                        clean_batched,
                        clean_timestep,
                        encoder_hidden_states=null_encoder,
                        timestep_cond=(
                            None
                            if timestep_cond_doubled is None
                            else torch.zeros_like(timestep_cond_doubled)
                        ),
                        added_cond_kwargs=null_kwargs,
                        return_dict=False,
                    )
            finally:
                for processor in memory_processors:
                    processor.set_clean_memory_capture(False)
            if any(
                processor.clean_reference_memory is None
                for processor in memory_processors
            ):
                raise RuntimeError("Clean-memory capture missed a configured processor")
            pipeline._ba_clean_memory_source_tensor = reference_latents

    # Runtime knobs for branched processors via call kwargs
    base_cross_attention_kwargs = getattr(pipeline, "_cross_attention_kwargs", None)
    runtime_cross_attention_kwargs = (
        dict(base_cross_attention_kwargs) if isinstance(base_cross_attention_kwargs, dict) else {}
    )
    # runtime_cross_attention_kwargs.update(
    #     {
    #         "ba_pose_adapt_ratio": float(getattr(pipeline, "pose_adapt_ratio", 0.25)),
    #         "ba_ca_mixing_for_face": bool(getattr(pipeline, "ca_mixing_for_face", True)),
    #         "ba_use_id_embeds": bool(getattr(pipeline, "use_id_embeds", True)),
    #         "ba_id_alpha": float(getattr(pipeline, "id_alpha", 0.3)),
    #         "ba_id_embeds": id_embeds,
    #     }
    # )
    
    # Single forward pass with doubled batch
    noise_pred = pipeline.unet(
        batched_latents,
        t_batched,
        encoder_hidden_states=encoder_hidden_states,
        timestep_cond=timestep_cond_doubled,
        cross_attention_kwargs=runtime_cross_attention_kwargs if runtime_cross_attention_kwargs else None,
        added_cond_kwargs=doubled_kwargs,
        return_dict=False,
    )[0]

    # --- quick check of cosine sim between halves
    # Split UNet output into halves (noise/merged vs face-pure)
    B2 = noise_pred.shape[0] // 2
    first, second = noise_pred[:B2].float(), noise_pred[B2:].float()

    if full_debug:
        # If CFG is on, each half is [uncond, cond]
        if pipeline.do_classifier_free_guidance and B2 % 2 == 0:
            fU, fC = first.chunk(2)
            sU, sC = second.chunk(2)
            def s2(x): return f"σ={x.std().item():.4f}"
            print(f"[2BP]   out halves: first({s2(first)})  second({s2(second)})  | first U/C {s2(fU)}/{s2(fC)}  second U/C {s2(sU)}/{s2(sC)}")
        else:
            print(f"[2BP]   out halves: first σ={first.std().item():.4f}  second σ={second.std().item():.4f}")

        # Mean cosine sim between halves → should NOT be ~1.0
        cos = torch.nn.functional.cosine_similarity(first.flatten(1), second.flatten(1), dim=1).mean().item()
        print(f"[2BP]   cos(first,second)={cos:.3f}")
    # --- end of quick check



    
    # Extract merged result (first half)
    noise_pred_merged = noise_pred[:batch_size]

    # Training consumes only the merged prediction. Keep the historical
    # branch tensors available by default for validation/debug callers.
    if not bool(getattr(pipeline, "compute_branch_debug_outputs", True)):
        return noise_pred_merged, None, None
    
    USE_SOFT_BLENDING = True
    
    if USE_SOFT_BLENDING:
        if mask4 is not None and mask4.shape[-2:] == noise_pred_merged.shape[-2:]:
            mask4 = gaussian_blur_mask(mask4, kernel_size=5) # Apply gaussian blur to mask for smoother transitions
    
    
    # For debugging: approximate branch outputs
    mask_4ch = mask4.repeat(1, 4, 1, 1).to(dtype=dtype)
    if mask_4ch.shape[0] != batch_size:
        cur = int(mask_4ch.shape[0])
        if cur <= 0:
            raise RuntimeError(f"Invalid mask batch size: {cur}")
        reps = (batch_size + cur - 1) // cur
        mask_4ch = mask_4ch.repeat(reps, 1, 1, 1)[:batch_size]
    
    noise_bg = noise_pred_merged * (1 - mask_4ch)
    noise_face = noise_pred_merged * mask_4ch
    
    # Debug logging
    if full_debug:
        if step_idx < 3 or step_idx % 10 == 0:
            print(f"[Branch] Step {step_idx}: "
                f"merged_norm={noise_pred_merged.std().item():.4f}, "
                f"face={noise_face.std().item():.4f}, "
                f"bg={noise_bg.std().item():.4f}")
   
    return noise_pred_merged, noise_face, noise_bg


def restore_original_processors(pipeline):
   """Restore original attention processors."""
   if hasattr(pipeline, '_original_attn_processors'):
       pipeline.unet.set_attn_processor(pipeline._original_attn_processors)
       delattr(pipeline, '_original_attn_processors')
       return True
   return False


def gaussian_blur_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur to mask for smoother transitions."""
    import torch.nn.functional as F
    
    # Create a simple Gaussian kernel
    sigma = kernel_size / 3.0
    kernel_1d = torch.exp(-torch.arange(kernel_size, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d[None, None, :, :].to(mask.device, mask.dtype)
    
    # Apply convolution
    mask_blurred = F.conv2d(mask, kernel_2d, padding=kernel_size // 2)
    
    return mask_blurred.clamp(0, 1)
