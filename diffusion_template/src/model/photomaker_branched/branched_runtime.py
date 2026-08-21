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
    residual_identity_ca = bool(
        getattr(pipeline, "ba_residual_identity_ca_v3_enabled", False)
    )
    hardcase_mode = str(
        getattr(pipeline, "ba_hardcase_mode", "off") or "off"
    ).lower()
    hardcase_groups = tuple(
        str(group)
        for group in (getattr(pipeline, "ba_hardcase_groups", None) or ())
    )
    if hardcase_mode not in {"off", "soft_router", "temporal_frequency"}:
        raise ValueError(f"Unsupported clean hard-case mode: {hardcase_mode}")
    if hardcase_mode != "off" and not hardcase_groups:
        raise RuntimeError("Clean hard-case routes require explicit U-Net groups")
    frequency_surface_enabled = bool(
        getattr(pipeline, "ba_frequency_surface_loss_enabled", False)
    )
    frequency_surface_groups = tuple(
        str(group) for group in (
            getattr(pipeline, "ba_frequency_surface_loss_groups", None) or ()
        )
    )
    null_key_enabled = bool(
        getattr(pipeline, "ba_null_key_router_enabled", False)
    )
    null_key_groups = tuple(
        str(group)
        for group in (getattr(pipeline, "ba_null_key_router_groups", None) or ())
    )
    # 18 Aug 2026 - CL27 reuses CL23 processors and enables its loss only in
    # the two declared up blocks; inference receives no supervision mask.
    if frequency_surface_enabled and (
        hardcase_mode != "temporal_frequency" or not frequency_surface_groups
    ):
        raise RuntimeError("CL27 requires temporal_frequency and explicit loss groups")
    if null_key_enabled and (
        hardcase_mode != "temporal_frequency" or not null_key_groups
    ):
        raise RuntimeError("CL39 requires temporal_frequency and explicit router groups")
    if bool(getattr(pipeline, "e13_family_contract", False)):
        # 10 Aug 2026 - E13C-CORE-01: Fail closed on the architectural
        # invariants shared by E13, BC_E13 and CL14.
        required = {
            "branched_sa": not disable_sa,
            "native_ca": disable_ca,
            "reference_only_kv": float(getattr(pipeline, "pose_adapt_ratio", 0.0)) == 0.0,
            "no_face_ca_mix": not bool(getattr(pipeline, "ca_mixing_for_face", False)),
            "all_sa_blocks": float(getattr(pipeline, "ba_patch_top_k", 1.0)) == 1.0,
        }
        failed = [name for name, valid in required.items() if not valid]
        if failed:
            raise RuntimeError(f"Invalid clean E13 processor route: {failed}")

    # Default to legacy (v1) when flag is not provided.
    use_attn_v2 = bool(getattr(pipeline, "use_attn_v2", False))
    # if use_attn_v2:
    #     from ._old2.attn_processor2 import BranchedAttnProcessor, BranchedCrossAttnProcessor
    # else:
    #     # from .attn_processor import BranchedAttnProcessor, BranchedCrossAttnProcessor
    #     # from .attn_processor_clean import BranchedAttnProcessor, BranchedCrossAttnProcessor

    from .attn_processor_cleanest import BranchedAttnProcessor, BranchedCrossAttnProcessor # New ver 25 Feb
    from .residual_identity_ca_processor_v3 import (
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
    has_branched = any(
        isinstance(p, (
            BranchedAttnProcessor,
            BranchedCrossAttnProcessor,
            ResidualIdentityCrossAttnProcessorV3,
        ))
        for p in current_procs.values()
    )

    def _resolve_attn_module(unet, proc_name):
        mod = unet
        for part in proc_name.rsplit(".processor", 1)[0].split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        return mod


    def _apply_runtime_flags(proc, pipe):
        # propagate key runtime knobs from model/pipeline onto processors
        # for k in ("pose_adapt_ratio", "ca_mixing_for_face", "train_branch_mode", "id_alpha", "use_id_embeds"):
        #     if hasattr(pipe, k):
        #         setattr(proc, k, getattr(pipe, k))
        
        # Keep only static toggles on processor instances.
        # Per-step runtime knobs are passed via UNet cross_attention_kwargs

        # Optional toggle for per-branch BA-specific adapters.
        if hasattr(pipe, "ba_weights_split"):
            setattr(proc, "ba_weights_split", getattr(pipe, "ba_weights_split"))
        if hasattr(pipe, "force_binary_masks"):
            setattr(proc, "force_binary_masks", bool(getattr(pipe, "force_binary_masks")))
        # 10 Aug 2026 - E13C-PERF-02: Processor-local mask caching removes
        # repeated interpolation only; it does not alter the mask tensor.
        setattr(proc, "cache_prepared_masks", bool(getattr(pipe, "cache_prepared_masks", False)))
        if hasattr(proc, "set_denoise_progress"):
            proc.set_denoise_progress(ba_denoise_progress)
        if hasattr(proc, "set_hardcase_telemetry_enabled"):
            proc.set_hardcase_telemetry_enabled(
                bool(getattr(pipe, "ba_hardcase_telemetry_enabled", False))
            )
        if hasattr(proc, "set_ownership_target_mask"):
            proc.set_ownership_target_mask(
                getattr(pipe, "_ba_ownership_target_mask", None)
            )
            
        

   
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
    patchable_sa_name_set = set(patchable_sa_names)

    identity_ca_names: list[str] = []
    if residual_identity_ca:
        groups = tuple(str(group) for group in (
            getattr(pipeline, "ba_residual_identity_ca_v3_groups", None) or ()
        ))
        # 13 Aug 2026 - CL14_CA-CORE-01: this is a corrected residual over
        # native CA, never a revival of the legacy reference-query CA route.
        if not disable_ca or groups != ("up_blocks.0", "up_blocks.1"):
            raise RuntimeError(
                "CL14_CA requires legacy CA off and residual CA only in up_blocks.0/1"
            )
        identity_ca_names = [
            name for name in pipeline.unet.attn_processors
            if name.endswith("attn2.processor")
            and any(name.startswith(f"{group}.") for group in groups)
        ]
        if not identity_ca_names:
            raise RuntimeError("CL14_CA selected zero residual identity-CA processors")
    identity_ca_name_set = set(identity_ca_names)
    setattr(pipeline, "_ba_identity_ca_processor_names", tuple(identity_ca_names))

    identity_token_indices = None
    if residual_identity_ca and class_tokens_mask is not None:
        token_mask = class_tokens_mask.detach().to(dtype=torch.bool)
        if token_mask.ndim == 1:
            token_mask = token_mask.unsqueeze(0)
        if token_mask.ndim != 2:
            raise RuntimeError("CL14_CA identity-token mask must be 2D")
        rows = [row.nonzero(as_tuple=False).flatten().tolist() for row in token_mask.cpu()]
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise RuntimeError("CL14_CA requires equal, nonzero ID-token counts")
        # 13 Aug 2026 - CL14_CA-PERF-02: validate/index the prompt once per
        # U-Net call instead of synchronizing independently in every CA layer.
        identity_token_indices = torch.tensor(
            rows, device=class_tokens_mask.device, dtype=torch.long
        )

    installed_identity_names = {
        name for name, processor in current_procs.items()
        if isinstance(processor, ResidualIdentityCrossAttnProcessorV3)
    }
    if has_branched and installed_identity_names != identity_ca_name_set:
        raise RuntimeError(
            "Installed residual identity-CA map differs from CL14_CA config: "
            f"installed={sorted(installed_identity_names)}, "
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
                    proc = BranchedAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=hidden_size,
                        scale=scale,
                        branched_attn_weight_mode=getattr(pipeline, "branched_attn_weight_mode", "shared"),
                        branched_attn_new_weight_kind=getattr(pipeline, "branched_attn_new_weight_kind", "full"),
                        branched_attn_lora_rank=int(
                            getattr(
                                pipeline,
                                "ba_hard_v1_lora_rank",
                                getattr(pipeline, "branched_attn_lora_rank", getattr(pipeline, "lora_rank", 16)),
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
                        hardcase_transition_cells=int(
                            getattr(pipeline, "ba_hardcase_transition_cells", 2)
                        ),
                        hardcase_frequency_low_early=float(
                            getattr(pipeline, "ba_hardcase_frequency_low_early", 0.50)
                        ),
                        hardcase_frequency_low_late=float(
                            getattr(pipeline, "ba_hardcase_frequency_low_late", 0.85)
                        ),
                        hardcase_frequency_high_early=float(
                            getattr(pipeline, "ba_hardcase_frequency_high_early", 0.75)
                        ),
                        hardcase_frequency_high_late=float(
                            getattr(pipeline, "ba_hardcase_frequency_high_late", 1.25)
                        ),
                        hardcase_telemetry_enabled=bool(
                            getattr(pipeline, "ba_hardcase_telemetry_enabled", False)
                        ),
                        frequency_surface_loss_enabled=(
                            frequency_surface_enabled
                            and any(
                                name.startswith(f"{group}.")
                                for group in frequency_surface_groups
                            )
                        ),
                        frequency_surface_top_low_band_factor=float(
                            getattr(
                                pipeline,
                                "ba_frequency_surface_top_low_band_factor",
                                0.25,
                            )
                        ),
                        frequency_surface_visible_floor_ratio=float(
                            getattr(
                                pipeline,
                                "ba_frequency_surface_visible_floor_ratio",
                                0.35,
                            )
                        ),
                        null_key_router_enabled=(
                            null_key_enabled
                            and any(
                                name.startswith(f"{group}.")
                                for group in null_key_groups
                            )
                        ),
                        null_key_entropy_threshold=float(
                            getattr(pipeline, "ba_null_key_entropy_threshold", 0.75)
                        ),
                        null_key_temperature=float(
                            getattr(pipeline, "ba_null_key_temperature", 0.08)
                        ),
                        null_key_max_abstention=float(
                            getattr(pipeline, "ba_null_key_max_abstention", 0.75)
                        ),
                        null_key_min_reference_fraction=float(
                            getattr(
                                pipeline,
                                "ba_null_key_min_reference_fraction",
                                0.25,
                            )
                        ),
                    )
                    proc.init_from_attention(_resolve_attn_module(pipeline.unet, name))
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
                    proc = ResidualIdentityCrossAttnProcessorV3(
                        hidden_size=hidden_size,
                        cross_attention_dim=int(cross_attention_dim),
                        rank=int(getattr(
                            pipeline, "ba_residual_identity_ca_v3_rank", 64
                        )),
                        gate_init=float(getattr(
                            pipeline, "ba_residual_identity_ca_v3_gate_init", 0.02
                        )),
                        gate_max=float(getattr(
                            pipeline, "ba_residual_identity_ca_v3_gate_max", 0.20
                        )),
                        trainable_dtype=torch.float32,
                    ).to(pipeline.device)
                    proc.init_from_attention(_resolve_attn_module(pipeline.unet, name))
                    proc.set_masks(_mask, _mref)
                    proc.set_class_tokens_mask(
                        class_tokens_mask, identity_token_indices
                    )
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
            if isinstance(proc, (
                BranchedAttnProcessor,
                BranchedCrossAttnProcessor,
                ResidualIdentityCrossAttnProcessorV3,
            )):
                patched_proc_names.append(name)
                # proc.set_masks(mask, mask_ref)
                proc.set_masks(_mask, _mref)
                _apply_runtime_flags(proc, pipeline)

                # (Re)apply id_embeds (zeros if missing); actual usage is gated by use_id_embeds
                if hasattr(proc, "id_embeds"):
                    proc.id_embeds = _idem
                if isinstance(proc, ResidualIdentityCrossAttnProcessorV3):
                    proc.set_class_tokens_mask(
                        class_tokens_mask, identity_token_indices
                    )
                elif hasattr(proc, "class_tokens_mask"):
                    proc.class_tokens_mask = class_tokens_mask
        setattr(pipeline, "_ba_patched_processor_names", tuple(patched_proc_names))

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


# 10 Aug 2026 - E13C-PIPE-02: This denoising/reference-noise path is copied from the sealed CL14 source so fixed seeds preserve the historical generation trajectory.
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
