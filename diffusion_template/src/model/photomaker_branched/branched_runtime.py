"""
branched_new.py - Simplified branched attention implementation with cross-attention
"""

import hashlib
import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Sequence
import os
from PIL import Image

from .debug_helpers import save_debug_images


def apply_ppr_reference_ca_mode(
    pipeline,
    face_prompt_embeds: torch.Tensor,
    face_prompt_attention_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor], str]:
    """Apply the inference-only reference-half cross-attention ablation."""
    mode = str(
        getattr(pipeline, "ba_ppr_reference_ca_mode", "original") or "original"
    ).lower()
    if mode == "original":
        return face_prompt_embeds, face_prompt_attention_mask, mode
    if mode != "zero":
        raise ValueError(
            "ba_ppr_reference_ca_mode must be one of: original, zero"
        )
    if not bool(getattr(pipeline, "ba_ppr_collect_diagnostics", False)):
        raise RuntimeError(
            "ba_ppr_reference_ca_mode=zero is diagnostic-only"
        )
    return torch.zeros_like(face_prompt_embeds), None, mode


def _record_reference_ca_fingerprint(
    pipeline,
    face_prompt_embeds: torch.Tensor,
    *,
    generation_batch_size: int,
    mode: str,
) -> None:
    if not bool(getattr(pipeline, "ba_ppr_collect_diagnostics", False)):
        return
    fingerprints = getattr(pipeline, "_ba_ppr_randomness_fingerprints", None)
    if not isinstance(fingerprints, dict):
        return
    sample_count = len(
        getattr(pipeline, "ba_ppr_diagnostic_sample_keys", ())
    )
    values = face_prompt_embeds
    if sample_count > 0 and generation_batch_size == 2 * sample_count:
        # CFG ordering is [unconditional B, conditional B].
        values = values[sample_count:]
    elif sample_count > 0:
        values = values[:sample_count]
    hashes, rms_values, nonzero_counts = [], [], []
    for sample in values.detach().contiguous().cpu():
        raw = sample.contiguous().view(torch.uint8).numpy().tobytes()
        hashes.append(hashlib.sha256(raw).hexdigest())
        rms_values.append(float(sample.float().square().mean().sqrt().item()))
        nonzero_counts.append(int(torch.count_nonzero(sample).item()))
    fingerprints["reference_ca_mode"] = [mode] * len(hashes)
    fingerprints["reference_ca_prompt_sha256"] = hashes
    fingerprints["reference_ca_prompt_rms"] = rms_values
    fingerprints["reference_ca_prompt_nonzero_count"] = nonzero_counts


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


def select_branched_self_attention_names(
    attn_processor_names: Sequence[str],
    policy: str,
) -> list[str]:
    """Select explicit self-attention sites for a processor variant."""
    policy = str(policy or "all").lower()
    if policy == "all":
        return [
            name for name in attn_processor_names
            if name.endswith("attn1.processor")
        ]
    if policy == "up_blocks_attn1":
        return [
            name for name in attn_processor_names
            if name.startswith("up_blocks.") and name.endswith("attn1.processor")
        ]
    if policy == "up_blocks0_attn1":
        return [
            name for name in attn_processor_names
            if name.startswith("up_blocks.0.") and name.endswith("attn1.processor")
        ]
    raise ValueError(f"Unknown ba_site_policy: {policy}")


def patch_unet_attention_processors(
    pipeline,
    mask: torch.Tensor,
    mask_ref: torch.Tensor,
    scale: float = 1.0,
    id_embeds: Optional[torch.Tensor] = None,
    class_tokens_mask: Optional[torch.Tensor] = None,
)-> None:
    """
    Patch UNet with branched attention processors for both self and cross attention.
    """
    disable_sa = bool(getattr(pipeline, "disable_branched_sa", False))
    disable_ca = bool(getattr(pipeline, "disable_branched_ca", False))

    # Default to legacy (v1) when flag is not provided.
    use_attn_v2 = bool(getattr(pipeline, "use_attn_v2", False))
    # if use_attn_v2:
    #     from ._old2.attn_processor2 import BranchedAttnProcessor, BranchedCrossAttnProcessor
    # else:
    #     # from .attn_processor import BranchedAttnProcessor, BranchedCrossAttnProcessor
    #     # from .attn_processor_clean import BranchedAttnProcessor, BranchedCrossAttnProcessor
    
    from .attn_processor_cleanest import BranchedAttnProcessor, BranchedCrossAttnProcessor # New ver 25 Feb
    from .packed_residual_attn_processor import (
        PackedResidualBranchedAttnProcessor,
        make_inner_core_mask,
    )

    variant = str(getattr(pipeline, "ba_processor_variant", "legacy") or "legacy").lower()
    if variant not in {"legacy", "packed_residual_v1"}:
        raise ValueError(f"Unknown ba_processor_variant: {variant}")
    site_policy = str(getattr(pipeline, "ba_site_policy", "all") or "all").lower()

    # print(f'[TEMP DEBUG] mask in patch_unet_attention_processors: {mask}')
    
    # Store original processors once
    if not hasattr(pipeline, '_original_attn_processors'):
        pipeline._original_attn_processors = {}
        for name, proc in pipeline.unet.attn_processors.items():
            pipeline._original_attn_processors[name] = proc
    
    # Check if already patched
    current_procs = pipeline.unet.attn_processors
    has_branched = any(
        bool(getattr(p, "_is_branched_processor", False))
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
        if isinstance(proc, PackedResidualBranchedAttnProcessor):
            runtime_scale = float(getattr(pipe, "ba_ppr_runtime_scale", 1.0))
            if not math.isfinite(runtime_scale) or runtime_scale < 0.0:
                raise ValueError(
                    f"ba_ppr_runtime_scale must be finite and non-negative, got {runtime_scale}"
                )
            proc.runtime_scale = runtime_scale
            proc.diagnostic_step = int(getattr(pipe, "_ba_ppr_step_idx", -1))
            proc.diagnostic_steps = tuple(
                int(step)
                for step in getattr(pipe, "ba_ppr_diagnostic_steps", (15, 25, 35, 49))
            )
            proc.diagnostic_variant = str(
                getattr(pipe, "ba_ppr_diagnostic_variant", "")
            )
            proc.diagnostic_sample_keys = tuple(
                str(value)
                for value in getattr(
                    pipe, "ba_ppr_diagnostic_sample_keys", ()
                )
            )
            proc.diagnostic_do_cfg = bool(
                getattr(pipe, "do_classifier_free_guidance", False)
            )
            proc.diagnostic_sink = getattr(
                pipe,
                "_ba_ppr_processor_diagnostics",
                None,
            )
            tensor_sites = tuple(
                getattr(pipe, "ba_ppr_tensor_diagnostic_sites", ())
            )
            proc.tensor_diagnostics = proc.processor_name in tensor_sites
            
        

   
    # Build safe, consistent context (batch, id_embeds)
    # Ensure masks are non-None to avoid runtime errors
    B = (mask.shape[0] if mask is not None else mask_ref.shape[0])
    dev, dt = pipeline.device, pipeline.unet.dtype
    _mask  = mask     if mask     is not None else torch.zeros(B, 1,  mask_ref.shape[-2], mask_ref.shape[-1], device=dev, dtype=dt)
    _mref  = mask_ref if mask_ref is not None else _mask
    _mcore = (
        make_inner_core_mask(
            _mask,
            erode_frac=float(getattr(pipeline, "ba_target_core_erode_frac", 0.10)),
        ).to(device=dev, dtype=dt)
        if variant == "packed_residual_v1"
        else None
    )
    # Always provide id_embeds so processor-local weights participate on every rank
    _idem = id_embeds.to(dev, dt) if id_embeds is not None else torch.zeros(B, 2048, device=dev, dtype=dt)   

    ba_patch_top_k = float(getattr(pipeline, "ba_patch_top_k", 1.0))
    all_processor_names = list(pipeline.unet.attn_processors.keys())
    if variant == "packed_residual_v1":
        if ba_patch_top_k != 1.0:
            raise ValueError(
                "packed_residual_v1 uses ba_site_policy and requires ba_patch_top_k=1.0"
            )
        patchable_sa_names = select_branched_self_attention_names(
            all_processor_names,
            site_policy,
        )
    else:
        patchable_sa_names = select_branched_processor_names(
            all_processor_names,
            include_self_attention=True,
            include_cross_attention=False,
            top_k=ba_patch_top_k,
            param_name="ba_patch_top_k",
        )
    patchable_sa_name_set = set(patchable_sa_names)

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
                    if variant == "packed_residual_v1":
                        proc = PackedResidualBranchedAttnProcessor(
                            hidden_size=hidden_size,
                            ref_kv_kind="lora",
                            ref_kv_rank=int(
                                getattr(
                                    pipeline,
                                    "branched_attn_lora_rank",
                                    getattr(pipeline, "lora_rank", 16),
                                )
                            ),
                            connector_rank=int(getattr(pipeline, "ba_connector_rank", 16)),
                            connector_input_mode=str(
                                getattr(
                                    pipeline,
                                    "ba_connector_input_mode",
                                    "reference_minus_target",
                                )
                            ),
                            null_memory_tokens=int(
                                getattr(pipeline, "ba_null_memory_tokens", 8)
                            ),
                            gate_max=float(getattr(pipeline, "ba_gate_max", 0.5)),
                            gate_init_logit=float(
                                getattr(pipeline, "ba_gate_init_logit", 0.0)
                            ),
                            delta_rms_cap=float(
                                getattr(pipeline, "ba_delta_rms_cap", 0.25)
                            ),
                            target_core_erode_frac=float(
                                getattr(pipeline, "ba_target_core_erode_frac", 0.10)
                            ),
                            collect_aux_losses=bool(
                                getattr(pipeline, "ba_collect_aux_losses", False)
                            ),
                            match_null_margin=float(
                                getattr(pipeline, "ba_match_null_margin", 0.02)
                            ),
                            cap_loss_target=float(
                                getattr(pipeline, "ba_cap_loss_target", 0.20)
                            ),
                            processor_name=name,
                            diagnostics=bool(getattr(pipeline, "ba_diagnostics", False)),
                        )
                    else:
                        proc = BranchedAttnProcessor(
                            hidden_size=hidden_size,
                            cross_attention_dim=hidden_size,
                            scale=scale,
                            branched_attn_weight_mode=getattr(pipeline, "branched_attn_weight_mode", "shared"),
                            branched_attn_new_weight_kind=getattr(pipeline, "branched_attn_new_weight_kind", "full"),
                            branched_attn_lora_rank=int(
                                getattr(pipeline, "branched_attn_lora_rank", getattr(pipeline, "lora_rank", 16))
                            ),
                            processor_name=name,
                            ba_sa_ref_token_mode=getattr(pipeline, "ba_sa_ref_token_mode", "full_grid"),
                            ba_sa_face_mode=getattr(pipeline, "ba_sa_face_mode", "reference"),
                            ba_sa_ref_layer_scope=getattr(pipeline, "ba_sa_ref_layer_scope", "all"),
                            ba_sa_roi_grid_size=int(getattr(pipeline, "ba_sa_roi_grid_size", 8)),
                            ba_sa_core_ratio=float(getattr(pipeline, "ba_sa_core_ratio", 0.7)),
                            ba_sa_mix_init=float(getattr(pipeline, "ba_sa_mix_init", 0.25)),
                        )
                    proc.init_from_attention(_resolve_attn_module(pipeline.unet, name))
                    proc = proc.to(pipeline.device, dtype=pipeline.unet.dtype)
                    if variant == "packed_residual_v1":
                        proc.set_masks(_mask, _mref, _mcore)
                    else:
                        proc.set_masks(_mask, _mref)
                    setattr(proc, "strict_face_routing", bool(getattr(pipeline, "strict_face_routing", False)))
                    _apply_runtime_flags(proc, pipeline)

                    # Wire id_embeds (zeros if missing); whether they are used is controlled by use_id_embeds
                    proc.id_embeds = _idem

                    new_procs[name] = proc
                    patched_proc_names.append(name)
                
            elif name.endswith("attn2.processor"):
                if disable_ca:
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
                    if variant == "legacy":
                        # Legacy no-op attributes retained for checkpoint parity.
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
        current_sa = [
            proc for proc in current_procs.values()
            if getattr(proc, "_branched_kind", None) == "self"
        ]
        expected_sa_class = (
            PackedResidualBranchedAttnProcessor
            if variant == "packed_residual_v1"
            else BranchedAttnProcessor
        )
        if current_sa and any(not isinstance(proc, expected_sa_class) for proc in current_sa):
            raise RuntimeError(
                f"Live branched processors do not match ba_processor_variant={variant}; "
                "processor reconstruction during a run is forbidden"
            )
        patched_proc_names: list[str] = []
        # Update masks on existing processors
        for name, proc in pipeline.unet.attn_processors.items():
            if bool(getattr(proc, "_is_branched_processor", False)):
                patched_proc_names.append(name)
                if isinstance(proc, PackedResidualBranchedAttnProcessor):
                    proc.set_masks(_mask, _mref, _mcore)
                else:
                    proc.set_masks(_mask, _mref)
                _apply_runtime_flags(proc, pipeline)

                # (Re)apply id_embeds (zeros if missing); actual usage is gated by use_id_embeds
                if hasattr(proc, "id_embeds"):
                    proc.id_embeds = _idem
                if hasattr(proc, "class_tokens_mask"):
                    proc.class_tokens_mask = class_tokens_mask
        setattr(pipeline, "_ba_patched_processor_names", tuple(patched_proc_names))

    patched_items = {
        name: proc
        for name, proc in pipeline.unet.attn_processors.items()
        if bool(getattr(proc, "_is_branched_processor", False))
    }
    patched_sa = sorted(
        name for name, proc in patched_items.items()
        if getattr(proc, "_branched_kind", None) == "self"
    )
    patched_ca = sorted(
        name for name, proc in patched_items.items()
        if getattr(proc, "_branched_kind", None) == "cross"
    )
    pipeline._ba_patched_sa_names = tuple(patched_sa)
    pipeline._ba_patched_ca_names = tuple(patched_ca)
    pipeline._ba_processor_object_ids = {
        name: id(proc) for name, proc in sorted(patched_items.items())
    }
    if not hasattr(pipeline, "_ba_processor_object_ids_initial"):
        pipeline._ba_processor_object_ids_initial = dict(
            pipeline._ba_processor_object_ids
        )
    if not bool(getattr(pipeline, "_ba_architecture_logged", False)):
        print(
            "[BA processor install] "
            f"variant={variant} site_policy={site_policy} "
            f"SA={len(patched_sa)} CA={len(patched_ca)}"
        )
        print(f"[BA processor install] SA names: {', '.join(patched_sa)}")
        print(f"[BA processor install] CA names: {', '.join(patched_ca)}")
        pipeline._ba_architecture_logged = True

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

    def _repeat_batch(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
        return tensor.repeat((int(repeats),) + (1,) * (tensor.ndim - 1))

    def _match_generation_batch(
        tensor: Optional[torch.Tensor],
        target_batch: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        cur_batch = int(tensor.shape[0])
        if cur_batch == target_batch:
            return tensor
        if cur_batch <= 0 or target_batch % cur_batch != 0:
            raise RuntimeError(
                f"{name} batch={cur_batch} is incompatible with generation batch={target_batch}"
            )
        return _repeat_batch(tensor, target_batch // cur_batch)

    def _match_reference_batch(
        tensor: Optional[torch.Tensor],
        target_batch: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        cur_batch = int(tensor.shape[0])
        if cur_batch == target_batch:
            return tensor
        if cur_batch > 0 and target_batch % cur_batch == 0:
            return _repeat_batch(tensor, target_batch // cur_batch)
        raise RuntimeError(
            f"{name} batch={cur_batch} is incompatible with reference batch={target_batch}"
        )

    # CFG doubles latent_model_input ([uncond, cond]) while masks/references are
    # prepared once per output image. NN4 caches reference noise at the base
    # image batch and duplicates it exactly across CFG. The opt-in toggle keeps
    # old experiment configs reproducible.
    pair_cfg_reference_noise = bool(
        getattr(pipeline, "ba_cfg_reference_noise_pairing", False)
    )
    do_cfg = bool(getattr(pipeline, "do_classifier_free_guidance", False))
    if pair_cfg_reference_noise and do_cfg:
        if batch_size % 2:
            raise RuntimeError(
                "CFG reference-noise pairing requires an even generation batch"
            )
        reference_batch_size = batch_size // 2
    else:
        reference_batch_size = batch_size

    mask4 = _match_generation_batch(mask4, batch_size, "mask4")
    reference_latents_base = _match_reference_batch(
        reference_latents,
        reference_batch_size,
        "reference_latents",
    )
    mask4_ref_base = _match_reference_batch(
        mask4_ref,
        reference_batch_size,
        "mask4_ref",
    )
    
    
    REF_NOISE_ONCE = True  # keep same ref noise across steps within one generation
    noise_cache_name = (
        "_ref_noise_base" if pair_cfg_reference_noise else "_ref_noise"
    )
    cached_noise = getattr(pipeline, noise_cache_name, None)
    # Setup-time diagnostic/reference noise historically uses `_ref_noise`.
    # Adopt it as the base cache when its shape is already correct.
    if (
        pair_cfg_reference_noise
        and cached_noise is None
        and hasattr(pipeline, "_ref_noise")
        and tuple(pipeline._ref_noise.shape)
        == tuple(reference_latents_base.shape)
    ):
        cached_noise = pipeline._ref_noise
        setattr(pipeline, noise_cache_name, cached_noise)

    if (
        cached_noise is None
        or tuple(cached_noise.shape) != tuple(reference_latents_base.shape)
    ):
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
                cached_noise = torch.randn_like(
                    reference_latents_base,
                    generator=ref_gen,
                )
            except TypeError:
                cached_noise = torch.randn(
                    reference_latents_base.shape,
                    generator=ref_gen,
                    device=reference_latents_base.device,
                    dtype=reference_latents_base.dtype,
                )
        else:
            # IMPORTANT: don't use a fresh unseeded torch.Generator() (it’s deterministic); use global RNG instead.
            cached_noise = torch.randn_like(reference_latents_base)
        setattr(pipeline, noise_cache_name, cached_noise)

    if pair_cfg_reference_noise:
        # Keep the historical name available to diagnostics, but never compare
        # it against the expanded CFG shape.
        pipeline._ref_noise = cached_noise
    if pair_cfg_reference_noise and do_cfg:
        reference_latents = torch.cat(
            [reference_latents_base, reference_latents_base],
            dim=0,
        )
        mask4_ref = torch.cat([mask4_ref_base, mask4_ref_base], dim=0)
        reference_noise = torch.cat([cached_noise, cached_noise], dim=0)
    else:
        reference_latents = reference_latents_base
        mask4_ref = mask4_ref_base
        reference_noise = cached_noise


    
    t_gen = t if torch.is_tensor(t) else torch.tensor([t], device=device, dtype=torch.long)
    if t_gen.ndim == 0:
        t_gen = t_gen.unsqueeze(0)
    if t_gen.shape[0] != batch_size:
        reps = (batch_size + t_gen.shape[0] - 1) // t_gen.shape[0]
        t_gen = t_gen.repeat(reps)[:batch_size]
    t_ref = t_gen
    
    ref_noised = pipeline.scheduler.add_noise(
        reference_latents,
        reference_noise,
        t_ref
    )

    
    ref_noised = pipeline.scheduler.scale_model_input(ref_noised, t_ref).to(latent_model_input.dtype) # critical: match UNet’s expected scaling at this timestep
    if pair_cfg_reference_noise and do_cfg:
        half = reference_batch_size
        if not torch.equal(reference_noise[:half], reference_noise[half:]):
            raise RuntimeError(
                "CFG unconditional/conditional reference noise differs"
            )
        if not torch.equal(ref_noised[:half], ref_noised[half:]):
            raise RuntimeError(
                "CFG unconditional/conditional noised reference differs"
            )

    diagnostic_steps = tuple(
        int(value)
        for value in getattr(
            pipeline,
            "ba_ppr_diagnostic_steps",
            (15, 25, 35, 49),
        )
    )
    if (
        bool(getattr(pipeline, "ba_ppr_collect_diagnostics", False))
        and int(step_idx) in diagnostic_steps
    ):
        fingerprints = getattr(
            pipeline,
            "_ba_ppr_randomness_fingerprints",
            None,
        )
        if isinstance(fingerprints, dict):
            if pair_cfg_reference_noise and do_cfg:
                half = reference_batch_size
                fingerprints["cfg_reference_noise_equal"] = bool(
                    torch.equal(
                        reference_noise[:half],
                        reference_noise[half:],
                    )
                )
                fingerprints["cfg_ref_noised_equal"] = bool(
                    torch.equal(
                        ref_noised[:half],
                        ref_noised[half:],
                    )
                )
                fingerprints["reference_noise_used_sha256"] = [
                    hashlib.sha256(
                        sample.detach()
                        .contiguous()
                        .cpu()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    ).hexdigest()
                    for sample in reference_noise
                ]
            sample_count = len(
                getattr(
                    pipeline,
                    "ba_ppr_diagnostic_sample_keys",
                    (),
                )
            )
            ref_noised_for_hash = ref_noised
            if (
                sample_count > 0
                and ref_noised.shape[0] == 2 * sample_count
            ):
                # CFG ordering is [unconditional B, conditional B].
                ref_noised_for_hash = ref_noised[sample_count:]
            hashes = []
            for sample in ref_noised_for_hash.detach().contiguous().cpu():
                raw = sample.contiguous().view(torch.uint8).numpy().tobytes()
                hashes.append(hashlib.sha256(raw).hexdigest())
            fingerprints[f"ref_noised_step_{int(step_idx)}_sha256"] = hashes

    if full_debug:
        if step_idx in (0, 1) or step_idx % 10 == 0:
            print(f"[2BP]   ref_noised:  {stat(ref_noised)}  Δ(noise,ref)σ={(latent_model_input.std()-ref_noised.std()).item():.4f}")

    
    # Create branched batch: [generation B, reference B].
    batched_latents = torch.cat([latent_model_input, ref_noised], dim=0)
    
    # Patch processors with masks
    pipeline._ba_ppr_step_idx = int(step_idx)
    patch_unet_attention_processors(
        pipeline, mask4, mask4_ref, scale,
        id_embeds=id_embeds if face_embed_strategy == "id_embeds" else None,
        class_tokens_mask=class_tokens_mask,
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
    t_batched = torch.cat([t_gen, t_ref], dim=0)
    
    # Prepare face prompt if not provided
    if face_prompt_embeds is None:
        face_prompt_embeds = encode_face_prompt(
            pipeline, 
            device, 
            batch_size,
            pipeline.do_classifier_free_guidance
        )

    
    face_prompt_attention_mask = None

    # Only mirror the main text into the face branch for legacy "id".
    # For "id_embeds" we keep actual "face" text and use the 2048-D ID features.
    if (face_embed_strategy or "face") in {"id"}:
        # keep dtype/device aligned with text encoder / UNet
        d, dev = prompt_embeds.dtype, prompt_embeds.device
        if not bool(
            getattr(pipeline, "ba_reference_ca_preserve_full_pm", False)
        ):
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
            token_mask = m.to(dtype=torch.bool)
            m = token_mask.unsqueeze(-1).to(dtype=d)       # [B,L,1]
            one = torch.tensor(1.0, device=dev, dtype=d)
            id_scale = torch.tensor(getattr(pipeline, "id_token_scale", 2.5),
                                   device=dev, dtype=d)
            ##### BRANCHED ATTENTION - FACE PROMPT MODE (B1) #####
            # "id_only" (legacy, default): zero every token except the ID tokens.
            #   Side effect: ~75/77 zero K/V tokens act as attention sinks in the
            #   ref branch cross-attention.
            # "full_boosted": keep the full fused prompt and boost the ID tokens
            #   (the pre-Feb-18 known-good variant).
            face_prompt_mode = str(getattr(pipeline, "ba_face_prompt_mode", "id_only") or "id_only").lower()
            use_face_prompt_attention_mask = bool(
                getattr(pipeline, "ba_face_prompt_attention_mask", False)
            )
            if use_face_prompt_attention_mask and face_prompt_mode != "id_only":
                raise ValueError(
                    "ba_face_prompt_attention_mask requires ba_face_prompt_mode=id_only"
                )
            if face_prompt_mode == "full_boosted":
                masked_face_prompt_embeds = (
                    face_prompt_embeds * (one - m) + face_prompt_embeds * m * id_scale
                )
            else:
                # Use only ID tokens for the face branch (no leakage from other words)
                masked_face_prompt_embeds = face_prompt_embeds * m * id_scale
            ##### BRANCHED ATTENTION - FACE PROMPT MODE (B1) #####

            ##### BRANCHED ATTENTION - UNCOND FACE PROMPT FIX (F1) #####
            # Legacy behavior also masks the NEGATIVE-prompt half with the
            # POSITIVE prompt's ID-token positions, which feeds garbage uncond
            # conditioning into the face branch; CFG then extrapolates
            # uncond + gs*(cond-uncond) in the face region. Training never sees
            # this pathway (no CFG at training). With ba_uncond_face_fix=True
            # the uncond half keeps the plain negative-prompt embeds and only
            # the cond half gets the ID-token masking.
            uncond_fix = bool(getattr(pipeline, "ba_uncond_face_fix", False))
            do_cfg = bool(getattr(pipeline, "do_classifier_free_guidance", False))
            fp_batch = int(face_prompt_embeds.shape[0])
            if (uncond_fix or use_face_prompt_attention_mask) and do_cfg and fp_batch % 2 == 0:
                half = fp_batch // 2
                face_prompt_embeds = torch.cat(
                    [face_prompt_embeds[:half], masked_face_prompt_embeds[half:]],
                    dim=0,
                )
                if use_face_prompt_attention_mask:
                    token_mask = token_mask.clone()
                    token_mask[:half] = True
            else:
                face_prompt_embeds = masked_face_prompt_embeds
            if use_face_prompt_attention_mask:
                if not bool(token_mask.any(dim=1).all()):
                    raise RuntimeError("Every face-prompt attention row must allow at least one token")
                face_prompt_attention_mask = token_mask
            ##### BRANCHED ATTENTION - UNCOND FACE PROMPT FIX (F1) #####

        else:
            if bool(getattr(pipeline, "ba_face_prompt_attention_mask", False)):
                raise RuntimeError(
                    "ba_face_prompt_attention_mask requires class_tokens_mask"
                )
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
    reference_token_text_mode = str(
        getattr(pipeline, "ba_reference_token_text_mode", "original")
        or "original"
    ).lower()
    if reference_token_text_mode == "zero":
        face_prompt_embeds = torch.zeros_like(face_prompt_embeds)
        face_prompt_attention_mask = None
    elif reference_token_text_mode != "original":
        raise ValueError(
            "ba_reference_token_text_mode must be one of: original, zero"
        )
    face_prompt_embeds, face_prompt_attention_mask, reference_ca_mode = (
        apply_ppr_reference_ca_mode(
            pipeline,
            face_prompt_embeds,
            face_prompt_attention_mask,
        )
    )
    if reference_token_text_mode != "original":
        reference_ca_mode = (
            reference_token_text_mode
            if reference_ca_mode == "original"
            else f"{reference_token_text_mode}+ppr_{reference_ca_mode}"
        )
    _record_reference_ca_fingerprint(
        pipeline,
        face_prompt_embeds,
        generation_batch_size=batch_size,
        mode=reference_ca_mode,
    )

    encoder_hidden_states = torch.cat([prompt_embeds, face_prompt_embeds], dim=0)

    if full_debug:
        # quick sanity – these should *not* be identical
        if (step_idx in (0, 1)) or (step_idx % 10 == 0):
            diff_mu = (prompt_embeds.detach().float() - face_prompt_embeds.detach().float()).abs().mean().item()
            print(f"[2BP]   encoder_hidden_states Δ(gen,face)μ={diff_mu:.4f}")


    doubled_kwargs = {}
    reference_pooled_text_mode = str(
        getattr(pipeline, "ba_reference_pooled_text_mode", "target")
        or "target"
    ).lower()
    if reference_pooled_text_mode not in {"target", "zero"}:
        raise ValueError(
            "ba_reference_pooled_text_mode must be one of: target, zero"
        )
    for k, v in added_cond_kwargs.items():
        if torch.is_tensor(v):
            if v.shape[0] == batch_size:
                reference_value = (
                    torch.zeros_like(v)
                    if k == "text_embeds"
                    and reference_pooled_text_mode == "zero"
                    else v
                )
                doubled_kwargs[k] = torch.cat([v, reference_value], dim=0)
            else:
                doubled_kwargs[k] = v
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
    
    # Store the explicit reference-token mask on the persistent CA processors.
    # This avoids passing unknown kwargs through diffusers while keeping the
    # target-half generation attention untouched.
    for name, processor in pipeline.unet.attn_processors.items():
        if (
            name.endswith("attn2.processor")
            and processor.__class__.__name__ == "BranchedCrossAttnProcessor"
        ):
            processor.face_prompt_attention_mask = face_prompt_attention_mask

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
    B2 = batch_size
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

        # Mean cosine sim between branches → should NOT be ~1.0
        second_for_debug = second[: first.shape[0]]
        cos = torch.nn.functional.cosine_similarity(first.flatten(1), second_for_debug.flatten(1), dim=1).mean().item()
        print(f"[2BP]   cos(first,second)={cos:.3f}")
    # --- end of quick check



    
    # Extract merged result (first half)
    noise_pred_merged = noise_pred[:batch_size]

    output_anchor_mode = str(
        getattr(pipeline, "ba_output_anchor_mode", "none") or "none"
    ).lower()
    force_base_output = bool(
        getattr(pipeline, "ba_ppr_force_base_output", False)
    )
    collect_diagnostics = bool(
        getattr(pipeline, "ba_ppr_collect_diagnostics", False)
    )
    needs_base_prediction = (
        output_anchor_mode == "base_outside_core"
        or force_base_output
        or collect_diagnostics
    )
    base_noise_pred = None
    core_mask = None
    pre_anchor_noise_pred = noise_pred_merged
    if needs_base_prediction:
        original_processors = getattr(pipeline, "_original_attn_processors", None)
        if not original_processors:
            raise RuntimeError(
                "PPR output control requires the original attention-processor registry"
            )

        branched_processors = dict(pipeline.unet.attn_processors)
        try:
            pipeline.unet.set_attn_processor(dict(original_processors))
            with torch.no_grad():
                base_noise_pred = pipeline.unet(
                    latent_model_input,
                    t_gen,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=(
                        runtime_cross_attention_kwargs
                        if runtime_cross_attention_kwargs
                        else None
                    ),
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
        finally:
            # Diffusers consumes this mapping with pop(); pass a copy because
            # the preserved registry is also inspected below.
            pipeline.unet.set_attn_processor(dict(branched_processors))

        from .packed_residual_attn_processor import make_inner_core_mask

        core_mask = make_inner_core_mask(
            mask4,
            erode_frac=float(
                getattr(pipeline, "ba_target_core_erode_frac", 0.10)
            ),
        ).to(device=noise_pred_merged.device, dtype=noise_pred_merged.dtype)
        if core_mask.shape[-2:] != noise_pred_merged.shape[-2:]:
            core_mask = F.interpolate(
                core_mask.float(),
                size=noise_pred_merged.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(noise_pred_merged.dtype)

    anchor_state = "none"
    if force_base_output:
        # Diagnostic A: return the actual ordinary single-target prediction.
        # The doubled call above is intentionally excluded from the scheduler
        # update even when its processor-local residual is zero.
        noise_pred_merged = base_noise_pred
        anchor_state = "diagnostic-force-base"
    elif output_anchor_mode == "base_outside_core":
        branch_is_exactly_off = False
        if not torch.is_grad_enabled():
            cached_branch_state = getattr(
                pipeline,
                "_ba_packed_branch_exactly_off",
                None,
            )
            if cached_branch_state is None:
                packed_processors = [
                    processor
                    for processor in branched_processors.values()
                    if processor.__class__.__name__
                    == "PackedResidualBranchedAttnProcessor"
                ]
                cached_branch_state = bool(packed_processors) and all(
                    int(torch.count_nonzero(processor.connector_up.weight).item()) == 0
                    for processor in packed_processors
                )
                pipeline._ba_packed_branch_exactly_off = cached_branch_state
            branch_is_exactly_off = bool(cached_branch_state)
        if branch_is_exactly_off:
            # A separate ordinary-batch pass is required for end-to-end BF16
            # parity: a doubled U-Net call is numerically different even when
            # every processor-local residual is exactly zero.
            noise_pred_merged = base_noise_pred
            anchor_state = "exact-zero-bypass"
        else:
            noise_pred_merged = base_noise_pred + core_mask * (
                noise_pred_merged - base_noise_pred
            )
            anchor_state = "base-outside-core"

    elif output_anchor_mode != "none":
        raise ValueError(f"Unknown ba_output_anchor_mode: {output_anchor_mode}")

    if force_base_output or output_anchor_mode == "base_outside_core":
        if not bool(getattr(pipeline, "_ba_output_anchor_logged", False)):
            print(
                "[BA output anchor] "
                f"mode={output_anchor_mode} state={anchor_state} "
                f"core_mean={core_mask.float().mean().item():.6f}"
            )
            pipeline._ba_output_anchor_logged = True

    diagnostic_steps = tuple(
        int(step)
        for step in getattr(
            pipeline,
            "ba_ppr_diagnostic_steps",
            (15, 25, 35, 49),
        )
    )
    if collect_diagnostics and int(step_idx) in diagnostic_steps:
        def _apply_cfg(prediction: torch.Tensor) -> torch.Tensor:
            if not bool(getattr(pipeline, "do_classifier_free_guidance", False)):
                return prediction.float()
            if prediction.shape[0] % 2 != 0:
                raise RuntimeError(
                    f"PPR diagnostics expected an even CFG batch, got {prediction.shape[0]}"
                )
            unconditional, conditional = prediction.float().chunk(2)
            guidance_scale = float(getattr(pipeline, "guidance_scale", 1.0))
            return unconditional + guidance_scale * (conditional - unconditional)

        def _masked_ratio(
            prediction: torch.Tensor,
            base: torch.Tensor,
            region: torch.Tensor,
        ) -> torch.Tensor:
            region = region.float()
            if region.shape[-2:] != prediction.shape[-2:]:
                region = F.interpolate(
                    region,
                    size=prediction.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if region.shape[0] != prediction.shape[0]:
                region = region[: prediction.shape[0]]
            channels = float(prediction.shape[1])
            count = (region.sum(dim=(1, 2, 3)) * channels).clamp_min(1.0)
            difference_rms = torch.sqrt(
                (
                    region
                    * (prediction.float() - base.float()).square()
                ).sum(dim=(1, 2, 3))
                / count
            )
            base_rms = torch.sqrt(
                (region * base.float().square()).sum(dim=(1, 2, 3)) / count
                + 1e-12
            )
            return difference_rms / (base_rms + 1e-12)

        base_cfg = _apply_cfg(base_noise_pred)
        pre_cfg = _apply_cfg(pre_anchor_noise_pred)
        post_cfg = _apply_cfg(noise_pred_merged)
        output_batch = int(base_cfg.shape[0])
        core_cfg = core_mask[:output_batch]
        bbox_cfg = (mask4[:output_batch] > 0.5).to(core_cfg.dtype)
        outside_bbox = 1.0 - bbox_cfg
        pre_core = _masked_ratio(pre_cfg, base_cfg, core_cfg)
        pre_outside = _masked_ratio(pre_cfg, base_cfg, outside_bbox)
        post_core = _masked_ratio(post_cfg, base_cfg, core_cfg)
        post_outside = _masked_ratio(post_cfg, base_cfg, outside_bbox)
        sample_keys = list(
            getattr(pipeline, "ba_ppr_diagnostic_sample_keys", ())
        )
        if len(sample_keys) != output_batch:
            sample_keys = [str(index) for index in range(output_batch)]
        sink = getattr(pipeline, "_ba_ppr_epsilon_diagnostics", None)
        if isinstance(sink, list):
            timestep_value = (
                int(t.flatten()[0].item())
                if torch.is_tensor(t)
                else int(t)
            )
            variant = str(
                getattr(pipeline, "ba_ppr_diagnostic_variant", "")
            )
            for sample_index, sample_key in enumerate(sample_keys):
                sink.append(
                    {
                        "record_type": "epsilon_ratio",
                        "variant": variant,
                        "sample": sample_key,
                        "step": int(step_idx),
                        "timestep": timestep_value,
                        "output_control": anchor_state,
                        "inside_core_pre_anchor": float(pre_core[sample_index].item()),
                        "outside_bbox_pre_anchor": float(pre_outside[sample_index].item()),
                        "inside_core_post_anchor": float(post_core[sample_index].item()),
                        "outside_bbox_post_anchor": float(post_outside[sample_index].item()),
                    }
                )
            tensor_sink = getattr(
                pipeline,
                "_ba_ppr_tensor_diagnostics",
                None,
            )
            if isinstance(tensor_sink, list):
                def _tensor_signature(value: torch.Tensor) -> dict[str, Any]:
                    value = value.detach().float().contiguous()
                    flat = value.flatten()
                    stride = max(1, math.ceil(flat.numel() / 512))
                    sketch = flat[::stride][:512].cpu()
                    raw = value.cpu().view(torch.uint8).numpy().tobytes()
                    return {
                        "shape": list(value.shape),
                        "rms": float(value.square().mean().sqrt().item()),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "sketch_stride": stride,
                        "sketch": sketch.tolist(),
                    }

                for sample_index, sample_key in enumerate(sample_keys):
                    tensor_sink.append(
                        {
                            "record_type": "epsilon_tensor_signature",
                            "variant": variant,
                            "sample": sample_key,
                            "step": int(step_idx),
                            "timestep": timestep_value,
                            "target_epsilon_pre_anchor": _tensor_signature(
                                pre_cfg[sample_index]
                            ),
                            "target_epsilon_post_anchor": _tensor_signature(
                                post_cfg[sample_index]
                            ),
                        }
                    )
    
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
