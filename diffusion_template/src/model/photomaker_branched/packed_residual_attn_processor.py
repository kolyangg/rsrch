"""Parity-preserving packed-reference self-attention for NN2-PPR1."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_processor_cleanest import _branch_batch_sizes, _clone_effective_linear


def _as_mask4(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        return mask[:, :1].float()
    batch = int(mask.shape[0])
    flat = mask.reshape(batch, -1).float()
    side = int(math.isqrt(flat.shape[1]))
    if side * side != flat.shape[1]:
        raise ValueError(f"Mask length {flat.shape[1]} is not square")
    return flat.reshape(batch, 1, side, side)


def make_inner_core_mask(
    mask4: torch.Tensor,
    erode_frac: float = 0.10,
) -> torch.Tensor:
    """Build a cosine-feathered inner core from a hard target bbox mask."""
    if not 0.0 <= float(erode_frac) < 0.5:
        raise ValueError(f"erode_frac must be in [0, 0.5), got {erode_frac}")
    source = _as_mask4(mask4)
    hard = source[:, 0] > 0.5
    core = torch.zeros_like(source)
    for sample_idx in range(hard.shape[0]):
        coords = hard[sample_idx].nonzero(as_tuple=False)
        if coords.numel() == 0:
            continue
        y0_tensor, x0_tensor = coords.amin(dim=0)
        y1_tensor, x1_tensor = coords.amax(dim=0) + 1
        y0, x0 = int(y0_tensor.item()), int(x0_tensor.item())
        y1, x1 = int(y1_tensor.item()), int(x1_tensor.item())
        height = y1 - y0
        width = x1 - x0
        ry = max(1, round(float(erode_frac) * height))
        rx = max(1, round(float(erode_frac) * width))

        ys = torch.arange(y0, y1, device=source.device, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=source.device, dtype=torch.float32)
        dy = torch.minimum(ys - y0, (y1 - 1) - ys)
        dx = torch.minimum(xs - x0, (x1 - 1) - xs)
        wy = 0.5 - 0.5 * torch.cos(
            math.pi * (dy / float(ry)).clamp(0.0, 1.0)
        )
        wx = 0.5 - 0.5 * torch.cos(
            math.pi * (dx / float(rx)).clamp(0.0, 1.0)
        )
        feather = torch.minimum(wy[:, None], wx[None, :])
        feather = feather * hard[sample_idx, y0:y1, x0:x1].float()
        core[sample_idx, 0, y0:y1, x0:x1] = feather
    return core


def pack_valid_tokens(
    hidden_states: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack valid per-sample tokens and return an additive padding mask."""
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected [B,L,D] hidden states, got {hidden_states.shape}")
    valid = valid.reshape(hidden_states.shape[0], hidden_states.shape[1]).bool()
    lengths = valid.sum(dim=1)
    max_length = max(int(lengths.max().item()), 1)
    packed = hidden_states.new_zeros(
        hidden_states.shape[0],
        max_length,
        hidden_states.shape[2],
    )
    additive_mask = hidden_states.new_full(
        (hidden_states.shape[0], 1, 1, max_length),
        float("-inf"),
    )
    sample_has_roi = lengths > 0
    for sample_idx in range(hidden_states.shape[0]):
        length = int(lengths[sample_idx].item())
        if length:
            packed[sample_idx, :length] = hidden_states[sample_idx, valid[sample_idx]]
            additive_mask[sample_idx, :, :, :length] = 0
        else:
            # Keep SDPA finite. sample_has_roi forces this row's final residual to zero.
            additive_mask[sample_idx, :, :, 0] = 0
    return packed, lengths, additive_mask, sample_has_roi


class PackedResidualBranchedAttnProcessor(nn.Module):
    """Ordinary doubled self-attention plus a bounded reference-face residual."""

    _is_branched_processor = True
    _branched_kind = "self"

    def __init__(
        self,
        hidden_size: int,
        *,
        ref_kv_kind: str = "lora",
        ref_kv_rank: int = 32,
        connector_rank: int = 16,
        gate_max: float = 0.5,
        gate_init_logit: float = 0.0,
        delta_rms_cap: float = 0.25,
        target_core_erode_frac: float = 0.10,
        processor_name: str = "",
        diagnostics: bool = False,
    ):
        super().__init__()
        if ref_kv_kind != "lora":
            raise ValueError("packed_residual_v1 requires LoRA reference K/V")
        if connector_rank <= 0:
            raise ValueError("connector_rank must be positive")
        if not 0.0 < gate_max <= 1.0:
            raise ValueError("gate_max must be in (0, 1]")
        if not 0.0 < delta_rms_cap <= 1.0:
            raise ValueError("delta_rms_cap must be in (0, 1]")
        if not 0.0 <= target_core_erode_frac < 0.5:
            raise ValueError("target_core_erode_frac must be in [0, 0.5)")

        self.hidden_size = int(hidden_size)
        self.ref_kv_kind = ref_kv_kind
        self.ref_kv_rank = int(ref_kv_rank)
        self.connector_rank = int(connector_rank)
        self.gate_max = float(gate_max)
        self.gate_init_logit = float(gate_init_logit)
        self.delta_rms_cap = float(delta_rms_cap)
        self.target_core_erode_frac = float(target_core_erode_frac)
        self.processor_name = str(processor_name)
        self.diagnostics = bool(diagnostics)
        self.has_cross_attention_kwargs = True

        self.ref_to_k: Optional[nn.Module] = None
        self.ref_to_v: Optional[nn.Module] = None
        self.connector_down: Optional[nn.Linear] = None
        self.connector_up: Optional[nn.Linear] = None
        self.gate_logit: Optional[nn.Parameter] = None
        self.mask: Optional[torch.Tensor] = None
        self.mask_ref: Optional[torch.Tensor] = None
        self.mask_core: Optional[torch.Tensor] = None
        self.last_diagnostics: dict[str, torch.Tensor | float | int] = {}
        self._diagnostic_calls = 0
        self.runtime_scale = 1.0
        self.diagnostic_step = -1
        self.diagnostic_steps = (15, 25, 35, 49)
        self.diagnostic_variant = ""
        self.diagnostic_sink = None

    def init_from_attention(self, attn) -> None:
        self.ref_to_k = _clone_effective_linear(
            attn.to_k,
            kind=self.ref_kv_kind,
            rank=self.ref_kv_rank,
        )
        self.ref_to_v = _clone_effective_linear(
            attn.to_v,
            kind=self.ref_kv_kind,
            rank=self.ref_kv_rank,
        )
        base_q = attn.to_q.get_base_layer() if hasattr(attn.to_q, "get_base_layer") else attn.to_q
        projection_dim = int(base_q.out_features)
        if projection_dim != self.hidden_size:
            raise ValueError(
                f"{self.processor_name}: configured hidden_size={self.hidden_size}, "
                f"attention projection={projection_dim}"
            )
        self.connector_down = nn.Linear(
            projection_dim,
            self.connector_rank,
            bias=False,
            device=base_q.weight.device,
            dtype=base_q.weight.dtype,
        )
        self.connector_up = nn.Linear(
            self.connector_rank,
            projection_dim,
            bias=False,
            device=base_q.weight.device,
            dtype=base_q.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.connector_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.connector_up.weight)
        self.gate_logit = nn.Parameter(
            torch.tensor(
                self.gate_init_logit,
                device=base_q.weight.device,
                dtype=base_q.weight.dtype,
            )
        )

    def set_masks(
        self,
        mask: Optional[torch.Tensor],
        mask_ref: Optional[torch.Tensor],
        mask_core: Optional[torch.Tensor] = None,
    ) -> None:
        self.mask = mask
        self.mask_ref = mask_ref
        self.mask_core = mask_core

    @staticmethod
    def _to_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, channels = tensor.shape
        return tensor.view(batch, length, heads, channels // heads).transpose(1, 2)

    @staticmethod
    def _from_heads(tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, length, channels = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, length, heads * channels)

    @staticmethod
    def _apply_q_norm(attn, query: torch.Tensor) -> torch.Tensor:
        norm = getattr(attn, "norm_q", None)
        return norm(query) if norm is not None else query

    @staticmethod
    def _apply_k_norm(attn, key: torch.Tensor) -> torch.Tensor:
        norm = getattr(attn, "norm_k", None)
        return norm(key) if norm is not None else key

    @staticmethod
    def _resize_mask(
        mask: torch.Tensor,
        *,
        target_length: int,
        batch_size: int,
        mode: str,
        binary: bool,
    ) -> torch.Tensor:
        side = int(math.isqrt(target_length))
        if side * side != target_length:
            raise ValueError(f"Attention length {target_length} is not square")
        source = _as_mask4(mask)
        kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
        resized = F.interpolate(source, size=(side, side), mode=mode, **kwargs)
        if resized.shape[0] != batch_size:
            repeats = (batch_size + resized.shape[0] - 1) // resized.shape[0]
            resized = resized.repeat(repeats, 1, 1, 1)[:batch_size]
        if binary:
            resized = resized > 0.5
        return resized.flatten(2).transpose(1, 2)

    def _base_self_attention_pre_out(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        prepared_mask = attention_mask
        if prepared_mask is not None:
            prepared_mask = attn.prepare_attention_mask(
                prepared_mask,
                sequence_length,
                batch_size,
            )
            prepared_mask = prepared_mask.view(
                batch_size,
                attn.heads,
                -1,
                prepared_mask.shape[-1],
            )

        query = self._to_heads(attn.to_q(hidden_states), attn.heads)
        key = self._to_heads(attn.to_k(hidden_states), attn.heads)
        value = self._to_heads(attn.to_v(hidden_states), attn.heads)
        query = self._apply_q_norm(attn, query)
        key = self._apply_k_norm(attn, key)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=prepared_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return self._from_heads(output).to(query.dtype), query

    @staticmethod
    def _masked_rms_cap(
        delta: torch.Tensor,
        *,
        base: torch.Tensor,
        mask: torch.Tensor,
        max_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_fp32 = mask.float()
        base_fp32 = base.float()
        delta_fp32 = delta.float()
        element_count = mask_fp32.sum(dim=(1, 2)) * float(base.shape[-1])
        denominator = element_count.clamp_min(1.0)
        base_rms = torch.sqrt(
            (mask_fp32 * base_fp32.square()).sum(dim=(1, 2)) / denominator + 1e-12
        )
        delta_rms = torch.sqrt(
            (mask_fp32 * delta_fp32.square()).sum(dim=(1, 2)) / denominator + 1e-12
        )
        has_core = element_count > 0
        cap_scale = (
            float(max_ratio) * base_rms / (delta_rms + 1e-12)
        ).clamp(max=1.0)
        cap_scale = torch.where(has_core, cap_scale, torch.ones_like(cap_scale)).detach()
        bounded = delta_fp32 * cap_scale[:, None, None]
        post_rms = torch.sqrt(
            (mask_fp32 * bounded.square()).sum(dim=(1, 2)) / denominator + 1e-12
        )
        pre_ratio = torch.where(
            has_core,
            delta_rms / (base_rms + 1e-12),
            torch.zeros_like(delta_rms),
        )
        post_ratio = torch.where(
            has_core,
            post_rms / (base_rms + 1e-12),
            torch.zeros_like(post_rms),
        )
        return bounded.to(delta.dtype), cap_scale, pre_ratio, post_ratio

    def _record_diagnostics(
        self,
        *,
        lengths: torch.Tensor,
        packed_length: int,
        gate: torch.Tensor,
        cap_scale: torch.Tensor,
        pre_ratio: torch.Tensor,
        post_ratio: torch.Tensor,
    ) -> None:
        self._diagnostic_calls += 1
        should_record = self.diagnostics and (
            self._diagnostic_calls == 1 or self._diagnostic_calls % 200 == 0
        )
        if not should_record:
            return
        valid_total = int(lengths.sum().item())
        capacity = max(int(lengths.numel()) * int(packed_length), 1)
        self.last_diagnostics = {
            "roi_tokens_min": int(lengths.min().item()),
            "roi_tokens_median": float(lengths.float().median().item()),
            "roi_tokens_max": int(lengths.max().item()),
            "padding_fraction": 1.0 - (valid_total / capacity),
            "gate": float(gate.detach().float().item()),
            "pre_cap_ratio_p50": float(pre_ratio.detach().float().median().item()),
            "pre_cap_ratio_max": float(pre_ratio.detach().float().max().item()),
            "post_cap_ratio_p50": float(post_ratio.detach().float().median().item()),
            "post_cap_ratio_max": float(post_ratio.detach().float().max().item()),
            "cap_fraction": float((cap_scale < 1.0).float().mean().item()),
        }
        values = self.last_diagnostics
        print(
            "[BA PPR diagnostics] "
            f"site={self.processor_name} calls={self._diagnostic_calls} "
            f"roi={values['roi_tokens_min']}/"
            f"{values['roi_tokens_median']:.0f}/{values['roi_tokens_max']} "
            f"padding={values['padding_fraction']:.3f} "
            f"gate={values['gate']:.4f} "
            f"pre={values['pre_cap_ratio_p50']:.4f}/"
            f"{values['pre_cap_ratio_max']:.4f} "
            f"post={values['post_cap_ratio_p50']:.4f}/"
            f"{values['post_cap_ratio_max']:.4f} "
            f"capped={values['cap_fraction']:.3f}"
        )

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        del args, kwargs
        if encoder_hidden_states is not None:
            raise ValueError("PackedResidualBranchedAttnProcessor handles self-attention only")
        if self.ref_to_k is None or self.ref_to_v is None:
            raise RuntimeError("init_from_attention must be called before the processor is used")
        if self.connector_down is None or self.connector_up is None or self.gate_logit is None:
            raise RuntimeError("Packed residual connector is not initialized")
        if self.mask is None or self.mask_ref is None or self.mask_core is None:
            raise ValueError("Packed residual attention requires target, reference, and core masks")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            total_batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                total_batch,
                channel,
                height * width,
            ).transpose(1, 2)
        else:
            total_batch = hidden_states.shape[0]
            channel = height = width = None

        generation_batch, _ = _branch_batch_sizes(self.mask, total_batch)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        base_all, query_all = self._base_self_attention_pre_out(
            attn,
            hidden_states,
            attention_mask,
        )
        target_base = base_all[:generation_batch]
        reference_base = base_all[generation_batch:]
        target_query = query_all[:generation_batch]
        reference_hidden = hidden_states[generation_batch:]
        sequence_length = target_base.shape[1]

        reference_valid = self._resize_mask(
            self.mask_ref,
            target_length=sequence_length,
            batch_size=generation_batch,
            mode="nearest",
            binary=True,
        ).squeeze(-1)
        packed, lengths, pad_mask, sample_has_roi = pack_valid_tokens(
            reference_hidden,
            reference_valid,
        )
        reference_key = self._to_heads(self.ref_to_k(packed), attn.heads)
        reference_value = self._to_heads(self.ref_to_v(packed), attn.heads)
        reference_key = self._apply_k_norm(attn, reference_key)
        reference_candidate = F.scaled_dot_product_attention(
            target_query,
            reference_key,
            reference_value,
            attn_mask=pad_mask.to(dtype=target_query.dtype, device=target_query.device),
            dropout_p=0.0,
            is_causal=False,
        )
        reference_candidate = self._from_heads(reference_candidate).to(target_base.dtype)

        raw_delta = self.connector_up(
            self.connector_down(reference_candidate - target_base)
        )
        target_core = self._resize_mask(
            self.mask_core,
            target_length=sequence_length,
            batch_size=generation_batch,
            mode="bilinear",
            binary=False,
        ).to(dtype=target_base.dtype, device=target_base.device)
        bounded_delta, cap_scale, pre_ratio, post_ratio = self._masked_rms_cap(
            raw_delta,
            base=target_base,
            mask=target_core,
            max_ratio=self.delta_rms_cap,
        )
        gate = self.gate_max * torch.sigmoid(self.gate_logit)
        applied_delta = (
            target_core
            * sample_has_roi[:, None, None].to(target_base.dtype)
            * gate
            * float(self.runtime_scale)
            * bounded_delta
        )
        target_output = target_base + applied_delta
        hidden_states = torch.cat([target_output, reference_base], dim=0)

        self._record_diagnostics(
            lengths=lengths,
            packed_length=packed.shape[1],
            gate=gate,
            cap_scale=cap_scale,
            pre_ratio=pre_ratio,
            post_ratio=post_ratio,
        )
        if (
            isinstance(self.diagnostic_sink, list)
            and int(self.diagnostic_step) in set(self.diagnostic_steps)
        ):
            mask_fp32 = target_core.float()
            count = (mask_fp32.sum(dim=(1, 2)) * target_base.shape[-1]).clamp_min(1.0)
            applied_rms = torch.sqrt(
                (mask_fp32 * applied_delta.float().square()).sum(dim=(1, 2)) / count
            )
            base_rms = torch.sqrt(
                (mask_fp32 * target_base.float().square()).sum(dim=(1, 2)) / count
                + 1e-12
            )
            ratios = applied_rms / (base_rms + 1e-12)
            self.diagnostic_sink.append(
                {
                    "record_type": "processor_applied_ratio",
                    "variant": self.diagnostic_variant,
                    "step": int(self.diagnostic_step),
                    "processor": self.processor_name,
                    "runtime_scale": float(self.runtime_scale),
                    "gate": float(gate.detach().float().item()),
                    "applied_ratio_min": float(ratios.min().item()),
                    "applied_ratio_p50": float(ratios.median().item()),
                    "applied_ratio_max": float(ratios.max().item()),
                }
            )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch,
                channel,
                height,
                width,
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
