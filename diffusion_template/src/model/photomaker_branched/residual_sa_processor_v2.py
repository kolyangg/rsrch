"""Versioned residual reference-conditioned branched self-attention.

The target lane keeps its frozen native self-attention message. Target queries
also attend explicit reference K/V, restricted to valid reference-face keys,
and a bounded face-local residual adds that message to the target lane.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_processor_cleanest import _clone_effective_linear


def _logit(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"gate_init must be in (0, 1), got {probability}")
    return math.log(probability / (1.0 - probability))


class ResidualLoRALinear(nn.Module):
    """Low-rank residual projection with no frozen/base linear term."""

    def __init__(
        self,
        features: int,
        *,
        rank: int,
        alpha: Optional[int] = None,
        device=None,
        dtype: torch.dtype = torch.float32,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        self.scaling = float(alpha if alpha is not None else rank) / float(rank)
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if zero_init_output:
            nn.init.zeros_(self.lora_B)
        else:
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        train_dtype = self.lora_A.dtype
        delta = F.linear(
            F.linear(hidden_states.to(dtype=train_dtype), self.lora_A),
            self.lora_B,
        )
        return (delta * self.scaling).to(dtype=input_dtype)


class ResidualBranchedSelfAttnProcessorV2(nn.Module):
    """Target-Q/reference-KV residual BA processor for doubled batches.

    Input and output layout is ``[target_batch, reference_batch]``. The target
    native self-attention path is frozen. Only the reference K/V LoRA deltas,
    branch-local residual projection, and bounded gate are trainable.
    """

    architecture_version = "residual_sa_v2"
    has_cross_attention_kwargs = True

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        ref_kv_rank: int = 32,
        output_rank: int = 32,
        gate_init: float = 0.10,
        gate_max: float = 1.0,
        gate_timestep: bool = True,
        gate_face_area: bool = True,
        trainable_dtype: torch.dtype = torch.float32,
        require_denoise_progress: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Residual BA-v2 requires PyTorch 2.0+")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if not 0.0 < float(gate_max) <= 1.0:
            raise ValueError(f"gate_max must be in (0, 1], got {gate_max}")

        self.hidden_size = int(hidden_size)
        self.cross_attention_dim = int(cross_attention_dim or hidden_size)
        self.scale = float(scale)
        self.ref_kv_rank = int(ref_kv_rank)
        self.output_rank = int(output_rank)
        self.gate_max = float(gate_max)
        self.gate_timestep = bool(gate_timestep)
        self.gate_face_area = bool(gate_face_area)
        self.trainable_dtype = trainable_dtype
        self.require_denoise_progress = bool(require_denoise_progress)

        self.ref_to_k = None
        self.ref_to_v = None
        self.ref_out = ResidualLoRALinear(
            self.hidden_size,
            rank=self.output_rank,
            device=None,
            dtype=trainable_dtype,
            zero_init_output=True,
        )
        self.gate_logit = nn.Parameter(
            torch.tensor(_logit(gate_init), dtype=trainable_dtype)
        )
        self.gate_t = nn.Parameter(
            torch.zeros((), dtype=trainable_dtype),
            requires_grad=self.gate_timestep,
        )
        self.gate_area = nn.Parameter(
            torch.zeros((), dtype=trainable_dtype),
            requires_grad=self.gate_face_area,
        )

        self.mask = None
        self.mask_ref = None
        self.ba_denoise_progress = None
        self.force_binary_masks = True
        self.cache_prepared_masks = False

    def init_from_attention(self, attn) -> None:
        # 2 Aug 2026 - AICODE-NOTE: target Q/K/V stay on the frozen effective
        # PhotoMaker path. Only explicit reference K/V receive LoRA deltas.
        self.ref_to_k = _clone_effective_linear(
            attn.to_k,
            kind="lora",
            rank=self.ref_kv_rank,
            trainable_dtype=self.trainable_dtype,
        )
        self.ref_to_v = _clone_effective_linear(
            attn.to_v,
            kind="lora",
            rank=self.ref_kv_rank,
            trainable_dtype=self.trainable_dtype,
        )

    def named_ba_trainables(
        self,
    ) -> Iterator[tuple[str, nn.Parameter, str]]:
        if self.ref_to_k is None or self.ref_to_v is None:
            raise RuntimeError("Residual BA-v2 processor was not initialized")
        for prefix, module in (
            ("ref_to_k", self.ref_to_k),
            ("ref_to_v", self.ref_to_v),
        ):
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter, "ref_kv"
        for name, parameter in self.ref_out.named_parameters():
            yield f"ref_out.{name}", parameter, "ref_output"
        yield "gate_logit", self.gate_logit, "gate"
        if self.gate_t.requires_grad:
            yield "gate_t", self.gate_t, "gate"
        if self.gate_area.requires_grad:
            yield "gate_area", self.gate_area, "gate"

    def set_masks(
        self,
        mask: Optional[torch.Tensor],
        mask_ref: Optional[torch.Tensor] = None,
    ) -> None:
        self.mask = mask
        self.mask_ref = mask_ref if mask_ref is not None else mask

    def set_denoise_progress(self, progress: Optional[torch.Tensor]) -> None:
        self.ba_denoise_progress = progress

    @staticmethod
    def _reshape_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        batch, seq_len, channels = tensor.shape
        if channels % heads:
            raise RuntimeError(
                f"Attention channels {channels} are not divisible by heads {heads}"
            )
        head_dim = channels // heads
        return tensor.view(batch, seq_len, heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)

    def _prepare_mask(
        self,
        mask: torch.Tensor,
        *,
        target_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        binary: bool,
    ) -> torch.Tensor:
        cache_key = (
            self.architecture_version,
            int(target_len),
            int(batch_size),
            bool(binary),
            str(device),
            str(dtype),
        )
        if self.cache_prepared_masks:
            prepared_cache = getattr(mask, "_ba_v2_prepared_mask_cache", None)
            if prepared_cache is not None and cache_key in prepared_cache:
                return prepared_cache[cache_key]

        side = int(math.isqrt(target_len))
        if side * side != target_len:
            raise RuntimeError(f"BA-v2 requires square spatial tokens, got {target_len}")
        batch = mask.shape[0]
        if mask.ndim == 4:
            mask_4d = mask[:, :1].float()
        else:
            flat = mask.reshape(batch, -1).float()
            source_side = int(math.isqrt(flat.shape[1]))
            if source_side * source_side != flat.shape[1]:
                raise RuntimeError(
                    f"BA-v2 mask length is not square: {flat.shape[1]}"
                )
            mask_4d = flat.reshape(batch, 1, source_side, source_side)
        resized = F.interpolate(
            mask_4d,
            size=(side, side),
            mode="bilinear",
            align_corners=False,
        )
        if binary:
            resized = resized > 0.5
        else:
            resized = resized.clamp(0.0, 1.0)
        prepared = resized.flatten(2).transpose(1, 2)
        if prepared.shape[0] != batch_size:
            if prepared.shape[0] <= 0:
                raise RuntimeError("BA-v2 received an empty mask batch")
            repeats = (batch_size + prepared.shape[0] - 1) // prepared.shape[0]
            prepared = prepared.repeat(repeats, 1, 1)[:batch_size]
        prepared = prepared.to(device=device, dtype=torch.bool if binary else dtype)

        if self.cache_prepared_masks:
            prepared_cache = getattr(mask, "_ba_v2_prepared_mask_cache", None)
            if prepared_cache is None:
                prepared_cache = {}
                mask._ba_v2_prepared_mask_cache = prepared_cache
            prepared_cache[cache_key] = prepared
        return prepared

    def _reference_key_bias(
        self,
        *,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mask_ref is None:
            raise RuntimeError("Residual BA-v2 requires a reference-face mask")
        keep = self._prepare_mask(
            self.mask_ref,
            target_len=seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            binary=True,
        ).squeeze(-1)
        valid_counts = keep.sum(dim=-1)
        if torch.any(valid_counts == 0):
            bad = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "Residual BA-v2 reference mask has zero valid attention keys "
                f"for batch indices {bad}"
            )
        bias = torch.zeros(
            batch_size, 1, 1, seq_len, device=device, dtype=dtype
        )
        return bias.masked_fill(~keep[:, None, None, :], torch.finfo(dtype).min)

    def _target_output_mask(
        self,
        *,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mask is None:
            raise RuntimeError("Residual BA-v2 requires a target-face mask")
        return self._prepare_mask(
            self.mask,
            target_len=seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            binary=bool(self.force_binary_masks),
        )

    def _bounded_gate(
        self,
        *,
        batch_size: int,
        target_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        train_dtype = self.gate_logit.dtype
        logits = self.gate_logit.expand(batch_size).to(device=device)

        progress = self.ba_denoise_progress
        if progress is None:
            if self.require_denoise_progress:
                raise RuntimeError("Residual BA-v2 requires denoise progress")
            progress = torch.full((batch_size,), 0.5, device=device)
        elif not torch.is_tensor(progress):
            progress = torch.tensor(progress, device=device)
        progress = progress.to(device=device, dtype=train_dtype).reshape(-1)
        if progress.numel() == 1:
            progress = progress.expand(batch_size)
        elif progress.numel() != batch_size:
            if batch_size % progress.numel() != 0:
                raise RuntimeError(
                    "BA-v2 denoise-progress batch mismatch: "
                    f"progress={progress.numel()}, target={batch_size}"
                )
            progress = progress.repeat(batch_size // progress.numel())
        if torch.any((progress < 0.0) | (progress > 1.0)):
            raise RuntimeError("BA-v2 denoise progress must be in [0, 1]")
        if self.gate_timestep:
            logits = logits + self.gate_t * (2.0 * progress - 1.0)

        if self.gate_face_area:
            area = target_mask.float().mean(dim=(1, 2)).to(dtype=train_dtype)
            log_area = torch.log(area.clamp_min(1.0e-4))
            logits = logits + self.gate_area * log_area

        return (self.gate_max * torch.sigmoid(logits)).view(batch_size, 1, 1)

    @staticmethod
    def _apply_output_projection(attn, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        del encoder_hidden_states, attention_mask, scale, kwargs
        if self.ref_to_k is None or self.ref_to_v is None:
            raise RuntimeError("Residual BA-v2 processor was not initialized")
        if hidden_states.shape[0] % 2:
            raise RuntimeError(
                "Residual BA-v2 expects [target, reference] doubled batches"
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            total_batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                total_batch, channels, height * width
            ).transpose(1, 2)
        elif input_ndim != 3:
            raise RuntimeError(f"Unsupported attention input rank: {input_ndim}")

        batch_size = hidden_states.shape[0] // 2
        target_hidden = hidden_states[:batch_size]
        reference_hidden = hidden_states[batch_size:]
        target_residual = residual[:batch_size]
        reference_residual = residual[batch_size:]

        if attn.group_norm is not None:
            target_hidden = attn.group_norm(
                target_hidden.transpose(1, 2)
            ).transpose(1, 2)
            reference_hidden = attn.group_norm(
                reference_hidden.transpose(1, 2)
            ).transpose(1, 2)

        heads = int(attn.heads)
        q_target = self._reshape_heads(attn.to_q(target_hidden), heads)
        k_target = self._reshape_heads(attn.to_k(target_hidden), heads)
        v_target = self._reshape_heads(attn.to_v(target_hidden), heads)

        base_message = F.scaled_dot_product_attention(
            q_target, k_target, v_target, dropout_p=0.0, is_causal=False
        )
        base_out = self._apply_output_projection(
            attn, self._merge_heads(base_message)
        )

        # 2 Aug 2026 - Invalid reference tokens are excluded from softmax;
        # multiplying hidden states by a bbox is not an attention key mask.
        k_reference = self._reshape_heads(self.ref_to_k(reference_hidden), heads)
        v_reference = self._reshape_heads(self.ref_to_v(reference_hidden), heads)
        key_bias = self._reference_key_bias(
            seq_len=reference_hidden.shape[1],
            batch_size=batch_size,
            device=q_target.device,
            dtype=q_target.dtype,
        )
        reference_message = F.scaled_dot_product_attention(
            q_target,
            k_reference,
            v_reference,
            attn_mask=key_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        reference_delta = self.ref_out(self._merge_heads(reference_message))
        target_mask = self._target_output_mask(
            seq_len=target_hidden.shape[1],
            batch_size=batch_size,
            device=base_out.device,
            dtype=base_out.dtype,
        )
        gate = self._bounded_gate(
            batch_size=batch_size,
            target_mask=target_mask,
            device=base_out.device,
        ).to(dtype=base_out.dtype)
        target_out = base_out + target_mask * gate * reference_delta * self.scale

        q_reference = self._reshape_heads(attn.to_q(reference_hidden), heads)
        k_reference_base = self._reshape_heads(attn.to_k(reference_hidden), heads)
        v_reference_base = self._reshape_heads(attn.to_v(reference_hidden), heads)
        reference_out = F.scaled_dot_product_attention(
            q_reference,
            k_reference_base,
            v_reference_base,
            dropout_p=0.0,
            is_causal=False,
        )
        reference_out = self._apply_output_projection(
            attn, self._merge_heads(reference_out)
        )

        if input_ndim == 4:
            target_out = target_out.transpose(-1, -2).reshape(
                batch_size, channels, height, width
            )
            reference_out = reference_out.transpose(-1, -2).reshape(
                batch_size, channels, height, width
            )

        if attn.residual_connection:
            target_out = target_out + target_residual
            reference_out = reference_out + reference_residual
        target_out = target_out / attn.rescale_output_factor
        reference_out = reference_out / attn.rescale_output_factor
        return torch.cat([target_out, reference_out], dim=0)
