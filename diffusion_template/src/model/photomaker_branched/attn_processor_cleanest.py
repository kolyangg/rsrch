"""Fixed E13 target-query/reference-KV self-attention processor."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchLoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.rank = int(rank)
        self.register_buffer(
            "base_weight",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "base_bias",
            torch.empty(out_features, device=device, dtype=dtype) if bias else None,
        )
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base_weight, self.base_bias) + F.linear(
            F.linear(x, self.lora_A),
            self.lora_B,
        )


def _clone_effective_linear(
    attn_linear,
    *,
    rank: int = 128,
):
    base = (
        attn_linear.get_base_layer()
        if hasattr(attn_linear, "get_base_layer")
        else attn_linear
    )
    cloned = BranchLoRALinear(
        base.in_features,
        base.out_features,
        rank=rank,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    with torch.no_grad():
        weight = base.weight.detach().clone()
        if hasattr(attn_linear, "lora_A") and "default" in attn_linear.lora_A:
            weight += attn_linear.get_delta_weight("default").detach().to(
                weight.device, weight.dtype
            )
        cloned.base_weight.copy_(weight)
        if base.bias is not None:
            cloned.base_bias.copy_(base.bias.detach())
    return cloned


def _branch_batch_sizes(mask, total_batch):
    if mask is None:
        if total_batch % 2 != 0:
            raise RuntimeError(f"Cannot infer branch sizes from total_batch={total_batch}")
        gen_batch = total_batch // 2
    else:
        gen_batch = int(mask.shape[0])
    ref_batch = total_batch - gen_batch
    if ref_batch != gen_batch:
        raise RuntimeError(
            f"Invalid branched batch: total={total_batch}, generation={gen_batch}, "
            f"reference={ref_batch}; expected one reference per sample"
        )
    return gen_batch, ref_batch


class BranchedAttnProcessor(nn.Module):
    """Hard replacement over a doubled ``[target, reference]`` batch."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.scale = scale
        self.mask = None
        self.mask_ref = None
        self.ref_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.noise_to_q = None
        self.noise_to_k = None
        self.noise_to_v = None
        # 10 Aug 2026 - E13C-PERF-02: Reuse resized masks within one forward;
        # the cache is attached to the current mask tensor and cannot cross
        # samples or steps, so attention values remain unchanged.
        self.has_cross_attention_kwargs = True

    def init_from_attention(self, attn) -> None:
        self.ref_to_q = _clone_effective_linear(attn.to_q)
        self.ref_to_k = _clone_effective_linear(attn.to_k)
        self.ref_to_v = _clone_effective_linear(attn.to_v)
        self.noise_to_q = _clone_effective_linear(attn.to_q)
        self.noise_to_k = _clone_effective_linear(attn.to_k)
        self.noise_to_v = _clone_effective_linear(attn.to_v)

    def _q_noise(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.noise_to_q if self.noise_to_q is not None else attn.to_q
        return layer(hidden_states)

    def _k_noise(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.noise_to_k if self.noise_to_k is not None else attn.to_k
        return layer(hidden_states)

    def _v_noise(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.noise_to_v if self.noise_to_v is not None else attn.to_v
        return layer(hidden_states)

    def _q_ref(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.ref_to_q if self.ref_to_q is not None else attn.to_q
        return layer(hidden_states)

    def _k_ref(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.ref_to_k if self.ref_to_k is not None else attn.to_k
        return layer(hidden_states)

    def _v_ref(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.ref_to_v if self.ref_to_v is not None else attn.to_v
        return layer(hidden_states)

    def set_masks(
        self,
        mask: Optional[torch.Tensor],
        mask_ref: Optional[torch.Tensor] = None,
    ) -> None:
        self.mask = mask
        self.mask_ref = mask_ref if mask_ref is not None else mask

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ) -> torch.Tensor:
        """Route target face queries to reference K/V in a doubled batch."""
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

        total_batch = hidden_states.shape[0]
        batch, ref_batch = _branch_batch_sizes(self.mask, total_batch)
        target, reference = hidden_states[:batch], hidden_states[batch:]
        if attn.group_norm is not None:
            target = attn.group_norm(target.transpose(1, 2)).transpose(1, 2)
            reference = attn.group_norm(reference.transpose(1, 2)).transpose(1, 2)

        if self.mask is None or self.mask_ref is None:
            raise ValueError("Branched self-attention requires target and reference masks")
        length = target.shape[1]
        target_mask = self._prepare_mask(self.mask, length, batch).to(
            device=target.device, dtype=target.dtype
        )
        reference_mask = self._prepare_mask(
            self.mask_ref, length, ref_batch
        ).to(device=reference.device, dtype=reference.dtype)
        target_mask_flat = target_mask.squeeze(1)

        heads = int(attn.heads)
        width = target.shape[-1] // heads
        def reshape(value):
            return value.view(value.shape[0], -1, heads, width).transpose(1, 2)
        query = reshape(self._q_noise(attn, target))

        background = F.scaled_dot_product_attention(
            query * (1.0 - target_mask),
            reshape(self._k_noise(attn, target)),
            reshape(self._v_noise(attn, target)),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(batch, length, -1)

        reference_face = reference * reference_mask.squeeze(1)
        face = F.scaled_dot_product_attention(
            query * target_mask,
            reshape(self._k_ref(attn, reference_face)),
            reshape(self._v_ref(attn, reference_face)),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(batch, length, -1)

        reference_out = F.scaled_dot_product_attention(
            reshape(self._q_ref(attn, reference)),
            reshape(self._k_ref(attn, reference)),
            reshape(self._v_ref(attn, reference)),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(ref_batch, length, -1)

        target_out = (
            background * (1.0 - target_mask_flat)
            + face * target_mask_flat * self.scale
        )
        hidden_states = attn.to_out[1](
            attn.to_out[0](torch.cat([target_out, reference_out], dim=0))
        )
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch, channels, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


    def _prepare_mask(
        self, mask: torch.Tensor, target_len: int, batch_size: int
    ) -> torch.Tensor:
        """Resize and cache one binary spatial attention mask."""
        cache_key = (
            int(target_len),
            int(batch_size),
            str(mask.device),
            str(mask.dtype),
        )
        prepared_cache = getattr(mask, "_ba_prepared_mask_cache", None)
        if prepared_cache is not None and cache_key in prepared_cache:
            return prepared_cache[cache_key]
        side = math.isqrt(target_len)
        if side * side != target_len:
            raise RuntimeError(f"Sequence length {target_len} is not square")

        source_batch = mask.shape[0]
        if mask.ndim == 4:
            mask_4d = mask[:, :1].float()
        else:
            flat = mask.reshape(source_batch, -1).float()
            h0 = int(math.isqrt(flat.shape[1]))
            if h0 * h0 != flat.shape[1]:
                raise RuntimeError(f"Mask length {flat.shape[1]} is not square")
            mask_4d = flat.reshape(source_batch, 1, h0, h0)

        resized = F.interpolate(
            mask_4d, size=(side, side), mode="bilinear", align_corners=False
        )
        flattened = (resized > 0.5).to(resized.dtype).flatten(2).transpose(1, 2)
        if flattened.shape[0] != batch_size:
            repeats = (batch_size + flattened.shape[0] - 1) // flattened.shape[0]
            flattened = flattened.repeat(repeats, 1, 1)[:batch_size]
        result = flattened.view(batch_size, 1, target_len, 1)
        prepared_cache = getattr(mask, "_ba_prepared_mask_cache", None)
        if prepared_cache is None:
            prepared_cache = {}
            mask._ba_prepared_mask_cache = prepared_cache
        prepared_cache[cache_key] = result
        return result
