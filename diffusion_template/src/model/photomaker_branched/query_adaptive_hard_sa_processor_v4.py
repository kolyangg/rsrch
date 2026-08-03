"""Query-adaptive hard-routed reference self-attention.

The target face uses target queries with explicit, true-key-masked reference
K/V. Native target self-attention is retained only outside the target face;
there is deliberately no native/reference face interpolation or gate.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_processor_cleanest import _clone_effective_linear
from .residual_sa_processor_v2 import (
    ResidualBranchedSelfAttnProcessorV2,
    ResidualLoRALinear,
)


class QueryAdaptiveHardBranchedSelfAttnProcessorV4(
    ResidualBranchedSelfAttnProcessorV2
):
    """Hard target-face target-Q/reference-KV routing for doubled batches."""

    architecture_version = "query_adaptive_hard_sa_v4"
    has_cross_attention_kwargs = True

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        branch_q_rank: int = 16,
        ref_kv_rank: int = 32,
        output_rank: int = 32,
        trainable_dtype: torch.dtype = torch.float32,
        telemetry_enabled: bool = False,
        telemetry_interval: int = 50,
    ) -> None:
        nn.Module.__init__(self)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Hard BA-v4 requires PyTorch 2.0+")
        if int(hidden_size) <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if min(int(branch_q_rank), int(ref_kv_rank), int(output_rank)) <= 0:
            raise ValueError("Hard BA-v4 ranks must be positive")
        if float(scale) != 1.0:
            raise ValueError(
                "Hard BA-v4 requires face_branch_scale=1.0; attenuation would "
                f"reintroduce a hidden mix-like control, got {scale}"
            )
        if int(telemetry_interval) <= 0:
            raise ValueError("telemetry_interval must be positive")

        self.hidden_size = int(hidden_size)
        self.cross_attention_dim = int(cross_attention_dim or hidden_size)
        self.scale = 1.0
        self.branch_q_rank = int(branch_q_rank)
        self.ref_kv_rank = int(ref_kv_rank)
        self.output_rank = int(output_rank)
        self.trainable_dtype = trainable_dtype

        self.branch_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.ref_out = ResidualLoRALinear(
            self.hidden_size,
            rank=self.output_rank,
            dtype=trainable_dtype,
            zero_init_output=True,
        )

        self.mask = None
        self.mask_ref = None
        self.ba_denoise_progress = None
        self.force_binary_masks = True
        self.cache_prepared_masks = False
        self.telemetry_enabled = bool(telemetry_enabled)
        self.telemetry_interval = int(telemetry_interval)
        self._telemetry_forward_count = 0
        self._latest_ba_telemetry = None

    def init_from_attention(self, attn) -> None:
        # 3 Aug 2026 - AICODE-NOTE: Hard v4 has no native/reference face mix.
        # Native target projections stay frozen; only the target query used by
        # the explicit reference branch and reference K/V/output deltas train.
        self.branch_to_q = _clone_effective_linear(
            attn.to_q,
            kind="lora",
            rank=self.branch_q_rank,
            trainable_dtype=self.trainable_dtype,
        )
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
        if self.branch_to_q is None or self.ref_to_k is None or self.ref_to_v is None:
            raise RuntimeError("Hard BA-v4 processor was not initialized")
        for name, parameter in self.branch_to_q.named_parameters():
            yield f"branch_to_q.{name}", parameter, "ref_query"
        for prefix, module in (
            ("ref_to_k", self.ref_to_k),
            ("ref_to_v", self.ref_to_v),
        ):
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter, "ref_kv"
        for name, parameter in self.ref_out.named_parameters():
            yield f"ref_out.{name}", parameter, "ref_output"

    def set_telemetry_enabled(self, enabled: bool) -> None:
        self.telemetry_enabled = bool(enabled)

    @staticmethod
    def _masked_rms(
        tensor: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = target_mask.float()
        denom = (mask.sum(dim=(1, 2)) * tensor.shape[-1]).clamp_min(1.0)
        energy = (tensor.float().square() * mask).sum(dim=(1, 2)) / denom
        return energy.clamp_min(0.0).sqrt()

    def _record_telemetry(
        self,
        *,
        native: torch.Tensor,
        reference: torch.Tensor,
        q_native: torch.Tensor,
        q_branch: torch.Tensor,
        target_mask: torch.Tensor,
        reference_key_bias: torch.Tensor,
    ) -> None:
        if not self.telemetry_enabled:
            return
        self._telemetry_forward_count += 1
        if (self._telemetry_forward_count - 1) % self.telemetry_interval:
            return
        with torch.no_grad():
            native_rms = self._masked_rms(native, target_mask).clamp_min(1.0e-8)
            reference_rms = self._masked_rms(reference, target_mask).clamp_min(
                1.0e-8
            )
            mask = target_mask.float()
            dot_denom = (mask.sum(dim=(1, 2)) * native.shape[-1]).clamp_min(1.0)
            dot = (reference.float() * native.float() * mask).sum(
                dim=(1, 2)
            ) / dot_denom
            cosine = dot / (reference_rms * native_rms).clamp_min(1.0e-8)
            query_denom = q_native.detach().float().square().mean(
                dim=(1, 2, 3)
            ).clamp_min(1.0e-12).sqrt()
            query_delta = (
                q_branch.detach().float() - q_native.detach().float()
            ).square().mean(dim=(1, 2, 3)).clamp_min(0.0).sqrt()
            progress = self.ba_denoise_progress
            if progress is None:
                progress_mean = native.new_tensor(float("nan"), dtype=torch.float32)
            else:
                progress_mean = torch.as_tensor(
                    progress, device=native.device, dtype=torch.float32
                ).mean()
            self._latest_ba_telemetry = {
                "reference_native_rms_ratio": (reference_rms / native_rms).mean(),
                "reference_native_cosine": cosine.mean(),
                "hard_face_native_leakage": native.new_tensor(0.0).float(),
                "branch_query_delta_rms_ratio": (query_delta / query_denom).mean(),
                "reference_valid_key_fraction": (
                    reference_key_bias == 0
                ).float().mean(),
                "denoise_progress_mean": progress_mean,
            }

    def latest_ba_telemetry(self) -> Optional[dict[str, torch.Tensor]]:
        return self._latest_ba_telemetry

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
        if self.branch_to_q is None or self.ref_to_k is None or self.ref_to_v is None:
            raise RuntimeError("Hard BA-v4 processor was not initialized")
        if hidden_states.shape[0] % 2:
            raise RuntimeError(
                "Hard BA-v4 expects [target, reference] doubled batches"
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
        q_native = self._reshape_heads(attn.to_q(target_hidden), heads)
        k_native = self._reshape_heads(attn.to_k(target_hidden), heads)
        v_native = self._reshape_heads(attn.to_v(target_hidden), heads)
        native_message = F.scaled_dot_product_attention(
            q_native, k_native, v_native, dropout_p=0.0, is_causal=False
        )
        native_out = self._apply_output_projection(
            attn, self._merge_heads(native_message)
        )

        target_mask = self._target_output_mask(
            seq_len=target_hidden.shape[1],
            batch_size=batch_size,
            device=native_out.device,
            dtype=native_out.dtype,
        ).to(dtype=native_out.dtype)
        q_branch = self._reshape_heads(self.branch_to_q(target_hidden), heads)
        k_reference = self._reshape_heads(self.ref_to_k(reference_hidden), heads)
        v_reference = self._reshape_heads(self.ref_to_v(reference_hidden), heads)
        key_bias = self._reference_key_bias(
            seq_len=reference_hidden.shape[1],
            batch_size=batch_size,
            device=q_branch.device,
            dtype=q_branch.dtype,
        )
        reference_message = F.scaled_dot_product_attention(
            q_branch,
            k_reference,
            v_reference,
            attn_mask=key_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        reference_message = self._merge_heads(reference_message)
        reference_face_out = self._apply_output_projection(attn, reference_message)
        reference_face_out = reference_face_out + self.ref_out(reference_message)

        # 3 Aug 2026 - Hard mask replacement is the core experiment invariant.
        # Inside the target face, no native self-attention message is mixed back
        # into the explicit target-Q/reference-KV branch.
        target_out = native_out * (1.0 - target_mask)
        target_out = target_out + reference_face_out * target_mask
        self._record_telemetry(
            native=native_out,
            reference=reference_face_out,
            q_native=q_native,
            q_branch=q_branch,
            target_mask=target_mask,
            reference_key_bias=key_bias,
        )

        q_reference = self._reshape_heads(attn.to_q(reference_hidden), heads)
        k_reference_base = self._reshape_heads(attn.to_k(reference_hidden), heads)
        v_reference_base = self._reshape_heads(attn.to_v(reference_hidden), heads)
        reference_lane_out = F.scaled_dot_product_attention(
            q_reference,
            k_reference_base,
            v_reference_base,
            dropout_p=0.0,
            is_causal=False,
        )
        reference_lane_out = self._apply_output_projection(
            attn, self._merge_heads(reference_lane_out)
        )

        if input_ndim == 4:
            target_out = target_out.transpose(-1, -2).reshape(
                batch_size, channels, height, width
            )
            reference_lane_out = reference_lane_out.transpose(-1, -2).reshape(
                batch_size, channels, height, width
            )

        if attn.residual_connection:
            target_out = target_out + target_residual
            reference_lane_out = reference_lane_out + reference_residual
        target_out = target_out / attn.rescale_output_factor
        reference_lane_out = reference_lane_out / attn.rescale_output_factor
        return torch.cat([target_out, reference_lane_out], dim=0)
