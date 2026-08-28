"""True reference-key masking shared by CL39-X01 and local side routes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class ValidKeyAttentionResult:
    message: torch.Tensor
    entropy: torch.Tensor | None
    eligible: torch.Tensor
    valid_fraction: torch.Tensor
    valid_count: torch.Tensor


def valid_key_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_key_mask: torch.Tensor,
    *,
    fallback: torch.Tensor | None = None,
    return_entropy: bool = False,
    entropy_chunk_size: int = 256,
) -> ValidKeyAttentionResult:
    """Apply SDPA over valid keys only and fail to an explicit native message."""
    if query.ndim != 4 or key.shape != value.shape or valid_key_mask.ndim != 2:
        raise ValueError("Expected Q/K/V [B,H,L,D] and valid_key_mask [B,L]")
    if key.shape[0] != query.shape[0] or key.shape[2] != valid_key_mask.shape[1]:
        raise ValueError("Valid-key attention batch/length mismatch")
    valid = valid_key_mask.to(device=query.device, dtype=torch.bool)
    count = valid.sum(dim=-1)
    eligible = count.gt(0)
    safe_valid = valid.clone()
    safe_valid[:, 0] |= ~eligible
    mask = safe_valid[:, None, None, :]
    message = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    fallback_value = torch.zeros_like(message) if fallback is None else fallback
    if fallback_value.shape != message.shape:
        raise ValueError("fallback must match the SDPA message shape")
    message = torch.where(eligible[:, None, None, None], message, fallback_value)

    entropy = None
    if return_entropy:
        chunks = []
        width = query.shape[-1]
        safe_count = count.clamp_min(2).float().log().view(-1, 1, 1)
        for start in range(0, query.shape[2], int(entropy_chunk_size)):
            logits = torch.matmul(
                query[:, :, start : start + entropy_chunk_size].detach().float(),
                key.detach().float().transpose(-1, -2),
            ) / math.sqrt(width)
            logits = logits.masked_fill(~safe_valid[:, None, None], -torch.inf)
            probability = torch.softmax(logits, dim=-1)
            ent = -(probability * probability.clamp_min(1.0e-12).log()).sum(-1)
            ent = (ent / safe_count).mean(dim=1).unsqueeze(-1)
            ent = torch.where(eligible[:, None, None], ent, torch.ones_like(ent))
            chunks.append(ent)
        entropy = torch.cat(chunks, dim=1).to(query.dtype)
    return ValidKeyAttentionResult(
        message=message,
        entropy=entropy,
        eligible=eligible[:, None, None, None],
        valid_fraction=count.float() / float(valid.shape[-1]),
        valid_count=count,
    )


def packed_attention_oracle(query, key, value, valid_key_mask):
    """Slow tiny-fixture oracle; intentionally excluded from production paths."""
    outputs = []
    for index in range(query.shape[0]):
        selected = valid_key_mask[index].bool()
        if not bool(selected.any()):
            outputs.append(torch.zeros_like(query[index : index + 1]))
            continue
        outputs.append(
            F.scaled_dot_product_attention(
                query[index : index + 1],
                key[index : index + 1, :, selected],
                value[index : index + 1, :, selected],
                dropout_p=0.0,
                is_causal=False,
            )
        )
    return torch.cat(outputs)
