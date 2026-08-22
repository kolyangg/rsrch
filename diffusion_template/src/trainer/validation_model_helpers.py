"""Small state-transfer helpers for alternate-base E13 validation."""

from __future__ import annotations

import torch


def copy_full_processor_state(train_unet, val_unet) -> int:
    """Copy every stateful attention processor by exact name."""
    train_processors = getattr(train_unet, "attn_processors", {})
    val_processors = getattr(val_unet, "attn_processors", {})
    copied = 0
    failures = []
    for name, train_processor in train_processors.items():
        # 22 Aug 2026 - Native Diffusers processors are stateless non-modules;
        # only installed BA processors participate in strict state transfer.
        state_dict = getattr(train_processor, "state_dict", None)
        if state_dict is None:
            continue
        state = state_dict()
        if not state:
            continue
        val_processor = val_processors.get(name)
        if val_processor is None:
            failures.append(f"missing:{name}")
            continue
        try:
            val_processor.load_state_dict(state, strict=True)
            copied += 1
        except Exception as exc:
            failures.append(f"{name}:{exc}")
    if failures or copied == 0:
        raise RuntimeError(
            "Strict E13 validation processor copy failed: "
            f"copied={copied}, failures={failures[:5]}"
        )
    return copied


def photomaker_default_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.unet.named_parameters()
        if ".default." in name and ("lora_A" in name or "lora_B" in name)
    }


def restore_photomaker_default(
    model, snapshot: dict[str, torch.Tensor]
) -> None:
    named = dict(model.unet.named_parameters())
    missing = sorted(set(snapshot) - set(named))
    if missing:
        raise RuntimeError(
            f"Validation model is missing PhotoMaker tensors: {missing[:5]}"
        )
    with torch.no_grad():
        for name, value in snapshot.items():
            named[name].copy_(value.to(named[name].device, named[name].dtype))
