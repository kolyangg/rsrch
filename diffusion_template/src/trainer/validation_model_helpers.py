"""Small state-transfer helpers for alternate-base E13 validation."""

from __future__ import annotations

import torch


class _NoParameterTouch:
    @staticmethod
    def named_parameters():
        return ()


_NO_PARAMETER_TOUCH = _NoParameterTouch()


def parameter_touch_source(model, enabled: bool):
    return model if enabled else _NO_PARAMETER_TOUCH


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


def selected_photomaker_default_snapshot(model):
    """Capture the pretrained PhotoMaker adapter only for E13 validation."""
    if not bool(getattr(model, "e13_family_contract", False)):
        return None
    snapshot = photomaker_default_snapshot(model)
    if len(snapshot) != 700:
        raise RuntimeError(
            "E13 validation shadow expected 700 PhotoMaker tensors, "
            f"got {len(snapshot)}"
        )
    return snapshot


def restore_selected_photomaker_default(model, snapshot) -> None:
    if snapshot is not None:
        restore_photomaker_default(model, snapshot)


def copy_selected_processor_state(train_unet, val_model) -> None:
    """Fail closed when an E13 alternate-base model misses processor state."""
    if not bool(getattr(val_model, "e13_family_contract", False)):
        return
    copied = copy_full_processor_state(train_unet, val_model.unet)
    expected = 106 if bool(
        getattr(val_model, "ba_residual_identity_ca_v3_enabled", False)
    ) else 70
    if copied != expected:
        raise RuntimeError(
            f"E13 validation expected {expected} stateful processors, got {copied}"
        )


def start_face_quality_session(
    *, config, checkpoint_dir, writer, logger, part, step, partition_count
):
    if config is None or not bool(config.get("enabled", True)):
        return None
    from src.metrics.face_quality_validation import FaceQualityValidationSession

    return FaceQualityValidationSession(
        config=config,
        checkpoint_dir=checkpoint_dir,
        writer=writer,
        logger=logger,
        part=part,
        step=step,
        partition_count=partition_count,
    )


def finish_face_quality_session(session, metrics, num_processes: int) -> None:
    if session is None:
        return
    for name, value in session.finalize(num_processes=num_processes).items():
        metrics.update(f"face_quality/{name}", value)
