"""CL45-only PCGrad projection for the unified clean training path.

Keeping this opt-in optimizer surgery outside the shared trainer makes the
CL14/19/23/27/39-44 update path visibly independent of CL45.
"""

from __future__ import annotations

import torch


def apply_ba_pcgrad_surrogate(trainer, batch) -> None:
    """Project a conflicting BA auxiliary gradient away from diffusion loss."""
    if not trainer.ba_pcgrad_enabled:
        return
    reference = batch["loss"].detach()
    zero = reference.new_tensor(0.0)
    metrics = {
        "ba/pcgrad/gradient_cosine": zero,
        "ba/pcgrad/conflict_fraction": zero,
        "ba/pcgrad/projection_norm": zero,
        "ba/pcgrad/main_norm": zero,
        "ba/pcgrad/aux_norm": zero,
    }
    batch.update(metrics)
    if int(batch.get("global_step", 0)) % trainer.ba_pcgrad_interval:
        return
    primary = batch.get("_loss_diffusion_graph")
    auxiliary = batch.get("_loss_ba_aux_graph")
    if primary is None or auxiliary is None or not auxiliary.requires_grad:
        raise RuntimeError("CL45 requires separate live diffusion and BA auxiliary graphs")
    ba_parameters = [
        parameter
        for group in trainer.optimizer.param_groups
        if str(group.get("name", "")) == "ba"
        for parameter in group.get("params", ())
    ]
    if len(ba_parameters) != 840:
        raise RuntimeError(
            f"CL45 expected 840 CL27 BA tensors, found {len(ba_parameters)}"
        )
    main_grads = torch.autograd.grad(
        primary, ba_parameters, retain_graph=True, allow_unused=True
    )
    aux_grads = torch.autograd.grad(
        auxiliary, ba_parameters, retain_graph=True, allow_unused=True
    )
    dot = zero
    main_sq = zero
    aux_sq = zero
    for main, aux in zip(main_grads, aux_grads):
        if main is not None:
            main_sq = main_sq + main.detach().float().square().sum()
        if aux is not None:
            aux_sq = aux_sq + aux.detach().float().square().sum()
        if main is not None and aux is not None:
            dot = dot + (main.detach().float() * aux.detach().float()).sum()
    main_norm = main_sq.clamp_min(0.0).sqrt()
    aux_norm = aux_sq.clamp_min(0.0).sqrt()
    coefficient = (-dot / (main_sq + trainer.ba_pcgrad_eps)).clamp_min(0.0)
    surrogate = batch["loss"]
    for parameter, main in zip(ba_parameters, main_grads):
        if main is not None:
            correction = coefficient * main.detach().to(parameter.dtype)
            surrogate = surrogate + (
                (parameter - parameter.detach()) * correction
            ).sum()
    batch["loss"] = surrogate
    batch.update(
        {
            "ba/pcgrad/gradient_cosine": (
                dot / (main_norm * aux_norm).clamp_min(trainer.ba_pcgrad_eps)
            ).detach(),
            "ba/pcgrad/conflict_fraction": (dot < 0).float().detach(),
            "ba/pcgrad/projection_norm": (coefficient * main_norm).detach(),
            "ba/pcgrad/main_norm": main_norm.detach(),
            "ba/pcgrad/aux_norm": aux_norm.detach(),
        }
    )
