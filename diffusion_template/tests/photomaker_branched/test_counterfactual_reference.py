import torch

from src.model.photomaker_branched.objectives.counterfactual_reference import (
    compute_counterfactual_reference_loss,
    derangement,
    deterministic_mode,
)


def test_derangement_and_sampling_are_deterministic():
    permutation = derangement(5)
    assert torch.all(permutation != torch.arange(5))
    first = deterministic_mode(global_step=17, rank=2, batch_size=4, probability=1.0, wrong_fraction=0.5)
    second = deterministic_mode(global_step=17, rank=2, batch_size=4, probability=1.0, wrong_fraction=0.5)
    assert first == second
    assert deterministic_mode(global_step=0, rank=0, batch_size=1, probability=1.0, wrong_fraction=1.0) is None


def test_counterfactual_detachment_and_outside_invariance():
    correct = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], requires_grad=True)
    counterfactual = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(correct)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    result = compute_counterfactual_reference_loss(
        pred_correct=correct, pred_counterfactual=counterfactual,
        target_noise=target, target_mask=mask, outside_weight=1.0,
        rank_weight=1.0, rank_margin=0.1, mode="wrong",
    )
    result.loss.backward()
    # The face comparison target is detached, so the counterfactual path only
    # receives outside-mask invariance gradients (zero in this fixture).
    assert torch.count_nonzero(counterfactual.grad) == 0
    assert correct.grad is not None and torch.isfinite(correct.grad).all()
    assert float(result.outside_loss) == 0.0
