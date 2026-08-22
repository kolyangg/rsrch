import math

from torch.optim.lr_scheduler import LambdaLR


class WarmupHoldCosineLR(LambdaLR):
    """Linear warmup, constant plateau, then cosine decay to ``min_factor``."""

    def __init__(
        self,
        warmup_steps: int,
        hold_steps: int,
        total_steps: int,
        min_factor: float = 0.1,
        *args,
        **kwargs,
    ):
        warmup_steps = int(warmup_steps)
        hold_steps = int(hold_steps)
        total_steps = int(total_steps)
        min_factor = float(min_factor)
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if not warmup_steps <= hold_steps < total_steps:
            raise ValueError(
                "Expected warmup_steps <= hold_steps < total_steps, got "
                f"{warmup_steps}, {hold_steps}, {total_steps}"
            )
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError("min_factor must be in [0, 1]")

        def lr_lambda(step: int) -> float:
            completed = int(step) + 1
            if completed < warmup_steps:
                return completed / warmup_steps
            if completed <= hold_steps:
                return 1.0
            progress = min(
                1.0,
                (completed - hold_steps) / (total_steps - hold_steps),
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        super().__init__(lr_lambda=lr_lambda, *args, **kwargs)
