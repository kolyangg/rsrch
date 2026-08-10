import math

from torch.optim.lr_scheduler import LambdaLR

class CustomLinearLR(LambdaLR):
    def __init__(self, warmup_steps, *args, **kwargs):
        super().__init__(
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0), 
            *args, 
            **kwargs
        )


class WarmupHoldCosineLR(LambdaLR):
    """The sealed E13 20-step warmup, 14k hold and 24k cosine schedule."""

    def __init__(self, warmup_steps, hold_steps, total_steps, min_factor=0.1,
                 *args, **kwargs):
        warmup_steps = int(warmup_steps)
        hold_steps = int(hold_steps)
        total_steps = int(total_steps)
        min_factor = float(min_factor)
        if not 0 < warmup_steps <= hold_steps < total_steps:
            raise ValueError("Expected 0 < warmup_steps <= hold_steps < total_steps")
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError("min_factor must be in [0, 1]")

        # 10 Aug 2026 - E13C-CFG-01: Preserve the exact E13 learning-rate
        # trajectory so dataset transfers remain controlled comparisons.
        def lr_lambda(step):
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
