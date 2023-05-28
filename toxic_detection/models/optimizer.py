import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(self, optimizer, *, warmup_steps=None, warmup_ratio=None, max_steps=None, min_lr=0.0, last_epoch=-1):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        

        step = self.last_epoch

        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        if (self.max_steps is not None) and (step > self.max_steps):
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs

class WarmupAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, last_epoch=-1, min_lr=0.0, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        delta_lr = self.base_lrs[0] - self.min_lr
        mult = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        out_lr = [self.min_lr + (1 - mult) * delta_lr for _ in self.base_lrs]
        return out_lr
    

