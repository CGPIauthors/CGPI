import torch

from all.approximation import FixedTarget

class FixedTargetEx(FixedTarget):
    def __call__(self, *inputs, **kwargs):
        with torch.no_grad():
            return self._target(*inputs, **kwargs)

