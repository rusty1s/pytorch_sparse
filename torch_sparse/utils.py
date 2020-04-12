from typing import Any
import torch

try:
    from typing_extensions import Final  # noqa
except ImportError:
    from torch.jit import Final  # noqa


def is_scalar(other: Any) -> bool:
    return isinstance(other, int) or isinstance(other, float)


def cartesian1d(x, y):
    a1, a2 = torch.meshgrid([x, y])
    coos = torch.stack([a1, a2]).T.reshape(-1, 2)
    return coos.split(1, dim=1)
