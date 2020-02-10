from typing import Any

try:
    from typing_extensions import Final  # noqa
except ImportError:
    from torch.jit import Final  # noqa


def is_scalar(other: Any) -> bool:
    return isinstance(other, int) or isinstance(other, float)
