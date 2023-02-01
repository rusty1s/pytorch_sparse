from typing import Any, Optional, Tuple

import torch

import torch_sparse.typing
from torch_sparse.typing import pyg_lib

try:
    from typing_extensions import Final  # noqa
except ImportError:
    from torch.jit import Final  # noqa


def index_sort(
        inputs: torch.Tensor,
        max_value: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""See pyg-lib documentation for more details:
    https://pyg-lib.readthedocs.io/en/latest/modules/ops.html"""
    if not torch_sparse.typing.WITH_INDEX_SORT:  # pragma: no cover
        return inputs.sort()
    return pyg_lib.ops.index_sort(inputs, max_value)


def is_scalar(other: Any) -> bool:
    return isinstance(other, int) or isinstance(other, float)
