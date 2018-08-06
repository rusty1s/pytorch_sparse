import torch
from torch_sparse import coalesce


def transpose(index, value, m, n):
    """Transpose of sparse matrix."""

    row, col = index
    index = torch.stack([col, row], dim=0)

    index, value = coalesce(index, value, n, m)

    return index, value
