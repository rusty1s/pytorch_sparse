import torch
from torch_sparse import coalesce


def transpose(index, value, m, n):
    row, col = index
    index = torch.stack([col, row], dim=0)

    index, value = coalesce(index, value, m, n)

    return index, value
