from typing import Tuple

import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.permute import permute
from torch_sparse.utils import cartesian1d


def metis_wgt(x):
    t1, t2 = cartesian1d(x, x)
    diff = t1 - t2
    diff = diff[diff != 0]
    if len(diff) == 0:
        return torch.ones(x.shape, dtype=torch.long)
    res = diff.abs().min()
    bod = x.max() - x.min()
    scale = (res / bod).item()
    tick, arange = scale.as_integer_ratio()
    x_ratio = (x - x.min()) / bod
    return (x_ratio * arange + tick).long()


def partition(src: SparseTensor, num_parts: int, recursive: bool = False
              ) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:
    rowptr, col = src.storage.rowptr().cpu(), src.storage.col().cpu()
    edge_wgt = src.storage.value().cpu()
    edge_wgt = metis_wgt(edge_wgt)
    cluster = torch.ops.torch_sparse.partition(rowptr, col, num_parts, edge_wgt,
                                               recursive)
    cluster = cluster.to(src.device())

    cluster, perm = cluster.sort()
    out = permute(src, perm)
    partptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

    return out, partptr, perm


SparseTensor.partition = partition
