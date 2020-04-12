from typing import Tuple

import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.permute import permute
from torch_sparse.utils import cartesian1d


def metis_wgt(x):
    t1, t2 = cartesian1d(x, x)
    diff = t1 - t2
    diff = diff[diff != 0]
    res = diff.abs().min()
    bod = x.max() - x.min()
    scale = (res / bod).item()
    tick, arange = scale.as_integer_ratio()
    x_ratio = (x - x.min()) / bod
    return (x_ratio * arange + tick).long(), tick, arange


def partition(src: SparseTensor, num_parts: int, recursive: bool = False
              ) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:
    rowptr, col = src.storage.rowptr().cpu(), src.storage.col().cpu()
    edge_wgt = src.storage.value().cpu()
    edge_wgt = metis_wgt(edge_wgt)[0]
    cluster = torch.ops.torch_sparse.partition(rowptr, col, num_parts, edge_wgt,
                                               recursive)
    cluster = cluster.to(src.device())

    cluster, perm = cluster.sort()
    out = permute(src, perm)
    partptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

    return out, partptr, perm


SparseTensor.partition = lambda self, num_parts, recursive=False: partition(
    self, num_parts, recursive)
