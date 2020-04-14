from typing import Tuple

import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.permute import permute


def cartesian1d(x, y):
    a1, a2 = torch.meshgrid([x, y])
    coos = torch.stack([a1, a2]).T.reshape(-1, 2)
    return coos.split(1, dim=1)


def metis_weight1(x):
    sorted_x = x.sort()[0]
    diff = sorted_x[1:] - sorted_x[:-1]
    if diff.sum() == 0:
        return None
    xmin, xmax = sorted_x[[0, -1]]
    srange = xmax - xmin
    min_diff = diff.min()
    scale = (min_diff / srange).item()
    tick, arange = scale.as_integer_ratio()
    x_ratio = (x - xmin) / srange
    return (x_ratio * arange + tick).long()


def metis_weight2(x):
    t1, t2 = cartesian1d(x, x)
    diff = t1 - t2
    diff = diff[diff != 0]
    if len(diff) == 0:
        return None
    xmin, xmax = x.min(), x.max()
    srange = xmax - xmin
    min_diff = diff.abs().min()
    scale = (min_diff / srange).item()
    tick, arange = scale.as_integer_ratio()
    x_ratio = (x - xmin) / srange
    return (x_ratio * arange + tick).long()


def metis_weight(x, sort_strategy=True):
    return metis_weight1(x) if sort_strategy else metis_weight2(x)


def partition(src: SparseTensor, num_parts: int, recursive: bool = False, sort_strategy=True,
              ) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:
    rowptr, col, value = src.csr()
    rowptr, col = rowptr.cpu(), col.cpu()
    if value is not None and value.dim() == 1:
        value = value.detach().cpu()
        value = metis_weight(value, sort_strategy)
    cluster = torch.ops.torch_sparse.partition(rowptr, col, value, num_parts,
                                               recursive)
    cluster = cluster.to(src.device())

    cluster, perm = cluster.sort()
    out = permute(src, perm)
    partptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

    return out, partptr, perm


SparseTensor.partition = partition
