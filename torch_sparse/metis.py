from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_sparse.permute import permute
from torch_sparse.tensor import SparseTensor


def weight2metis(weight: Tensor) -> Optional[Tensor]:
    sorted_weight = weight.sort()[0]
    diff = sorted_weight[1:] - sorted_weight[:-1]
    if diff.sum() == 0:
        return None
    weight_min, weight_max = sorted_weight[0], sorted_weight[-1]
    srange = weight_max - weight_min
    min_diff = diff.min()
    scale = (min_diff / srange).item()
    tick, arange = scale.as_integer_ratio()
    weight_ratio = (weight - weight_min).div_(srange).mul_(arange).add_(tick)
    return weight_ratio.to(torch.long)


def partition(
    src: SparseTensor,
    num_parts: int,
    recursive: bool = False,
    weighted: bool = False,
    node_weight: Optional[Tensor] = None,
    balance_edge: bool = False,
) -> Tuple[SparseTensor, Tensor, Tensor]:

    assert num_parts >= 1
    if num_parts == 1:
        partptr = torch.tensor([0, src.size(0)], device=src.device())
        perm = torch.arange(src.size(0), device=src.device())
        return src, partptr, perm

    if balance_edge and node_weight:
        raise ValueError("Cannot set 'balance_edge' and 'node_weight' at the "
                         "same time in 'torch_sparse.partition'")

    rowptr, col, value = src.csr()
    rowptr, col = rowptr.cpu(), col.cpu()

    if value is not None and weighted:
        assert value.numel() == col.numel()
        value = value.view(-1).detach().cpu()
        if value.is_floating_point():
            value = weight2metis(value)
    else:
        value = None

    if balance_edge:
        node_weight = col.new_zeros(rowptr.numel() - 1)
        node_weight.scatter_add_(0, col, torch.ones_like(col))

    if node_weight is not None:
        assert node_weight.numel() == rowptr.numel() - 1
        node_weight = node_weight.view(-1).detach().cpu()
        if node_weight.is_floating_point():
            node_weight = weight2metis(node_weight)
        cluster = torch.ops.torch_sparse.partition2(rowptr, col, value,
                                                    node_weight, num_parts,
                                                    recursive)
    else:
        cluster = torch.ops.torch_sparse.partition(rowptr, col, value,
                                                   num_parts, recursive)
    cluster = cluster.to(src.device())

    cluster, perm = cluster.sort()
    out = permute(src, perm)
    partptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

    return out, partptr, perm


SparseTensor.partition = partition
