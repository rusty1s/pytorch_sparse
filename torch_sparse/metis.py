from typing import Tuple, Optional

import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.permute import permute

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def degree(index: Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.
    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.
    :rtype: :class:`Tensor`
    Example:
        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)



def weight2metis(weight: torch.Tensor) -> Optional[torch.Tensor]:
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
    src: SparseTensor, num_parts: int, recursive: bool = False,
    weighted: bool = False, node_weight: Optional[torch.Tensor] = None,
    balance_edge: bool = False
) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:

    assert num_parts >= 1
    if num_parts == 1:
        partptr = torch.tensor([0, src.size(0)], device=src.device())
        perm = torch.arange(src.size(0), device=src.device())
        return src, partptr, perm

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
        idx = col
        deg = degree(idx, src.size(0), dtype=torch.long)
        node_weight = deg

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
