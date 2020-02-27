from typing import Tuple

import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.permute import permute


def partition(
    src: SparseTensor, num_parts: int, recursive: bool = False
) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:

    rowptr, col = src.storage.rowptr().cpu(), src.storage.col().cpu()
    cluster = torch.ops.torch_sparse.partition(rowptr, col, num_parts,
                                               recursive)
    cluster = cluster.to(src.device())

    cluster, perm = cluster.sort()
    out = permute(src, perm)
    partptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

    return out, partptr, perm


SparseTensor.partition = lambda self, num_parts, recursive=False: partition(
    self, num_parts, recursive)
