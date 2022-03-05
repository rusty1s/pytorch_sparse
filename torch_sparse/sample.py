from typing import Optional, Tuple

import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.tensor import DynamicSparseTensor


def sample(src: SparseTensor, num_neighbors: int,
           subset: Optional[torch.Tensor] = None) -> torch.Tensor:

    rowptr, col, _ = src.csr()
    rowcount = src.storage.rowcount()

    if subset is not None:
        rowcount = rowcount[subset]
        rowptr = rowptr[subset]
    else:
        rowptr = rowptr[:-1]

    rand = torch.rand((rowcount.size(0), num_neighbors), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.view(-1, 1))

    return col[rand]


def sample_adj(src: SparseTensor, subset: torch.Tensor, num_neighbors: int,
               replace: bool = False) -> Tuple[SparseTensor, torch.Tensor]:

    rowptr, col, value = src.csr()

    rowptr, col, n_id, e_id = torch.ops.torch_sparse.sample_adj(
        rowptr, col, subset, num_neighbors, replace)

    if value is not None:
        value = value[e_id]

    out = SparseTensor(rowptr=rowptr, row=None, col=col, value=value,
                       sparse_sizes=(subset.size(0), n_id.size(0)),
                       is_sorted=True)

    return out, n_id


def sample_adj_dynamic(src: DynamicSparseTensor, subset: torch.Tensor, num_neighbors: int,
               replace: bool = False, realtime: bool = True) -> Tuple[SparseTensor, torch.Tensor]:
    if realtime:
        return src.storage.sample_realtime(subset, num_neighbors, replace)
    else:
        return src.storage.sample_stale(subset, num_neighbors, replace)


SparseTensor.sample = sample
SparseTensor.sample_adj = sample_adj

DynamicSparseTensor.sample_adj = sample_adj_dynamic
