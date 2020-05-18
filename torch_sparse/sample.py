from typing import Optional

import torch
from torch_sparse.tensor import SparseTensor


def sample(src: SparseTensor, num_neighbors: int,
           subset: Optional[torch.Tensor] = None) -> torch.Tensor:

    rowptr, col, _ = src.csr()
    rowcount = src.storage.rowcount()

    if subset is not None:
        rowcount = rowcount[subset]
        rowptr = rowptr[subset]

    rand = torch.rand((rowcount.size(0), num_neighbors), device=col.device)
    rand = rand.mul_(rowcount.to(rand.dtype)).to(torch.long).add_(rowptr)

    return col[rand]


SparseTensor.sample = sample
