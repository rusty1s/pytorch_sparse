import scipy.sparse as sp
from typing import Tuple, Optional

import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.permute import permute


def reverse_cuthill_mckee(src: SparseTensor,
                          is_symmetric: Optional[bool] = None
                          ) -> Tuple[SparseTensor, torch.Tensor]:

    if is_symmetric is None:
        is_symmetric = src.is_symmetric()

    if not is_symmetric:
        src = src.to_symmetric()

    sp_src = src.to_scipy(layout='csr')
    perm = sp.csgraph.reverse_cuthill_mckee(sp_src, symmetric_mode=True).copy()
    perm = torch.from_numpy(perm).to(torch.long).to(src.device())

    out = permute(src, perm)

    return out, perm


SparseTensor.reverse_cuthill_mckee = reverse_cuthill_mckee
