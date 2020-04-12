from typing import Tuple, List

import torch
from torch_sparse.tensor import SparseTensor


def padded_index(src: SparseTensor, binptr: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.
                            Tensor, List[int], List[int]]:
    return torch.ops.torch_sparse.padded_index(src.storage.rowptr(),
                                               src.storage.col(),
                                               src.storage.rowcount(), binptr)


def padded_index_select(src: torch.Tensor, index: torch.Tensor,
                        fill_value: float = 0.) -> torch.Tensor:
    fill_value = torch.tensor(fill_value, dtype=src.dtype)
    return torch.ops.torch_sparse.padded_index_select(src, index, fill_value)


SparseTensor.padded_index = padded_index
