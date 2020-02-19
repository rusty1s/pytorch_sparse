import torch
from torch_sparse.tensor import SparseTensor


def permute(src: SparseTensor, perm: torch.Tensor) -> SparseTensor:
    assert src.is_quadratic()
    return src.index_select(0, perm).index_select(1, perm)


SparseTensor.permute = lambda self, perm: permute(self, perm)
