from torch_sparse.tensor import SparseTensor
from torch_sparse.narrow import narrow


def select(src: SparseTensor, dim: int, idx: int) -> SparseTensor:
    return narrow(src, dim, start=idx, length=1)


SparseTensor.select = lambda self, dim, idx: select(self, dim, idx)
