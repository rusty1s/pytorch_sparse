from typing import Optional

import torch
from torch_scatter import gather_csr
from torch_sparse.tensor import SparseTensor


def mul(src: SparseTensor, other: torch.Tensor) -> SparseTensor:
    rowptr, col, value = src.csr()
    if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise...
        other = gather_csr(other.squeeze(1), rowptr)
        pass
    elif other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise...
        other = other.squeeze(0)[col]
    else:
        raise ValueError(
            f'Size mismatch: Expected size ({src.size(0)}, 1, ...) or '
            f'(1, {src.size(1)}, ...), but got size {other.size()}.')

    if value is not None:
        value = other.to(value.dtype).mul_(value)
    else:
        value = other
    return src.set_value(value, layout='coo')


def mul_(src: SparseTensor, other: torch.Tensor) -> SparseTensor:
    rowptr, col, value = src.csr()
    if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise...
        other = gather_csr(other.squeeze(1), rowptr)
        pass
    elif other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise...
        other = other.squeeze(0)[col]
    else:
        raise ValueError(
            f'Size mismatch: Expected size ({src.size(0)}, 1, ...) or '
            f'(1, {src.size(1)}, ...), but got size {other.size()}.')

    if value is not None:
        value = value.mul_(other.to(value.dtype))
    else:
        value = other
    return src.set_value_(value, layout='coo')


def mul_nnz(src: SparseTensor, other: torch.Tensor,
            layout: Optional[str] = None) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.mul(other.to(value.dtype))
    else:
        value = other
    return src.set_value(value, layout=layout)


def mul_nnz_(src: SparseTensor, other: torch.Tensor,
             layout: Optional[str] = None) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.mul_(other.to(value.dtype))
    else:
        value = other
    return src.set_value_(value, layout=layout)


SparseTensor.mul = lambda self, other: mul(self, other)
SparseTensor.mul_ = lambda self, other: mul_(self, other)
SparseTensor.mul_nnz = lambda self, other, layout=None: mul_nnz(
    self, other, layout)
SparseTensor.mul_nnz_ = lambda self, other, layout=None: mul_nnz_(
    self, other, layout)
SparseTensor.__mul__ = SparseTensor.mul
SparseTensor.__rmul__ = SparseTensor.mul
SparseTensor.__imul__ = SparseTensor.mul_
