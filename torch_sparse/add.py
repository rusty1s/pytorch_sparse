from typing import Optional

import torch
from torch import Tensor
from torch_scatter import gather_csr
from torch_sparse.tensor import SparseTensor


@torch.jit._overload  # noqa: F811
def add(src, other):  # noqa: F811
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


@torch.jit._overload  # noqa: F811
def add(src, other):  # noqa: F811
    # type: (SparseTensor, SparseTensor) -> SparseTensor
    pass


def add(src, other):  # noqa: F811
    if isinstance(other, Tensor):
        rowptr, col, value = src.csr()
        if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise.
            other = gather_csr(other.squeeze(1), rowptr)
        elif other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise.
            other = other.squeeze(0)[col]
        else:
            raise ValueError(
                f'Size mismatch: Expected size ({src.size(0)}, 1, ...) or '
                f'(1, {src.size(1)}, ...), but got size {other.size()}.')
        if value is not None:
            value = other.to(value.dtype).add_(value)
        else:
            value = other.add_(1)
        return src.set_value(value, layout='coo')

    elif isinstance(other, SparseTensor):
        rowA, colA, valueA = src.coo()
        rowB, colB, valueB = other.coo()

        row = torch.cat([rowA, rowB], dim=0)
        col = torch.cat([colA, colB], dim=0)

        value: Optional[Tensor] = None
        if valueA is not None and valueB is not None:
            value = torch.cat([valueA, valueB], dim=0)

        M = max(src.size(0), other.size(0))
        N = max(src.size(1), other.size(1))
        sparse_sizes = (M, N)

        out = SparseTensor(row=row, col=col, value=value,
                           sparse_sizes=sparse_sizes)
        out = out.coalesce(reduce='sum')
        return out

    else:
        raise NotImplementedError


def add_(src: SparseTensor, other: torch.Tensor) -> SparseTensor:
    rowptr, col, value = src.csr()
    if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise.
        other = gather_csr(other.squeeze(1), rowptr)
    elif other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise.
        other = other.squeeze(0)[col]
    else:
        raise ValueError(
            f'Size mismatch: Expected size ({src.size(0)}, 1, ...) or '
            f'(1, {src.size(1)}, ...), but got size {other.size()}.')

    if value is not None:
        value = value.add_(other.to(value.dtype))
    else:
        value = other.add_(1)
    return src.set_value_(value, layout='coo')


def add_nnz(src: SparseTensor, other: torch.Tensor,
            layout: Optional[str] = None) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.add(other.to(value.dtype))
    else:
        value = other.add(1)
    return src.set_value(value, layout=layout)


def add_nnz_(src: SparseTensor, other: torch.Tensor,
             layout: Optional[str] = None) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.add_(other.to(value.dtype))
    else:
        value = other.add(1)
    return src.set_value_(value, layout=layout)


SparseTensor.add = lambda self, other: add(self, other)
SparseTensor.add_ = lambda self, other: add_(self, other)
SparseTensor.add_nnz = lambda self, other, layout=None: add_nnz(
    self, other, layout)
SparseTensor.add_nnz_ = lambda self, other, layout=None: add_nnz_(
    self, other, layout)
SparseTensor.__add__ = SparseTensor.add
SparseTensor.__radd__ = SparseTensor.add
SparseTensor.__iadd__ = SparseTensor.add_
