from typing import Optional

import torch
from torch import Tensor
from torch_scatter import gather_csr

from torch_sparse.tensor import SparseTensor


@torch.jit._overload  # noqa: F811
def mul(src, other):  # noqa: F811
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


@torch.jit._overload  # noqa: F811
def mul(src, other):  # noqa: F811
    # type: (SparseTensor, SparseTensor) -> SparseTensor
    pass


def mul(src, other):  # noqa: F811
    if isinstance(other, Tensor):
        rowptr, col, value = src.csr()
        if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise...
            other = gather_csr(other.squeeze(1), rowptr)
            pass
        # Col-wise...
        elif other.size(0) == 1 and other.size(1) == src.size(1):
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

    assert isinstance(other, SparseTensor)

    if not src.is_coalesced():
        raise ValueError("The `src` tensor is not coalesced")
    if not other.is_coalesced():
        raise ValueError("The `other` tensor is not coalesced")

    rowA, colA, valueA = src.coo()
    rowB, colB, valueB = other.coo()

    row = torch.cat([rowA, rowB], dim=0)
    col = torch.cat([colA, colB], dim=0)

    if valueA is not None and valueB is not None:
        value = torch.cat([valueA, valueB], dim=0)
    else:
        raise ValueError('Both sparse tensors must contain values')

    M = max(src.size(0), other.size(0))
    N = max(src.size(1), other.size(1))
    sparse_sizes = (M, N)

    # Sort indices:
    idx = col.new_full((col.numel() + 1, ), -1)
    idx[1:] = row * sparse_sizes[1] + col
    perm = idx[1:].argsort()
    idx[1:] = idx[1:][perm]

    row, col, value = row[perm], col[perm], value[perm]

    valid_mask = idx[1:] == idx[:-1]
    valid_idx = valid_mask.nonzero().view(-1)

    return SparseTensor(
        row=row[valid_mask],
        col=col[valid_mask],
        value=value[valid_idx - 1] * value[valid_idx],
        sparse_sizes=sparse_sizes,
    )


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


def mul_nnz(
    src: SparseTensor,
    other: torch.Tensor,
    layout: Optional[str] = None,
) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.mul(other.to(value.dtype))
    else:
        value = other
    return src.set_value(value, layout=layout)


def mul_nnz_(
    src: SparseTensor,
    other: torch.Tensor,
    layout: Optional[str] = None,
) -> SparseTensor:
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
