from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_sparse.tensor import SparseTensor


def spmm_sum(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row = src.storage._row
    csr2csc = src.storage._csr2csc
    colptr = src.storage._colptr

    if value is not None:
        value = value.to(other.dtype)

    if value is not None and value.requires_grad:
        row = src.storage.row()

    if other.requires_grad:
        row = src.storage.row()
        csr2csc = src.storage.csr2csc()
        colptr = src.storage.colptr()

    return torch.ops.torch_sparse.spmm_sum(row, rowptr, col, value, colptr,
                                           csr2csc, other)


def spmm_add(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    return spmm_sum(src, other)


def spmm_mean(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row = src.storage._row
    rowcount = src.storage._rowcount
    csr2csc = src.storage._csr2csc
    colptr = src.storage._colptr

    if value is not None:
        value = value.to(other.dtype)

    if value is not None and value.requires_grad:
        row = src.storage.row()

    if other.requires_grad:
        row = src.storage.row()
        rowcount = src.storage.rowcount()
        csr2csc = src.storage.csr2csc()
        colptr = src.storage.colptr()

    return torch.ops.torch_sparse.spmm_mean(row, rowptr, col, value, rowcount,
                                            colptr, csr2csc, other)


def spmm_min(src: SparseTensor,
             other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rowptr, col, value = src.csr()

    if value is not None:
        value = value.to(other.dtype)

    return torch.ops.torch_sparse.spmm_min(rowptr, col, value, other)


def spmm_max(src: SparseTensor,
             other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rowptr, col, value = src.csr()

    if value is not None:
        value = value.to(other.dtype)

    return torch.ops.torch_sparse.spmm_max(rowptr, col, value, other)


def spmm(src: SparseTensor,
         other: torch.Tensor,
         reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return spmm_sum(src, other)
    elif reduce == 'mean':
        return spmm_mean(src, other)
    elif reduce == 'min':
        return spmm_min(src, other)[0]
    elif reduce == 'max':
        return spmm_max(src, other)[0]
    else:
        raise ValueError


def spspmm_sum(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    A = src.to_torch_sparse_coo_tensor()
    B = other.to_torch_sparse_coo_tensor()
    C = torch.sparse.mm(A, B)
    edge_index = C._indices()
    row, col = edge_index[0], edge_index[1]
    value: Optional[Tensor] = None
    if src.has_value() or other.has_value():
        value = C._values()

    return SparseTensor(
        row=row,
        col=col,
        value=value,
        sparse_sizes=(C.size(0), C.size(1)),
        is_sorted=True,
        trust_data=True,
    )


def spspmm_add(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    return spspmm_sum(src, other)


def spspmm(src: SparseTensor,
           other: SparseTensor,
           reduce: str = "sum") -> SparseTensor:
    if reduce == 'sum' or reduce == 'add':
        return spspmm_sum(src, other)
    elif reduce == 'mean' or reduce == 'min' or reduce == 'max':
        raise NotImplementedError
    else:
        raise ValueError


@torch.jit._overload  # noqa: F811
def matmul(src, other, reduce):  # noqa: F811
    # type: (SparseTensor, torch.Tensor, str) -> torch.Tensor
    pass


@torch.jit._overload  # noqa: F811
def matmul(src, other, reduce):  # noqa: F811
    # type: (SparseTensor, SparseTensor, str) -> SparseTensor
    pass


def matmul(src, other, reduce="sum"):  # noqa: F811
    """Matrix product of a sparse tensor with either another sparse tensor or a
     dense tensor. The sparse tensor represents an adjacency matrix and is
     stored as a list of edges. This method multiplies elements along the rows
     of the adjacency matrix with the column of the other matrix. In regular
     matrix multiplication, the products are then summed together, but this
     method allows us to use other aggregation functions as well.

    Args:
        src (:class:`SparseTensor`): The sparse tensor.
        other (:class:`Tensor` or :class:`SparseTensor`): The second matrix.
        reduce (string, optional): The function to reduce along the rows of
            :obj:`src` and columns of :obj:`other`. Can be :obj:`"sum"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`.
            (default: :obj:`"sum"`)

    :rtype: (:class:`Tensor`)
    """
    if isinstance(other, torch.Tensor):
        return spmm(src, other, reduce)
    elif isinstance(other, SparseTensor):
        return spspmm(src, other, reduce)
    raise ValueError


SparseTensor.spmm = lambda self, other, reduce="sum": spmm(self, other, reduce)
SparseTensor.spspmm = lambda self, other, reduce="sum": spspmm(
    self, other, reduce)
SparseTensor.matmul = lambda self, other, reduce="sum": matmul(
    self, other, reduce)
SparseTensor.__matmul__ = lambda self, other: matmul(self, other, 'sum')
