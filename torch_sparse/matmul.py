import warnings
import os.path as osp
from typing import Optional, Union, Tuple

import torch
from torch_sparse.tensor import SparseTensor

try:
    torch.ops.load_library(
        osp.join(osp.dirname(osp.abspath(__file__)), '_spmm.so'))
except OSError:
    warnings.warn('Failed to load `spmm` binaries.')

    def spmm_sum_placeholder(row: Optional[torch.Tensor], rowptr: torch.Tensor,
                             col: torch.Tensor, value: Optional[torch.Tensor],
                             colptr: Optional[torch.Tensor],
                             csr2csc: Optional[torch.Tensor],
                             mat: torch.Tensor) -> torch.Tensor:
        raise ImportError
        return mat

    def spmm_mean_placeholder(row: Optional[torch.Tensor],
                              rowptr: torch.Tensor, col: torch.Tensor,
                              value: Optional[torch.Tensor],
                              rowcount: Optional[torch.Tensor],
                              colptr: Optional[torch.Tensor],
                              csr2csc: Optional[torch.Tensor],
                              mat: torch.Tensor) -> torch.Tensor:
        raise ImportError
        return mat

    def spmm_min_max_placeholder(rowptr: torch.Tensor, col: torch.Tensor,
                                 value: Optional[torch.Tensor],
                                 mat: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise ImportError
        return mat, mat

    torch.ops.torch_sparse.spmm_sum = spmm_sum_placeholder
    torch.ops.torch_sparse.spmm_mean = spmm_mean_placeholder
    torch.ops.torch_sparse.spmm_min = spmm_min_max_placeholder
    torch.ops.torch_sparse.spmm_max = spmm_min_max_placeholder

try:
    torch.ops.load_library(
        osp.join(osp.dirname(osp.abspath(__file__)), '_spspmm.so'))
except OSError:
    warnings.warn('Failed to load `spspmm` binaries.')

    def spspmm_sum_placeholder(
            rowptrA: torch.Tensor, colA: torch.Tensor,
            valueA: Optional[torch.Tensor], rowptrB: torch.Tensor,
            colB: torch.Tensor, valueB: Optional[torch.Tensor], K: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise ImportError
        return rowptrA, colA, valueA

    torch.ops.torch_sparse.spspmm_sum = spspmm_sum_placeholder


@torch.jit.script
def spmm_sum(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row = src.storage._row
    csr2csc = src.storage._csr2csc
    colptr = src.storage._colptr

    if value is not None and value.requires_grad:
        row = src.storage.row()

    if other.requires_grad:
        row = src.storage.row()
        csr2csc = src.storage.csr2csc()
        colptr = src.storage.colptr()

    return torch.ops.torch_sparse.spmm_sum(row, rowptr, col, value, colptr,
                                           csr2csc, other)


@torch.jit.script
def spmm_add(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    return spmm_sum(src, other)


@torch.jit.script
def spmm_mean(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row = src.storage._row
    rowcount = src.storage._rowcount
    csr2csc = src.storage._csr2csc
    colptr = src.storage._colptr

    if value is not None and value.requires_grad:
        row = src.storage.row()

    if other.requires_grad:
        row = src.storage.row()
        rowcount = src.storage.rowcount()
        csr2csc = src.storage.csr2csc()
        colptr = src.storage.colptr()

    return torch.ops.torch_sparse.spmm_mean(row, rowptr, col, value, rowcount,
                                            colptr, csr2csc, other)


@torch.jit.script
def spmm_min(src: SparseTensor,
             other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rowptr, col, value = src.csr()
    return torch.ops.torch_sparse.spmm_min(rowptr, col, value, other)


@torch.jit.script
def spmm_max(src: SparseTensor,
             other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rowptr, col, value = src.csr()
    return torch.ops.torch_sparse.spmm_max(rowptr, col, value, other)


@torch.jit.script
def spmm(src: SparseTensor, other: torch.Tensor,
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


@torch.jit.script
def spspmm_sum(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    assert src.sparse_size(1) == other.sparse_size(0)
    rowptrA, colA, valueA = src.csr()
    rowptrB, colB, valueB = other.csr()
    M, K = src.sparse_size(0), other.sparse_size(1)
    rowptrC, colC, valueC = torch.ops.torch_sparse.spspmm_sum(
        rowptrA, colA, valueA, rowptrB, colB, valueB, K)
    return SparseTensor(row=None, rowptr=rowptrC, col=colC, value=valueC,
                        sparse_sizes=torch.Size([M, K]), is_sorted=True)


@torch.jit.script
def spspmm_add(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    return spspmm_sum(src, other)


@torch.jit.script
def spspmm(src: SparseTensor, other: SparseTensor,
           reduce: str = "sum") -> SparseTensor:
    if reduce == 'sum' or reduce == 'add':
        return spspmm_sum(src, other)
    elif reduce == 'mean' or reduce == 'min' or reduce == 'max':
        raise NotImplementedError
    else:
        raise ValueError


def matmul(src: SparseTensor, other: Union[torch.Tensor, SparseTensor],
           reduce: str = "sum"):
    if torch.is_tensor(other):
        return spmm(src, other, reduce)
    elif isinstance(other, SparseTensor):
        return spspmm(src, other, reduce)
    else:
        raise ValueError


SparseTensor.spmm = lambda self, other, reduce=None: spmm(self, other, reduce)
SparseTensor.spspmm = lambda self, other, reduce=None: spspmm(
    self, other, reduce)
SparseTensor.matmul = lambda self, other, reduce=None: matmul(
    self, other, reduce)
SparseTensor.__matmul__ = lambda self, other: matmul(self, other, 'sum')
