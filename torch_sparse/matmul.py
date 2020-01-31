import warnings
import os.path as osp
from typing import Optional, Union

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

    torch.ops.torch_sparse.spmm_sum = spmm_sum_placeholder


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
def spmm(src: SparseTensor, other: torch.Tensor,
         reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return spmm_sum(src, other)
    else:
        raise ValueError


def matmul(src: SparseTensor, other: Union[torch.Tensor, SparseTensor],
           reduce: str = "sum"):
    if torch.is_tensor(other):
        return spmm(src, other, reduce)
    else:
        raise ValueError


SparseTensor.spmm = lambda self, other, reduce=None: spmm(self, other, reduce)
SparseTensor.matmul = lambda self, other, reduce=None: matmul(
    self, other, reduce)
SparseTensor.__matmul__ = lambda self, other: matmul(self, other, 'sum')

# class SPMM(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, row, rowptr, col, value, mat, rowcount, colptr, csr2csc,
#                 reduce):
#         if mat.is_cuda:
#             out, arg_out = torch.ops.torch_sparse_cuda.spmm(
#                 rowptr, col, value, mat, reduce)
#         else:
#             out, arg_out = torch.ops.torch_sparse_cpu.spmm(
#                 rowptr, col, value, mat, reduce)

#         ctx.reduce = reduce
#         ctx.save_for_backward(row, rowptr, col, value, mat, rowcount, colptr,
#                               csr2csc, arg_out)

#         if reduce == 'min' or reduce == 'max':
#             ctx.mark_non_differentiable(arg_out)
#             return out, arg_out
#         else:
#             return out

#     @staticmethod
#     def backward(ctx, grad_out, *args):
#         (row, rowptr, col, value, mat, rowcount, colptr, csr2csc,
#          arg_out) = ctx.saved_tensors

#         invalid_arg_mask = arg_out_ind = None
#         if ctx.reduce in ['min', 'max'] and (ctx.needs_input_grad[3]
#                                              or ctx.needs_input_grad[4]):
#             invalid_arg_mask = arg_out == col.size(0)
#             arg_out_ind = arg_out.masked_fill(invalid_arg_mask, -1)

#         grad_value = None
#         if ctx.needs_input_grad[3]:
#             if ctx.reduce in ['sum', 'add', 'mean']:
#                 grad_value = ext(grad_out.is_cuda).spmm_val_bw(
#                     row, rowptr, col, mat, grad_out, ctx.reduce)

#             elif ctx.reduce in ['min', 'max']:
#                 col_tmp = col[arg_out_ind.flatten()].view_as(arg_out)
#                 out = mat.gather(-2, col_tmp).mul_(grad_out)
#                 out.masked_fill_(invalid_arg_mask, 0)
#                 grad_value = scatter_add(out.flatten(), arg_out.flatten(),
#                                          dim=0, dim_size=value.numel() + 1)
#                 grad_value = grad_value[:-1]

#         grad_mat = None
#         if ctx.needs_input_grad[4]:
#             if ctx.reduce in ['sum', 'add']:
#                 value = value[csr2csc] if value is not None else value
#                 grad_mat, _ = ext(grad_out.is_cuda).spmm(
#                     colptr, row[csr2csc], value, grad_out, 'sum')

#             elif ctx.reduce == 'mean':
#                 count = rowcount[row].to(mat.dtype).clamp_(min=1)
#                 value = count.pow_(-1) if value is None else value / count
#                 row = row[csr2csc]
#                 value = value[csr2csc] if value is not None else value
#                 grad_mat, _ = ext(grad_out.is_cuda).spmm(
#                     colptr, row, value, grad_out, 'sum')

#             elif ctx.reduce in ['min', 'max']:
#                 if value is not None:
#                     value = value[arg_out_ind.flatten()].view_as(arg_out)
#                     value = value.mul_(grad_out)
#                 else:
#                     value = grad_out
#                 value.masked_fill_(invalid_arg_mask, 0)
#                 col_tmp = col[arg_out_ind.flatten()].view_as(arg_out)
#                 grad_mat = scatter_add(value, col_tmp, dim=-2,
#                                        dim_size=mat.size(-2))

#         return None, None, None, grad_value, grad_mat, None, None, None, None

# class SPSPMM(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, rowptrA, colA, valueA, rowptrB, colB, valueB, M, N, K):
#         if rowptrA.is_cuda:
#             rowptrC, colC, valueC = ext(True).spspmm(rowptrA, colA, valueA,
#                                                      rowptrB, colB, valueB, M,
#                                                      N, K)
#         else:
#             dtype = None
#             if valueA is not None:
#                 dtype = valueA.dtype
#             if valueB is not None:
#                 dtype = valueB.dtype

#             if valueA is None:
#                 valueA = torch.ones(colA.numel(), dtype=dtype)
#             A = scipy.sparse.csr_matrix((valueA, colA, rowptrA), (M, N))

#             if valueB is None:
#                 valueB = torch.ones(colB.numel(), dtype=dtype)
#             B = scipy.sparse.csr_matrix((valueB, colB, rowptrB), (N, K))

#             C = A @ B

#             rowptrC = torch.from_numpy(C.indptr).to(torch.int64)
#             colC = torch.from_numpy(C.indices).to(torch.int64)
#             valueC = torch.from_numpy(C.data)
#             valueC = valueC.to(dtype) if dtype is not None else None

#         ctx.mark_non_differentiable(rowptrC, colC)

#         # We cannot return `NoneType` in torch.autograd :(
#         if valueC is None:
#             return rowptrC, colC
#         else:
#             return rowptrC, colC, valueC

#     @staticmethod
#     def backward(ctx, grad_indexC, grad_rowptrC, *args):
#         grad_valueA = None
#         if ctx.needs_input_grad[2]:
#             raise NotImplementedError

#         grad_valueB = None
#         if ctx.needs_input_grad[5]:
#             raise NotImplementedError

#         return (None, None, grad_valueA, None, None, grad_valueB, None, None,
#                 None)

# def matmul(src, other, reduce='sum'):
#     assert src.dim() == 2 and src.size(-1) == other.size(-2)

#     # Sparse-Dense Matrix Multiplication.
#     if torch.is_tensor(other):
#         assert reduce in ['sum', 'add', 'mean', 'min', 'max']
#         rowptr, col, value = src.csr()

#         row = None
#         if reduce in ['sum', 'add', 'mean'] and (src.requires_grad
#                                                  or other.requires_grad):
#             row = src.storage.row

#         rowcount = None
#         if other.requires_grad and reduce in ['mean']:
#             rowcount = src.storage.rowcount

#         csr2csc = colptr = None
#         if other.requires_grad and reduce in ['sum', 'add', 'mean']:
#             csr2csc, colptr = src.storage.csr2csc, src.storage.colptr

#         return SPMM.apply(row, rowptr, col, value, other, rowcount, colptr,
#                           csr2csc, reduce)

#     # Sparse-Sparse Matrix Multiplication.
#     elif isinstance(other, src.__class__):
#         assert reduce in ['sum', 'add']
#         assert src.dim() == 2 and other.dim() == 2
#         data = SPSPMM.apply(*src.csr(), *other.csr(), src.size(0), src.size(1),
#                             other.size(1))
#         (rowptr, col), value = data[:2], data[2] if len(data) == 3 else None
#         sparse_size = torch.Size([src.size(0), other.size(1)])
#         return src.__class__(rowptr=rowptr, col=col, value=value,
#                              sparse_size=sparse_size, is_sorted=True)

#     raise ValueError
