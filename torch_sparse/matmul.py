import torch
import scipy.sparse
from torch_sparse import spmm_cpu
from torch_scatter import scatter_add

try:
    from torch_sparse import spmm_cuda
except ImportError:
    spmm_cuda = None

try:
    from torch_sparse import spspmm_cuda
except ImportError:
    spspmm_cuda = None


def spmm(is_cuda):
    return spmm_cuda if is_cuda else spmm_cpu


class SPMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, row, rowptr, col, value, mat, rowcount, colptr, csr2csc,
                reduce):
        out, arg_out = spmm(mat.is_cuda).spmm(rowptr, col, value, mat, reduce)

        ctx.reduce = reduce
        ctx.save_for_backward(row, rowptr, col, value, mat, rowcount, colptr,
                              csr2csc, arg_out)

        if reduce == 'min' or reduce == 'max':
            ctx.mark_non_differentiable(arg_out)
            return out, arg_out
        else:
            return out

    @staticmethod
    def backward(ctx, grad_out, *args):
        (row, rowptr, col, value, mat, rowcount, colptr, csr2csc,
         arg_out) = ctx.saved_tensors

        invalid_arg_mask = arg_out_ind = None
        if ctx.reduce in ['min', 'max'] and (ctx.needs_input_grad[5]
                                             or ctx.needs_input_grad[6]):
            invalid_arg_mask = arg_out == row.size(0)
            arg_out_ind = arg_out.masked_fill(invalid_arg_mask, -1)

        grad_value = None
        if ctx.needs_input_grad[3]:
            if ctx.reduce in ['sum', 'add']:
                grad_value = spmm(grad_out.is_cuda).spmm_val_bw(
                    row, rowptr, col, mat, grad_out, ctx.reduce)

            if ctx.reduce == 'mean':
                grad_value = spmm(grad_out.is_cuda).spmm_val_bw(
                    row, rowptr, col, mat, grad_out, ctx.reduce)

            elif ctx.reduce in ['min', 'max']:
                col = col[arg_out_ind.flatten()].view_as(arg_out)
                out = mat.gather(-2, col).mul_(grad_out)
                out.masked_fill_(invalid_arg_mask, 0)
                grad_value = scatter_add(out.flatten(), arg_out.flatten(),
                                         dim=0, dim_size=value.numel() + 1)
                grad_value = grad_value[:-1]

        grad_mat = None
        if ctx.needs_input_grad[4]:
            if ctx.reduce in ['sum', 'add']:
                value = value[csr2csc] if value is not None else value
                grad_mat, _ = spmm(grad_out.is_cuda).spmm(
                    colptr, row[csr2csc], value, grad_out, 'sum')

            elif ctx.reduce == 'mean':
                count = rowcount[row].to(mat.dtype).clamp_(min=1)
                value = count.pow_(-1) if value is None else value / count
                row = row[csr2csc]
                value = value[csr2csc] if value is not None else value
                grad_mat, _ = spmm(grad_out.is_cuda).spmm(
                    colptr, row, value, grad_out, 'sum')

            elif ctx.reduce in ['min', 'max']:
                if value is not None:
                    value = value[arg_out_ind.flatten()].view_as(arg_out)
                    value = value.mul_(grad_out)
                else:
                    value = grad_out
                value.masked_fill_(invalid_arg_mask, 0)
                col = col[arg_out_ind.flatten()].view_as(arg_out)
                grad_mat = scatter_add(value, col, dim=-2,
                                       dim_size=mat.size(-2))

        return None, None, None, grad_value, grad_mat, None, None, None, None


class SPSPMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptrA, colA, valueA, rowptrB, colB, valueB, M, N, K):
        if rowptrA.is_cuda:
            rowptrC, colC, valueC = spspmm_cuda.spspmm(rowptrA, colA, valueA,
                                                       rowptrB, colB, valueB,
                                                       M, N, K)
        else:
            dtype = None
            if valueA is not None:
                dtype = valueA.dtype
            if valueB is not None:
                dtype = valueB.dtype

            if valueA is None:
                valueA = torch.ones(colA.numel(), dtype=dtype)
            A = scipy.sparse.csr_matrix((valueA, colA, rowptrA), (M, N))

            if valueB is None:
                valueB = torch.ones(colB.numel(), dtype=dtype)
            B = scipy.sparse.csr_matrix((valueB, colB, rowptrB), (N, K))

            C = A @ B

            rowptrC = torch.from_numpy(C.indptr).to(torch.int64)
            colC = torch.from_numpy(C.indices).to(torch.int64)
            valueC = torch.from_numpy(C.data)
            valueC = valueC.to(dtype) if dtype is not None else valueC

        ctx.mark_non_differentiable(rowptrC, colC)

        # We cannot return `NoneType` in torch.autograd :(
        if valueC is None:
            return rowptrC, colC
        else:
            return rowptrC, colC, valueC

    @staticmethod
    def backward(ctx, grad_indexC, grad_rowptrC, *args):
        grad_valueA = None
        if ctx.needs_input_grad[2]:
            raise NotImplementedError

        grad_valueB = None
        if ctx.needs_input_grad[5]:
            raise NotImplementedError

        return (None, None, grad_valueA, None, None, grad_valueB, None, None,
                None)


def matmul(src, other, reduce='sum'):
    assert src.dim() == 2 and src.size(-1) == other.size(-2)

    # Sparse-Dense Matrix Multiplication.
    if torch.is_tensor(other):
        assert reduce in ['sum', 'add', 'mean', 'min', 'max']
        rowptr, col, value = src.csr()

        row = None
        if reduce in ['sum', 'add'] and (src.requires_grad
                                         or other.reuqires_grad):
            row = src.storage.row

        rowcount = None
        if other.requires_grad and reduce in ['mean']:
            rowcount = src.storage.rowcount

        csr2csc = colptr = None
        if other.requires_grad and reduce in ['sum', 'add', 'mean']:
            csr2csc, colptr = src.storage.csr2csc, src.storage.colptr

        return SPMM.apply(row, rowptr, col, value, other, rowcount, colptr,
                          csr2csc, reduce)

    # Sparse-Sparse Matrix Multiplication.
    elif isinstance(other, src.__class__):
        assert reduce in ['sum', 'add']
        assert src.dim() == 2 and other.dim() == 2
        data = SPSPMM.apply(*src.csr(), *other.csr(), src.size(0), src.size(1),
                            other.size(1))
        (rowptr, col), value = data[:2], data[2] if len(data) == 3 else None
        sparse_size = torch.Size([src.size(0), other.size(1)])
        return src.__class__(rowptr=rowptr, col=col, value=value,
                             sparse_size=sparse_size, is_sorted=True)

    raise ValueError
