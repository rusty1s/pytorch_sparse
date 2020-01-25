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
    def forward(ctx, index, rowcount, rowptr, colptr, csr2csc, value, mat,
                reduce):
        out, arg_out = spmm(mat.is_cuda).spmm(rowptr, index[1], value, mat,
                                              reduce)

        ctx.reduce = reduce
        ctx.save_for_backward(index, rowcount, rowptr, colptr, csr2csc, value,
                              mat, arg_out)

        if reduce == 'min' or reduce == 'max':
            ctx.mark_non_differentiable(arg_out)
            return out, arg_out
        else:
            return out

    @staticmethod
    def backward(ctx, grad_out, *args):
        data = ctx.saved_tensors
        index, rowcount, rowptr, colptr, csr2csc, value, mat, arg_out = data

        invalid_arg_mask = arg_out_ind = None
        if ctx.reduce in ['min', 'max'] and (ctx.needs_input_grad[5]
                                             or ctx.needs_input_grad[6]):
            invalid_arg_mask = arg_out == index.size(1)
            arg_out_ind = arg_out.masked_fill(invalid_arg_mask, -1)

        grad_value = None
        if ctx.needs_input_grad[5]:
            if ctx.reduce in ['sum', 'add']:
                grad_value = spmm(grad_out.is_cuda).spmm_val_bw(
                    index, rowptr, mat, grad_out, ctx.reduce)

            if ctx.reduce == 'mean':
                grad_value = spmm(grad_out.is_cuda).spmm_val_bw(
                    index, rowptr, mat, grad_out, ctx.reduce)

            elif ctx.reduce in ['min', 'max']:
                col = index[1][arg_out_ind.flatten()].view_as(arg_out)
                out = mat.gather(-2, col).mul_(grad_out)
                out.masked_fill_(invalid_arg_mask, 0)
                grad_value = scatter_add(out.flatten(), arg_out.flatten(),
                                         dim=0, dim_size=value.numel() + 1)
                grad_value = grad_value[:-1]

        grad_mat = None
        if ctx.needs_input_grad[6]:
            if ctx.reduce in ['sum', 'add']:
                value = value[csr2csc] if value is not None else value
                grad_mat, _ = spmm(grad_out.is_cuda).spmm(
                    colptr, index[0][csr2csc], value, grad_out, 'sum')

            elif ctx.reduce == 'mean':
                count = rowcount[index[0]].to(mat.dtype).clamp_(min=1)
                value = count.pow_(-1) if value is None else value / count
                row = index[0][csr2csc]
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
                col = index[1][arg_out_ind.flatten()].view_as(arg_out)
                grad_mat = scatter_add(value, col, dim=-2,
                                       dim_size=mat.size(-2))

        return None, None, None, None, None, grad_value, grad_mat, None


class SPSPMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptrA, colA, valueA, rowptrB, colB, valueB, M, N, K):
        if rowptrA.is_cuda:
            indexC, rowptrC, valueC = spspmm_cuda.spspmm(
                rowptrA, colA, valueA, rowptrB, colB, valueB, M, N, K)
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

            valueC = torch.from_numpy(
                C.data).to(dtype) if dtype is not None else None
            rowptrC = torch.from_numpy(C.indptr).to(torch.int64)
            C = C.tocoo()
            rowC = torch.from_numpy(C.row).to(torch.int64)
            colC = torch.from_numpy(C.col).to(torch.int64)
            indexC = torch.stack([rowC, colC], dim=0)

        ctx.mark_non_differentiable(indexC, rowptrC)

        # We cannot return `NoneType` in torch.autograd :(
        if valueC is None:
            return indexC, rowptrC
        else:
            return indexC, rowptrC, valueC

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
        (index, value), rowptr = src.coo(), src.storage.rowptr

        rowcount = None
        if other.requires_grad and reduce in ['mean']:
            rowcount = src.storage.rowcount

        csr2csc = colptr = None
        if other.requires_grad and reduce in ['sum', 'add', 'mean']:
            csr2csc, colptr = src.storage.csr2csc, src.storage.colptr

        return SPMM.apply(index, rowcount, rowptr, colptr, csr2csc, value,
                          other, reduce)

    # Sparse-Sparse Matrix Multiplication.
    elif isinstance(other, src.__class__):
        assert reduce in ['sum', 'add']
        assert src.dim() == 2 and other.dim() == 2
        data = SPSPMM.apply(*src.csr(), *other.csr(), src.size(0), src.size(1),
                            other.size(1))
        data = data if len(data) == 3 else data + (None, )
        sparse_size = torch.Size([src.size(0), other.size(1)])
        out = src.__class__(data[0], data[2], sparse_size, is_sorted=True)
        out.storage._rowptr = data[1]
        return out

    raise ValueError
