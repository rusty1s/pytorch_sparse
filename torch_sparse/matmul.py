import torch

from torch_sparse import spmm_cpu
from torch_scatter import scatter_add

try:
    from torch_sparse import spmm_cuda
except ImportError:
    spmm_cuda = None


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
                    rowptr, index[1], mat, grad_out, ctx.reduce)

            if ctx.reduce == 'mean':
                grad_value = spmm(grad_out.is_cuda).spmm_val_bw(
                    rowptr, index[1], mat, grad_out, ctx.reduce)

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
                row = index[0][csr2csc]
                value = value[csr2csc] if value is not None else value
                grad_mat, _ = spmm(grad_out.is_cuda).spmm(
                    colptr, row, value, grad_out, 'sum')

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


def matmul(src, other, reduce='sum'):
    assert src.dim() == 2 and src.size(-1) == other.size(-2)

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

    elif isinstance(other, src.__class__):
        assert reduce in ['sum', 'add']

    raise ValueError
