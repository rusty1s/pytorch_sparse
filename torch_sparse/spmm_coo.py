import torch
from torch_sparse._spmm_coo_cuda import spmm_coo_sum,spmm_coo_mean,spmm_coo_max,spmm_coo_min

def spmm_coo(row, col, mat, dim_size, value=None, reduce='sum'):
    """Matrix product of sparse matrix with dense matrix in coo format.

    Args:
        row (:class:`LongTensor`): The rows of sparse matrix.
        col (:class:`LongTensor`): The columns of sparse matrix.
        value (:class:`Tensor`) The values of indices in sparse matrix.
        mat (:class:`Tensor`): The dense matrix.
        dim_size (:class`Int`): The 0. dimension of result matrix.
        reduce (:class`String`); The reduction used in multiplication.

    :rtype: :class:`Tensor`
    """
    if reduce == 'sum':
        return spmm_coo_sum(row, col, value, mat, dim_size)
    elif reduce == 'mean':
        return spmm_coo_mean(row, col, value,mat, dim_size)
    elif reduce == 'max':
        return spmm_coo_max(row, col,value, mat, dim_size)[0]
    elif reduce == 'min':
        return spmm_coo_min(row, col,value, mat, dim_size)[0]
    else:
        raise ValueError
