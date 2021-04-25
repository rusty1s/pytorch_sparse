import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import matmul


def spspmm(indexA, valueA, indexB, valueB, m, k, n, coalesced=False):
    """Matrix product of two sparse tensors. Both input sparse matrices need to
    be coalesced (use the :obj:`coalesced` attribute to force).

    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of first sparse matrix.
        k (int): The second dimension of first sparse matrix and first
            dimension of second sparse matrix.
        n (int): The second dimension of second sparse matrix.
        coalesced (bool, optional): If set to :obj:`True`, will coalesce both
            input sparse matrices. (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    C = matmul(A, B)
    row, col, value = C.coo()

    return torch.stack([row, col], dim=0), value
