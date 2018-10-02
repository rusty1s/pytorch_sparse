import torch
from torch import from_numpy
import scipy.sparse
from torch_sparse import transpose

if torch.cuda.is_available():
    import spspmm_cuda


def spspmm(indexA, valueA, indexB, valueB, m, k, n):
    """Matrix product of two sparse tensors. Both input sparse matrices need to
    be coalesced.

    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of first sparse matrix.
        k (int): The second dimension of first sparse matrix and first
            dimension of second sparse matrix.
        n (int): The second dimension of second sparse matrix.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    return SpSpMM.apply(indexA, valueA, indexB, valueB, m, k, n)


class SpSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indexA, valueA, indexB, valueB, m, k, n):
        indexC, valueC = mm(indexA, valueA, indexB, valueB, m, k, n)
        ctx.m, ctx.k, ctx.n = m, k, n
        ctx.save_for_backward(indexA, valueA, indexB, valueB, indexC)
        return indexC, valueC

    @staticmethod
    def backward(ctx, grad_indexC, grad_valueC):
        m, k, n = ctx.m, ctx.k, ctx.n
        indexA, valueA, indexB, valueB, indexC = ctx.saved_variables

        grad_valueA = grad_valueB = None

        if ctx.needs_input_grad[1]:
            indexB_T, valueB_T = transpose(indexB, valueB, k, n)
            grad_indexA, grad_valueA = mm(indexC, grad_valueC, indexB_T,
                                          valueB_T, m, n, k)
            grad_valueA = lift(grad_indexA, grad_valueA, indexA, k)

        if ctx.needs_input_grad[3]:
            indexA_T, valueA_T = transpose(indexA, valueA, m, k)
            grad_indexB, grad_valueB = mm(indexA_T, valueA_T, indexC,
                                          grad_valueC, k, m, n)
            grad_valueB = lift(grad_indexB, grad_valueB, indexB, n)

        return None, grad_valueA, None, grad_valueB, None, None, None


def mm(indexA, valueA, indexB, valueB, m, k, n):
    assert valueA.dtype == valueB.dtype

    if indexA.is_cuda:
        return spspmm_cuda.spspmm(indexA, valueA, indexB, valueB, m, k, n)

    A = to_scipy(indexA, valueA, m, k)
    B = to_scipy(indexB, valueB, k, n)
    indexC, valueC = from_scipy(A.tocsr().dot(B.tocsr()).tocoo())

    return indexC, valueC


def to_scipy(index, value, m, n):
    (row, col), data = index.detach(), value.detach()
    return scipy.sparse.coo_matrix((data, (row, col)), (m, n))


def from_scipy(A):
    row, col, value = from_numpy(A.row), from_numpy(A.col), from_numpy(A.data)
    index = torch.stack([row, col], dim=0).to(torch.long)
    return index, value


def lift(indexA, valueA, indexB, n):  # pragma: no cover
    idxA = indexA[0] * n + indexA[1]
    idxB = indexB[0] * n + indexB[1]

    max_value = max(idxA.max().item(), idxB.max().item()) + 1
    valueB = valueA.new_zeros(max_value)

    valueB[idxA] = valueA
    valueB = valueB[idxB]

    return valueB
