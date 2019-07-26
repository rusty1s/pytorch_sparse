import torch
from torch_sparse import transpose, to_scipy, from_scipy, coalesce

import torch_sparse.spspmm_cpu

if torch.cuda.is_available():
    import torch_sparse.spspmm_cuda


def spspmm(indexA, valueA, indexB, valueB, m, k, n, coalesced=False):
    """Matrix product of two sparse tensors. Both input sparse matrices need to
    be coalesced (use the :obj:`coalesced` attribute to force).

    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of first corresponding dense matrix.
        k (int): The second dimension of first corresponding dense matrix and
            first dimension of second corresponding dense matrix.
        n (int): The second dimension of second corresponding dense matrix.
        coalesced (bool, optional): If set to :obj:`True`, will coalesce both
            input sparse matrices. (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if indexA.is_cuda and coalesced:
        indexA, valueA = coalesce(indexA, valueA, m, k)
        indexB, valueB = coalesce(indexB, valueB, k, n)

    index, value = SpSpMM.apply(indexA, valueA, indexB, valueB, m, k, n)
    return index.detach(), value


class SpSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indexA, valueA, indexB, valueB, m, k, n):
        indexC, valueC = mm(indexA, valueA, indexB, valueB, m, k, n)
        ctx.m, ctx.k, ctx.n = m, k, n
        ctx.save_for_backward(indexA, valueA, indexB, valueB, indexC)
        return indexC, valueC

    @staticmethod
    def backward(ctx, grad_indexC, grad_valueC):
        m, k = ctx.m, ctx.k
        n = ctx.n
        indexA, valueA, indexB, valueB, indexC = ctx.saved_tensors

        grad_valueA = grad_valueB = None

        if not grad_valueC.is_cuda:
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[1]:
                grad_valueC = grad_valueC.clone()

            if ctx.needs_input_grad[1]:
                grad_valueA = torch_sparse.spspmm_cpu.spspmm_bw(
                    indexA, indexC.detach(), grad_valueC, indexB.detach(),
                    valueB, m, k)

            if ctx.needs_input_grad[3]:
                indexA, valueA = transpose(indexA, valueA, m, k)
                indexC, grad_valueC = transpose(indexC, grad_valueC, m, n)
                grad_valueB = torch_sparse.spspmm_cpu.spspmm_bw(
                    indexB, indexA.detach(), valueA, indexC.detach(),
                    grad_valueC, k, n)
        else:
            if ctx.needs_input_grad[1]:
                grad_valueA = torch_sparse.spspmm_cuda.spspmm_bw(
                    indexA, indexC.detach(), grad_valueC.clone(),
                    indexB.detach(), valueB, m, k)

            if ctx.needs_input_grad[3]:
                indexA_T, valueA_T = transpose(indexA, valueA, m, k)
                grad_indexB, grad_valueB = mm(indexA_T, valueA_T, indexC,
                                              grad_valueC, k, m, n)
                grad_valueB = lift(grad_indexB, grad_valueB, indexB, n)

        return None, grad_valueA, None, grad_valueB, None, None, None


def mm(indexA, valueA, indexB, valueB, m, k, n):
    assert valueA.dtype == valueB.dtype

    if indexA.is_cuda:
        return torch_sparse.spspmm_cuda.spspmm(indexA, valueA, indexB, valueB,
                                               m, k, n)

    A = to_scipy(indexA, valueA, m, k)
    B = to_scipy(indexB, valueB, k, n)
    C = A.dot(B).tocoo().tocsr().tocoo()  # Force coalesce.
    indexC, valueC = from_scipy(C)
    return indexC, valueC


def lift(indexA, valueA, indexB, n):  # pragma: no cover
    idxA = indexA[0] * n + indexA[1]
    idxB = indexB[0] * n + indexB[1]

    max_value = max(idxA.max().item(), idxB.max().item()) + 1
    valueB = valueA.new_zeros(max_value)

    valueB[idxA] = valueA
    valueB = valueB[idxB]

    return valueB
