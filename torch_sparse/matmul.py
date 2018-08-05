import torch
import scipy.sparse
from torch_sparse import transpose

if torch.cuda.is_available():
    import matmul_cuda


class SpSpMM(torch.autograd.Function):
    """Sparse matrix product of two sparse tensors with autograd support."""

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
            indexB, valueB = transpose(indexB, valueB, k, n)
            _, grad_valueA = mm(indexC, grad_valueC, indexB, valueB, m, n, k)
            # TODO: Filter values.

        if ctx.needs_input_grad[4]:
            indexA, valueA = transpose(indexA, valueA, m, k)
            _, grad_valueB = mm(indexA, valueA, indexC, grad_valueC, k, m, n)
            # TODO: Filter values.

        return None, grad_valueA, None, grad_valueB, None, None, None


spspmm = SpSpMM.apply


def mm(indexA, valueA, indexB, valueB, m, k, n):
    assert valueA.dtype == valueB.dtype

    if indexA.is_cuda:
        return matmul_cuda.spspmm(indexA, valueA, indexB, valueB, m, k, n)

    A = to_scipy(indexA, valueA, m, k)
    B = to_scipy(indexB, valueB, k, n)
    indexC, valueC = from_scipy(A.tocsr().dot(B.tocsr()).tocoo())

    return indexC, valueC


def to_scipy(index, value, m, n):
    (row, col), data = index.detach(), value.detach()
    return scipy.sparse.coo_matrix((data, (row, col)), (m, n))


def from_scipy(A):
    row, col, value = A.row, A.col, A.data
    index = torch.stack([row, col], dim=0).to(torch.long)
    return index, value
