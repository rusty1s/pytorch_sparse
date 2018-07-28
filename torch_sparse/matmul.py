import torch
from torch import from_numpy
from scipy.sparse import coo_matrix


class SpSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix1, matrix2):
        ctx.save_for_backawrd(matrix1, matrix2)
        return mm(matrix1, matrix2)

    @staticmethod
    def backward(ctx, grad_out):
        matrix1, matrix2 = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_matrix1 = mm(grad_out, matrix2.t())

        if ctx.needs_input_grad[0]:
            grad_matrix2 = mm(matrix1.t(), grad_out)

        return grad_matrix1, grad_matrix2


spspmm = SpSpMM.apply


def mm(A, B):
    if A[0].is_cuda:
        pass
    else:
        return mm_cpu(A, B)


def mm_cpu(A, B):
    A, B, = to_csr(A), to_csr(B)
    C = A.dot(B).tocoo()
    row, col, value = from_numpy(C.row), from_numpy(C.col), from_numpy(C.data)
    return torch.stack([row, col], dim=0), value


def to_csr(A):
    (row, col), value, size = A
    row, col, value = row.numpy(), col.numpy(), value.numpy()
    return coo_matrix((value, (row, col)), shape=(size[0], size[1])).tocsr()
