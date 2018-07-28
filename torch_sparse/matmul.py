import torch
from torch import from_numpy
from scipy.sparse import coo_matrix


class SparseSparseMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix1, matrix2):
        ctx.save_for_backawrd(matrix1, matrix2)
        return matmul(matrix1, matrix2)

    @staticmethod
    def backward(ctx, grad_out):
        matrix1, matrix2 = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_matrix1 = matmul(grad_out, matrix2.t())

        if ctx.needs_input_grad[0]:
            grad_matrix2 = matmul(matrix1.t(), grad_out)

        return grad_matrix1, grad_matrix2


sparse_sparse_matmul = SparseSparseMatmul.apply


def matmul(A, B):
    if A[0].is_cuda:
        pass
    else:
        return matmul_cpu(A, B)


def matmul_cpu(A, B):
    A, B, = to_csr(A), to_csr(B)
    C = A.dot(B).tocoo()
    row, col, value = from_numpy(C.row), from_numpy(C.col), from_numpy(C.data)
    return torch.stack([row, col], dim=0), value


def to_csr(A):
    (row, col), value, size = A
    row, col, value = row.numpy(), col.numpy(), value.numpy()
    return coo_matrix((value, (row, col)), shape=(size[0], size[1])).tocsr()
