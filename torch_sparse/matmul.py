import torch
import scipy.sparse

if torch.cuda.is_available():
    import matmul_cuda


class SpSpMM(torch.autograd.Function):
    """Sparse matrix product of two sparse tensors with autograd support."""

    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return mm(A, B)

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_variables
        grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            grad_A = mm(grad_C, B.t().coalesce())

        if ctx.needs_input_grad[1]:
            grad_B = mm(A.t(), grad_C)

        return grad_A, grad_B


spspmm = SpSpMM.apply


def mm(A, B):
    assert A.dtype == B.dtype
    assert A.size(1) == B.size(0)
    return mm_cuda(A, B) if A.is_cuda else mm_cpu(A, B)


def mm_cuda(A, B):
    index, value = matmul_cuda.spspmm(A, B)
    size = torch.Size([A.size(0), B.size(1)])
    return torch.sparse_coo_tensor(index, value, size, device=value.device)


def mm_cpu(A, B):
    return from_scipy(to_scipy(A).dot(to_scipy(B)))


def to_scipy(A):
    (row, col), data, shape = A._indices(), A._values(), tuple(A.size())
    row, col, data = row.detach(), col.detach(), data.detach()
    return scipy.sparse.coo_matrix((data, (row, col)), shape).tocsr()


def from_scipy(A):
    A = A.tocoo()
    row, col, value, size = A.row, A.col, A.data, torch.Size(A.shape)
    row, col = torch.from_numpy(row).long(), torch.from_numpy(col).long()
    value = torch.from_numpy(value)
    index = torch.stack([row, col], dim=0)
    return torch.sparse_coo_tensor(index, value, size)
