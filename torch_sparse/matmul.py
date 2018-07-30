import torch
import scipy.sparse
from torch_sparse import transpose

if torch.cuda.is_available():
    import matmul_cuda


def spspmm(indexA, valueA, sizeA, indexB, valueB, sizeB):
    assert valueA.dtype == valueB.dtype
    assert len(sizeA) == len(sizeB) == 2
    assert sizeA[1] == sizeB[0]

    index, value = SpSpMM.apply(indexA, valueA, sizeA, indexB, valueB, sizeB)
    size = torch.Size([sizeA[0], sizeB[1]])

    return index, value, size


class SpSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indexA, valueA, sizeA, indexB, valueB, sizeB):
        index, value = mm(indexA, valueA, sizeA, indexB, valueB, sizeB)

        ctx.sizeA, ctx.sizeB = sizeA, sizeB
        ctx.save_for_backward(indexA, valueA, indexB, valueB, index)

        return index, value

    @staticmethod
    def backward(ctx, grad_index, grad_value):
        indexA, valueA, indexB, valueB, index = ctx.saved_variables
        grad_valueA = grad_valueB = None
        grad = (index, grad_value, torch.Size([ctx.sizeA[0], ctx.sizeB[1]]))

        if ctx.needs_input_grad[1]:
            B_tranposed = transpose(indexB, valueB, ctx.sizeB)
            _, grad_valueA = mm(*grad, *B_tranposed)

        if ctx.needs_input_grad[4]:
            A_tranposed = transpose(indexA, valueA, ctx.sizeA)
            _, grad_valueB = mm(*A_tranposed, *grad)

        return None, grad_valueA, None, None, grad_valueB, None


def mm(indexA, valueA, sizeA, indexB, valueB, sizeB):
    if valueA.is_cuda:
        return mm_cuda(indexA, valueA, sizeA, indexB, valueB, sizeB)
    else:
        return mm_cpu(indexA, valueA, sizeA, indexB, valueB, sizeB)


def mm_cuda(indexA, valueA, sizeA, indexB, valueB, sizeB):
    A = torch.sparse_coo_tensor(indexA, valueA, sizeA)
    B = torch.sparse_coo_tensor(indexB, valueB, sizeB)

    index, value = matmul_cuda.spspmm(A, B)

    return index, value


def mm_cpu(indexA, valueA, sizeA, indexB, valueB, sizeB):
    A, B, = to_scipy(indexA, valueA, sizeA), to_scipy(indexB, valueB, sizeB)
    C = A.tocsr().dot(B.tocsr()).tocoo()

    row, col = torch.from_numpy(C.row).long(), torch.from_numpy(C.col).long()
    index = torch.stack([row, col], dim=0)
    value = torch.from_numpy(C.data).type_as(valueA)

    return index, value


def to_scipy(index, value, size):
    (row, col), value = index.detach().numpy(), value.detach().numpy()
    return scipy.sparse.coo_matrix((value, (row, col)), tuple(size))
