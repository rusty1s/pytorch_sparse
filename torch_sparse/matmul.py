import torch
from torch import from_numpy
from scipy.sparse import coo_matrix


class SpSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e1, v1, s1, e2, v2, s2):
        e, v = mm(e1, v1, s1, e2, v2, s2)

        ctx.s1, ctx.s2 = s1, s2
        ctx.save_for_backward(e1, v1, e2, v2, e)

        return e, v

    @staticmethod
    def backward(ctx, grad_e, grad_v):
        e1, v1, e2, v2, e = ctx.saved_variables
        grad_v1 = grad_v2 = None
        grad = (e, grad_v, torch.Size([ctx.s1[0], ctx.s2[1]]))

        if ctx.needs_input_grad[1]:
            e2 = torch.stack([e2[1], e2[0]], dim=0)
            _, grad_v1 = mm(*grad, e2, v2, torch.Size([ctx.s2[1], ctx.s2[0]]))

        if ctx.needs_input_grad[4]:
            e1 = torch.stack([e1[1], e1[0]], dim=0)
            _, grad_v2 = mm(e1, v1, torch.Size([ctx.s1[1], ctx.s1[0]]), *grad)

        return None, grad_v1, None, None, grad_v2, None


spspmm = SpSpMM.apply


def mm(e1, v1, s1, e2, v2, s2):
    if e1.is_cuda:
        pass
    else:
        return mm_cpu(e1, v1, s1, e2, v2, s2)


def mm_cpu(e1, v1, s1, e2, v2, s2):
    matrix1, matrix2, = to_csr(e1, v1, s1), to_csr(e2, v2, s2)
    out = matrix1.dot(matrix2).tocoo()
    row, col = from_numpy(out.row).long(), from_numpy(out.col).long()
    return torch.stack([row, col], dim=0), from_numpy(out.data)


def to_csr(index, value, size):
    index, value = index.detach().numpy(), value.detach().numpy()
    shape = (size[0], size[1])
    return coo_matrix((value, (index[0], index[1])), shape).tocsr()
