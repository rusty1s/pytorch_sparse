import torch


class _SparseTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, index, value, size):
        ctx.size = size
        ctx.save_for_backward(index)
        return torch.sparse_coo_tensor(index, value, size, device=value.device)

    @staticmethod
    def backward(ctx, grad_out):
        index = ctx.saved_variables[0]
        grad_in = None

        if ctx.needs_input_grad[1]:
            value = grad_out._values()
            id1 = index[0] * ctx.size[1] + index[1]
            index = grad_out._indices()
            id2 = index[0] * ctx.size[1] + index[1]

            grad_in = value.new_zeros(id1.max().item() + 1)
            grad_in[id2] = value
            grad_in = grad_in[id1]

        return None, grad_in, None


SparseTensor = _SparseTensor.apply


class ToValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return A._values()

    @staticmethod
    def backward(ctx, grad_out):
        A = ctx.saved_variables[0]
        grad_in = None

        if ctx.needs_input_grad[0]:
            grad_in = torch.sparse_coo_tensor(
                A._indices(), grad_out, A.size(), device=grad_out.device)

        return grad_in


to_value = ToValue.apply
