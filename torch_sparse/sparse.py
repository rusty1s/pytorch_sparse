import torch


class SparseCooTensor(torch.autograd.Function):
    """Constructs Sparse matrix with autograd capabilities w.r.t. to value."""

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


sparse_coo_tensor = SparseCooTensor.apply


class ToValue(torch.autograd.Function):
    """Extract values of sparse tensors with autograd support."""

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
