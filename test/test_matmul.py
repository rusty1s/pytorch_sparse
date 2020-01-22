from itertools import product

import pytest
import torch
from torch.autograd import gradcheck

from torch_sparse.matmul import matmul
from torch_sparse.tensor import SparseTensor
import torch_scatter

from .utils import tensor, devices, dtypes

devices = ['cpu']
dtypes = [torch.float]

reductions = ['sum', 'mean', 'min', 'max']
# grad_reductions = ['sum', 'mean']


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_spmm_forward(dtype, device):
    src_dense = torch.randn((5, 4), dtype=dtype, device=device)
    src = SparseTensor.from_dense(src_dense)
    src.requires_grad_()
    src_dense = src_dense.clone().requires_grad_()

    other = torch.randn((4, 8), dtype=dtype, device=device)
    other.requires_grad_()

    out1 = matmul(src, other)
    grad_out = torch.randn_like(out1)
    out1.backward(grad_out)

    other.grad = None
    out2 = torch.matmul(src_dense, other)
    out2.backward(grad_out)

    # assert torch.allclose(out1, out2)
    # assert torch.allclose(src.storage.value.grad.view(5, 4), src_dense.grad)


@pytest.mark.parametrize('dtype,device,reduce',
                         product(dtypes, devices, reductions))
def test_spmm(dtype, device, reduce):
    src = torch.ones((5, 4), dtype=dtype, device=device)

    src[2] = 0
    src = SparseTensor.from_dense(src).requires_grad_()
    src.set_value_(None)

    other = torch.randn((2, 4, 2), dtype=dtype, device=device,
                        requires_grad=True)

    (row, col), value = src.coo()

    out1 = other.index_select(-2, col)  # * value.unsqueeze(-1)
    func = 'add' if reduce == 'sum' else reduce
    out1 = getattr(torch_scatter, f'scatter_{func}')(out1, row, dim=-2)
    out1 = out1[0] if isinstance(out1, tuple) else out1

    grad_out = torch.randn_like(out1)
    out1.backward(grad_out)
    # grad_value1 = value.grad
    # value.grad = None
    grad_other1 = other.grad
    other.grad = None

    print(reduce)
    out2 = matmul(src, other, reduce)
    out2 = out2[0] if isinstance(out2, tuple) else out2

    out2.backward(grad_out)
    # grad_value2 = value.grad
    # value.grad = None
    grad_other2 = other.grad
    other.grad = None

    # assert torch.allclose(out1, out2)
    # assert torch.allclose(grad_value1, grad_value2)
    assert torch.allclose(grad_other1, grad_other2)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_spmm_backward(dtype, device):
    src_dense = torch.randn((5, 4), dtype=torch.double, device=device)
    src = SparseTensor.from_dense(src_dense)
    src.requires_grad_()

    other = torch.randn((4, 8), dtype=torch.double, device=device)
    other.requires_grad_()

    # assert gradcheck(matmul, (src, other, "sum"))
