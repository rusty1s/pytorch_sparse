from itertools import product

import pytest
import torch

from torch_sparse.matmul import matmul
from torch_sparse.tensor import SparseTensor
import torch_scatter

from .utils import devices, grad_dtypes

devices = ['cpu', 'cuda']
grad_dtypes = [torch.float]
reductions = ['sum', 'mean', 'min', 'max']
reductions = ['min', 'max']


@pytest.mark.parametrize('dtype,device,reduce',
                         product(grad_dtypes, devices, reductions))
def test_spmm(dtype, device, reduce):
    src = torch.randn((10, 8), dtype=dtype, device=device)
    src[2:4, :] = 0  # Remove multiple rows.
    src[:, 2:4] = 0  # Remove multiple columns.
    src = SparseTensor.from_dense(src).requires_grad_()
    (row, col), value = src.coo()

    other = torch.randn((2, 8, 2), dtype=dtype, device=device,
                        requires_grad=True)

    src_col = other.index_select(-2, col) * value.unsqueeze(-1)
    func = 'add' if reduce == 'sum' else reduce
    expected = getattr(torch_scatter, f'scatter_{func}')(src_col, row, dim=-2)
    expected = expected[0] if isinstance(expected, tuple) else expected
    if reduce == 'min':
        expected[expected > 1000] = 0
    if reduce == 'max':
        expected[expected < -1000] = 0

    grad_out = torch.randn_like(expected)

    expected.backward(grad_out)
    expected_grad_value = value.grad
    value.grad = None
    expected_grad_other = other.grad
    other.grad = None

    out = matmul(src, other, reduce)
    out = out[0] if isinstance(out, tuple) else out
    out.backward(grad_out)

    assert torch.allclose(expected, out)
    assert torch.allclose(expected_grad_value, value.grad)
    assert torch.allclose(expected_grad_other, other.grad)
