from itertools import product

import pytest
import torch
import torch_scatter

from torch_sparse.matmul import matmul, spspmm
from torch_sparse.tensor import SparseTensor
from torch_sparse.testing import devices, grad_dtypes, reductions


@pytest.mark.parametrize('dtype,device,reduce',
                         product(grad_dtypes, devices, reductions))
def test_spmm(dtype, device, reduce):
    if device == torch.device('cuda:0') and dtype == torch.bfloat16:
        return  # Not yet implemented.

    src = torch.randn((10, 8), dtype=dtype, device=device)
    src[2:4, :] = 0  # Remove multiple rows.
    src[:, 2:4] = 0  # Remove multiple columns.
    src = SparseTensor.from_dense(src).requires_grad_()
    row, col, value = src.coo()

    other = torch.randn((2, 8, 2), dtype=dtype, device=device,
                        requires_grad=True)

    src_col = other.index_select(-2, col) * value.unsqueeze(-1)
    expected = torch_scatter.scatter(src_col, row, dim=-2, reduce=reduce)
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
    out.backward(grad_out)

    atol = 1e-7
    if dtype == torch.float16 or dtype == torch.bfloat16:
        atol = 1e-1

    assert torch.allclose(expected, out, atol=atol)
    assert torch.allclose(expected_grad_value, value.grad, atol=atol)
    assert torch.allclose(expected_grad_other, other.grad, atol=atol)


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_spspmm(dtype, device):
    if dtype in {torch.half, torch.bfloat16}:
        return  # Not yet implemented.

    src = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype,
                       device=device)

    src = SparseTensor.from_dense(src)
    out = matmul(src, src)
    assert out.sizes() == [3, 3]
    assert out.has_value()
    rowptr, col, value = out.csr()
    assert rowptr.tolist() == [0, 1, 2, 3]
    assert col.tolist() == [0, 1, 2]
    assert value.tolist() == [1, 1, 1]

    src.set_value_(None)
    out = matmul(src, src)
    assert out.sizes() == [3, 3]
    assert not out.has_value()
    rowptr, col, value = out.csr()
    assert rowptr.tolist() == [0, 1, 2, 3]
    assert col.tolist() == [0, 1, 2]

    torch.jit.script(spspmm)
