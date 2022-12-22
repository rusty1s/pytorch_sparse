from itertools import product

import pytest
import torch

from torch_sparse import SparseTensor, spspmm
from torch_sparse.testing import devices, grad_dtypes, tensor


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_spspmm(dtype, device):
    if dtype in {torch.half, torch.bfloat16}:
        return  # Not yet implemented.

    indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], device=device)
    valueA = tensor([1, 2, 3, 4, 5], dtype, device)
    indexB = torch.tensor([[0, 2], [1, 0]], device=device)
    valueB = tensor([2, 4], dtype, device)

    indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
    assert indexC.tolist() == [[0, 1, 2], [0, 1, 1]]
    assert valueC.tolist() == [8, 6, 8]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_sparse_tensor_spspmm(dtype, device):
    if dtype in {torch.half, torch.bfloat16}:
        return  # Not yet implemented.

    x = SparseTensor(
        row=torch.tensor(
            [0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9],
            device=device),
        col=torch.tensor(
            [0, 5, 10, 15, 1, 2, 3, 7, 13, 6, 9, 5, 10, 15, 11, 14, 5, 15],
            device=device),
        value=torch.tensor([
            1, 3**-0.5, 3**-0.5, 3**-0.5, 1, 1, 1, -2**-0.5, -2**-0.5,
            -2**-0.5, -2**-0.5, 6**-0.5, -6**0.5 / 3, 6**-0.5, -2**-0.5,
            -2**-0.5, 2**-0.5, -2**-0.5
        ], dtype=dtype, device=device),
    )

    expected = torch.eye(10, device=device).to(dtype)

    out = x @ x.to_dense().t()
    assert torch.allclose(out, expected, atol=1e-2)

    out = x @ x.t()
    out = out.to_dense()
    assert torch.allclose(out, expected, atol=1e-2)
