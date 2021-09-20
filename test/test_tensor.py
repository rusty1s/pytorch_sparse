from itertools import product

import pytest
import torch
from torch_sparse import SparseTensor

from .utils import grad_dtypes, devices


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_getitem(dtype, device):
    mat = torch.randn(50, 40, dtype=dtype, device=device)
    mat = SparseTensor.from_dense(mat)

    idx1 = torch.randint(0, 50, (10, ), dtype=torch.long, device=device)
    idx2 = torch.randint(0, 40, (10, ), dtype=torch.long, device=device)

    assert mat[:10, :10].sizes() == [10, 10]
    assert mat[..., :10].sizes() == [50, 10]
    assert mat[idx1, idx2].sizes() == [10, 10]
    assert mat[idx1.tolist()].sizes() == [10, 40]


@pytest.mark.parametrize('device', devices)
def test_to_symmetric(device):
    row = torch.tensor([0, 0, 0, 1, 1], device=device)
    col = torch.tensor([0, 1, 2, 0, 2], device=device)
    value = torch.arange(1, 6, device=device)
    mat = SparseTensor(row=row, col=col, value=value)
    assert not mat.is_symmetric()

    mat = mat.to_symmetric()

    assert mat.is_symmetric()
    assert mat.to_dense().tolist() == [
        [2, 6, 3],
        [6, 0, 5],
        [3, 5, 0],
    ]


def test_equal():
    row = torch.tensor([0, 0, 0, 1, 1])
    col = torch.tensor([0, 1, 2, 0, 2])
    value = torch.arange(1, 6)
    matA = SparseTensor(row=row, col=col, value=value)
    matB = SparseTensor(row=row, col=col, value=value)
    col = torch.tensor([0, 1, 2, 0, 1])
    matC = SparseTensor(row=row, col=col, value=value)

    assert id(matA) != id(matB)
    assert matA == matB

    assert id(matA) != id(matC)
    assert matA != matC
