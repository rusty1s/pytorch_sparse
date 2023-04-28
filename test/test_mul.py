from itertools import product

import pytest
import torch

from torch_sparse import SparseTensor, mul
from torch_sparse.testing import devices, dtypes, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_sparse_mul(dtype, device):
    rowA = torch.tensor([0, 0, 1, 2, 2], device=device)
    colA = torch.tensor([0, 2, 1, 0, 1], device=device)
    valueA = tensor([1, 2, 4, 1, 3], dtype, device)
    A = SparseTensor(row=rowA, col=colA, value=valueA)

    rowB = torch.tensor([0, 0, 1, 2, 2], device=device)
    colB = torch.tensor([1, 2, 2, 1, 2], device=device)
    valueB = tensor([2, 3, 1, 2, 4], dtype, device)
    B = SparseTensor(row=rowB, col=colB, value=valueB)

    C = A * B
    rowC, colC, valueC = C.coo()

    assert rowC.tolist() == [0, 2]
    assert colC.tolist() == [2, 1]
    assert valueC.tolist() == [6, 6]

    @torch.jit.script
    def jit_mul(A: SparseTensor, B: SparseTensor) -> SparseTensor:
        return mul(A, B)

    jit_mul(A, B)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_sparse_mul_empty(dtype, device):
    rowA = torch.tensor([0], device=device)
    colA = torch.tensor([1], device=device)
    valueA = tensor([1], dtype, device)
    A = SparseTensor(row=rowA, col=colA, value=valueA)

    rowB = torch.tensor([1], device=device)
    colB = torch.tensor([0], device=device)
    valueB = tensor([2], dtype, device)
    B = SparseTensor(row=rowB, col=colB, value=valueB)

    C = A * B
    rowC, colC, valueC = C.coo()

    assert rowC.tolist() == []
    assert colC.tolist() == []
    assert valueC.tolist() == []
