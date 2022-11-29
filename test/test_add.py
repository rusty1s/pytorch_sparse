from itertools import product

import pytest
import torch

from torch_sparse import SparseTensor, add
from torch_sparse.testing import devices, dtypes, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_add(dtype, device):
    rowA = torch.tensor([0, 0, 1, 2, 2], device=device)
    colA = torch.tensor([0, 2, 1, 0, 1], device=device)
    valueA = tensor([1, 2, 4, 1, 3], dtype, device)
    A = SparseTensor(row=rowA, col=colA, value=valueA)

    rowB = torch.tensor([0, 0, 1, 2, 2], device=device)
    colB = torch.tensor([1, 2, 2, 1, 2], device=device)
    valueB = tensor([2, 3, 1, 2, 4], dtype, device)
    B = SparseTensor(row=rowB, col=colB, value=valueB)

    C = A + B
    rowC, colC, valueC = C.coo()

    assert rowC.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert colC.tolist() == [0, 1, 2, 1, 2, 0, 1, 2]
    assert valueC.tolist() == [1, 2, 5, 4, 1, 1, 5, 4]

    @torch.jit.script
    def jit_add(A: SparseTensor, B: SparseTensor) -> SparseTensor:
        return add(A, B)

    jit_add(A, B)
