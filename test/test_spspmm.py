from itertools import product

import pytest
import torch
from torch_sparse import spspmm

from .utils import grad_dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_spspmm(dtype, device):
    indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], device=device)
    valueA = tensor([1, 2, 3, 4, 5], dtype, device)
    indexB = torch.tensor([[0, 2], [1, 0]], device=device)
    valueB = tensor([2, 4], dtype, device)

    indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
    assert indexC.tolist() == [[0, 1, 2], [0, 1, 1]]
    assert valueC.tolist() == [8, 6, 8]
