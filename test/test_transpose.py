from itertools import product

import pytest
import torch

from torch_sparse import transpose
from torch_sparse.testing import devices, dtypes, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_transpose_matrix(dtype, device):
    row = torch.tensor([1, 0, 1, 2], device=device)
    col = torch.tensor([0, 1, 1, 0], device=device)
    index = torch.stack([row, col], dim=0)
    value = tensor([1, 2, 3, 4], dtype, device)

    index, value = transpose(index, value, m=3, n=2)
    assert index.tolist() == [[0, 0, 1, 1], [1, 2, 0, 1]]
    assert value.tolist() == [1, 4, 2, 3]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_transpose(dtype, device):
    row = torch.tensor([1, 0, 1, 0, 2, 1], device=device)
    col = torch.tensor([0, 1, 1, 1, 0, 0], device=device)
    index = torch.stack([row, col], dim=0)
    value = tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], dtype,
                   device)

    index, value = transpose(index, value, m=3, n=2)
    assert index.tolist() == [[0, 0, 1, 1], [1, 2, 0, 1]]
    assert value.tolist() == [[7, 9], [5, 6], [6, 8], [3, 4]]
