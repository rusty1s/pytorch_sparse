from itertools import product

import pytest
import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse import view

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_view_matrix(dtype, device):
    row = torch.tensor([0, 1, 1], device=device)
    col = torch.tensor([1, 0, 2], device=device)
    index = torch.stack([row, col], dim=0)
    value = tensor([1, 2, 3], dtype, device)

    index, value = view(index, value, m=2, n=3, new_n=2)
    assert index.tolist() == [[0, 1, 2], [1, 1, 1]]
    assert value.tolist() == [1, 2, 3]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_view_sparse_tensor(dtype, device):
    options = torch.tensor(0, dtype=dtype, device=device)

    mat = SparseTensor.eye(4, options=options).view(2, 8)
    assert mat.storage.sparse_sizes() == (2, 8)
    assert mat.storage.row().tolist() == [0, 0, 1, 1]
    assert mat.storage.col().tolist() == [0, 5, 2, 7]
    assert mat.storage.value().tolist() == [1, 1, 1, 1]
