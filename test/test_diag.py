from itertools import product

import pytest
import torch

from torch_sparse.tensor import SparseTensor
from torch_sparse.testing import devices, dtypes, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_remove_diag(dtype, device):
    row, col = tensor([[0, 0, 1, 2], [0, 1, 2, 2]], torch.long, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    mat = SparseTensor(row=row, col=col, value=value)
    mat.fill_cache_()

    mat = mat.remove_diag()
    assert mat.storage.row().tolist() == [0, 1]
    assert mat.storage.col().tolist() == [1, 2]
    assert mat.storage.value().tolist() == [2, 3]
    assert mat.storage.num_cached_keys() == 2
    assert mat.storage.rowcount().tolist() == [1, 1, 0]
    assert mat.storage.colcount().tolist() == [0, 1, 1]

    mat = SparseTensor(row=row, col=col, value=value)
    mat.fill_cache_()

    mat = mat.remove_diag(k=1)
    assert mat.storage.row().tolist() == [0, 2]
    assert mat.storage.col().tolist() == [0, 2]
    assert mat.storage.value().tolist() == [1, 4]
    assert mat.storage.num_cached_keys() == 2
    assert mat.storage.rowcount().tolist() == [1, 0, 1]
    assert mat.storage.colcount().tolist() == [1, 0, 1]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_set_diag(dtype, device):
    row, col = tensor([[0, 0, 9, 9], [0, 1, 0, 1]], torch.long, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    mat = SparseTensor(row=row, col=col, value=value)

    mat = mat.set_diag(tensor([-8, -8], dtype, device), k=-1)
    mat = mat.set_diag(tensor([-8], dtype, device), k=1)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_fill_diag(dtype, device):
    row, col = tensor([[0, 0, 9, 9], [0, 1, 0, 1]], torch.long, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    mat = SparseTensor(row=row, col=col, value=value)

    mat = mat.fill_diag(-8, k=-1)
    mat = mat.fill_diag(-8, k=1)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_get_diag(dtype, device):
    row, col = tensor([[0, 0, 1, 2], [0, 1, 2, 2]], torch.long, device)
    value = tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype, device)
    mat = SparseTensor(row=row, col=col, value=value)
    assert mat.get_diag().tolist() == [[1, 1], [0, 0], [4, 4]]

    row, col = tensor([[0, 0, 1, 2], [0, 1, 2, 2]], torch.long, device)
    mat = SparseTensor(row=row, col=col)
    assert mat.get_diag().tolist() == [1, 0, 1]
