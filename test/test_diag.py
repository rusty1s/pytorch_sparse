from itertools import product

import pytest
import torch
from torch_sparse.tensor import SparseTensor

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_remove_diag(dtype, device):
    index = tensor([
        [0, 0, 1, 2],
        [0, 1, 2, 2],
    ], torch.long, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    mat = SparseTensor(index, value)
    mat.fill_cache_()

    mat = mat.remove_diag()
    assert mat.storage.index.tolist() == [[0, 1], [1, 2]]
    assert mat.storage.value.tolist() == [2, 3]
    assert len(mat.cached_keys()) == 2
    assert mat.storage.rowcount.tolist() == [1, 1, 0]
    assert mat.storage.colcount.tolist() == [0, 1, 1]

    mat = SparseTensor(index, value)
    mat.fill_cache_()

    mat = mat.remove_diag(k=1)
    assert mat.storage.index.tolist() == [[0, 2], [0, 2]]
    assert mat.storage.value.tolist() == [1, 4]
    assert len(mat.cached_keys()) == 2
    assert mat.storage.rowcount.tolist() == [1, 0, 1]
    assert mat.storage.colcount.tolist() == [1, 0, 1]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_set_diag(dtype, device):
    index = tensor([
        [0, 0, 9, 9],
        [0, 1, 0, 1],
    ], torch.long, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    mat = SparseTensor(index, value)

    k = -8
    mat = mat.set_diag(k)
