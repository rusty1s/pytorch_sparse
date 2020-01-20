from itertools import product

import pytest
import torch
from torch_sparse.tensor import SparseTensor

from torch_sparse.diag_cpu import non_diag_mask

from .utils import dtypes, devices, tensor

dtypes = [torch.float]
devices = ['cpu']


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

    print()
    k = -8
    print("k = ", k)
    mat = mat.remove_diag(k)
    print(mat.to_dense())

    # row, col = mat.storage.index
    # print('k', k)
    # mask = row != col - k
    # index = index[:, mask]

    # row, col = index
    # print(row)
    # print(col)

    mask = non_diag_mask(mat.storage.index, mat.size(0), mat.size(1), k)
    print(mask)

    # bla = col - row
    # print(bla)

    # DETECT VORZEICHEN WECHSEL

    # mask = row.new_ones(index.size(1) + 3, dtype=torch.bool)
    # mask[1:] = row[1:] != row[:-1]
    # # mask = row[1:] != row[:-1]
    # print(mask)

    # mask = (row <= col)
    # print(row)
    # print(col)
    # print(mask)
    # mask = (row[1:] == row[:-1])
    # print(mask)

    # UNION
    # idx1 = ...
    # idx2 = ...
