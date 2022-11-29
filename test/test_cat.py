import pytest
import torch

from torch_sparse.cat import cat
from torch_sparse.tensor import SparseTensor
from torch_sparse.testing import devices, tensor


@pytest.mark.parametrize('device', devices)
def test_cat(device):
    row, col = tensor([[0, 0, 1], [0, 1, 2]], torch.long, device)
    mat1 = SparseTensor(row=row, col=col)
    mat1.fill_cache_()

    row, col = tensor([[0, 0, 1, 2], [0, 1, 1, 0]], torch.long, device)
    mat2 = SparseTensor(row=row, col=col)
    mat2.fill_cache_()

    out = cat([mat1, mat2], dim=0)
    assert out.to_dense().tolist() == [[1, 1, 0], [0, 0, 1], [1, 1, 0],
                                       [0, 1, 0], [1, 0, 0]]
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.has_rowcount()
    assert out.storage.num_cached_keys() == 1

    out = cat([mat1, mat2], dim=1)
    assert out.to_dense().tolist() == [[1, 1, 0, 1, 1], [0, 0, 1, 0, 1],
                                       [0, 0, 0, 1, 0]]
    assert out.storage.has_row()
    assert not out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 2

    out = cat([mat1, mat2], dim=(0, 1))
    assert out.to_dense().tolist() == [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                       [0, 0, 0, 1, 1], [0, 0, 0, 0, 1],
                                       [0, 0, 0, 1, 0]]
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 5

    value = torch.randn((mat1.nnz(), 4), device=device)
    mat1 = mat1.set_value_(value, layout='coo')
    out = cat([mat1, mat1], dim=-1)
    assert out.storage.value().size() == (mat1.nnz(), 8)
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 5
