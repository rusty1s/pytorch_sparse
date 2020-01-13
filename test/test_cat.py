import pytest
import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.cat import cat

from .utils import devices, tensor


@pytest.mark.parametrize('device', devices)
def test_cat(device):
    index = tensor([[0, 0, 1], [0, 1, 2]], torch.long, device)
    mat1 = SparseTensor(index)
    mat1.fill_cache_()

    index = tensor([[0, 0, 1, 2], [0, 1, 1, 0]], torch.long, device)
    mat2 = SparseTensor(index)
    mat2.fill_cache_()

    out = cat([mat1, mat2], dim=0)
    assert out.to_dense().tolist() == [[1, 1, 0], [0, 0, 1], [1, 1, 0],
                                       [0, 1, 0], [1, 0, 0]]
    assert len(out.storage.cached_keys()) == 2
    assert out.storage.has_rowcount()
    assert out.storage.has_rowptr()

    out = cat([mat1, mat2], dim=1)
    assert out.to_dense().tolist() == [[1, 1, 0, 1, 1], [0, 0, 1, 0, 1],
                                       [0, 0, 0, 1, 0]]
    assert len(out.storage.cached_keys()) == 2
    assert out.storage.has_colcount()
    assert out.storage.has_colptr()

    out = cat([mat1, mat2], dim=(0, 1))
    assert out.to_dense().tolist() == [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                       [0, 0, 0, 1, 1], [0, 0, 0, 0, 1],
                                       [0, 0, 0, 1, 0]]
    assert len(out.storage.cached_keys()) == 6

    mat1.set_value_(torch.randn((mat1.nnz(), 4), device=device))
    out = cat([mat1, mat1], dim=-1)
    assert out.storage.value.size() == (mat1.nnz(), 8)
    assert len(out.storage.cached_keys()) == 6
