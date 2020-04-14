import pytest
import torch
from torch_sparse.tensor import SparseTensor

from .utils import devices


@pytest.mark.parametrize('device', devices)
def test_metis(device):
    weighted_mat = SparseTensor.from_dense(torch.randn((6, 6), device=device))
    mat, partptr, perm = weighted_mat.partition(num_parts=2, recursive=False, sort_strategy=True)
    assert partptr.numel() == 3
    assert perm.numel() == 6

    mat, partptr, perm = weighted_mat.partition(num_parts=2, recursive=False, sort_strategy=False)
    assert partptr.numel() == 3
    assert perm.numel() == 6

    unweighted_mat = SparseTensor.from_dense(torch.ones((6, 6), device=device))
    mat, partptr, perm = unweighted_mat.partition(num_parts=2, recursive=True, sort_strategy=True)
    assert partptr.numel() == 3
    assert perm.numel() == 6

    unweighted_mat = unweighted_mat.set_value(None)
    mat, partptr, perm = unweighted_mat.partition(num_parts=2, recursive=True)
    assert partptr.numel() == 3
    assert perm.numel() == 6
