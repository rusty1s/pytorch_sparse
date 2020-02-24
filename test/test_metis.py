import pytest
import torch
from torch_sparse.tensor import SparseTensor

from .utils import devices


@pytest.mark.parametrize('device', devices)
def test_metis(device):
    mat = SparseTensor.from_dense(torch.randn((6, 6), device=device))
    mat, partptr, perm = mat.partition(num_parts=2, recursive=False)
    assert partptr.numel() == 3
    assert perm.numel() == 6

    mat, partptr, perm = mat.partition(num_parts=2, recursive=True)
    assert partptr.numel() == 3
    assert perm.numel() == 6
