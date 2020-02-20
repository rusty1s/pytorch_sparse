import pytest
import torch
from torch_sparse.tensor import SparseTensor

from .utils import devices


@pytest.mark.parametrize('device', devices)
def test_metis(device):
    mat = SparseTensor.from_dense(torch.randn((6, 6), device=device))
    mat, partptr, perm = mat.partition_kway(num_parts=2)
    assert partptr.numel() == 3
    assert perm.numel() == 6
