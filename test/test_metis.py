import pytest
import torch
from torch_sparse.tensor import SparseTensor

from .utils import devices


@pytest.mark.parametrize('device', devices)
def test_metis(device):
    mat = SparseTensor.from_dense(torch.randn((6, 6), device=device))
    mat, partptr, perm = mat.partition_kway(num_parts=2)
    assert partptr.tolist() == [0, 3, 6]
    assert perm.tolist() == [0, 1, 2, 3, 4, 5]
