from itertools import product

import pytest
import torch

from torch_sparse.tensor import SparseTensor
from torch_sparse.testing import devices

try:
    rowptr = torch.tensor([0, 1])
    col = torch.tensor([0])
    torch.ops.torch_sparse.partition(rowptr, col, None, 1, True)
    with_metis = True
except RuntimeError:
    with_metis = False


@pytest.mark.skipif(not with_metis, reason='Not compiled with METIS support')
@pytest.mark.parametrize('device,weighted', product(devices, [False, True]))
def test_metis(device, weighted):
    mat1 = torch.randn(6 * 6, device=device).view(6, 6)
    mat2 = torch.arange(6 * 6, dtype=torch.long, device=device).view(6, 6)
    mat3 = torch.ones(6 * 6, device=device).view(6, 6)

    vec1 = None
    vec2 = torch.rand(6, device=device)

    for mat, vec in product([mat1, mat2, mat3], [vec1, vec2]):
        mat = SparseTensor.from_dense(mat)

        _, partptr, perm = mat.partition(num_parts=1, recursive=False,
                                         weighted=weighted, node_weight=vec)
        assert partptr.numel() == 2
        assert perm.numel() == 6

        _, partptr, perm = mat.partition(num_parts=2, recursive=False,
                                         weighted=weighted, node_weight=vec)
        assert partptr.numel() == 3
        assert perm.numel() == 6
