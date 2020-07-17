from itertools import product

import pytest
import torch
from torch_sparse import SparseTensor

from .utils import grad_dtypes, devices


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_getitem(dtype, device):
    mat = torch.randn(50, 40, dtype=dtype, device=device)
    mat = SparseTensor.from_dense(mat)

    idx1 = torch.randint(0, 50, (10, ), dtype=torch.long, device=device)
    idx2 = torch.randint(0, 40, (10, ), dtype=torch.long, device=device)

    assert mat[:10, :10].sizes() == [10, 10]
    assert mat[..., :10].sizes() == [50, 10]
    assert mat[idx1, idx2].sizes() == [10, 10]
    assert mat[idx1.tolist()].sizes() == [10, 40]
