from itertools import product

import pytest
import torch
from torch_sparse import spspmm, spmm

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_spmm_spspmm(dtype, device):
    row = torch.tensor([0, 0, 1, 2, 2], device=device)
    col = torch.tensor([0, 2, 1, 0, 1], device=device)
    index = torch.stack([row, col], dim=0)
    value = tensor([1, 2, 4, 1, 3], dtype, device)
    x = tensor([[1, 4], [2, 5], [3, 6]], dtype, device)

    value = value.requires_grad_(True)

    out_index, out_value = spspmm(index, value, index, value, 3, 3, 3)
    out = spmm(out_index, out_value, 3, 3, x)
    assert out.size() == (3, 2)
