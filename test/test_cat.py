from itertools import product

import pytest
import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.cat import cat

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_cat(dtype, device):
    index = tensor([[0, 0, 1], [0, 1, 2]], torch.long, device)
    mat1 = SparseTensor(index)

    index = tensor([[0, 0, 1, 2], [0, 1, 1, 0]], torch.long, device)
    mat2 = SparseTensor(index)

    cat([mat1, mat2], dim=(0, 1))
