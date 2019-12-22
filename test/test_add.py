from itertools import product

import pytest
import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.add import add

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_add(dtype, device):
    print()
    index = tensor([[0, 0, 1], [0, 1, 2]], torch.long, device)
    mat1 = SparseTensor(index)

    index = tensor([[0, 0, 1, 2], [0, 1, 1, 0]], torch.long, device)
    mat2 = SparseTensor(index)

    add(mat1, mat2)
