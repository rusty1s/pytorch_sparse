import time
from itertools import product
from scipy.io import loadmat
import numpy as np

import pytest
import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.add import sparse_add

from .utils import dtypes, devices, tensor

devices = ['cpu']
dtypes = [torch.float]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_index_select(dtype, device):
    row = torch.tensor([0, 0, 1, 1, 2])
    col = torch.tensor([0, 1, 1, 2, 1])
    mat = SparseTensor(row=row, col=col)
    print()
    print(mat.to_dense())
    pass

    mat = mat.index_select(0, torch.tensor([0, 2]))
    print(mat.to_dense())
