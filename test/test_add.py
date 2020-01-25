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
def test_sparse_add(dtype, device):
    name = ('DIMACS10', 'citationCiteseer')[1]
    mat_scipy = loadmat(f'benchmark/{name}.mat')['Problem'][0][0][2].tocsr()
    mat = SparseTensor.from_scipy(mat_scipy)

    mat1 = mat[:, 0:100000]
    mat2 = mat[:, 100000:200000]
    print(mat1.shape)
    print(mat2.shape)

    # 0.0159 to beat
    t = time.perf_counter()
    mat = sparse_add(mat1, mat2)
    print(time.perf_counter() - t)
    print(mat.nnz())

    mat1 = mat_scipy[:, 0:100000]
    mat2 = mat_scipy[:, 100000:200000]
    t = time.perf_counter()
    mat = mat1 + mat2
    print(time.perf_counter() - t)
    print(mat.nnz)

    # mat1 + mat2

    # mat1 = mat1.tocoo()
    # mat2 = mat2.tocoo()

    # row1, col1 = mat1.row, mat1.col
    # row2, col2 = mat2.row, mat2.col

    # idx1 = row1 * 100000 + col1
    # idx2 = row2 * 100000 + col2

    # t = time.perf_counter()
    # np.union1d(idx1, idx2)
    # print(time.perf_counter() - t)

    # index = tensor([[0, 0, 1], [0, 1, 2]], torch.long, device)
    # mat1 = SparseTensor(index)
    # print()
    # print(mat1.to_dense())

    # index = tensor([[0, 0, 1, 2], [0, 1, 1, 0]], torch.long, device)
    # mat2 = SparseTensor(index)
    # print(mat2.to_dense())

    # add(mat1, mat2)
