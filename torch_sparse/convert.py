import numpy as np
import scipy.sparse
import torch
from torch import from_numpy


def to_torch_sparse(index, value, m, n):
    return torch.sparse_coo_tensor(index.detach(), value, (m, n))


def from_torch_sparse(A):
    return A.indices().detach(), A.values()


def to_scipy(index, value, m, n):
    assert not index.is_cuda and not value.is_cuda
    (row, col), data = index.detach(), value.detach()
    return scipy.sparse.coo_matrix((data, (row, col)), (m, n))


def from_scipy(A):
    A = A.tocoo()
    row, col, value = A.row.astype(np.int64), A.col.astype(np.int64), A.data
    row, col, value = from_numpy(row), from_numpy(col), from_numpy(value)
    index = torch.stack([row, col], dim=0)
    return index, value
