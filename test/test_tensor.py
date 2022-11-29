from itertools import product

import pytest
import torch

from torch_sparse import SparseTensor
from torch_sparse.testing import devices, grad_dtypes


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_getitem(dtype, device):
    m = 50
    n = 40
    k = 10
    mat = torch.randn(m, n, dtype=dtype, device=device)
    mat = SparseTensor.from_dense(mat)

    idx1 = torch.randint(0, m, (k, ), dtype=torch.long, device=device)
    idx2 = torch.randint(0, n, (k, ), dtype=torch.long, device=device)
    bool1 = torch.zeros(m, dtype=torch.bool, device=device)
    bool2 = torch.zeros(n, dtype=torch.bool, device=device)
    bool1.scatter_(0, idx1, 1)
    bool2.scatter_(0, idx2, 1)
    # idx1 and idx2 may have duplicates
    k1_bool = bool1.nonzero().size(0)
    k2_bool = bool2.nonzero().size(0)

    idx1np = idx1.cpu().numpy()
    idx2np = idx2.cpu().numpy()
    bool1np = bool1.cpu().numpy()
    bool2np = bool2.cpu().numpy()

    idx1list = idx1np.tolist()
    idx2list = idx2np.tolist()
    bool1list = bool1np.tolist()
    bool2list = bool2np.tolist()

    assert mat[:k, :k].sizes() == [k, k]
    assert mat[..., :k].sizes() == [m, k]

    assert mat[idx1, idx2].sizes() == [k, k]
    assert mat[idx1np, idx2np].sizes() == [k, k]
    assert mat[idx1list, idx2list].sizes() == [k, k]

    assert mat[bool1, bool2].sizes() == [k1_bool, k2_bool]
    assert mat[bool1np, bool2np].sizes() == [k1_bool, k2_bool]
    assert mat[bool1list, bool2list].sizes() == [k1_bool, k2_bool]

    assert mat[idx1].sizes() == [k, n]
    assert mat[idx1np].sizes() == [k, n]
    assert mat[idx1list].sizes() == [k, n]

    assert mat[bool1].sizes() == [k1_bool, n]
    assert mat[bool1np].sizes() == [k1_bool, n]
    assert mat[bool1list].sizes() == [k1_bool, n]


@pytest.mark.parametrize('device', devices)
def test_to_symmetric(device):
    row = torch.tensor([0, 0, 0, 1, 1], device=device)
    col = torch.tensor([0, 1, 2, 0, 2], device=device)
    value = torch.arange(1, 6, device=device)
    mat = SparseTensor(row=row, col=col, value=value)
    assert not mat.is_symmetric()

    mat = mat.to_symmetric()

    assert mat.is_symmetric()
    assert mat.to_dense().tolist() == [
        [2, 6, 3],
        [6, 0, 5],
        [3, 5, 0],
    ]


def test_equal():
    row = torch.tensor([0, 0, 0, 1, 1])
    col = torch.tensor([0, 1, 2, 0, 2])
    value = torch.arange(1, 6)
    matA = SparseTensor(row=row, col=col, value=value)
    matB = SparseTensor(row=row, col=col, value=value)
    col = torch.tensor([0, 1, 2, 0, 1])
    matC = SparseTensor(row=row, col=col, value=value)

    assert id(matA) != id(matB)
    assert matA == matB

    assert id(matA) != id(matC)
    assert matA != matC
