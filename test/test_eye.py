from itertools import product

import pytest

from torch_sparse.tensor import SparseTensor
from torch_sparse.testing import devices, dtypes


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_eye(dtype, device):
    mat = SparseTensor.eye(3, dtype=dtype, device=device)
    assert mat.device() == device
    assert mat.storage.sparse_sizes() == (3, 3)
    assert mat.storage.row().tolist() == [0, 1, 2]
    assert mat.storage.rowptr().tolist() == [0, 1, 2, 3]
    assert mat.storage.col().tolist() == [0, 1, 2]
    assert mat.storage.value().tolist() == [1, 1, 1]
    assert mat.storage.value().dtype == dtype
    assert mat.storage.num_cached_keys() == 0

    mat = SparseTensor.eye(3, has_value=False, device=device)
    assert mat.device() == device
    assert mat.storage.sparse_sizes() == (3, 3)
    assert mat.storage.row().tolist() == [0, 1, 2]
    assert mat.storage.rowptr().tolist() == [0, 1, 2, 3]
    assert mat.storage.col().tolist() == [0, 1, 2]
    assert mat.storage.value() is None
    assert mat.storage.num_cached_keys() == 0

    mat = SparseTensor.eye(3, 4, fill_cache=True, device=device)
    assert mat.device() == device
    assert mat.storage.sparse_sizes() == (3, 4)
    assert mat.storage.row().tolist() == [0, 1, 2]
    assert mat.storage.rowptr().tolist() == [0, 1, 2, 3]
    assert mat.storage.col().tolist() == [0, 1, 2]
    assert mat.storage.num_cached_keys() == 5
    assert mat.storage.rowcount().tolist() == [1, 1, 1]
    assert mat.storage.colptr().tolist() == [0, 1, 2, 3, 3]
    assert mat.storage.colcount().tolist() == [1, 1, 1, 0]
    assert mat.storage.csr2csc().tolist() == [0, 1, 2]
    assert mat.storage.csc2csr().tolist() == [0, 1, 2]

    mat = SparseTensor.eye(4, 3, fill_cache=True, device=device)
    assert mat.device() == device
    assert mat.storage.sparse_sizes() == (4, 3)
    assert mat.storage.row().tolist() == [0, 1, 2]
    assert mat.storage.rowptr().tolist() == [0, 1, 2, 3, 3]
    assert mat.storage.col().tolist() == [0, 1, 2]
    assert mat.storage.num_cached_keys() == 5
    assert mat.storage.rowcount().tolist() == [1, 1, 1, 0]
    assert mat.storage.colptr().tolist() == [0, 1, 2, 3]
    assert mat.storage.colcount().tolist() == [1, 1, 1]
    assert mat.storage.csr2csc().tolist() == [0, 1, 2]
    assert mat.storage.csc2csr().tolist() == [0, 1, 2]
