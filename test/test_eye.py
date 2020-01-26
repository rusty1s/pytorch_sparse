from itertools import product

import pytest
from torch_sparse.tensor import SparseTensor

from .utils import dtypes, devices


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_eye(dtype, device):
    mat = SparseTensor.eye(3, dtype=dtype, device=device)
    assert mat.storage.row.tolist() == [0, 1, 2]
    assert mat.storage.rowptr.tolist() == [0, 1, 2, 3]
    assert mat.storage.col.tolist() == [0, 1, 2]
    assert mat.storage.value.tolist() == [1, 1, 1]
    assert len(mat.cached_keys()) == 0

    mat = SparseTensor.eye(3, dtype=dtype, device=device, has_value=False)
    assert mat.storage.row.tolist() == [0, 1, 2]
    assert mat.storage.rowptr.tolist() == [0, 1, 2, 3]
    assert mat.storage.col.tolist() == [0, 1, 2]
    assert mat.storage.value is None
    assert len(mat.cached_keys()) == 0

    mat = SparseTensor.eye(3, 4, dtype=dtype, device=device, fill_cache=True)
    assert mat.storage.row.tolist() == [0, 1, 2]
    assert mat.storage.rowptr.tolist() == [0, 1, 2, 3]
    assert mat.storage.col.tolist() == [0, 1, 2]
    assert len(mat.cached_keys()) == 5
    assert mat.storage.rowcount.tolist() == [1, 1, 1]
    assert mat.storage.colptr.tolist() == [0, 1, 2, 3, 3]
    assert mat.storage.colcount.tolist() == [1, 1, 1, 0]
    assert mat.storage.csr2csc.tolist() == [0, 1, 2]
    assert mat.storage.csc2csr.tolist() == [0, 1, 2]

    mat = SparseTensor.eye(4, 3, dtype=dtype, device=device, fill_cache=True)
    assert mat.storage.row.tolist() == [0, 1, 2]
    assert mat.storage.rowptr.tolist() == [0, 1, 2, 3, 3]
    assert mat.storage.col.tolist() == [0, 1, 2]
    assert len(mat.cached_keys()) == 5
    assert mat.storage.rowcount.tolist() == [1, 1, 1, 0]
    assert mat.storage.colptr.tolist() == [0, 1, 2, 3]
    assert mat.storage.colcount.tolist() == [1, 1, 1]
    assert mat.storage.csr2csc.tolist() == [0, 1, 2]
    assert mat.storage.csc2csr.tolist() == [0, 1, 2]
