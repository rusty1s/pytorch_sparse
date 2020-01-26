import copy
from itertools import product

import pytest
import torch
from torch_sparse.storage import SparseStorage, no_cache

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_storage(dtype, device):
    row, col = tensor([[0, 0, 1, 1], [0, 1, 0, 1]], torch.long, device)

    storage = SparseStorage(row=row, col=col)
    assert storage.row.tolist() == [0, 0, 1, 1]
    assert storage.col.tolist() == [0, 1, 0, 1]
    assert storage.value is None
    assert storage.sparse_size == (2, 2)

    row, col = tensor([[0, 0, 1, 1], [1, 0, 1, 0]], torch.long, device)
    value = tensor([2, 1, 4, 3], dtype, device)
    storage = SparseStorage(row=row, col=col, value=value)
    assert storage.row.tolist() == [0, 0, 1, 1]
    assert storage.col.tolist() == [0, 1, 0, 1]
    assert storage.value.tolist() == [1, 2, 3, 4]
    assert storage.sparse_size == (2, 2)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_caching(dtype, device):
    row, col = tensor([[0, 0, 1, 1], [0, 1, 0, 1]], torch.long, device)
    storage = SparseStorage(row=row, col=col)

    assert storage._row.tolist() == row.tolist()
    assert storage._col.tolist() == col.tolist()
    assert storage._value is None

    assert storage._rowcount is None
    assert storage._rowptr is None
    assert storage._colcount is None
    assert storage._colptr is None
    assert storage._csr2csc is None
    assert storage.cached_keys() == []

    storage.fill_cache_()
    assert storage._rowcount.tolist() == [2, 2]
    assert storage._rowptr.tolist() == [0, 2, 4]
    assert storage._colcount.tolist() == [2, 2]
    assert storage._colptr.tolist() == [0, 2, 4]
    assert storage._csr2csc.tolist() == [0, 2, 1, 3]
    assert storage._csc2csr.tolist() == [0, 2, 1, 3]
    assert storage.cached_keys() == [
        'rowcount', 'colptr', 'colcount', 'csr2csc', 'csc2csr'
    ]

    storage = SparseStorage(row=row, rowptr=storage.rowptr, col=col,
                            value=storage.value,
                            sparse_size=storage.sparse_size,
                            rowcount=storage.rowcount, colptr=storage.colptr,
                            colcount=storage.colcount, csr2csc=storage.csr2csc,
                            csc2csr=storage.csc2csr)

    assert storage._rowcount.tolist() == [2, 2]
    assert storage._rowptr.tolist() == [0, 2, 4]
    assert storage._colcount.tolist() == [2, 2]
    assert storage._colptr.tolist() == [0, 2, 4]
    assert storage._csr2csc.tolist() == [0, 2, 1, 3]
    assert storage._csc2csr.tolist() == [0, 2, 1, 3]
    assert storage.cached_keys() == [
        'rowcount', 'colptr', 'colcount', 'csr2csc', 'csc2csr'
    ]

    storage.clear_cache_()
    assert storage._rowcount is None
    assert storage._rowptr is not None
    assert storage._colcount is None
    assert storage._colptr is None
    assert storage._csr2csc is None
    assert storage.cached_keys() == []

    with no_cache():
        storage.fill_cache_()
    assert storage.cached_keys() == []

    @no_cache()
    def do_something(storage):
        return storage.fill_cache_()

    storage = do_something(storage)
    assert storage.cached_keys() == []


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_utility(dtype, device):
    row, col = tensor([[0, 0, 1, 1], [1, 0, 1, 0]], torch.long, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    storage = SparseStorage(row=row, col=col, value=value)

    assert storage.has_value()

    storage.set_value_(value, layout='csc')
    assert storage.value.tolist() == [1, 3, 2, 4]
    storage.set_value_(value, layout='coo')
    assert storage.value.tolist() == [1, 2, 3, 4]

    storage = storage.set_value(value, layout='csc')
    assert storage.value.tolist() == [1, 3, 2, 4]
    storage = storage.set_value(value, layout='coo')
    assert storage.value.tolist() == [1, 2, 3, 4]

    storage = storage.sparse_resize(3, 3)
    assert storage.sparse_size == (3, 3)

    new_storage = copy.copy(storage)
    assert new_storage != storage
    assert new_storage.col.data_ptr() == storage.col.data_ptr()

    new_storage = storage.clone()
    assert new_storage != storage
    assert new_storage.col.data_ptr() != storage.col.data_ptr()

    new_storage = copy.deepcopy(storage)
    assert new_storage != storage
    assert new_storage.col.data_ptr() != storage.col.data_ptr()

    storage.apply_value_(lambda x: x + 1)
    assert storage.value.tolist() == [2, 3, 4, 5]
    storage = storage.apply_value(lambda x: x + 1)
    assert storage.value.tolist() == [3, 4, 5, 6]

    storage.apply_(lambda x: x.to(torch.long))
    assert storage.col.dtype == torch.long
    assert storage.value.dtype == torch.long

    storage = storage.apply(lambda x: x.to(torch.long))
    assert storage.col.dtype == torch.long
    assert storage.value.dtype == torch.long

    storage.clear_cache_()
    assert storage.map(lambda x: x.numel()) == [4, 4, 4]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_coalesce(dtype, device):
    row, col = tensor([[0, 0, 0, 1, 1], [0, 1, 1, 0, 1]], torch.long, device)
    value = tensor([1, 1, 1, 3, 4], dtype, device)
    storage = SparseStorage(row=row, col=col, value=value)

    assert storage.row.tolist() == row.tolist()
    assert storage.col.tolist() == col.tolist()
    assert storage.value.tolist() == value.tolist()

    assert not storage.is_coalesced()
    storage = storage.coalesce()
    assert storage.is_coalesced()

    assert storage.row.tolist() == [0, 0, 1, 1]
    assert storage.col.tolist() == [0, 1, 0, 1]
    assert storage.value.tolist() == [1, 2, 3, 4]
