import copy
from itertools import product

import pytest
import torch
from torch_sparse.storage import SparseStorage

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_storage(dtype, device):
    index = tensor([[0, 0, 1, 1], [0, 1, 0, 1]], torch.long, device)

    storage = SparseStorage(index)
    assert storage.index.tolist() == index.tolist()
    assert storage.row.tolist() == [0, 0, 1, 1]
    assert storage.col.tolist() == [0, 1, 0, 1]
    assert storage.value is None
    assert storage.sparse_size() == (2, 2)

    index = tensor([[0, 0, 1, 1], [1, 0, 1, 0]], torch.long, device)
    value = tensor([2, 1, 4, 3], dtype, device)
    storage = SparseStorage(index, value)
    assert storage.index.tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert storage.row.tolist() == [0, 0, 1, 1]
    assert storage.col.tolist() == [0, 1, 0, 1]
    assert storage.value.tolist() == [1, 2, 3, 4]
    assert storage.sparse_size() == (2, 2)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_caching(dtype, device):
    index = tensor([[0, 0, 1, 1], [0, 1, 0, 1]], torch.long, device)
    storage = SparseStorage(index)

    assert storage._index.tolist() == index.tolist()
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
        'rowcount', 'rowptr', 'colcount', 'colptr', 'csr2csc', 'csc2csr'
    ]

    storage = SparseStorage(index, storage.value, storage.sparse_size(),
                            storage.rowcount, storage.rowptr, storage.colcount,
                            storage.colptr, storage.csr2csc, storage.csc2csr)

    assert storage._rowcount.tolist() == [2, 2]
    assert storage._rowptr.tolist() == [0, 2, 4]
    assert storage._colcount.tolist() == [2, 2]
    assert storage._colptr.tolist() == [0, 2, 4]
    assert storage._csr2csc.tolist() == [0, 2, 1, 3]
    assert storage._csc2csr.tolist() == [0, 2, 1, 3]
    assert storage.cached_keys() == [
        'rowcount', 'rowptr', 'colcount', 'colptr', 'csr2csc', 'csc2csr'
    ]

    storage.clear_cache_()
    assert storage._rowcount is None
    assert storage._rowptr is None
    assert storage._colcount is None
    assert storage._colptr is None
    assert storage._csr2csc is None
    assert storage.cached_keys() == []


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_utility(dtype, device):
    index = tensor([[0, 0, 1, 1], [1, 0, 1, 0]], torch.long, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    storage = SparseStorage(index, value)

    assert storage.has_value()

    storage.set_value_(value, layout='csc')
    assert storage.value.tolist() == [1, 3, 2, 4]
    storage.set_value_(value, layout='coo')
    assert storage.value.tolist() == [1, 2, 3, 4]

    storage = storage.set_value(value, layout='csc')
    assert storage.value.tolist() == [1, 3, 2, 4]
    storage = storage.set_value(value, layout='coo')
    assert storage.value.tolist() == [1, 2, 3, 4]

    storage.sparse_resize_(3, 3)
    assert storage.sparse_size() == (3, 3)

    new_storage = copy.copy(storage)
    assert new_storage != storage
    assert new_storage.index.data_ptr() == storage.index.data_ptr()

    new_storage = storage.clone()
    assert new_storage != storage
    assert new_storage.index.data_ptr() != storage.index.data_ptr()

    new_storage = copy.deepcopy(storage)
    assert new_storage != storage
    assert new_storage.index.data_ptr() != storage.index.data_ptr()

    storage.apply_value_(lambda x: x + 1)
    assert storage.value.tolist() == [2, 3, 4, 5]
    storage = storage.apply_value(lambda x: x + 1)
    assert storage.value.tolist() == [3, 4, 5, 6]

    storage.apply_(lambda x: x.to(torch.long))
    assert storage.index.dtype == torch.long
    assert storage.value.dtype == torch.long

    storage = storage.apply(lambda x: x.to(torch.long))
    assert storage.index.dtype == torch.long
    assert storage.value.dtype == torch.long

    storage.clear_cache_()
    assert storage.map(lambda x: x.numel()) == [8, 4]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_coalesce(dtype, device):
    index = tensor([[0, 0, 0, 1, 1], [0, 1, 1, 0, 1]], torch.long, device)
    value = tensor([1, 1, 1, 3, 4], dtype, device)
    storage = SparseStorage(index, value)

    assert storage.index.tolist() == index.tolist()
    assert storage.value.tolist() == value.tolist()

    assert not storage.is_coalesced()
    storage = storage.coalesce()
    assert storage.is_coalesced()

    assert storage.index.tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert storage.value.tolist() == [1, 2, 3, 4]
