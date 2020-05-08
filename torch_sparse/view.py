import torch

from torch_sparse.storage import SparseStorage
from torch_sparse.tensor import SparseTensor


def _view(src: SparseTensor, n: int, layout: str = 'csr') -> SparseTensor:
    row, col, value = src.coo()
    sparse_sizes = src.storage.sparse_sizes()

    if sparse_sizes[0] * sparse_sizes[1] % n == 0:
        raise RuntimeError(
            f"shape '[-1, {n}]' is invalid for input of size {sparse_sizes[0] * sparse_sizes[1]}")

    assert layout == 'csr' or layout == 'csc'

    if layout == 'csr':
        idx = sparse_sizes[1] * row + col
        row = idx // n
        col = idx % n
        sparse_sizes = (sparse_sizes[0] * sparse_sizes[1] // n, n)
    if layout == 'csc':
        idx = sparse_sizes[0] * col + row
        row = idx % n
        col = idx // n
        sparse_sizes = (n, sparse_sizes[0] * sparse_sizes[1] // n)

    storage = SparseStorage(
        row=row,
        rowptr=src.storage._rowptr,
        col=col,
        value=value,
        sparse_sizes=sparse_sizes,
        rowcount=src.storage._rowcount,
        colptr=src.storage._colptr,
        colcount=src.storage._colcount,
        csr2csc=src.storage._csr2csc,
        csc2csr=src.storage._csc2csr,
        is_sorted=True,
    )

    return src.from_storage(storage)


SparseTensor.view = lambda self, m, n: _view(self, n, layout='csr')

###############################################################################


def view(index, value, m, n, new_n):
    assert m * n % new_n == 0

    row, col = index
    idx = n * row + col
    row = idx // new_n
    col = idx % new_n

    return torch.stack([row, col], dim=0), value
