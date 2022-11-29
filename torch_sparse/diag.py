from typing import Optional

import torch
from torch import Tensor

from torch_sparse.storage import SparseStorage
from torch_sparse.tensor import SparseTensor


def remove_diag(src: SparseTensor, k: int = 0) -> SparseTensor:
    row, col, value = src.coo()
    inv_mask = row != col if k == 0 else row != (col - k)
    new_row, new_col = row[inv_mask], col[inv_mask]

    if value is not None:
        value = value[inv_mask]

    rowcount = src.storage._rowcount
    colcount = src.storage._colcount
    if rowcount is not None or colcount is not None:
        mask = ~inv_mask
        if rowcount is not None:
            rowcount = rowcount.clone()
            rowcount[row[mask]] -= 1
        if colcount is not None:
            colcount = colcount.clone()
            colcount[col[mask]] -= 1

    storage = SparseStorage(row=new_row, rowptr=None, col=new_col, value=value,
                            sparse_sizes=src.sparse_sizes(), rowcount=rowcount,
                            colptr=None, colcount=colcount, csr2csc=None,
                            csc2csr=None, is_sorted=True)
    return src.from_storage(storage)


def set_diag(src: SparseTensor, values: Optional[Tensor] = None,
             k: int = 0) -> SparseTensor:
    src = remove_diag(src, k=k)
    row, col, value = src.coo()

    mask = torch.ops.torch_sparse.non_diag_mask(row, col, src.size(0),
                                                src.size(1), k)
    inv_mask = ~mask

    start, num_diag = -k if k < 0 else 0, mask.numel() - row.numel()
    diag = torch.arange(start, start + num_diag, device=row.device)

    new_row = row.new_empty(mask.size(0))
    new_row[mask] = row
    new_row[inv_mask] = diag

    new_col = col.new_empty(mask.size(0))
    new_col[mask] = col
    new_col[inv_mask] = diag.add_(k)

    new_value: Optional[Tensor] = None
    if value is not None:
        new_value = value.new_empty((mask.size(0), ) + value.size()[1:])
        new_value[mask] = value
        if values is not None:
            new_value[inv_mask] = values
        else:
            new_value[inv_mask] = torch.ones((num_diag, ), dtype=value.dtype,
                                             device=value.device)

    rowcount = src.storage._rowcount
    if rowcount is not None:
        rowcount = rowcount.clone()
        rowcount[start:start + num_diag] += 1

    colcount = src.storage._colcount
    if colcount is not None:
        colcount = colcount.clone()
        colcount[start + k:start + num_diag + k] += 1

    storage = SparseStorage(row=new_row, rowptr=None, col=new_col,
                            value=new_value, sparse_sizes=src.sparse_sizes(),
                            rowcount=rowcount, colptr=None, colcount=colcount,
                            csr2csc=None, csc2csr=None, is_sorted=True)
    return src.from_storage(storage)


def fill_diag(src: SparseTensor, fill_value: float,
              k: int = 0) -> SparseTensor:
    num_diag = min(src.sparse_size(0), src.sparse_size(1) - k)
    if k < 0:
        num_diag = min(src.sparse_size(0) + k, src.sparse_size(1))

    value = src.storage.value()
    if value is not None:
        sizes = [num_diag] + src.sizes()[2:]
        return set_diag(src, value.new_full(sizes, fill_value), k)
    else:
        return set_diag(src, None, k)


def get_diag(src: SparseTensor) -> Tensor:
    row, col, value = src.coo()

    if value is None:
        value = torch.ones(row.size(0), device=row.device)

    sizes = list(value.size())
    sizes[0] = min(src.size(0), src.size(1))

    out = value.new_zeros(sizes)

    mask = row == col
    out[row[mask]] = value[mask]
    return out


SparseTensor.remove_diag = lambda self, k=0: remove_diag(self, k)
SparseTensor.set_diag = lambda self, values=None, k=0: set_diag(
    self, values, k)
SparseTensor.fill_diag = lambda self, fill_value, k=0: fill_diag(
    self, fill_value, k)
SparseTensor.get_diag = lambda self: get_diag(self)
