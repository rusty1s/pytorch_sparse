from typing import Optional

import torch
from torch_sparse.storage import SparseStorage, get_layout
from torch_sparse.tensor import SparseTensor


def masked_select(src: SparseTensor, dim: int,
                  mask: torch.Tensor) -> SparseTensor:
    dim = src.dim() + dim if dim < 0 else dim

    assert mask.dim() == 1
    storage = src.storage

    if dim == 0:
        row, col, value = src.coo()
        rowcount = src.storage.rowcount()

        rowcount = rowcount[mask]

        mask = mask[row]
        row = torch.arange(rowcount.size(0),
                           device=row.device).repeat_interleave(rowcount)

        col = col[mask]

        if value is not None:
            value = value[mask]

        sparse_sizes = (rowcount.size(0), src.sparse_size(1))

        storage = SparseStorage(row=row, rowptr=None, col=col, value=value,
                                sparse_sizes=sparse_sizes, rowcount=rowcount,
                                colcount=None, colptr=None, csr2csc=None,
                                csc2csr=None, is_sorted=True)
        return src.from_storage(storage)

    elif dim == 1:
        row, col, value = src.coo()
        csr2csc = src.storage.csr2csc()
        row = row[csr2csc]
        col = col[csr2csc]
        colcount = src.storage.colcount()

        colcount = colcount[mask]

        mask = mask[col]
        col = torch.arange(colcount.size(0),
                           device=col.device).repeat_interleave(colcount)
        row = row[mask]
        csc2csr = (colcount.size(0) * row + col).argsort()
        row, col = row[csc2csr], col[csc2csr]

        if value is not None:
            value = value[csr2csc][mask][csc2csr]

        sparse_sizes = (src.sparse_size(0), colcount.size(0))

        storage = SparseStorage(row=row, rowptr=None, col=col, value=value,
                                sparse_sizes=sparse_sizes, rowcount=None,
                                colcount=colcount, colptr=None, csr2csc=None,
                                csc2csr=csc2csr, is_sorted=True)
        return src.from_storage(storage)

    else:
        value = src.storage.value()
        if value is not None:
            idx = mask.nonzero().flatten()
            return src.set_value(value.index_select(dim - 1, idx),
                                 layout='coo')
        else:
            raise ValueError


def masked_select_nnz(src: SparseTensor, mask: torch.Tensor,
                      layout: Optional[str] = None) -> SparseTensor:
    assert mask.dim() == 1

    if get_layout(layout) == 'csc':
        mask = mask[src.storage.csc2csr()]

    row, col, value = src.coo()
    row, col = row[mask], col[mask]

    if value is not None:
        value = value[mask]

    return SparseTensor(row=row, rowptr=None, col=col, value=value,
                        sparse_sizes=src.sparse_sizes(), is_sorted=True)


SparseTensor.masked_select = lambda self, dim, mask: masked_select(
    self, dim, mask)
tmp = lambda self, mask, layout=None: masked_select_nnz(  # noqa
    self, mask, layout)
SparseTensor.masked_select_nnz = tmp
