from typing import Optional

import torch
from torch_scatter import gather_csr
from torch_sparse.storage import SparseStorage, get_layout
from torch_sparse.tensor import SparseTensor


def index_select(src: SparseTensor, dim: int,
                 idx: torch.Tensor) -> SparseTensor:
    dim = src.dim() + dim if dim < 0 else dim
    assert idx.dim() == 1

    if dim == 0:
        old_rowptr, col, value = src.csr()
        rowcount = src.storage.rowcount()

        rowcount = rowcount[idx]

        rowptr = col.new_zeros(idx.size(0) + 1)
        torch.cumsum(rowcount, dim=0, out=rowptr[1:])

        row = torch.arange(idx.size(0),
                           device=col.device).repeat_interleave(rowcount)

        perm = torch.arange(row.size(0), device=row.device)
        perm += gather_csr(old_rowptr[idx] - rowptr[:-1], rowptr)

        col = col[perm]

        if value is not None:
            value = value[perm]

        sparse_sizes = (idx.size(0), src.sparse_size(1))

        storage = SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                                sparse_sizes=sparse_sizes, rowcount=rowcount,
                                colptr=None, colcount=None, csr2csc=None,
                                csc2csr=None, is_sorted=True)
        return src.from_storage(storage)

    elif dim == 1:
        old_colptr, row, value = src.csc()
        colcount = src.storage.colcount()

        colcount = colcount[idx]

        colptr = row.new_zeros(idx.size(0) + 1)
        torch.cumsum(colcount, dim=0, out=colptr[1:])

        col = torch.arange(idx.size(0),
                           device=row.device).repeat_interleave(colcount)

        perm = torch.arange(col.size(0), device=col.device)
        perm += gather_csr(old_colptr[idx] - colptr[:-1], colptr)

        row = row[perm]
        csc2csr = (idx.size(0) * row + col).argsort()
        row, col = row[csc2csr], col[csc2csr]

        if value is not None:
            value = value[perm][csc2csr]

        sparse_sizes = (src.sparse_size(0), idx.size(0))

        storage = SparseStorage(row=row, rowptr=None, col=col, value=value,
                                sparse_sizes=sparse_sizes, rowcount=None,
                                colptr=colptr, colcount=colcount, csr2csc=None,
                                csc2csr=csc2csr, is_sorted=True)
        return src.from_storage(storage)

    else:
        value = src.storage.value()
        if value is not None:
            return src.set_value(value.index_select(dim - 1, idx),
                                 layout='coo')
        else:
            raise ValueError


def index_select_nnz(src: SparseTensor, idx: torch.Tensor,
                     layout: Optional[str] = None) -> SparseTensor:
    assert idx.dim() == 1

    if get_layout(layout) == 'csc':
        idx = src.storage.csc2csr()[idx]

    row, col, value = src.coo()
    row, col = row[idx], col[idx]

    if value is not None:
        value = value[idx]

    return SparseTensor(row=row, rowptr=None, col=col, value=value,
                        sparse_sizes=src.sparse_sizes(), is_sorted=True)


SparseTensor.index_select = lambda self, dim, idx: index_select(self, dim, idx)
tmp = lambda self, idx, layout=None: index_select_nnz(  # noqa
    self, idx, layout)
SparseTensor.index_select_nnz = tmp
