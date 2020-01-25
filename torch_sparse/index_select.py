import torch
from torch_scatter import gather_csr

from torch_sparse.storage import get_layout


def index_select(src, dim, idx):
    dim = src.dim() + dim if dim < 0 else dim

    assert idx.dim() == 1

    if dim == 0:
        old_rowptr, col, value = src.csr()
        rowcount = src.storage.rowcount

        rowcount = rowcount[idx]

        rowptr = col.new_zeros(idx.size(0) + 1)
        torch.cumsum(rowcount, dim=0, out=rowptr[1:])

        row = torch.arange(idx.size(0),
                           device=col.device).repeat_interleave(rowcount)

        perm = torch.arange(row.size(0), device=row.device)
        perm += gather_csr(old_rowptr[idx] - rowptr[:-1], rowptr)

        col = col[perm]

        if src.has_value():
            value = value[perm]

        sparse_size = torch.Size([idx.size(0), src.sparse_size(1)])

        storage = src.storage.__class__(row=row, rowptr=rowptr, col=col,
                                        value=value, sparse_size=sparse_size,
                                        rowcount=rowcount, is_sorted=True)

    elif dim == 1:
        old_colptr, row, value = src.csc()
        colcount = src.storage.colcount

        colcount = colcount[idx]
        col = torch.arange(idx.size(0),
                           device=row.device).repeat_interleave(colcount)

        colptr = row.new_zeros(idx.size(0) + 1)
        torch.cumsum(colcount, dim=0, out=colptr[1:])

        perm = torch.arange(col.size(0), device=col.device)
        perm += gather_csr(old_colptr[idx] - colptr[:-1], colptr)

        row = row[perm]
        csc2csr = (idx.size(0) * row + col).argsort()
        row, col = row[csc2csr], col[csc2csr]

        if src.has_value():
            value = value[perm][csc2csr]

        sparse_size = torch.Size([src.sparse_size(0), idx.size(0)])

        storage = src.storage.__class__(row=row, col=col, value=value,
                                        sparse_size=sparse_size, colptr=colptr,
                                        colcount=colcount, csc2csr=csc2csr,
                                        is_sorted=True)

    else:
        storage = src.storage.apply_value(
            lambda x: x.index_select(dim - 1, idx))

    return src.from_storage(storage)


def index_select_nnz(src, idx, layout=None):
    assert idx.dim() == 1

    if get_layout(layout) == 'csc':
        idx = idx[src.storage.csc2csr]

    row, col, value = src.coo()
    row, col = row[idx], col[idx]

    if src.has_value():
        value = value[idx]

    # There is no other information we can maintain...
    storage = src.storage.__class__(row=row, col=col, value=value,
                                    sparse_size=src.sparse_size(),
                                    is_sorted=True)

    return src.from_storage(storage)
