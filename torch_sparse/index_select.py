import torch

from torch_sparse.storage import get_layout


def index_select(src, dim, idx):
    dim = src.dim() + dim if dim < 0 else dim

    assert idx.dim() == 1

    if dim == 0:
        (row, col), value = src.coo()
        rowcount = src.storage.rowcount
        old_rowptr = src.storage.rowptr

        rowcount = rowcount[idx]
        tmp = torch.arange(rowcount.size(0), device=rowcount.device)
        row = tmp.repeat_interleave(rowcount)

        # Creates an "arange interleave" tensor of col indices.
        rowptr = torch.cat([row.new_zeros(1), rowcount.cumsum(0)], dim=0)
        perm = torch.arange(row.size(0), device=row.device)
        perm += (old_rowptr[idx] - rowptr[:-1])[row]

        col = col[perm]
        index = torch.stack([row, col], dim=0)

        if src.has_value():
            value = value[perm]

        sparse_size = torch.Size([rowcount.size(0), src.sparse_size(1)])

        storage = src.storage.__class__(index, value, sparse_size,
                                        rowcount=rowcount, rowptr=rowptr,
                                        is_sorted=True)

    elif dim == 1:
        old_colptr, row, value = src.csc()
        colcount = src.storage.colcount

        colcount = colcount[idx]
        tmp = torch.arange(colcount.size(0), device=row.device)
        col = tmp.repeat_interleave(colcount)

        # Creates an "arange interleave" tensor of row indices.
        colptr = torch.cat([col.new_zeros(1), colcount.cumsum(0)], dim=0)
        perm = torch.arange(col.size(0), device=col.device)
        perm += (old_colptr[idx] - colptr[:-1])[col]

        row = row[perm]
        csc2csr = (colcount.size(0) * row + col).argsort()
        index = torch.stack([row, col], dim=0)[:, csc2csr]

        if src.has_value():
            value = value[perm][csc2csr]

        sparse_size = torch.Size([src.sparse_size(0), colcount.size(0)])

        storage = src.storage.__class__(index, value, sparse_size,
                                        colcount=colcount, colptr=colptr,
                                        csc2csr=csc2csr, is_sorted=True)

    else:
        storage = src.storage.apply_value(
            lambda x: x.index_select(dim - 1, idx))

    return src.from_storage(storage)


def index_select_nnz(src, idx, layout=None):
    assert idx.dim() == 1

    if get_layout(layout) == 'csc':
        idx = idx[src.storage.csc2csr]

    index, value = src.coo()

    index = index[:, idx]
    if src.has_value():
        value = value[idx]

    # There is no other information we can maintain...
    storage = src.storage.__class__(index, value, src.sparse_size(),
                                    is_sorted=True)

    return src.from_storage(storage)
