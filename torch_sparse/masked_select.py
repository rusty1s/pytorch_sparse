import torch

from torch_sparse.storage import get_layout


def masked_select(src, dim, mask):
    dim = src.dim() + dim if dim < 0 else dim

    assert mask.dim() == 1
    storage = src.storage

    if dim == 0:
        row, col, value = src.coo()
        rowcount = src.storage.rowcount

        rowcount = rowcount[mask]

        mask = mask[row]
        row = torch.arange(rowcount.size(0),
                           device=row.device).repeat_interleave(rowcount)
        col = col[mask]

        if src.has_value():
            value = value[mask]

        sparse_size = torch.Size([rowcount.size(0), src.sparse_size(1)])

        storage = src.storage.__class__(row=row, col=col, value=value,
                                        sparse_size=sparse_size,
                                        rowcount=rowcount, is_sorted=True)

    elif dim == 1:
        row, col, value = src.coo()
        csr2csc = src.storage.csr2csc
        row, col = row[csr2csc], col[csr2csc]
        colcount = src.storage.colcount

        colcount = colcount[mask]

        mask = mask[col]
        col = torch.arange(colcount.size(0),
                           device=col.device).repeat_interleave(colcount)
        row = row[mask]
        csc2csr = (colcount.size(0) * row + col).argsort()
        row, col = row[csc2csr], col[csc2csr]

        if src.has_value():
            value = value[csr2csc][mask][csc2csr]

        sparse_size = torch.Size([src.sparse_size(0), colcount.size(0)])

        storage = src.storage.__class__(row=row, col=col, value=value,
                                        sparse_size=sparse_size,
                                        colcount=colcount, csc2csr=csc2csr,
                                        is_sorted=True)

    else:
        idx = mask.nonzero().view(-1)
        storage = src.storage.apply_value(
            lambda x: x.index_select(dim - 1, idx))

    return src.from_storage(storage)


def masked_select_nnz(src, mask, layout=None):
    assert mask.dim() == 1

    if get_layout(layout) == 'csc':
        mask = mask[src.storage.csc2csr]

    row, col, value = src.coo()
    row, col = row[mask], col[mask]

    if src.has_value():
        value = value[mask]

    # There is no other information we can maintain...
    storage = src.storage.__class__(row=row, col=col, value=value,
                                    sparse_size=src.sparse_size(),
                                    is_sorted=True)

    return src.from_storage(storage)
