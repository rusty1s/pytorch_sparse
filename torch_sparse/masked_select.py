import torch

from torch_sparse.storage import get_layout


def masked_select(src, dim, mask):
    dim = src.dim() + dim if dim < 0 else dim

    assert mask.dim() == 1
    storage = src.storage

    if dim == 0:
        (row, col), value = src.coo()
        rowcount = src.storage.rowcount

        row_mask = mask[row]
        rowcount = rowcount[mask]
        idx = torch.arange(rowcount.size(0), device=rowcount.device)
        row = idx.repeat_interleave(rowcount)
        col = col[row_mask]
        index = torch.stack([row, col], dim=0)

        if src.has_value():
            value = value[row_mask]

        sparse_size = torch.Size([rowcount.size(0), src.sparse_size(1)])

        storage = src.storage.__class__(
            index, value, sparse_size, rowcount=rowcount, is_sorted=True)

    elif dim == 1:
        csr2csc = src.storage.csr2csc
        row = src.storage.row[csr2csc]
        col = src.storage.col[csr2csc]
        colcount = src.storage.colcount

        col_mask = mask[col]
        colcount = colcount[mask]
        tmp = torch.arange(colcount.size(0), device=row.device)
        col = tmp.repeat_interleave(colcount)
        row = row[col_mask]
        csc2csr = (colcount.size(0) * row + col).argsort()
        index = torch.stack([row, col], dim=0)[:, csc2csr]

        value = src.storage.value
        if src.has_value():
            value = value[csr2csc][col_mask][csc2csr]

        sparse_size = torch.Size([src.sparse_size(0), colcount.size(0)])

        storage = src.storage.__class__(
            index,
            value,
            sparse_size,
            colcount=colcount,
            csc2csr=csc2csr,
            is_sorted=True)

    else:
        idx = mask.nonzero().view(-1)
        storage = src.storage.apply_value(lambda x: x.index_select(
            dim - 1, idx))

    return src.from_storage(storage)


def masked_select_nnz(src, mask, layout=None):
    assert mask.dim() == 1

    if get_layout(layout) == 'csc':
        mask = mask[src.storage.csc2csr]

    index, value = src.coo()

    index = index[:, mask]
    if src.has_value():
        value = value[mask]

    # There is no other information we can maintain...
    storage = src.storage.__class__(
        index, value, src.sparse_size(), is_sorted=True)

    return src.from_storage(storage)
