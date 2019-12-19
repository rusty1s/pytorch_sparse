import torch

from torch_sparse.storage import get_layout
import torch_sparse.arange_interleave_cpu as arange_interleave_cpu


def arange_interleave(start, repeat):
    assert start.device == repeat.device
    assert repeat.dtype == torch.long
    assert start.dim() == 1
    assert repeat.dim() == 1
    assert start.numel() == repeat.numel()
    if start.is_cuda:
        raise NotImplementedError
    return arange_interleave_cpu.arange_interleave(start, repeat)


def index_select(src, dim, idx):
    dim = src.dim() - dim if dim < 0 else dim

    assert idx.dim() == 1
    idx = idx.to(src.device)

    if dim == 0:
        (_, col), value = src.coo()
        rowcount = src.storage.rowcount
        rowptr = src.storage.rowptr

        rowcount = rowcount[idx]
        tmp = torch.arange(rowcount.size(0), device=rowcount.device)
        row = tmp.repeat_interleave(rowcount)
        perm = arange_interleave(rowptr[idx], rowcount)
        col = col[perm]
        index = torch.stack([row, col], dim=0)

        if src.has_value():
            value = value[perm]

        sparse_size = torch.Size([rowcount.size(0), src.sparse_size(1)])

        storage = src.storage.__class__(index, value, sparse_size,
                                        rowcount=rowcount, is_sorted=True)

    elif dim == 1:
        colptr, row, value = src.csc()
        colcount = src.storage.colcount

        colcount = colcount[idx]
        tmp = torch.arange(colcount.size(0), device=row.device)
        col = tmp.repeat_interleave(colcount)
        perm = arange_interleave(colptr[idx], colcount)
        row = row[perm]
        csc2csr = (colcount.size(0) * row + col).argsort()
        index = torch.stack([row, col], dim=0)[:, csc2csr]

        if src.has_value():
            value = value[perm][csc2csr]

        sparse_size = torch.Size([src.sparse_size(0), colcount.size(0)])

        storage = src.storage.__class__(index, value, sparse_size,
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

    index, value = src.coo()

    index = index[:, idx]
    if src.has_value():
        value = value[idx]

    # There is no other information we can maintain...
    storage = src.storage.__class__(index, value, src.sparse_size(),
                                    is_sorted=True)

    return src.from_storage(storage)
