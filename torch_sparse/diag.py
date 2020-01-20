import torch

from torch_sparse import diag_cpu

try:
    from torch_sparse import diag_cuda
except ImportError:
    diag_cuda = None


def remove_diag(src, k=0):
    index, value = src.coo()
    row, col = index

    inv_mask = row != col if k == 0 else row != (col - k)

    index = index[:, inv_mask]

    if src.has_value():
        value = value[inv_mask]

    if src.storage.has_rowcount() or src.storage.has_colcount():
        mask = ~inv_mask

    rowcount = None
    if src.storage.has_rowcount():
        rowcount = src.storage.rowcount.clone()
        rowcount[row[mask]] -= 1

    colcount = None
    if src.storage.has_colcount():
        colcount = src.storage.colcount.clone()
        colcount[col[mask]] -= 1

    storage = src.storage.__class__(index, value,
                                    sparse_size=src.sparse_size(),
                                    rowcount=rowcount, colcount=colcount,
                                    is_sorted=True)
    return src.__class__.from_storage(storage)


def set_diag(src, value=None, k=0):
    src = src.remove_diag(k=0)

    index, value = src.coo()

    func = diag_cuda if index.is_cuda else diag_cpu
    mask = func.non_diag_mask(index, src.size(0), src.size(1), k)
    inv_mask = ~mask

    new_index = index.new_empty((2, mask.size(0)))
    new_index[:, mask] = index

    num_diag = mask.numel() - index.size(1)
    start = -k if k < 0 else 0

    diag_row = torch.arange(start, start + num_diag, device=src.device)
    new_index[0, inv_mask] = diag_row
    diag_col = diag_row.add_(k)
    new_index[1, inv_mask] = diag_col

    new_value = None
    if src.has_value():
        new_value = torch.new_empty((mask.size(0), ) + mask.size()[1:])
        new_value[mask] = value
        new_value[inv_mask] = 1

    rowcount = None
    if src.storage.has_rowcount():
        rowcount = src.storage.rowcount.clone()
        rowcount[start:start + num_diag] += 1

    colcount = None
    if src.storage.has_colcount():
        colcount = src.storage.colcount.clone()
        colcount[start + k:start + num_diag + k] += 1

    storage = src.storage.__class__(new_index, new_value,
                                    sparse_size=src.sparse_size(),
                                    rowcount=rowcount, colcount=colcount,
                                    is_sorted=True)
    return src.__class__.from_storage(storage)
