import torch

from torch_sparse import diag_cpu

try:
    from torch_sparse import diag_cuda
except ImportError:
    diag_cuda = None


def remove_diag(src, k=0):
    row, col, value = src.coo()
    inv_mask = row != col if k == 0 else row != (col - k)
    row, col = row[inv_mask], col[inv_mask]

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

    storage = src.storage.__class__(row=row, col=col, value=value,
                                    sparse_size=src.sparse_size(),
                                    rowcount=rowcount, colcount=colcount,
                                    is_sorted=True)
    return src.__class__.from_storage(storage)


def set_diag(src, values=None, k=0):
    if values is not None and not src.has_value():
        raise ValueError('Sparse matrix has no values')

    src = src.remove_diag(k=0)

    row, col, value = src.coo()

    func = diag_cuda if row.is_cuda else diag_cpu
    mask = func.non_diag_mask(row, col, src.size(0), src.size(1), k)
    inv_mask = ~mask

    start, num_diag = -k if k < 0 else 0, mask.numel() - row.numel()
    diag = torch.arange(start, start + num_diag, device=src.device)

    new_row = row.new_empty(mask.size(0))
    new_row[mask] = row
    new_row[inv_mask] = diag

    new_col = col.new_empty(mask.size(0))
    new_col[mask] = row
    new_col[inv_mask] = diag.add_(k)

    new_value = None
    if src.has_value():
        new_value = torch.new_empty((mask.size(0), ) + value.size()[1:])
        new_value[mask] = value
        new_value[inv_mask] = values if values is not None else 1

    rowcount = None
    if src.storage.has_rowcount():
        rowcount = src.storage.rowcount.clone()
        rowcount[start:start + num_diag] += 1

    colcount = None
    if src.storage.has_colcount():
        colcount = src.storage.colcount.clone()
        colcount[start + k:start + num_diag + k] += 1

    storage = src.storage.__class__(row=new_row, col=new_col, value=new_value,
                                    sparse_size=src.sparse_size(),
                                    rowcount=rowcount, colcount=colcount,
                                    is_sorted=True)

    return src.__class__.from_storage(storage)
