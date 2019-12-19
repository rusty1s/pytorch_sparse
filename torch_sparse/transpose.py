import torch
from torch_sparse import to_scipy, from_scipy, coalesce


def transpose(index, value, m, n, coalesced=True):
    """Transposes dimensions 0 and 1 of a sparse tensor.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of corresponding dense matrix.
        n (int): The second dimension of corresponding dense matrix.
        coalesced (bool, optional): If set to :obj:`False`, will not coalesce
            the output. (default: :obj:`True`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    if value.dim() == 1 and not value.is_cuda:
        mat = to_scipy(index, value, m, n).tocsc()
        (col, row), value = from_scipy(mat)
        index = torch.stack([row, col], dim=0)
        return index, value

    row, col = index
    index = torch.stack([col, row], dim=0)
    if coalesced:
        index, value = coalesce(index, value, n, m)
    return index, value


def t(src):
    (row, col), value = src.coo()
    csr2csc = src.storage.csr2csc

    storage = src.storage.__class__(
        index=torch.stack([col, row], dim=0)[:, csr2csc],
        value=value[csr2csc] if src.has_value() else None,
        sparse_size=src.sparse_size()[::-1],
        rowcount=src.storage._colcount,
        rowptr=src.storage._colptr,
        colcount=src.storage._rowcount,
        colptr=src.storage._rowptr,
        csr2csc=src.storage._csc2csr,
        csc2csr=csr2csc,
        is_sorted=True,
    )

    return src.from_storage(storage)
