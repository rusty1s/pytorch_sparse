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


def t(mat):
    ((row, col), value), perm = mat.coo(), mat._storage.csr_to_csc
    storage = mat._storage.__class__(
        index=torch.stack([col, row], dim=0)[:, perm],
        value=value[perm] if mat.has_value() else None,
        sparse_size=mat.sparse_size()[::-1],
        rowptr=mat._storage._colptr,
        colptr=mat._storage._rowptr,
        csr_to_csc=mat._storage._csc_to_csr,
        csc_to_csr=perm,
        is_sorted=True)
    return mat.__class__.from_storage(storage)
