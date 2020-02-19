import torch

from torch_sparse.storage import SparseStorage
from torch_sparse.tensor import SparseTensor


def t(src: SparseTensor) -> SparseTensor:
    csr2csc = src.storage.csr2csc()

    row, col, value = src.coo()

    if value is not None:
        value = value[csr2csc]

    sparse_sizes = src.storage.sparse_sizes()

    storage = SparseStorage(
        row=col[csr2csc],
        rowptr=src.storage._colptr,
        col=row[csr2csc],
        value=value,
        sparse_sizes=(sparse_sizes[1], sparse_sizes[0]),
        rowcount=src.storage._colcount,
        colptr=src.storage._rowptr,
        colcount=src.storage._rowcount,
        csr2csc=src.storage._csc2csr,
        csc2csr=csr2csc,
        is_sorted=True,
    )

    return src.from_storage(storage)


SparseTensor.t = lambda self: t(self)

###############################################################################


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

    row, col = index
    row, col = col, row

    if coalesced:
        sparse_sizes = (n, m)
        storage = SparseStorage(row=row, col=col, value=value,
                                sparse_sizes=sparse_sizes, is_sorted=False)
        storage = storage.coalesce()
        row, col, value = storage.row(), storage.col(), storage.value()

    return torch.stack([row, col], dim=0), value
