import torch
from torch_sparse.storage import SparseStorage


def coalesce(index, value, m, n, op="add"):
    """Row-wise sorts :obj:`value` and removes duplicate entries. Duplicate
    entries are removed by scattering them together. For scattering, any
    operation of `"torch_scatter"<https://github.com/rusty1s/pytorch_scatter>`_
    can be used.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        n (int): The second dimension of sparse matrix.
        op (string, optional): The scatter operation to use. (default:
            :obj:`"add"`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    storage = SparseStorage(row=index[0], col=index[1], value=value,
                            sparse_sizes=(m, n), is_sorted=False)
    storage = storage.coalesce(reduce=op)
    return torch.stack([storage.row(), storage.col()], dim=0), storage.value()
