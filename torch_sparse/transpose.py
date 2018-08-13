import torch
from torch_sparse import coalesce


def transpose(index, value, m, n):
    """Transposes dimensions 0 and 1 of a sparse matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        n (int): The second dimension of sparse matrix.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    row, col = index
    index = torch.stack([col, row], dim=0)

    index, value = coalesce(index, value, n, m)

    return index, value
