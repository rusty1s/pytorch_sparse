import torch
from torch_sparse import to_scipy, from_scipy, coalesce


def transpose(index, value, m, n):
    """Transposes dimensions 0 and 1 of a sparse tensor.

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


def transpose_matrix(index, value, m, n):
    """Transposes dimensions 0 and 1 of a sparse matrix, where :args:`value` is
    one-dimensional.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        n (int): The second dimension of sparse matrix.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    assert value.dim() == 1

    if index.is_cuda:
        return transpose(index, value, m, n)
    else:
        mat = to_scipy(index, value, m, n).tocsc()
        (col, row), value = from_scipy(mat)
        index = torch.stack([row, col], dim=0)
        return index, value
