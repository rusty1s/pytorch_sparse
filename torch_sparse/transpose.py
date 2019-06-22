import torch
from torch_sparse import to_scipy, from_scipy, coalesce


def transpose(index, value, m, nm coalesce=True):
    """Transposes dimensions 0 and 1 of a sparse tensor.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of corresponding dense matrix.
        n (int): The second dimension of corresponding dense matrix.
        coalesce (bool, optional): To return coalesced index and value or not (default: :obj:`True`) 
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    if value.dim() == 1 and not value.is_cuda:
        mat = to_scipy(index, value, m, n).tocsc()
        (col, row), value = from_scipy(mat)
        index = torch.stack([row, col], dim=0)
        return index, value

    row, col = index
    index = torch.stack([col, row], dim=0)
    if coalesce:
        index, value = coalesce(index, value, n, m)
    return index, value
