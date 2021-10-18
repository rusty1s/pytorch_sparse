import torch
from torch_sparse import coalesce


def spadd(indexA, valueA, indexB, valueB, m, n):
    """Matrix addition of two sparse matrices.

    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of the sparse matrices.
        n (int): The second dimension of the sparse matrices.
    """
    index = torch.cat([indexA, indexB], dim=-1)
    value = torch.cat([valueA, valueB], dim=0)
    return coalesce(index=index, value=value, m=m, n=n, op='add')
