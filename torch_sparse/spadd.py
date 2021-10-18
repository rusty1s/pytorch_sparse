import torch
from torch_sparse import coalesce
from torch_sparse.tensor import SparseTensor
from torch_sparse.storage import SparseStorage


def spadd(index_src, val_src, index_other, val_other, 
          m=None, n=None): 
    """Matrix addition of two sparse matrices.
    Args:
        index_src   (:class:`LongTensor`): The index tensor of source sparse matrix.
        val_src     (:class:`Tensor`):     The value tensor of source sparse matrix.
        index_other (:class:`LongTensor`): The index tensor of the other sparse matrix.
        val_other   (:class:`Tensor`):     The value tensor of the other sparse matrix.
        m (int): The first dimension of sparse matrices.
        n (int): The second dimension of sparse matrces.
    """
    index = torch.cat((index_src, index_other), dim=-1)
    value = torch.cat((val_src, val_other), dim=-1)
    return coalesce(index=index, value=value, m=m, n=n)


def spadd(src: SparseTensor, other: SparseTensor) -> (torch.LongTensor, torch.Tensor):
    storage = SparseStorage(row=torch.cat([src.storage._row, other.storage._row], dim=-1), 
                            col=torch.cat([src.storage._col, other.storage._col], dim=-1), 
                            value=torch.cat([src.storage._value, other.storage._value], dim=-1),
                            sparse_sizes=(m, n), 
                            is_sorted=False)
    storage = storage.coalesce(reduce="add")
    return torch.stack([storage.row(), storage.col()], dim=0), storage.value()
    
