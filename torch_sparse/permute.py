import torch
from torch_sparse.storage import SparseStorage
from torch_sparse.tensor import SparseTensor


def permute(src: SparseTensor, perm: torch.Tensor) -> SparseTensor:
    assert src.is_symmetric()

    row, col, value = src.coo()

    row = perm[row]
    col = perm[col]

    if value is not None:
        value = value[row]

    rowcount = src.storage._rowcount
    if rowcount is not None:
        rowcount = rowcount[perm]

    colcount = src.storage._colcount
    if colcount is not None:
        colcount = colcount[perm]

    storage = SparseStorage(row=row, rowptr=None, col=col, value=value,
                            sparse_sizes=src.sparse_sizes(), rowcount=rowcount,
                            colptr=None, colcount=colcount, csr2csc=None,
                            csc2csr=None, is_sorted=False)
    return src.from_storage(storage)


SparseTensor.permute = lambda self, perm: permute(self, perm)
