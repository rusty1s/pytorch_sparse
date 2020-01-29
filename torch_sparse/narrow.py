import torch
from torch_sparse.storage import SparseStorage
from torch_sparse.tensor import SparseTensor


@torch.jit.script
def narrow(src: SparseTensor, dim: int, start: int,
           length: int) -> SparseTensor:
    dim = src.dim() + dim if dim < 0 else dim
    start = src.size(dim) + start if start < 0 else start

    if dim == 0:
        rowptr, col, value = src.csr()

        rowptr = rowptr.narrow(0, start=start, length=length + 1)
        row_start = rowptr[0]
        rowptr = rowptr - row_start
        row_length = rowptr[-1]

        row = src.storage._row
        if row is not None:
            row = row.narrow(0, row_start, row_length) - start

        col = col.narrow(0, row_start, row_length)

        if value is not None:
            value = value.narrow(0, row_start, row_length)

        sparse_sizes = torch.Size([length, src.sparse_size(1)])

        rowcount = src.storage._rowcount
        if rowcount is not None:
            rowcount = rowcount.narrow(0, start=start, length=length)

        storage = SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                                sparse_sizes=sparse_sizes, rowcount=rowcount,
                                colptr=None, colcount=None, csr2csc=None,
                                csc2csr=None, is_sorted=True)
        return src.from_storage(storage)

    elif dim == 1:
        # This is faster than accessing `csc()` contrary to the `dim=0` case.
        row, col, value = src.coo()
        mask = (col >= start) & (col < start + length)

        row = row[mask]
        col = col[mask] - start

        if value is not None:
            value = value[mask]

        sparse_sizes = torch.Size([src.sparse_size(0), length])

        colptr = src.storage._colptr
        if colptr is not None:
            colptr = colptr.narrow(0, start=start, length=length + 1)
            colptr = colptr - colptr[0]

        colcount = src.storage._colcount
        if colcount is not None:
            colcount = colcount.narrow(0, start=start, length=length)

        storage = SparseStorage(row=row, rowptr=None, col=col, value=value,
                                sparse_sizes=sparse_sizes, rowcount=None,
                                colptr=colptr, colcount=colcount, csr2csc=None,
                                csc2csr=None, is_sorted=True)
        return src.from_storage(storage)

    else:
        value = src.storage.value()
        if value is not None:
            return src.set_value(value.narrow(dim - 1, start, length),
                                 layout='coo')
        else:
            raise ValueError


SparseTensor.narrow = lambda self, dim, start, length: narrow(
    self, dim, start, length)
