from typing import Tuple

from torch_sparse.storage import SparseStorage
from torch_sparse.tensor import SparseTensor


def narrow(src: SparseTensor, dim: int, start: int,
           length: int) -> SparseTensor:
    if dim < 0:
        dim = src.dim() + dim

    if start < 0:
        start = src.size(dim) + start

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

        sparse_sizes = (length, src.sparse_size(1))

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

        sparse_sizes = (src.sparse_size(0), length)

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


def __narrow_diag__(src: SparseTensor, start: Tuple[int, int],
                    length: Tuple[int, int]) -> SparseTensor:
    # This function builds the inverse operation of `cat_diag` and should hence
    # only be used on *diagonally stacked* sparse matrices.
    # That's the reason why this method is marked as *private*.

    rowptr, col, value = src.csr()

    rowptr = rowptr.narrow(0, start=start[0], length=length[0] + 1)
    row_start = int(rowptr[0])
    rowptr = rowptr - row_start
    row_length = int(rowptr[-1])

    row = src.storage._row
    if row is not None:
        row = row.narrow(0, row_start, row_length) - start[0]

    col = col.narrow(0, row_start, row_length) - start[1]

    if value is not None:
        value = value.narrow(0, row_start, row_length)

    sparse_sizes = length

    rowcount = src.storage._rowcount
    if rowcount is not None:
        rowcount = rowcount.narrow(0, start[0], length[0])

    colptr = src.storage._colptr
    if colptr is not None:
        colptr = colptr.narrow(0, start[1], length[1] + 1)
        colptr = colptr - int(colptr[0])  # i.e. `row_start`

    colcount = src.storage._colcount
    if colcount is not None:
        colcount = colcount.narrow(0, start[1], length[1])

    csr2csc = src.storage._csr2csc
    if csr2csc is not None:
        csr2csc = csr2csc.narrow(0, row_start, row_length) - row_start

    csc2csr = src.storage._csc2csr
    if csc2csr is not None:
        csc2csr = csc2csr.narrow(0, row_start, row_length) - row_start

    storage = SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                            sparse_sizes=sparse_sizes, rowcount=rowcount,
                            colptr=colptr, colcount=colcount, csr2csc=csr2csc,
                            csc2csr=csc2csr, is_sorted=True)
    return src.from_storage(storage)


SparseTensor.narrow = lambda self, dim, start, length: narrow(
    self, dim, start, length)
SparseTensor.__narrow_diag__ = lambda self, start, length: __narrow_diag__(
    self, start, length)
