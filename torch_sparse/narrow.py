from typing import Tuple

import torch
from torch_sparse.storage import SparseStorage
from torch_sparse.tensor import SparseTensor


def narrow(src: SparseTensor, dim: int, start: int,
           length: int, bidim: False, 
           n_active_nodes: None) -> SparseTensor:
    r"""It resizes :obj:`src` SparseTensor along one 
    specified dimension, or two dimensions at the same time
    if the SparseTensor has 2 dimensions. In the 
    latter case, it returns a squared SparseTensor.
    In the case of 1-dimensional resizing,
    :obj:'start' and :obj:'lenght' are supported
    In the case of bidimensional resizing, resizing
    always starts from 0, and it returns a square matrix;
    :obj:`src` must be a bidimensional SparseTensor
    for bidim option to work.
    Args:
        src (SparseTensor): SparseTensor to be manipulated
        dim (int): dimension along which the resizing is done.
                    Not used in case bidim is True
        start (int): first position to include in the ouput.
                    Not used in case bidim is True (set to 0)
        length (int): last position to include in the output.
        bidim (bool):  it applies bidimensional and squared narrowing
                    of the src, which must be a bidimensional SparseTensor.
        n_active_nodes (int): Required only if bidim option is True;
                    used for efficient manipulation of the rowptr in bidim case.
    """
    if bidim: 
        if len(src.storage._sparse_sizes) != 2:
            raise NotImplementedError 
        
        start = 0 # only start at 0 is supported for bidim resizing
        if src.storage._rowptr is None:
            rowptr, col, value = src.csr()

        # rowptr
        rowptr = torch.narrow(src.storage._rowptr, 
                              0, start, length + 1).clone()
        rowptr[(n_active_nodes + 1):] = rowptr[n_active_nodes]

        # col and value
        col = torch.narrow(src.storage._col, 
                           0, start, rowptr[-1])
        value = torch.narrow(src.storage._value,
                    0, start, rowptr[-1]
                        ) if src.storage._value is not None else None

        # indeces for conversion to csc
        csr2csc = src.storage._csr2csc[src.storage._csr2csc < len(col)] \
            if src.storage._csr2csc is not None else None
        
        # update storage and edge_index
        storage = SparseStorage(row=None, rowptr=rowptr, col=col,
                                value=value, sparse_sizes=(length, length),
                                rowcount=None, colptr=None,
                                colcount=None, csr2csc=csr2csc,
                                csc2csr=None, is_sorted=True,
                                trust_data=False)
        return src.from_storage(storage)
      
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


SparseTensor.narrow = lambda self, dim, start, length, \
                        bidim=False, n_active_nodes=None: narrow(
                        self, dim, start, length, 
                        bidim, n_active_nodes)
SparseTensor.__narrow_diag__ = lambda self, start, length: __narrow_diag__(
    self, start, length)
