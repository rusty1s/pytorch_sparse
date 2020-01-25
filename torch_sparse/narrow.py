import torch


def narrow(src, dim, start, length):
    dim = src.dim() + dim if dim < 0 else dim
    start = src.size(dim) + start if start < 0 else start

    if dim == 0:
        rowptr, col, value = src.csr()
        # rowptr = src.storage.rowptr

        # Maintain `rowcount`...
        rowcount = src.storage._rowcount
        if rowcount is not None:
            rowcount = rowcount.narrow(0, start=start, length=length)

        rowptr = rowptr.narrow(0, start=start, length=length + 1)
        row_start = rowptr[0]
        rowptr = rowptr - row_start
        row_length = rowptr[-1]

        row = src.storage._row
        if row is not None:
            row = row.narrow(0, row_start, row_length) - start
        col = col.narrow(0, row_start, row_length)

        if src.has_value():
            value = value.narrow(0, row_start, row_length)

        sparse_size = torch.Size([length, src.sparse_size(1)])

        storage = src.storage.__class__(row=row, rowptr=rowptr, col=col,
                                        value=value, sparse_size=sparse_size,
                                        rowcount=rowcount, is_sorted=True)

    elif dim == 1:
        # This is faster than accessing `csc()` contrary to the `dim=0` case.
        row, col, value = src.coo()
        mask = (col >= start) & (col < start + length)

        row, col = row[mask], col[mask] - start

        # Maintain `colcount`...
        colcount = src.storage._colcount
        if colcount is not None:
            colcount = colcount.narrow(0, start=start, length=length)

        # Maintain `colptr`...
        colptr = src.storage._colptr
        if colptr is not None:
            colptr = colptr.narrow(0, start=start, length=length + 1)
            colptr = colptr - colptr[0]

        if src.has_value():
            value = value[mask]

        sparse_size = torch.Size([src.sparse_size(0), length])

        storage = src.storage.__class__(row=row, col=col, value=value,
                                        sparse_size=sparse_size, colptr=colptr,
                                        colcount=colcount, is_sorted=True)

    else:
        storage = src.storage.apply_value(
            lambda x: x.narrow(dim - 1, start, length))

    return src.from_storage(storage)
