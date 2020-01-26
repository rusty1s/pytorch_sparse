import torch


def cat(tensors, dim):
    assert len(tensors) > 0
    has_row = tensors[0].storage.has_row()
    has_value = tensors[0].has_value()
    has_rowcount = tensors[0].storage.has_rowcount()
    has_colptr = tensors[0].storage.has_colptr()
    has_colcount = tensors[0].storage.has_colcount()
    has_csr2csc = tensors[0].storage.has_csr2csc()
    has_csc2csr = tensors[0].storage.has_csc2csr()

    rows, rowptrs, cols, values, sparse_size, nnzs = [], [], [], [], [0, 0], 0
    rowcounts, colcounts, colptrs, csr2cscs, csc2csrs = [], [], [], [], []

    if isinstance(dim, int):
        dim = tensors[0].dim() + dim if dim < 0 else dim
    else:
        dim = tuple([tensors[0].dim() + d if d < 0 else d for d in dim])

    if dim == 0:
        for tensor in tensors:
            rowptr, col, value = tensor.csr()
            rowptr = rowptr if len(rowptrs) == 0 else rowptr[1:]
            rowptrs += [rowptr + nnzs]
            cols += [col]
            values += [value]

            if has_row:
                rows += [tensor.storage.row + sparse_size[0]]

            if has_rowcount:
                rowcounts += [tensor.storage.rowcount]

            sparse_size[0] += tensor.sparse_size(0)
            sparse_size[1] = max(sparse_size[1], tensor.sparse_size(1))
            nnzs += tensor.nnz()

        storage = tensors[0].storage.__class__(
            row=torch.cat(rows) if has_row else None,
            rowptr=torch.cat(rowptrs), col=torch.cat(cols),
            value=torch.cat(values, dim=0) if has_value else None,
            sparse_size=sparse_size,
            rowcount=torch.cat(rowcounts) if has_rowcount else None,
            is_sorted=True)

    elif dim == 1:
        for tensor in tensors:
            row, col, value = tensor.coo()
            rows += [row]
            cols += [col + sparse_size[1]]
            values += [value]

            if has_colcount:
                colcounts += [tensor.storage.colcount]

            if has_colptr:
                colptr = tensor.storage.colptr
                colptr = colptr if len(colptrs) == 0 else colptr[1:]
                colptrs += [colptr + nnzs]

            sparse_size[0] = max(sparse_size[0], tensor.sparse_size(0))
            sparse_size[1] += tensor.sparse_size(1)
            nnzs += tensor.nnz()

        storage = tensors[0].storage.__class__(
            row=torch.cat(rows),
            col=torch.cat(cols),
            value=torch.cat(values, dim=0) if has_value else None,
            sparse_size=sparse_size,
            colcount=torch.cat(colcounts) if has_colcount else None,
            colptr=torch.cat(colptrs) if has_colptr else None,
            is_sorted=False,
        )

    elif dim == (0, 1) or dim == (1, 0):
        for tensor in tensors:
            rowptr, col, value = tensor.csr()
            rowptr = rowptr if len(rowptrs) == 0 else rowptr[1:]
            rowptrs += [rowptr + nnzs]
            cols += [col + sparse_size[1]]
            values += [value]

            if has_row:
                rows += [tensor.storage.row + sparse_size[0]]

            if has_rowcount:
                rowcounts += [tensor.storage.rowcount]

            if has_colcount:
                colcounts += [tensor.storage.colcount]

            if has_colptr:
                colptr = tensor.storage.colptr
                colptr = colptr if len(colptrs) == 0 else colptr[1:]
                colptrs += [colptr + nnzs]

            if has_csr2csc:
                csr2cscs += [tensor.storage.csr2csc + nnzs]

            if has_csc2csr:
                csc2csrs += [tensor.storage.csc2csr + nnzs]

            sparse_size[0] += tensor.sparse_size(0)
            sparse_size[1] += tensor.sparse_size(1)
            nnzs += tensor.nnz()

        storage = tensors[0].storage.__class__(
            row=torch.cat(rows) if has_row else None,
            rowptr=torch.cat(rowptrs),
            col=torch.cat(cols),
            value=torch.cat(values, dim=0) if has_value else None,
            sparse_size=sparse_size,
            rowcount=torch.cat(rowcounts) if has_rowcount else None,
            colptr=torch.cat(colptrs) if has_colptr else None,
            colcount=torch.cat(colcounts) if has_colcount else None,
            csr2csc=torch.cat(csr2cscs) if has_csr2csc else None,
            csc2csr=torch.cat(csc2csrs) if has_csc2csr else None,
            is_sorted=True,
        )

    elif isinstance(dim, int) and dim > 1 and dim < tensors[0].dim():
        for tensor in tensors:
            values += [tensor.storage.value]

        old_storage = tensors[0].storage
        storage = old_storage.__class__(
            row=old_storage._row,
            rowptr=old_storage._rowptr,
            col=old_storage._col,
            value=torch.cat(values, dim=dim - 1),
            sparse_size=old_storage.sparse_size,
            rowcount=old_storage._rowcount,
            colptr=old_storage._colptr,
            colcount=old_storage._colcount,
            csr2csc=old_storage._csr2csc,
            csc2csr=old_storage._csc2csr,
            is_sorted=True,
        )

    else:
        raise IndexError(
            (f'Dimension out of range: Expected to be in range of '
             f'[{-tensors[0].dim()}, {tensors[0].dim() - 1}, but got {dim}]'))

    return tensors[0].__class__.from_storage(storage)
