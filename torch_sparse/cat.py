import torch


def cat(tensors, dim):
    assert len(tensors) > 0
    has_value = tensors[0].has_value()
    has_rowcount = tensors[0].storage.has_rowcount()
    has_rowptr = tensors[0].storage.has_rowptr()
    has_colcount = tensors[0].storage.has_colcount()
    has_colptr = tensors[0].storage.has_colptr()
    has_csr2csc = tensors[0].storage.has_csr2csc()
    has_csc2csr = tensors[0].storage.has_csc2csr()

    rows, cols, values, sparse_size = [], [], [], [0, 0]
    rowcounts, rowptrs, colcounts, colptrs = [], [], [], []
    csr2cscs, csc2csrs, nnzs = [], [], 0

    if isinstance(dim, int):
        dim = tensors[0].dim() + dim if dim < 0 else dim
    else:
        dim = tuple([tensors[0].dim() + d if d < 0 else d for d in dim])

    if dim == 0:
        for tensor in tensors:
            (row, col), value = tensor.coo()
            rows += [row + sparse_size[0]]
            cols += [col]
            values += [value]
            sparse_size[0] += tensor.sparse_size(0)
            sparse_size[1] = max(sparse_size[1], tensor.sparse_size(1))

            if has_rowcount:
                rowcounts += [tensor.storage.rowcount]

            if has_rowptr:
                rowptr = tensor.storage.rowptr
                rowptr = rowptr if len(rowptrs) == 0 else rowptr[1:]
                rowptrs += [rowptr + nnzs]

            nnzs += tensor.nnz()

        storage = tensors[0].storage.__class__(
            torch.stack([torch.cat(rows), torch.cat(cols)], dim=0),
            value=torch.cat(values, dim=0) if has_value else None,
            sparse_size=sparse_size,
            rowcount=torch.cat(rowcounts) if has_rowcount else None,
            rowptr=torch.cat(rowptrs) if has_rowptr else None, is_sorted=True)

    elif dim == 1:
        for tensor in tensors:
            (row, col), value = tensor.coo()
            rows += [row]
            cols += [col + sparse_size[1]]
            values += [value]
            sparse_size[0] = max(sparse_size[0], tensor.sparse_size(0))
            sparse_size[1] += tensor.sparse_size(1)

            if has_colcount:
                colcounts += [tensor.storage.colcount]

            if has colptr:
                colptr = tensor.storage.colptr
                colptr = colptr if len(colptrs) == 0 else colptr[1:]
                colptrs += [colptr + nnzs]

            nnzs += tensor.nnz()

        storage = tensors[0].storage.__class__(
            torch.stack([torch.cat(rows), torch.cat(cols)], dim=0),
            value=torch.cat(values, dim=0) if has_value else None,
            sparse_size=sparse_size,
            colcount=torch.cat(colcounts) if has_colcount else None,
            colptr=torch.cat(colptrs) if has_colptr else None, is_sorted=False)

    elif dim == (0, 1) or (1, 0):
        for tensor in tensors:
            (row, col), value = tensor.coo()
            rows += [row + sparse_size[0]]
            cols += [col + sparse_size[1]]
            values += [value] if has_value else []
            sparse_size[0] += tensor.sparse_size(0)
            sparse_size[1] += tensor.sparse_size(1)

            if has_rowcount:
                rowcounts += [tensor.storage.rowcount]

            if has_rowptr:
                rowptr = tensor.storage.rowptr
                rowptr = rowptr if len(rowptrs) == 0 else rowptr[1:]
                rowptrs += [rowptr + nnzs]

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

            nnzs += tensor.nnz()

        storage = tensors[0].storage.__class__(
            torch.stack([torch.cat(rows), torch.cat(cols)], dim=0),
            value=torch.cat(values, dim=0) if has_value else None,
            sparse_size=sparse_size,
            rowcount=torch.cat(rowcounts) if has_rowcount else None,
            rowptr=torch.cat(rowptrs) if has_rowptr else None,
            colcount=torch.cat(colcounts) if has_colcount else None,
            colptr=torch.cat(colptrs) if has_colptr else None,
            csr2csc=torch.cat(csr2cscs) if has_csr2csc else None,
            csc2csr=torch.cat(csc2csrs) if has_csc2csr else None,
            is_sorted=True)

    elif isinstance(dim, int) and dim > 1 and dim < tensors[0].dim():
        for tensor in tensors:
            values += [tensor.storage.value]
            sparse_size[0] = max(sparse_size[0], tensor.sparse_size(0))
            sparse_size[1] = max(sparse_size[1], tensor.sparse_size(1))

        old_storage = tensors[0].storage
        storage = old_storage.storage.__class__(
            tensors[0].storage.index,
            value=torch.cat(values, dim=dim - 1),
            sparse_size=sparse_size,
            rowcount=old_storage._rowcount,
            rowptr=old_storage._rowcount,
            colcount=old_storage._rowcount,
            colptr=old_storage._rowcount,
            csr2csc=old_storage._csr2csc,
            csc2csr=old_storage._csc2csr,
            is_sorted=True)

    else:
        raise IndexError(
            (f'Dimension out of range: Expected to be in range of '
             f'[{-tensors[0].dim()}, {tensors[0].dim() - 1}, but got {dim}]'))

    return tensors[0].__class__.from_storage(storage)
