import torch


def cat(tensors, dim):
    assert len(tensors) > 0
    has_value = tensors[0].has_value()
    has_rowcount = tensors[0].storage._rowcount is not None
    has_rowptr = tensors[0].storage._rowptr is not None
    has_colcount = tensors[0].storage._colcount is not None
    has_colptr = tensors[0].storage._colptr is not None
    has_csr2csc = tensors[0].storage._csr2csc is not None
    has_csc2csr = tensors[0].storage._csc2csr is not None

    rows, cols, values, sparse_size = [], [], [], [0, 0]
    rowcounts, rowptrs, colcounts, colptrs = [], [], [], []
    csr2cscs, csc2csrs, nnzs = [], [], 0

    if dim == 0:
        for tensor in tensors:
            (row, col), value = tensor.coo()
            rows += [row + sparse_size[0]]
            cols += [col]
            values += [value] if has_value else []
            sparse_size[0] += tensor.sparse_size(0)
            sparse_size[1] = max(sparse_size[1], tensor.sparse_size(1))

            rowcounts += [tensor.storage.rowcount] if has_rowcount else []

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
            rowptr=torch.cat(rowptrs) if has_rowptr else None,
            is_sorted=True)

    if dim == 1:
        raise NotImplementedError

    if dim == (0, 1) or (1, 0):
        for tensor in tensors:
            (row, col), value = tensor.coo()
            rows += [row + sparse_size[0]]
            cols += [col + sparse_size[1]]
            values += [value] if has_value else []
            sparse_size[0] += tensor.sparse_size(0)
            sparse_size[1] += tensor.sparse_size(1)

            rowcounts += [tensor.storage.rowcount] if has_rowcount else []
            colcounts += [tensor.storage.colcount] if has_colcount else []

            if has_rowptr:
                rowptr = tensor.storage.rowptr
                rowptr = rowptr if len(rowptrs) == 0 else rowptr[1:]
                rowptrs += [rowptr + nnzs]

            if has_colptr:
                colptr = tensor.storage.colptr
                colptr = colptr if len(colptrs) == 0 else colptr[1:]
                colptrs += [colptr + nnzs]

            csr2cscs += [tensor.storage.csr2csc + nnzs] if has_csr2csc else []
            csc2csrs += [tensor.storage.csc2csr + nnzs] if has_csc2csr else []

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

    else:
        raise NotImplementedError

    return tensors[0].__class__.from_storage(storage)
