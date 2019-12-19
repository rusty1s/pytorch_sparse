import torch


def narrow(src, dim, start, length):
    if dim == 0:
        (row, col), value = src.coo()
        rowptr, _, _ = src.csr()

        rowptr = rowptr.narrow(0, start=start, length=length + 1)
        row_start = rowptr[0]
        rowptr = rowptr - row_start
        row_length = rowptr[-1]

        row = row.narrow(0, row_start, row_length) - start
        col = col.narrow(0, row_start, row_length)
        index = torch.stack([row, col], dim=0)
        if src.has_value():
            value = value.narrow(0, row_start, row_length)
        sparse_size = torch.Size([length, src.sparse_size(1)])

        storage = src._storage.__class__(
            index, value, sparse_size, rowptr, is_sorted=True)

    elif dim == 1:
        # This is faster than accessing `csc()` in analogy to the `dim=0` case.
        (row, col), value = src.coo()
        mask = (col >= start) & (col < start + length)

        index = torch.stack([row, col - start], dim=0)[:, mask]
        if src.has_value():
            value = value[mask]
        sparse_size = torch.Size([src.sparse_size(0), length])

        storage = src._storage.__class__(
            index, value, sparse_size, is_sorted=True)

    else:
        storage = src._storage.apply_value(lambda x: x.narrow(
            dim - 1, start, length))

    return src.__class__.from_storage(storage)
