def remove_diag(src, k=0):
    index, value = src.coo()
    row, col = index

    inv_mask = row != col if k == 0 else row != (col - k)

    index = index[:, inv_mask]

    if src.has_value():
        value = value[inv_mask]

    if src.storage.has_rowcount() or src.storage.has_colcount():
        mask = ~inv_mask

    rowcount = None
    if src.storage.has_rowcount():
        rowcount = src.storage.rowcount.clone()
        rowcount[row[mask]] -= 1

    # TODO: Maintain `rowptr`.

    colcount = None
    if src.storage.has_colcount():
        colcount = src.storage.colcount.clone()
        colcount[col[mask]] -= 1

    # TODO: Maintain `colptr`.

    storage = src.storage.__class__(index, value,
                                    sparse_size=src.sparse_size(),
                                    rowcount=rowcount, colcount=colcount,
                                    is_sorted=True)
    return src.__class__.from_storage(storage)


def set_diag(src, value=None, k=0):
    pass
