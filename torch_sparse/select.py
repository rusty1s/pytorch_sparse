def select(src, dim, idx):
    return src.narrow(dim, start=idx, length=1)
