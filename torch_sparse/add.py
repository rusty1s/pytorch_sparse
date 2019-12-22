import torch


def union(mat1, mat2):
    offset = mat1.nnz() + 1
    value1 = torch.ones(mat1.nnz(), dtype=torch.long, device=mat2.device)
    value2 = value1.new_full((mat2.nnz(), ), offset)
    size = max(mat1.size(0), mat2.size(0)), max(mat1.size(1), mat2.size(1))

    if not mat1.is_cuda:
        mat1 = mat1.set_value(value1, layout='coo').to_scipy(layout='csr')
        mat1.resize(*size)

        mat2 = mat2.set_value(value2, layout='coo').to_scipy(layout='csr')
        mat2.resize(*size)

        out = mat1 + mat2
        rowptr = torch.from_numpy(out.indptr).to(torch.long)
        out = out.tocoo()
        row = torch.from_numpy(out.row).to(torch.long)
        col = torch.from_numpy(out.col).to(torch.long)
        value = torch.from_numpy(out.data)
    else:
        raise NotImplementedError

    mask1 = value % offset > 0
    mask2 = value >= offset

    return rowptr, torch.stack([row, col], dim=0), mask1, mask2


def add(src, other):
    if isinstance(other, int) or isinstance(other, float):
        return add_nnz(src, other)

    elif torch.is_tensor(other):
        (row, col), value = src.coo()
        if other.size(0) == src.size(0) and other.size(1) == 1:
            val = other.squeeze(1).repeat_interleave(
                row, 0) + (value if src.has_value() else 1)
        if other.size(0) == 1 and other.size(1) == src.size(1):
            val = other.squeeze(0)[col] + (value if src.has_value() else 1)
        else:
            raise ValueError(f'Size mismatch: Expected size ({src.size(0)}, 1,'
                             f' ...) or (1, {src.size(1)}, ...), but got size '
                             f'{other.size()}.')
        return src.set_value(val, layout='coo')

    elif isinstance(other, src.__class__):
        rowptr, index, src_offset, other_offset = union(src, other)
        raise NotImplementedError

    raise ValueError('Argument `other` needs to be of type `int`, `float`, '
                     '`torch.tensor` or `torch_sparse.SparseTensor`.')


def add_nnz(src, other, layout=None):
    if isinstance(other, int) or isinstance(other, float):
        return src.set_value(src.storage.value + other if src.has_value(
        ) else torch.full((src.nnz(), ), 1 + other, device=src.device))
    elif torch.is_tensor(other):
        return src.set_value(src.storage.value +
                             other if src.has_value() else other + 1)

    raise ValueError('Argument `other` needs to be of type `int`, `float` or '
                     '`torch.tensor`.')
