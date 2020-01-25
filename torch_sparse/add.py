import torch
from torch_scatter import gather_csr


def sparse_add(matA, matB):
    nnzA, nnzB = matA.nnz(), matB.nnz()
    valA = torch.full((nnzA, ), 1, dtype=torch.uint8, device=matA.device)
    valB = torch.full((nnzB, ), 2, dtype=torch.uint8, device=matB.device)

    if matA.is_cuda:
        pass
    else:
        matA_ = matA.set_value(valA, layout='csr').to_scipy(layout='csr')
        matB_ = matB.set_value(valB, layout='csr').to_scipy(layout='csr')
        matC_ = matA_ + matB_
        rowptr = torch.from_numpy(matC_.indptr).to(torch.long)
        matC_ = matC_.tocoo()
        row = torch.from_numpy(matC_.row).to(torch.long)
        col = torch.from_numpy(matC_.col).to(torch.long)
        index = torch.stack([row, col], dim=0)
        valC_ = torch.from_numpy(matC_.data)

    value = None
    if matA.has_value() or matB.has_value():
        maskA, maskB = valC_ != 2, valC_ >= 2

        size = matA.size() if matA.dim() >= matB.dim() else matA.size()
        size = (valC_.size(0), ) + size[2:]

        value = torch.zeros(size, dtype=matA.dtype, device=matA.device)
        value[maskA] += matA.storage.value if matA.has_value() else 1
        value[maskB] += matB.storage.value if matB.has_value() else 1

    storage = matA.storage.__class__(index, value, matA.sparse_size(),
                                     rowptr=rowptr, is_sorted=True)

    return matA.__class__.from_storage(storage)


def add(src, other):
    if isinstance(other, int) or isinstance(other, float):
        return add_nnz(src, other)

    elif torch.is_tensor(other):
        rowptr, col, value = src.csr()
        if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise...
            other = gather_csr(other.squeeze(1), rowptr)
            value = other.add_(src.storage.value if src.has_value() else 1)
            return src.set_value(value, layout='csr')

        if other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise...
            other = other.squeeze(0)[col]
            value = other.add_(src.storage.value if src.has_value() else 1)
            return src.set_value(value, layout='coo')

        raise ValueError(f'Size mismatch: Expected size ({src.size(0)}, 1,'
                         f' ...) or (1, {src.size(1)}, ...), but got size '
                         f'{other.size()}.')

    elif isinstance(other, src.__class__):
        raise NotImplementedError

    raise ValueError('Argument `other` needs to be of type `int`, `float`, '
                     '`torch.tensor` or `torch_sparse.SparseTensor`.')


def add_(src, other):
    if isinstance(other, int) or isinstance(other, float):
        return add_nnz_(src, other)

    elif torch.is_tensor(other):
        rowptr, col, value = src.csr()
        if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise...
            other = gather_csr(other.squeeze(1), rowptr)
            if src.has_value():
                value = src.storage.value.add_(other)
            else:
                value = other.add_(1)
            return src.set_value_(value, layout='csr')

        if other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise...
            other = other.squeeze(0)[col]
            if src.has_value():
                value = src.storage.value.add_(other)
            else:
                value = other.add_(1)
            return src.set_value_(value, layout='coo')

        raise ValueError(f'Size mismatch: Expected size ({src.size(0)}, 1,'
                         f' ...) or (1, {src.size(1)}, ...), but got size '
                         f'{other.size()}.')

    elif isinstance(other, src.__class__):
        raise NotImplementedError

    raise ValueError('Argument `other` needs to be of type `int`, `float`, '
                     '`torch.tensor` or `torch_sparse.SparseTensor`.')


def add_nnz(src, other, layout=None):
    if isinstance(other, int) or isinstance(other, float):
        if src.has_value():
            value = src.storage.value + other
        else:
            value = torch.full((src.nnz(), ), 1 + other, device=src.device)
        return src.set_value(value, layout='coo')

    if torch.is_tensor(other):
        if src.has_value():
            value = src.storage.value + other
        else:
            value = other + 1
        return src.set_value(value, layout='coo')

    raise ValueError('Argument `other` needs to be of type `int`, `float` or '
                     '`torch.tensor`.')


def add_nnz_(src, other, layout=None):
    if isinstance(other, int) or isinstance(other, float):
        if src.has_value():
            value = src.storage.value.add_(other)
        else:
            value = torch.full((src.nnz(), ), 1 + other, device=src.device)
        return src.set_value_(value, layout='coo')

    if torch.is_tensor(other):
        if src.has_value():
            value = src.storage.value.add_(other)
        else:
            value = other + 1  # No inplace operation possible.
        return src.set_value_(value, layout='coo')

    raise ValueError('Argument `other` needs to be of type `int`, `float` or '
                     '`torch.tensor`.')
