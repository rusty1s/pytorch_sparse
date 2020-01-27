import torch
from torch_scatter import gather_csr
from torch_sparse.utils import is_scalar


def mul(src, other):
    if is_scalar(other):
        return mul_nnz(src, other)

    elif torch.is_tensor(other):
        rowptr, col, value = src.csr()
        if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise...
            other = gather_csr(other.squeeze(1), rowptr)
            if src.has_value():
                value = other.mul_(src.storage.value)
            else:
                value = other
            return src.set_value(value, layout='csr')

        if other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise...
            other = other.squeeze(0)[col]
            if src.has_value():
                value = other.mul_(src.storage.value)
            else:
                value = other
            return src.set_value(value, layout='coo')

        raise ValueError(f'Size mismatch: Expected size ({src.size(0)}, 1,'
                         f' ...) or (1, {src.size(1)}, ...), but got size '
                         f'{other.size()}.')

    elif isinstance(other, src.__class__):
        raise NotImplementedError

    raise ValueError('Argument `other` needs to be of type `int`, `float`, '
                     '`torch.tensor` or `torch_sparse.SparseTensor`.')


def mul_(src, other):
    if is_scalar(other):
        return mul_nnz_(src, other)

    elif torch.is_tensor(other):
        rowptr, col, value = src.csr()
        if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise...
            other = gather_csr(other.squeeze(1), rowptr)
            if src.has_value():
                value = src.storage.value.mul_(other)
            else:
                value = other
            return src.set_value_(value, layout='csr')

        if other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise...
            other = other.squeeze(0)[col]
            if src.has_value():
                value = src.storage.value.mul_(other)
            else:
                value = other
            return src.set_value_(value, layout='coo')

        raise ValueError(f'Size mismatch: Expected size ({src.size(0)}, 1,'
                         f' ...) or (1, {src.size(1)}, ...), but got size '
                         f'{other.size()}.')

    elif isinstance(other, src.__class__):
        raise NotImplementedError

    raise ValueError('Argument `other` needs to be of type `int`, `float`, '
                     '`torch.tensor` or `torch_sparse.SparseTensor`.')


def mul_nnz(src, other, layout=None):
    if torch.is_tensor(other) or is_scalar(other):
        if src.has_value():
            value = src.storage.value * other
        else:
            value = other
        return src.set_value(value, layout='coo')

    raise ValueError('Argument `other` needs to be of type `int`, `float` or '
                     '`torch.tensor`.')


def mul_nnz_(src, other, layout=None):
    if torch.is_tensor(other) or is_scalar(other):
        if src.has_value():
            value = src.storage.value.mul_(other)
        else:
            value = other
        return src.set_value_(value, layout='coo')

    raise ValueError('Argument `other` needs to be of type `int`, `float` or '
                     '`torch.tensor`.')
