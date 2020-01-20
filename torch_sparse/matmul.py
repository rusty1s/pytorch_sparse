import torch


def matmul(src, other, reduce='add'):
    if torch.is_tensor(other):
        pass
    if isinstance(other, src.__class__):
        if reduce != 'add':
            raise NotImplementedError(
                (f'Reduce argument "{reduce}" not implemented for sparse-'
                 f'sparse matrix multiplication'))
