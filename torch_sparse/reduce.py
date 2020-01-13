import torch
import torch_scatter
from torch_scatter import segment_csr


def __reduce__(src, dim=None, reduce='add', deterministic=False):
    if dim is None and src.has_value():
        func = getattr(torch, 'sum' if reduce == 'add' else reduce)
        return func(src.storage.value)

    if dim is None and not src.has_value():
        assert reduce in ['add', 'mean', 'min', 'max']
        value = src.nnz() if reduce == 'add' else 1
        return torch.tensor(value, device=src.device)

    dims = [dim] if isinstance(dim, int) else sorted(list(dim))
    assert dims[-1] < src.dim()

    rowptr, col, value = src.csr()

    sparse_dims = tuple(set([d for d in dims if d < 2]))
    dense_dims = tuple(set([d - 1 for d in dims if d > 1]))

    if len(sparse_dims) == 2 and src.has_value():
        func = getattr(torch, 'sum' if reduce == 'add' else reduce)
        return func(value, dim=(0, ) + dense_dims)

    if len(sparse_dims) == 2 and not src.has_value():
        assert reduce in ['add', 'mean', 'min', 'max']
        value = src.nnz() if reduce == 'add' else 1
        return torch.tensor(value, device=src.device)

    if len(dense_dims) > 0 and len(sparse_dims) == 0:  # src.has_value()
        func = getattr(torch, 'sum' if reduce == 'add' else reduce)
        dense_dims = dense_dims[0] if len(dense_dims) == 1 else dense_dims
        value = func(value, dim=dense_dims)
        if isinstance(value, tuple):
            return (src.set_value(value[0], layout='csr'), ) + value[1:]
        return src.set_value(value, layout='csr')

    if len(dense_dims) > 0 and len(sparse_dims) > 0:
        func = getattr(torch, 'sum' if reduce == 'add' else reduce)
        dense_dims = dense_dims[0] if len(dense_dims) == 1 else dense_dims
        value = func(value, dim=dense_dims)
        value = value[0] if isinstance(value, tuple) else value

    if sparse_dims[0] == 1 and src.has_value():
        out = segment_csr(value, rowptr)
        out = out[0] if len(dense_dims) > 0 and isinstance(out, tuple) else out
        return out

    if sparse_dims[0] == 1 and not src.has_value():
        assert reduce in ['add', 'mean', 'min', 'max']
        if reduce == 'add':
            return src.storage.rowcount.to(torch.get_default_dtype())
        elif reduce == 'min' or 'max':
            return torch.ones(src.size(0), device=src.device), None
        else:
            return torch.ones(src.size(0), device=src.device)

    deterministic = src.storage._csr2csc is not None or deterministic

    if sparse_dims[0] == 0 and deterministic and src.has_value():
        csr2csc = src.storage.csr2csc
        out = segment_csr(value[csr2csc], src.storage.colptr)
        out = out[0] if len(dense_dims) > 0 and isinstance(out, tuple) else out
        return out

    if sparse_dims[0] == 0 and src.has_value():
        func = getattr(torch_scatter, f'scatter_{reduce}')
        out = func(value, col, dim=0, dim_size=src.sparse_size(1))
        out = out[0] if len(dense_dims) > 0 and isinstance(out, tuple) else out
        return out

    if sparse_dims[0] == 0 and not src.has_value():
        assert reduce in ['add', 'mean', 'min', 'max']
        if reduce == 'add':
            return src.storage.colcount.to(torch.get_default_dtype())
        elif reduce == 'min' or 'max':
            return torch.ones(src.size(1), device=src.device), None
        else:
            return torch.ones(src.size(1), device=src.device)


def sum(src, dim=None, deterministic=False):
    return __reduce__(src, dim, reduce='add', deterministic=deterministic)


def mean(src, dim=None, deterministic=False):
    return __reduce__(src, dim, reduce='mean', deterministic=deterministic)


def min(src, dim=None, deterministic=False):
    return __reduce__(src, dim, reduce='min', deterministic=deterministic)


def max(src, dim=None, deterministic=False):
    return __reduce__(src, dim, reduce='max', deterministic=deterministic)
