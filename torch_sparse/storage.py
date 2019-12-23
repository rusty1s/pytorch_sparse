import warnings

import torch
import torch_scatter
from torch_scatter import scatter_add, segment_add

__cache_flag__ = {'enabled': True}


def is_cache_enabled():
    return __cache_flag__['enabled']


def set_cache_enabled(mode):
    __cache_flag__['enabled'] = mode


class no_cache(object):
    def __enter__(self):
        self.prev = is_cache_enabled()
        set_cache_enabled(False)

    def __exit__(self, *args):
        set_cache_enabled(self.prev)
        return False

    def __call__(self, func):
        def decorate_no_cache(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_cache


class cached_property(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        value = getattr(obj, f'_{self.func.__name__}', None)
        if value is None:
            value = self.func(obj)
            if is_cache_enabled():
                setattr(obj, f'_{self.func.__name__}', value)
        return value


def optional(func, src):
    return func(src) if src is not None else src


layouts = ['coo', 'csr', 'csc']


def get_layout(layout=None):
    if layout is None:
        layout = 'coo'
        warnings.warn('`layout` argument unset, using default layout '
                      '"coo". This may lead to unexpected behaviour.')
    assert layout in layouts
    return layout


class SparseStorage(object):
    cache_keys = [
        'rowcount', 'rowptr', 'colcount', 'colptr', 'csr2csc', 'csc2csr'
    ]

    def __init__(self, index, value=None, sparse_size=None, rowcount=None,
                 rowptr=None, colcount=None, colptr=None, csr2csc=None,
                 csc2csr=None, is_sorted=False):

        assert index.dtype == torch.long
        assert index.dim() == 2 and index.size(0) == 2
        index = index.contiguous()

        if value is not None:
            assert value.device == index.device
            assert value.size(0) == index.size(1)
            value = value.contiguous()

        if sparse_size is None:
            sparse_size = torch.Size((index.max(dim=-1)[0] + 1).tolist())

        if rowcount is not None:
            assert rowcount.dtype == torch.long
            assert rowcount.device == index.device
            assert rowcount.dim() == 1 and rowcount.numel() == sparse_size[0]

        if rowptr is not None:
            assert rowptr.dtype == torch.long
            assert rowptr.device == index.device
            assert rowptr.dim() == 1 and rowptr.numel() - 1 == sparse_size[0]

        if colcount is not None:
            assert colcount.dtype == torch.long
            assert colcount.device == index.device
            assert colcount.dim() == 1 and colcount.numel() == sparse_size[1]

        if colptr is not None:
            assert colptr.dtype == torch.long
            assert colptr.device == index.device
            assert colptr.dim() == 1 and colptr.numel() - 1 == sparse_size[1]

        if csr2csc is not None:
            assert csr2csc.dtype == torch.long
            assert csr2csc.device == index.device
            assert csr2csc.dim() == 1
            assert csr2csc.numel() == index.size(1)

        if csc2csr is not None:
            assert csc2csr.dtype == torch.long
            assert csc2csr.device == index.device
            assert csc2csr.dim() == 1
            assert csc2csr.numel() == index.size(1)

        if not is_sorted:
            idx = sparse_size[1] * index[0] + index[1]
            # Only sort if necessary...
            if (idx < torch.cat([idx.new_zeros(1), idx[:-1]], dim=0)).any():
                perm = idx.argsort()
                index = index[:, perm]
                value = None if value is None else value[perm]
                csr2csc = None
                csc2csr = None

        self._index = index
        self._value = value
        self._sparse_size = sparse_size
        self._rowcount = rowcount
        self._rowptr = rowptr
        self._colcount = colcount
        self._colptr = colptr
        self._csr2csc = csr2csc
        self._csc2csr = csc2csr

    @property
    def index(self):
        return self._index

    @property
    def row(self):
        return self._index[0]

    @property
    def col(self):
        return self._index[1]

    def has_value(self):
        return self._value is not None

    @property
    def value(self):
        return self._value

    def set_value_(self, value, layout=None):
        assert value.device == self._index.device
        assert value.size(0) == self._index.size(1)
        if value is not None and get_layout(layout) == 'csc':
            value = value[self.csc2csr]
        self._value = value
        return self

    def set_value(self, value, layout=None):
        assert value.device == self._index.device
        assert value.size(0) == self._index.size(1)
        if value is not None and get_layout(layout) == 'csc':
            value = value[self.csc2csr]
        return self.__class__(
            self._index,
            value,
            self._sparse_size,
            self._rowcount,
            self._rowptr,
            self._colcount,
            self._colptr,
            self._csr2csc,
            self._csc2csr,
            is_sorted=True,
        )

    def sparse_size(self, dim=None):
        return self._sparse_size if dim is None else self._sparse_size[dim]

    def sparse_resize_(self, *sizes):
        assert len(sizes) == 2
        self._sparse_size = sizes
        return self

    @cached_property
    def rowcount(self):
        one = torch.ones_like(self.row)
        return segment_add(one, self.row, dim=0, dim_size=self._sparse_size[0])

    @cached_property
    def rowptr(self):
        rowcount = self.rowcount
        return torch.cat([rowcount.new_zeros(1), rowcount.cumsum(0)], dim=0)

    @cached_property
    def colcount(self):
        one = torch.ones_like(self.col)
        return scatter_add(one, self.col, dim=0, dim_size=self._sparse_size[1])

    @cached_property
    def colptr(self):
        colcount = self.colcount
        return torch.cat([colcount.new_zeros(1), colcount.cumsum(0)], dim=0)

    @cached_property
    def csr2csc(self):
        idx = self._sparse_size[0] * self.col + self.row
        return idx.argsort()

    @cached_property
    def csc2csr(self):
        return self.csr2csc.argsort()

    def is_coalesced(self):
        idx = self.sparse_size(1) * self.row + self.col
        mask = idx > torch.cat([idx.new_full((1, ), -1), idx[:-1]], dim=0)
        return mask.all().item()

    def coalesce(self, reduce='add'):
        idx = self.sparse_size(1) * self.row + self.col
        mask = idx > torch.cat([idx.new_full((1, ), -1), idx[:-1]], dim=0)

        if mask.all():  # Already coalesced
            return self

        index = self.index[:, mask]

        value = self.value
        if self.has_value():
            assert reduce in ['add', 'mean', 'min', 'max']
            idx = mask.cumsum(0) - 1
            op = getattr(torch_scatter, f'scatter_{reduce}')
            value = op(value, idx, dim=0, dim_size=idx[-1].item() + 1)
            value = value[0] if isinstance(value, tuple) else value

        return self.__class__(index, value, self.sparse_size(), is_sorted=True)

    def cached_keys(self):
        return [
            key for key in self.cache_keys
            if getattr(self, f'_{key}', None) is not None
        ]

    def fill_cache_(self, *args):
        for arg in args or self.cache_keys:
            getattr(self, arg)
        return self

    def clear_cache_(self, *args):
        for arg in args or self.cache_keys:
            setattr(self, f'_{arg}', None)
        return self

    def __copy__(self):
        return self.apply(lambda x: x)

    def clone(self):
        return self.apply(lambda x: x.clone())

    def __deepcopy__(self, memo):
        new_storage = self.clone()
        memo[id(self)] = new_storage
        return new_storage

    def apply_value_(self, func):
        self._value = optional(func, self._value)
        return self

    def apply_value(self, func):
        return self.__class__(
            self._index,
            optional(func, self._value),
            self._sparse_size,
            self._rowcount,
            self._rowptr,
            self._colcount,
            self._colptr,
            self._csr2csc,
            self._csc2csr,
            is_sorted=True,
        )

    def apply_(self, func):
        self._index = func(self._index)
        self._value = optional(func, self._value)
        for key in self.cached_keys():
            setattr(self, f'_{key}', func(getattr(self, f'_{key}')))
        return self

    def apply(self, func):
        return self.__class__(
            func(self._index),
            optional(func, self._value),
            self._sparse_size,
            optional(func, self._rowcount),
            optional(func, self._rowptr),
            optional(func, self._colcount),
            optional(func, self._colptr),
            optional(func, self._csr2csc),
            optional(func, self._csc2csr),
            is_sorted=True,
        )

    def map(self, func):
        data = [func(self.index)]
        if self.has_value():
            data += [func(self.value)]
        data += [func(getattr(self, f'_{key}')) for key in self.cached_keys()]
        return data
