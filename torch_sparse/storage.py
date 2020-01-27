import warnings

import torch
from torch_scatter import segment_csr, scatter_add
from .utils import ext

__cache__ = {'enabled': True}


def is_cache_enabled():
    return __cache__['enabled']


def set_cache_enabled(mode):
    __cache__['enabled'] = mode


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
    cache_keys = ['rowcount', 'colptr', 'colcount', 'csr2csc', 'csc2csr']

    def __init__(self, row=None, rowptr=None, col=None, value=None,
                 sparse_size=None, rowcount=None, colptr=None, colcount=None,
                 csr2csc=None, csc2csr=None, is_sorted=False):

        assert row is not None or rowptr is not None
        assert col is not None
        assert col.dtype == torch.long
        assert col.dim() == 1
        col = col.contiguous()

        if sparse_size is None:
            M = rowptr.numel() - 1 if row is None else row.max().item() + 1
            N = col.max().item() + 1
            sparse_size = torch.Size([M, N])

        if row is not None:
            assert row.dtype == torch.long
            assert row.device == col.device
            assert row.dim() == 1
            assert row.numel() == col.numel()
            row = row.contiguous()

        if rowptr is not None:
            assert rowptr.dtype == torch.long
            assert rowptr.device == col.device
            assert rowptr.dim() == 1
            assert rowptr.numel() - 1 == sparse_size[0]
            rowptr = rowptr.contiguous()

        if value is not None:
            assert value.device == col.device
            assert value.size(0) == col.size(0)
            value = value.contiguous()

        if rowcount is not None:
            assert rowcount.dtype == torch.long
            assert rowcount.device == col.device
            assert rowcount.dim() == 1
            assert rowcount.numel() == sparse_size[0]
            rowcount = rowcount.contiguous()

        if colptr is not None:
            assert colptr.dtype == torch.long
            assert colptr.device == col.device
            assert colptr.dim() == 1
            assert colptr.numel() - 1 == sparse_size[1]
            colptr = colptr.contiguous()

        if colcount is not None:
            assert colcount.dtype == torch.long
            assert colcount.device == col.device
            assert colcount.dim() == 1
            assert colcount.numel() == sparse_size[1]
            colcount = colcount.contiguous()

        if csr2csc is not None:
            assert csr2csc.dtype == torch.long
            assert csr2csc.device == col.device
            assert csr2csc.dim() == 1
            assert csr2csc.numel() == col.size(0)
            csr2csc = csr2csc.contiguous()

        if csc2csr is not None:
            assert csc2csr.dtype == torch.long
            assert csc2csr.device == col.device
            assert csc2csr.dim() == 1
            assert csc2csr.numel() == col.size(0)
            csc2csr = csc2csr.contiguous()

        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._value = value
        self._sparse_size = sparse_size
        self._rowcount = rowcount
        self._colptr = colptr
        self._colcount = colcount
        self._csr2csc = csr2csc
        self._csc2csr = csc2csr

        if not is_sorted:
            idx = self.col.new_zeros(col.numel() + 1)
            idx[1:] = sparse_size[1] * self.row + self.col
            if (idx[1:] < idx[:-1]).any():
                perm = idx[1:].argsort()
                self._row = self.row[perm]
                self._col = self.col[perm]
                self._value = self.value[perm] if self.has_value() else None
                self._csr2csc = None
                self._csc2csr = None

    def has_row(self):
        return self._row is not None

    @property
    def row(self):
        if self._row is None:
            self._row = ext(self.col.is_cuda).ptr2ind(self.rowptr,
                                                      self.col.numel())
        return self._row

    def has_rowptr(self):
        return self._rowptr is not None

    @property
    def rowptr(self):
        if self._rowptr is None:
            self._rowptr = ext(self.col.is_cuda).ind2ptr(
                self.row, self.sparse_size[0])
        return self._rowptr

    @property
    def col(self):
        return self._col

    def has_value(self):
        return self._value is not None

    @property
    def value(self):
        return self._value

    def set_value_(self, value, layout=None, dtype=None):
        if isinstance(value, int) or isinstance(value, float):
            value = torch.full((self.col.numel(), ), dtype=dtype,
                               device=self.col.device)

        elif torch.is_tensor(value) and get_layout(layout) == 'csc':
            value = value[self.csc2csr]

        if torch.is_tensor(value):
            value = value if dtype is None else value.to(dtype)
            assert value.device == self.col.device
            assert value.size(0) == self.col.numel()

        self._value = value
        return self

    def set_value(self, value, layout=None, dtype=None):
        if isinstance(value, int) or isinstance(value, float):
            value = torch.full((self.col.numel(), ), dtype=dtype,
                               device=self.col.device)

        elif torch.is_tensor(value) and get_layout(layout) == 'csc':
            value = value[self.csc2csr]

        if torch.is_tensor(value):
            value = value if dtype is None else value.to(dtype)
            assert value.device == self.col.device
            assert value.size(0) == self.col.numel()

        return self.__class__(row=self._row, rowptr=self._rowptr, col=self.col,
                              value=value, sparse_size=self._sparse_size,
                              rowcount=self._rowcount, colptr=self._colptr,
                              colcount=self._colcount, csr2csc=self._csr2csc,
                              csc2csr=self._csc2csr, is_sorted=True)

    @property
    def sparse_size(self):
        return self._sparse_size

    def sparse_resize(self, *sizes):
        old_sparse_size, nnz = self.sparse_size, self.col.numel()

        diff_0 = sizes[0] - old_sparse_size[0]
        rowcount, rowptr = self._rowcount, self._rowptr
        if diff_0 > 0:
            if rowptr is not None:
                rowptr = torch.cat([rowptr, rowptr.new_full((diff_0, ), nnz)])
            if rowcount is not None:
                rowcount = torch.cat([rowcount, rowcount.new_zeros(diff_0)])
        else:
            if rowptr is not None:
                rowptr = rowptr[:-diff_0]
            if rowcount is not None:
                rowcount = rowcount[:-diff_0]

        diff_1 = sizes[1] - old_sparse_size[1]
        colcount, colptr = self._colcount, self._colptr
        if diff_1 > 0:
            if colptr is not None:
                colptr = torch.cat([colptr, colptr.new_full((diff_1, ), nnz)])
            if colcount is not None:
                colcount = torch.cat([colcount, colcount.new_zeros(diff_1)])
        else:
            if colptr is not None:
                colptr = colptr[:-diff_1]
            if colcount is not None:
                colcount = colcount[:-diff_1]

        return self.__class__(row=self._row, rowptr=rowptr, col=self.col,
                              value=self.value, sparse_size=sizes,
                              rowcount=rowcount, colptr=colptr,
                              colcount=colcount, csr2csc=self._csr2csc,
                              csc2csr=self._csc2csr, is_sorted=True)

    def has_rowcount(self):
        return self._rowcount is not None

    @cached_property
    def rowcount(self):
        return self.rowptr[1:] - self.rowptr[:-1]

    def has_colptr(self):
        return self._colptr is not None

    @cached_property
    def colptr(self):
        if self.has_csr2csc():
            return ext(self.col.is_cuda).ind2ptr(self.col[self.csr2csc],
                                                 self.sparse_size[1])
        else:
            colptr = self.col.new_zeros(self.sparse_size[1] + 1)
            torch.cumsum(self.colcount, dim=0, out=colptr[1:])
            return colptr

    def has_colcount(self):
        return self._colcount is not None

    @cached_property
    def colcount(self):
        if self.has_colptr():
            return self.colptr[1:] - self.colptr[:-1]
        else:
            return scatter_add(torch.ones_like(self.col), self.col,
                               dim_size=self.sparse_size[1])

    def has_csr2csc(self):
        return self._csr2csc is not None

    @cached_property
    def csr2csc(self):
        idx = self.sparse_size[0] * self.col + self.row
        return idx.argsort()

    def has_csc2csr(self):
        return self._csc2csr is not None

    @cached_property
    def csc2csr(self):
        return self.csr2csc.argsort()

    def is_coalesced(self):
        idx = self.col.new_full((self.col.numel() + 1, ), -1)
        idx[1:] = self.sparse_size[1] * self.row + self.col
        return (idx[1:] > idx[:-1]).all().item()

    def coalesce(self, reduce='add'):
        idx = self.col.new_full((self.col.numel() + 1, ), -1)
        idx[1:] = self.sparse_size[1] * self.row + self.col
        mask = idx[1:] > idx[:-1]

        if mask.all():  # Skip if indices are already coalesced.
            return self

        row = self.row[mask]
        col = self.col[mask]

        value = self.value
        if self.has_value():
            ptr = mask.nonzero().flatten()
            ptr = torch.cat([ptr, ptr.new_full((1, ), value.size(0))])
            value = segment_csr(value, ptr, reduce=reduce)
            value = value[0] if isinstance(value, tuple) else value

        return self.__class__(row=row, col=col, value=value,
                              sparse_size=self.sparse_size, is_sorted=True)

    def cached_keys(self):
        return [
            key for key in self.cache_keys
            if getattr(self, f'_{key}', None) is not None
        ]

    def fill_cache_(self, *args):
        for arg in args or self.cache_keys + ['row', 'rowptr']:
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
        self._value = optional(func, self.value)
        return self

    def apply_value(self, func):
        return self.__class__(row=self._row, rowptr=self._rowptr, col=self.col,
                              value=optional(func, self.value),
                              sparse_size=self.sparse_size,
                              rowcount=self._rowcount, colptr=self._colptr,
                              colcount=self._colcount, csr2csc=self._csr2csc,
                              csc2csr=self._csc2csr, is_sorted=True)

    def apply_(self, func):
        self._row = optional(func, self._row)
        self._rowptr = optional(func, self._rowptr)
        self._col = func(self.col)
        self._value = optional(func, self.value)
        for key in self.cached_keys():
            setattr(self, f'_{key}', func(getattr(self, f'_{key}')))
        return self

    def apply(self, func):
        return self.__class__(
            row=optional(func, self._row),
            rowptr=optional(func, self._rowptr),
            col=func(self.col),
            value=optional(func, self.value),
            sparse_size=self.sparse_size,
            rowcount=optional(func, self._rowcount),
            colptr=optional(func, self._colptr),
            colcount=optional(func, self._colcount),
            csr2csc=optional(func, self._csr2csc),
            csc2csr=optional(func, self._csc2csr),
            is_sorted=True,
        )

    def map(self, func):
        data = []
        if self.has_row():
            data += [func(self.row)]
        if self.has_rowptr():
            data += [func(self.rowptr)]
        data += [func(self.col)]
        if self.has_value():
            data += [func(self.value)]
        data += [func(getattr(self, f'_{key}')) for key in self.cached_keys()]
        return data
