import warnings

import torch
from torch_scatter import scatter_add, segment_add


def optional(func, src):
    return func(src) if src is not None else src


class cached_property(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        value = getattr(obj, f'_{self.func.__name__}', None)
        if value is None:
            value = self.func(obj)
            setattr(obj, f'_{self.func.__name__}', value)
        return value


class SparseStorage(object):
    cache_keys = ['rowptr', 'colptr', 'csr_to_csc', 'csc_to_csr']

    def __init__(self, index, value=None, sparse_size=None, rowptr=None,
                 colptr=None, csr_to_csc=None, csc_to_csr=None,
                 is_sorted=False):

        assert index.dtype == torch.long
        assert index.dim() == 2 and index.size(0) == 2

        if value is not None:
            assert value.device == index.device
            assert value.size(0) == index.size(1)
            value = value.contiguous()

        if sparse_size is None:
            sparse_size = torch.Size((index.max(dim=-1)[0] + 1).tolist())

        if rowptr is not None:
            assert rowptr.dtype == torch.long and rowptr.device == index.device
            assert rowptr.dim() == 1 and rowptr.numel() - 1 == sparse_size[0]

        if colptr is not None:
            assert colptr.dtype == torch.long and colptr.device == index.device
            assert colptr.dim() == 1 and colptr.numel() - 1 == sparse_size[1]

        if csr_to_csc is not None:
            assert csr_to_csc.dtype == torch.long
            assert csr_to_csc.device == index.device
            assert csr_to_csc.dim() == 1
            assert csr_to_csc.numel() == index.size(1)

        if csc_to_csr is not None:
            assert csc_to_csr.dtype == torch.long
            assert csc_to_csr.device == index.device
            assert csc_to_csr.dim() == 1
            assert csc_to_csr.numel() == index.size(1)

        if not is_sorted:
            idx = sparse_size[1] * index[0] + index[1]
            # Only sort if necessary...
            if (idx <= torch.cat([idx.new_zeros(1), idx[:-1]], dim=0)).any():
                perm = idx.argsort()
                index = index[:, perm]
                value = None if value is None else value[perm]
                rowptr = None
                colptr = None
                csr_to_csc = None
                csc_to_csr = None

        self._index = index
        self._value = value
        self._sparse_size = sparse_size
        self._rowptr = rowptr
        self._colptr = colptr
        self._csr_to_csc = csr_to_csc
        self._csc_to_csr = csc_to_csr

    @property
    def index(self):
        return self._index

    @property
    def row(self):
        return self._index[0]

    @property
    def col(self):
        return self._index[1]

    @property
    def has_value(self):
        return self._value is not None

    @property
    def value(self):
        return self._value

    def set_value_(self, value, layout=None):
        if layout is None:
            layout = 'coo'
            warnings.warn('`layout` argument unset, using default layout '
                          '"coo". This may lead to unexpected behaviour.')
        assert layout in ['coo', 'csr', 'csc']
        assert value.device == self._index.device
        assert value.size(0) == self._index.size(1)
        if value is not None and layout == 'csc':
            value = value[self.csc_to_csr]
        return self.apply_value_(lambda x: value)

    def set_value(self, value, layout=None):
        if layout is None:
            layout = 'coo'
            warnings.warn('`layout` argument unset, using default layout '
                          '"coo". This may lead to unexpected behaviour.')
        assert layout in ['coo', 'csr', 'csc']
        assert value.device == self._index.device
        assert value.size(0) == self._index.size(1)
        if value is not None and layout == 'csc':
            value = value[self.csc_to_csr]
        return self.apply_value(lambda x: value)

    def sparse_size(self, dim=None):
        return self._sparse_size if dim is None else self._sparse_size[dim]

    def sparse_resize_(self, *sizes):
        assert len(sizes) == 2
        self._sparse_size == sizes
        return self

    @cached_property
    def rowptr(self):
        row = self.row
        ones = torch.ones_like(row)
        out_deg = segment_add(ones, row, dim=0, dim_size=self._sparse_size[0])
        return torch.cat([row.new_zeros(1), out_deg.cumsum(0)], dim=0)

    @cached_property
    def colptr(self):
        col = self.col
        ones = torch.ones_like(col)
        in_deg = scatter_add(ones, col, dim=0, dim_size=self._sparse_size[1])
        return torch.cat([col.new_zeros(1), in_deg.cumsum(0)], dim=0)

    @cached_property
    def csr_to_csc(self):
        idx = self._sparse_size[0] * self.col + self.row
        return idx.argsort()

    @cached_property
    def csc_to_csr(self):
        return self.csr_to_csc.argsort()

    def compute_cache_(self, *args):
        for arg in args or self.cache_keys:
            getattr(self, arg)
        return self

    def clear_cache_(self, *args):
        for arg in args or self.cache_keys:
            setattr(self, f'_{arg}', None)
        return self

    def clone(self):
        return self.apply(lambda x: x.clone())

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('SparseStorage', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def apply_value_(self, func):
        self._value = optional(func, self._value)
        return self

    def apply_value(self, func):
        return self.__class__(
            self._index,
            optional(func, self._value),
            self._sparse_size,
            self._rowptr,
            self._colptr,
            self._csr_to_csc,
            self._csc_to_csr,
            is_sorted=True,
        )

    def apply_(self, func):
        self._index = func(self._index)
        self._value = optional(func, self._value)
        for key in self.cache_keys:
            setattr(self, f'_{key}', optional(func, getattr(self, f'_{key}')))

    def apply(self, func):
        return self.__class__(
            func(self._index),
            optional(func, self._value),
            self._sparse_size,
            optional(func, self._rowptr),
            optional(func, self._colptr),
            optional(func, self._csr_to_csc),
            optional(func, self._csc_to_csr),
            is_sorted=True,
        )


if __name__ == '__main__':
    from torch_geometric.datasets import Reddit, Planetoid  # noqa
    import time  # noqa
    import copy  # noqa

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dataset = Reddit('/tmp/Reddit')
    dataset = Planetoid('/tmp/Cora', 'Cora')
    data = dataset[0].to(device)
    edge_index = data.edge_index

    storage = SparseStorage(edge_index, is_sorted=True)
    t = time.perf_counter()
    storage.compute_cache_()
    print(time.perf_counter() - t)
    t = time.perf_counter()
    storage.clear_cache_()
    storage.compute_cache_()
    print(time.perf_counter() - t)
    print(storage)
    storage = storage.clone()
    print(storage)
    # storage = copy.copy(storage)
    # print(storage)
    # storage = copy.deepcopy(storage)
    # print(storage)
    storage.compute_cache_()
    storage.clear_cache_()
