import warnings
from typing import Optional, List, Dict, Union, Any

import torch
from torch_scatter import segment_csr, scatter_add
from torch_sparse.utils import Final, is_scalar

# __cache__ = {'enabled': True}

# def is_cache_enabled():
#     return __cache__['enabled']

# def set_cache_enabled(mode):
#     __cache__['enabled'] = mode

# class no_cache(object):
#     def __enter__(self):
#         self.prev = is_cache_enabled()
#         set_cache_enabled(False)

#     def __exit__(self, *args):
#         set_cache_enabled(self.prev)
#         return False

#     def __call__(self, func):
#         def decorate_no_cache(*args, **kwargs):
#             with self:
#                 return func(*args, **kwargs)

#         return decorate_no_cache


def optional(func, src):
    return func(src) if src is not None else src


layouts: Final[List[str]] = ['coo', 'csr', 'csc']


def get_layout(layout: Optional[str] = None) -> str:
    if layout is None:
        layout = 'coo'
        warnings.warn('`layout` argument unset, using default layout '
                      '"coo". This may lead to unexpected behaviour.')
    assert layout == 'coo' or layout == 'csr' or layout == 'csc'
    return layout


@torch.jit.script
class SparseStorage(object):
    _row: Optional[torch.Tensor]
    _rowptr: Optional[torch.Tensor]
    _col: torch.Tensor
    _value: Optional[torch.Tensor]
    _sparse_size: List[int]
    _rowcount: Optional[torch.Tensor]
    _colptr: Optional[torch.Tensor]
    _colcount: Optional[torch.Tensor]
    _csr2csc: Optional[torch.Tensor]
    _csc2csr: Optional[torch.Tensor]

    def __init__(self, row: Optional[torch.Tensor] = None,
                 rowptr: Optional[torch.Tensor] = None,
                 col: Optional[torch.Tensor] = None,
                 value: Optional[torch.Tensor] = None,
                 sparse_size: Optional[List[int]] = None,
                 rowcount: Optional[torch.Tensor] = None,
                 colptr: Optional[torch.Tensor] = None,
                 colcount: Optional[torch.Tensor] = None,
                 csr2csc: Optional[torch.Tensor] = None,
                 csc2csr: Optional[torch.Tensor] = None,
                 is_sorted: bool = False):

        assert row is not None or rowptr is not None
        assert col is not None
        assert col.dtype == torch.long
        assert col.dim() == 1
        col = col.contiguous()

        if sparse_size is None:
            if rowptr is not None:
                M = rowptr.numel() - 1
            elif row is not None:
                M = row.max().item() + 1
            else:
                raise ValueError
            N = col.max().item() + 1
            sparse_size = torch.Size([int(M), int(N)])
        else:
            assert len(sparse_size) == 2

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
            idx = col.new_zeros(col.numel() + 1)
            idx[1:] = sparse_size[1] * self.row() + col
            if (idx[1:] < idx[:-1]).any():
                perm = idx[1:].argsort()
                self._row = self.row()[perm]
                self._col = col[perm]
                if value is not None:
                    self._value = value[perm]
                self._csr2csc = None
                self._csc2csr = None

    def has_row(self) -> bool:
        return self._row is not None

    def row(self):
        row = self._row
        if row is not None:
            return row

        rowptr = self._rowptr
        if rowptr is not None:
            if rowptr.is_cuda:
                row = torch.ops.torch_sparse_cuda.ptr2ind(
                    rowptr, self._col.numel())
            else:
                if rowptr.is_cuda:
                    row = torch.ops.torch_sparse_cuda.ptr2ind(
                        rowptr, self._col.numel())
                else:
                    row = torch.ops.torch_sparse_cpu.ptr2ind(
                        rowptr, self._col.numel())
            self._row = row
            return row

        raise ValueError

    def has_rowptr(self) -> bool:
        return self._rowptr is not None

    def rowptr(self) -> torch.Tensor:
        rowptr = self._rowptr
        if rowptr is not None:
            return rowptr

        row = self._row
        if row is not None:
            if row.is_cuda:
                rowptr = torch.ops.torch_sparse_cuda.ind2ptr(
                    row, self._sparse_size[0])
            else:
                rowptr = torch.ops.torch_sparse_cpu.ind2ptr(
                    row, self._sparse_size[0])
            self._rowptr = rowptr
            return rowptr

        raise ValueError

    def col(self) -> torch.Tensor:
        return self._col

    def has_value(self) -> bool:
        return self._value is not None

    def value(self) -> Optional[torch.Tensor]:
        return self._value

    def set_value_(self, value: Optional[torch.Tensor],
                   layout: Optional[str] = None):
        if value is not None:
            if get_layout(layout) == 'csc2csr':
                value = value[self.csc2csr()]
            value = value.contiguous()
            assert value.device == self._col.device
            assert value.size(0) == self._col.numel()

        self._value = value
        return self

    def set_value(self, value: Optional[torch.Tensor],
                  layout: Optional[str] = None):
        if value is not None:
            if get_layout(layout) == 'csc2csr':
                value = value[self.csc2csr()]
            value = value.contiguous()
            assert value.device == self._col.device
            assert value.size(0) == self._col.numel()

        return SparseStorage(row=self._row, rowptr=self._rowptr, col=self._col,
                             value=value, sparse_size=self._sparse_size,
                             rowcount=self._rowcount, colptr=self._colptr,
                             colcount=self._colcount, csr2csc=self._csr2csc,
                             csc2csr=self._csc2csr, is_sorted=True)

    def fill_value_(self, fill_value: float, dtype=Optional[torch.dtype]):
        value = torch.empty(self._col.numel(), dtype, device=self._col.device)
        return self.set_value_(value.fill_(fill_value), layout='csr')

    def fill_value(self, fill_value: float, dtype=Optional[torch.dtype]):
        value = torch.empty(self._col.numel(), dtype, device=self._col.device)
        return self.set_value(value.fill_(fill_value), layout='csr')

    def sparse_size(self) -> List[int]:
        return self._sparse_size

    def sparse_resize(self, sparse_size: List[int]):
        assert len(sparse_size) == 2
        old_sparse_size, nnz = self._sparse_size, self._col.numel()

        diff_0 = sparse_size[0] - old_sparse_size[0]
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

        diff_1 = sparse_size[1] - old_sparse_size[1]
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

        return SparseStorage(row=self._row, rowptr=rowptr, col=self._col,
                             value=self._value, sparse_size=sparse_size,
                             rowcount=rowcount, colptr=colptr,
                             colcount=colcount, csr2csc=self._csr2csc,
                             csc2csr=self._csc2csr, is_sorted=True)

    def has_rowcount(self) -> bool:
        return self._rowcount is not None

    def rowcount(self) -> torch.Tensor:
        rowcount = self._rowcount
        if rowcount is not None:
            return rowcount

        rowptr = self.rowptr()
        rowcount = rowptr[1:] - rowptr[1:]
        self._rowcount = rowcount
        return rowcount

    def has_colptr(self) -> bool:
        return self._colptr is not None

    def colptr(self) -> torch.Tensor:
        colptr = self._colptr
        if colptr is not None:
            return colptr

        csr2csc = self._csr2csc
        if csr2csc is not None:
            colptr = torch.ops.torch_sparse_cpu.ind2ptr(
                self._col[csr2csc], self._sparse_size[1])
        else:
            colptr = self._col.new_zeros(self._sparse_size[1] + 1)
            torch.cumsum(self.colcount(), dim=0, out=colptr[1:])
        self._colptr = colptr
        return colptr

    def has_colcount(self) -> bool:
        return self._colcount is not None

    def colcount(self) -> torch.Tensor:
        colcount = self._colcount
        if colcount is not None:
            return colcount

        colptr = self._colptr
        if colptr is not None:
            colcount = colptr[1:] - colptr[1:]
        else:
            raise NotImplementedError
            # colcount = scatter_add(torch.ones_like(self._col), self._col,
            #                        dim_size=self._sparse_size[1])
        self._colcount = colcount
        return colcount

    def has_csr2csc(self) -> bool:
        return self._csr2csc is not None

    def csr2csc(self) -> torch.Tensor:
        csr2csc = self._csr2csc
        if csr2csc is not None:
            return csr2csc

        idx = self._sparse_size[0] * self._col + self.row()
        csr2csc = idx.argsort()
        self._csr2csc = csr2csc
        return csr2csc

    def has_csc2csr(self) -> bool:
        return self._csc2csr is not None

    def csc2csr(self) -> torch.Tensor:
        csc2csr = self._csc2csr
        if csc2csr is not None:
            return csc2csr

        csc2csr = self.csr2csc().argsort()
        self._csc2csr = csc2csr
        return csc2csr

    def is_coalesced(self) -> bool:
        idx = self._col.new_full((self._col.numel() + 1, ), -1)
        idx[1:] = self._sparse_size[1] * self.row() + self._col
        return bool((idx[1:] > idx[:-1]).all())

    def coalesce(self, reduce: str = "add"):
        idx = self._col.new_full((self._col.numel() + 1, ), -1)
        idx[1:] = self._sparse_size[1] * self.row() + self._col
        mask = idx[1:] > idx[:-1]

        if mask.all():  # Skip if indices are already coalesced.
            return self

        row = self.row()[mask]
        col = self._col[mask]

        value = self._value
        if value is not None:
            ptr = mask.nonzero().flatten()
            ptr = torch.cat([ptr, ptr.new_full((1, ), value.size(0))])
            raise NotImplementedError
            # value = segment_csr(value, ptr, reduce=reduce)
            value = value[0] if isinstance(value, tuple) else value

        return SparseStorage(row=row, rowptr=None, col=col, value=value,
                             sparse_size=self._sparse_size, rowcount=None,
                             colptr=None, colcount=None, csr2csc=None,
                             csc2csr=None, is_sorted=True)

    def fill_cache_(self):
        self.row()
        self.rowptr()
        self.rowcount()
        self.colptr()
        self.colcount()
        self.csr2csc()
        self.csc2csr()
        return self

    def clear_cache_(self):
        self._rowcount = None
        self._colptr = None
        self._colcount = None
        self._csr2csc = None
        self._csc2csr = None
        return self

    def copy(self):
        return SparseStorage(row=self._row, rowptr=self._rowptr, col=self._col,
                             value=self._value, sparse_size=self._sparse_size,
                             rowcount=self._rowcount, colptr=self._colptr,
                             colcount=self._colcount, csr2csc=self._csr2csc,
                             csc2csr=self._csc2csr, is_sorted=True)

    def clone(self):
        row = self._row
        if row is not None:
            row = row.clone()
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.clone()
        value = self._value
        if value is not None:
            value = value.clone()
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.clone()
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.clone()
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.clone()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.clone()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.clone()
        return SparseStorage(row=row, rowptr=rowptr, col=self._col.clone(),
                             value=value, sparse_size=self._sparse_size,
                             rowcount=rowcount, colptr=colptr,
                             colcount=colcount, csr2csc=csr2csc,
                             csc2csr=csc2csr, is_sorted=True)
