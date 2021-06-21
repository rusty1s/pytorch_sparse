import warnings
from typing import Optional, List, Tuple

import torch
from torch_scatter import segment_csr, scatter_add
from torch_sparse.utils import Final

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
    _sparse_sizes: Tuple[int, int]
    _rowcount: Optional[torch.Tensor]
    _colptr: Optional[torch.Tensor]
    _colcount: Optional[torch.Tensor]
    _csr2csc: Optional[torch.Tensor]
    _csc2csr: Optional[torch.Tensor]

    def __init__(self, row: Optional[torch.Tensor] = None,
                 rowptr: Optional[torch.Tensor] = None,
                 col: Optional[torch.Tensor] = None,
                 value: Optional[torch.Tensor] = None,
                 sparse_sizes: Optional[Tuple[int, int]] = None,
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

        if sparse_sizes is None:
            if rowptr is not None:
                M = rowptr.numel() - 1
            elif row is not None and row.numel() > 0:
                M = row.max().item() + 1
            elif row is not None and row.numel() == 0:
                M = 0
            else:
                raise ValueError
            if col.numel() > 0:
                N = col.max().item() + 1
            else:
                N = 0
            sparse_sizes = (int(M), int(N))
        else:
            assert len(sparse_sizes) == 2
            if row is not None and row.numel() > 0:
                assert row.max().item() < sparse_sizes[0]
            if col.numel() > 0:
                assert col.max().item() < sparse_sizes[1]

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
            assert rowptr.numel() - 1 == sparse_sizes[0]
            rowptr = rowptr.contiguous()

        if value is not None:
            assert value.device == col.device
            assert value.size(0) == col.size(0)
            value = value.contiguous()

        if rowcount is not None:
            assert rowcount.dtype == torch.long
            assert rowcount.device == col.device
            assert rowcount.dim() == 1
            assert rowcount.numel() == sparse_sizes[0]
            rowcount = rowcount.contiguous()

        if colptr is not None:
            assert colptr.dtype == torch.long
            assert colptr.device == col.device
            assert colptr.dim() == 1
            assert colptr.numel() - 1 == sparse_sizes[1]
            colptr = colptr.contiguous()

        if colcount is not None:
            assert colcount.dtype == torch.long
            assert colcount.device == col.device
            assert colcount.dim() == 1
            assert colcount.numel() == sparse_sizes[1]
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
        self._sparse_sizes = tuple(sparse_sizes)
        self._rowcount = rowcount
        self._colptr = colptr
        self._colcount = colcount
        self._csr2csc = csr2csc
        self._csc2csr = csc2csr

        if not is_sorted:
            idx = self._col.new_zeros(self._col.numel() + 1)
            idx[1:] = self.row()
            idx[1:] *= self._sparse_sizes[1]
            idx[1:] += self._col
            if (idx[1:] < idx[:-1]).any():
                perm = idx[1:].argsort()
                self._row = self.row()[perm]
                self._col = self._col[perm]
                if value is not None:
                    self._value = value[perm]
                self._csr2csc = None
                self._csc2csr = None

    @classmethod
    def empty(self):
        row = torch.tensor([], dtype=torch.long)
        col = torch.tensor([], dtype=torch.long)
        return SparseStorage(row=row, rowptr=None, col=col, value=None,
                             sparse_sizes=(0, 0), rowcount=None, colptr=None,
                             colcount=None, csr2csc=None, csc2csr=None,
                             is_sorted=True)

    def has_row(self) -> bool:
        return self._row is not None

    def row(self):
        row = self._row
        if row is not None:
            return row

        rowptr = self._rowptr
        if rowptr is not None:
            row = torch.ops.torch_sparse.ptr2ind(rowptr, self._col.numel())
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
            rowptr = torch.ops.torch_sparse.ind2ptr(row, self._sparse_sizes[0])
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
            if get_layout(layout) == 'csc':
                value = value[self.csc2csr()]
            value = value.contiguous()
            assert value.device == self._col.device
            assert value.size(0) == self._col.numel()

        self._value = value
        return self

    def set_value(self, value: Optional[torch.Tensor],
                  layout: Optional[str] = None):
        if value is not None:
            if get_layout(layout) == 'csc':
                value = value[self.csc2csr()]
            value = value.contiguous()
            assert value.device == self._col.device
            assert value.size(0) == self._col.numel()

        return SparseStorage(row=self._row, rowptr=self._rowptr, col=self._col,
                             value=value, sparse_sizes=self._sparse_sizes,
                             rowcount=self._rowcount, colptr=self._colptr,
                             colcount=self._colcount, csr2csc=self._csr2csc,
                             csc2csr=self._csc2csr, is_sorted=True)

    def sparse_sizes(self) -> Tuple[int, int]:
        return self._sparse_sizes

    def sparse_size(self, dim: int) -> int:
        return self._sparse_sizes[dim]

    def sparse_resize(self, sparse_sizes: Tuple[int, int]):
        assert len(sparse_sizes) == 2
        old_sparse_sizes, nnz = self._sparse_sizes, self._col.numel()

        diff_0 = sparse_sizes[0] - old_sparse_sizes[0]
        rowcount, rowptr = self._rowcount, self._rowptr
        if diff_0 > 0:
            if rowptr is not None:
                rowptr = torch.cat([rowptr, rowptr.new_full((diff_0, ), nnz)])
            if rowcount is not None:
                rowcount = torch.cat([rowcount, rowcount.new_zeros(diff_0)])
        elif diff_0 < 0:
            if rowptr is not None:
                rowptr = rowptr[:-diff_0]
            if rowcount is not None:
                rowcount = rowcount[:-diff_0]

        diff_1 = sparse_sizes[1] - old_sparse_sizes[1]
        colcount, colptr = self._colcount, self._colptr
        if diff_1 > 0:
            if colptr is not None:
                colptr = torch.cat([colptr, colptr.new_full((diff_1, ), nnz)])
            if colcount is not None:
                colcount = torch.cat([colcount, colcount.new_zeros(diff_1)])
        elif diff_1 < 0:
            if colptr is not None:
                colptr = colptr[:-diff_1]
            if colcount is not None:
                colcount = colcount[:-diff_1]

        return SparseStorage(row=self._row, rowptr=rowptr, col=self._col,
                             value=self._value, sparse_sizes=sparse_sizes,
                             rowcount=rowcount, colptr=colptr,
                             colcount=colcount, csr2csc=self._csr2csc,
                             csc2csr=self._csc2csr, is_sorted=True)

    def sparse_reshape(self, num_rows: int, num_cols: int):
        assert num_rows > 0 or num_rows == -1
        assert num_cols > 0 or num_cols == -1
        assert num_rows > 0 or num_cols > 0

        total = self.sparse_size(0) * self.sparse_size(1)

        if num_rows == -1:
            num_rows = total // num_cols

        if num_cols == -1:
            num_cols = total // num_rows

        assert num_rows * num_cols == total

        idx = self.sparse_size(1) * self.row() + self.col()

        row = idx // num_cols
        col = idx % num_cols
        assert row.dtype == torch.long and col.dtype == torch.long

        return SparseStorage(row=row, rowptr=None, col=col, value=self._value,
                             sparse_sizes=(num_rows, num_cols), rowcount=None,
                             colptr=None, colcount=None, csr2csc=None,
                             csc2csr=None, is_sorted=True)

    def has_rowcount(self) -> bool:
        return self._rowcount is not None

    def rowcount(self) -> torch.Tensor:
        rowcount = self._rowcount
        if rowcount is not None:
            return rowcount

        rowptr = self.rowptr()
        rowcount = rowptr[1:] - rowptr[:-1]
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
            colptr = torch.ops.torch_sparse.ind2ptr(self._col[csr2csc],
                                                    self._sparse_sizes[1])
        else:
            colptr = self._col.new_zeros(self._sparse_sizes[1] + 1)
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
            colcount = colptr[1:] - colptr[:-1]
        else:
            colcount = scatter_add(torch.ones_like(self._col), self._col,
                                   dim_size=self._sparse_sizes[1])
        self._colcount = colcount
        return colcount

    def has_csr2csc(self) -> bool:
        return self._csr2csc is not None

    def csr2csc(self) -> torch.Tensor:
        csr2csc = self._csr2csc
        if csr2csc is not None:
            return csr2csc

        idx = self._sparse_sizes[0] * self._col + self.row()
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
        idx[1:] = self._sparse_sizes[1] * self.row() + self._col
        return bool((idx[1:] > idx[:-1]).all())

    def coalesce(self, reduce: str = "add"):
        idx = self._col.new_full((self._col.numel() + 1, ), -1)
        idx[1:] = self._sparse_sizes[1] * self.row() + self._col
        mask = idx[1:] > idx[:-1]

        if mask.all():  # Skip if indices are already coalesced.
            return self

        row = self.row()[mask]
        col = self._col[mask]

        value = self._value
        if value is not None:
            ptr = mask.nonzero().flatten()
            ptr = torch.cat([ptr, ptr.new_full((1, ), value.size(0))])
            value = segment_csr(value, ptr, reduce=reduce)

        return SparseStorage(row=row, rowptr=None, col=col, value=value,
                             sparse_sizes=self._sparse_sizes, rowcount=None,
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

    def cached_keys(self) -> List[str]:
        keys: List[str] = []
        if self.has_rowcount():
            keys.append('rowcount')
        if self.has_colptr():
            keys.append('colptr')
        if self.has_colcount():
            keys.append('colcount')
        if self.has_csr2csc():
            keys.append('csr2csc')
        if self.has_csc2csr():
            keys.append('csc2csr')
        return keys

    def num_cached_keys(self) -> int:
        return len(self.cached_keys())

    def copy(self):
        return SparseStorage(row=self._row, rowptr=self._rowptr, col=self._col,
                             value=self._value,
                             sparse_sizes=self._sparse_sizes,
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
        col = self._col.clone()
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

        return SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                             sparse_sizes=self._sparse_sizes,
                             rowcount=rowcount, colptr=colptr,
                             colcount=colcount, csr2csc=csr2csc,
                             csc2csr=csc2csr, is_sorted=True)

    def type_as(self, tensor: torch.Tensor):
        value = self._value
        if value is not None:
            if tensor.dtype == value.dtype:
                return self
            else:
                return self.set_value(value.type_as(tensor), layout='coo')
        else:
            return self

    def device_as(self, tensor: torch.Tensor, non_blocking: bool = False):
        if tensor.device == self._col.device:
            return self

        row = self._row
        if row is not None:
            row = row.to(tensor.device, non_blocking=non_blocking)
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.to(tensor.device, non_blocking=non_blocking)
        col = self._col.to(tensor.device, non_blocking=non_blocking)
        value = self._value
        if value is not None:
            value = value.to(tensor.device, non_blocking=non_blocking)
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.to(tensor.device, non_blocking=non_blocking)
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.to(tensor.device, non_blocking=non_blocking)
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.to(tensor.device, non_blocking=non_blocking)
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.to(tensor.device, non_blocking=non_blocking)
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.to(tensor.device, non_blocking=non_blocking)

        return SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                             sparse_sizes=self._sparse_sizes,
                             rowcount=rowcount, colptr=colptr,
                             colcount=colcount, csr2csc=csr2csc,
                             csc2csr=csc2csr, is_sorted=True)

    def cuda(self):
        new_col = self._col.cuda()
        if new_col.device == self._col.device:
            return self

        row = self._row
        if row is not None:
            row = row.cuda()
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.cuda()
        value = self._value
        if value is not None:
            value = value.cuda()
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.cuda()
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.cuda()
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.cuda()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.cuda()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.cuda()

        return SparseStorage(row=row, rowptr=rowptr, col=new_col, value=value,
                             sparse_sizes=self._sparse_sizes,
                             rowcount=rowcount, colptr=colptr,
                             colcount=colcount, csr2csc=csr2csc,
                             csc2csr=csc2csr, is_sorted=True)

    def pin_memory(self):
        row = self._row
        if row is not None:
            row = row.pin_memory()
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.pin_memory()
        col = self._col.pin_memory()
        value = self._value
        if value is not None:
            value = value.pin_memory()
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.pin_memory()
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.pin_memory()
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.pin_memory()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.pin_memory()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.pin_memory()

        return SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                             sparse_sizes=self._sparse_sizes,
                             rowcount=rowcount, colptr=colptr,
                             colcount=colcount, csr2csc=csr2csc,
                             csc2csr=csc2csr, is_sorted=True)

    def is_pinned(self) -> bool:
        is_pinned = True
        row = self._row
        if row is not None:
            is_pinned = is_pinned and row.is_pinned()
        rowptr = self._rowptr
        if rowptr is not None:
            is_pinned = is_pinned and rowptr.is_pinned()
        is_pinned = self._col.is_pinned()
        value = self._value
        if value is not None:
            is_pinned = is_pinned and value.is_pinned()
        rowcount = self._rowcount
        if rowcount is not None:
            is_pinned = is_pinned and rowcount.is_pinned()
        colptr = self._colptr
        if colptr is not None:
            is_pinned = is_pinned and colptr.is_pinned()
        colcount = self._colcount
        if colcount is not None:
            is_pinned = is_pinned and colcount.is_pinned()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            is_pinned = is_pinned and csr2csc.is_pinned()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            is_pinned = is_pinned and csc2csr.is_pinned()
        return is_pinned


def share_memory_(self) -> SparseStorage:
    row = self._row
    if row is not None:
        row.share_memory_()
    rowptr = self._rowptr
    if rowptr is not None:
        rowptr.share_memory_()
    self._col.share_memory_()
    value = self._value
    if value is not None:
        value.share_memory_()
    rowcount = self._rowcount
    if rowcount is not None:
        rowcount.share_memory_()
    colptr = self._colptr
    if colptr is not None:
        colptr.share_memory_()
    colcount = self._colcount
    if colcount is not None:
        colcount.share_memory_()
    csr2csc = self._csr2csc
    if csr2csc is not None:
        csr2csc.share_memory_()
    csc2csr = self._csc2csr
    if csc2csr is not None:
        csc2csr.share_memory_()


def is_shared(self) -> bool:
    is_shared = True
    row = self._row
    if row is not None:
        is_shared = is_shared and row.is_shared()
    rowptr = self._rowptr
    if rowptr is not None:
        is_shared = is_shared and rowptr.is_shared()
    is_shared = is_shared and self._col.is_shared()
    value = self._value
    if value is not None:
        is_shared = is_shared and value.is_shared()
    rowcount = self._rowcount
    if rowcount is not None:
        is_shared = is_shared and rowcount.is_shared()
    colptr = self._colptr
    if colptr is not None:
        is_shared = is_shared and colptr.is_shared()
    colcount = self._colcount
    if colcount is not None:
        is_shared = is_shared and colcount.is_shared()
    csr2csc = self._csr2csc
    if csr2csc is not None:
        is_shared = is_shared and csr2csc.is_shared()
    csc2csr = self._csc2csr
    if csc2csr is not None:
        is_shared = is_shared and csc2csr.is_shared()
    return is_shared


SparseStorage.share_memory_ = share_memory_
SparseStorage.is_shared = is_shared
