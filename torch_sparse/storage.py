import inspect

import torch
from torch import Size
from torch_scatter import scatter_add, segment_add


class SparseStorage(object):
    def __init__(self, row, col, value=None, sparse_size=None, rowptr=None,
                 colptr=None, arg_csr_to_csc=None, arg_csc_to_csr=None,
                 is_sorted=False):

        assert row.dtype == torch.long and col.dtype == torch.long
        assert row.device == row.device
        assert row.dim() == 1 and col.dim() == 1 and row.numel() == col.numel()

        if sparse_size is None:
            sparse_size = Size((row.max().item() + 1, col.max().item() + 1))

        if not is_sorted:
            idx = sparse_size[1] * row + col
            # Only sort if necessary...
            if (idx <= torch.cat([idx.new_zeros(1), idx[:-1]], dim=0)).any():
                perm = idx.argsort()
                row = row[perm]
                col = col[perm]
                value = None if value is None else value[perm]
                rowptr = None
                colptr = None
                arg_csr_to_csc = None
                arg_csc_to_csr = None

        if value is not None:
            assert row.device == value.device and value.size(0) == row.size(0)
            value = value.contiguous()

        ones = None
        if rowptr is None:
            ones = torch.ones_like(row)
            out_deg = segment_add(ones, row, dim=0, dim_size=sparse_size[0])
            rowptr = torch.cat([row.new_zeros(1), out_deg.cumsum(0)], dim=0)
        else:
            assert rowptr.dtype == torch.long and rowptr.device == row.device
            assert rowptr.dim() == 1 and rowptr.numel() - 1 == sparse_size[0]

        if colptr is None:
            ones = torch.ones_like(col) if ones is None else ones
            in_deg = scatter_add(ones, col, dim=0, dim_size=sparse_size[1])
            colptr = torch.cat([col.new_zeros(1), in_deg.cumsum(0)], dim=0)
        else:
            assert colptr.dtype == torch.long and colptr.device == col.device
            assert colptr.dim() == 1 and colptr.numel() - 1 == sparse_size[1]

        if arg_csr_to_csc is None:
            idx = sparse_size[0] * col + row
            arg_csr_to_csc = idx.argsort()
        else:
            assert arg_csr_to_csc.dtype == torch.long
            assert arg_csr_to_csc.device == row.device
            assert arg_csr_to_csc.dim() == 1
            assert arg_csr_to_csc.numel() == row.numel()

        if arg_csc_to_csr is None:
            arg_csc_to_csr = arg_csr_to_csc.argsort()
        else:
            assert arg_csc_to_csr.dtype == torch.long
            assert arg_csc_to_csr.device == row.device
            assert arg_csc_to_csr.dim() == 1
            assert arg_csc_to_csr.numel() == row.numel()

        self.__row = row
        self.__col = col
        self.__value = value
        self.__sparse_size = sparse_size
        self.__rowptr = rowptr
        self.__colptr = colptr
        self.__arg_csr_to_csc = arg_csr_to_csc
        self.__arg_csc_to_csr = arg_csc_to_csr

    @property
    def _row(self):
        return self.__row

    @property
    def _col(self):
        return self.__col

    @property
    def _index(self):
        return torch.stack([self.__row, self.__col], dim=0)

    @property
    def _rowptr(self):
        return self.__rowptr

    @property
    def _colptr(self):
        return self.__colptr

    @property
    def _arg_csr_to_csc(self):
        return self.__arg_csr_to_csc

    @property
    def _arg_csc_to_csr(self):
        return self.__arg_csc_to_csr

    @property
    def _value(self):
        return self.__value

    @property
    def has_value(self):
        return self.__value is not None

    def sparse_size(self, dim=None):
        return self.__sparse_size if dim is None else self.__sparse_size[dim]

    def size(self, dim=None):
        size = self.__sparse_size
        size += () if self.__value is None else self.__value.size()[1:]
        return size if dim is None else size[dim]

    def dim(self):
        return len(self.size())

    @property
    def shape(self):
        return self.size()

    def sparse_resize_(self, *sizes):
        assert len(sizes) == 2
        self.__sparse_size == sizes
        return self

    def nnz(self):
        return self.__row.size(0)

    def density(self):
        return self.nnz() / (self.__sparse_size[0] * self.__sparse_size[1])

    def sparsity(self):
        return 1 - self.density()

    def avg_row_length(self):
        return self.nnz() / self.__sparse_size[0]

    def avg_col_length(self):
        return self.nnz() / self.__sparse_size[1]

    def numel(self):
        return self.nnz() if self.__value is None else self.__value.numel()

    def clone(self):
        return self._apply(lambda x: x.clone())

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('SparseStorage', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def pin_memory(self):
        return self._apply(lambda x: x.pin_memory())

    def is_pinned(self):
        return all([x.is_pinned for x in self.__attributes])

    def share_memory_(self):
        return self._apply_(lambda x: x.share_memory_())

    def is_shared(self):
        return all([x.is_shared for x in self.__attributes])

    @property
    def device(self):
        return self.__row.device

    def cpu(self):
        return self._apply(lambda x: x.cpu())

    def cuda(self, device=None, non_blocking=False, **kwargs):
        return self._apply(lambda x: x.cuda(device, non_blocking, **kwargs))

    @property
    def is_cuda(self):
        return self.__row.is_cuda

    @property
    def dtype(self):
        return None if self.__value is None else self.__value.dtype

    def to(self, *args, **kwargs):
        if 'device' in kwargs:
            out = self._apply(lambda x: x.to(kwargs['device'], **kwargs))
            del kwargs['device']

        for arg in args[:]:
            if isinstance(arg, str) or isinstance(arg, torch.device):
                out = self._apply(lambda x: x.to(arg, **kwargs))
                args.remove(arg)

        if len(args) > 0 and len(kwargs) > 0:
            out = self.type(*args, **kwargs)

        return out

    def type(self, dtype=None, non_blocking=False, **kwargs):
        return self.dtype if dtype is None else self._apply_value(
            lambda x: x.type(dtype, non_blocking, **kwargs))

    def is_floating_point(self):
        return self.__value is None or torch.is_floating_point(self.__value)

    def bfloat16(self):
        return self._apply_value(lambda x: x.bfloat16())

    def bool(self):
        return self._apply_value(lambda x: x.bool())

    def byte(self):
        return self._apply_value(lambda x: x.byte())

    def char(self):
        return self._apply_value(lambda x: x.char())

    def half(self):
        return self._apply_value(lambda x: x.half())

    def float(self):
        return self._apply_value(lambda x: x.float())

    def double(self):
        return self._apply_value(lambda x: x.double())

    def short(self):
        return self._apply_value(lambda x: x.short())

    def int(self):
        return self._apply_value(lambda x: x.int())

    def long(self):
        return self._apply_value(lambda x: x.long())

    def __state(self):
        return {
            key: getattr(self, f'_{self.__class__.__name__}__{key}')
            for key in inspect.getfullargspec(self.__init__)[0][1:-1]
        }

    def _apply_value(self, func):
        if self.__value is None:
            return self

        state = self.__state()
        state['value'] == func(self.__value)
        return self.__class__(is_sorted=True, **state)

    def _apply_value_(self, func):
        self.__value = None if self.__value is None else func(self.__value)
        return self

    def _apply(self, func):
        state = self.__state().items()
        state = {k: func(v) if torch.is_tensor(v) else v for k, v in state}
        return self.__class__(is_sorted=True, **state)

    def _apply_(self, func):
        for k, v in self.__state().items():
            v = func(v) if torch.is_tensor(v) else v
            setattr(self, f'_{self.__class__.__name__}__{k}', v)
        return self


if __name__ == '__main__':
    from torch_geometric.datasets import Reddit  # noqa
    import time  # noqa

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = Reddit('/tmp/Reddit')
    data = dataset[0].to(device)
    edge_index = data.edge_index
    row, col = edge_index

    storage = SparseStorage(row, col)
