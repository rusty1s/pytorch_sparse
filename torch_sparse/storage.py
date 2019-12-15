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

        if colptr is None:
            ones = torch.ones_like(col) if ones is None else ones
            in_deg = scatter_add(ones, col, dim=0, dim_size=sparse_size[1])
            colptr = torch.cat([col.new_zeros(1), in_deg.cumsum(0)], dim=0)

        if arg_csr_to_csc is None:
            idx = sparse_size[0] * col + row
            arg_csr_to_csc = idx.argsort()

        if arg_csr_to_csc is None:
            arg_csc_to_csr = arg_csr_to_csc.argsort()

        self.__row = row
        self.__col = col
        self.__value = value
        self.__sparse_size = sparse_size
        self.__rowptr = rowptr
        self.__colptr = colptr
        self.__arg_csr_to_csc = arg_csr_to_csc
        self.__arg_csc_to_csr = arg_csc_to_csr

    @property
    def row(self):
        return self.__row

    @property
    def col(self):
        return self.__col

    def index(self):
        return torch.stack([self.__row, self.__col], dim=0)

    @property
    def rowptr(self):
        return self.__rowptr

    @property
    def colptr(self):
        return self.__colptr

    @property
    def arg_csr_to_csc(self):
        return self.__arg_csr_to_csc

    @property
    def arg_csc_to_csr(self):
        return self.__arg_csc_to_csr

    @property
    def value(self):
        return self.__value

    @property
    def has_value(self):
        return self.__value is not None

    def sparse_size(self, dim=None):
        return self.__sparse_size if dim is None else self.__sparse_size[dim]

    def size(self, dim=None):
        size = self.__sparse_size
        size += () if self.has_value is None else self.__value.size()[1:]
        return size if dim is None else size[dim]

    @property
    def shape(self):
        return self.size()

    def sparse_resize_(self, *sizes):
        assert len(sizes) == 2
        self.__sparse_size == sizes

    def clone(self):
        raise NotImplementedError

    def copy_(self):
        raise NotImplementedError

    def pin_memory(self):
        raise NotImplementedError

    def is_pinned(self):
        raise NotImplementedError

    def share_memory_(self):
        raise NotImplementedError

    def is_shared(self):
        raise NotImplementedError

    @property
    def device(self):
        return self.__row.device

    def cpu(self):
        pass

    def cuda(device=None, non_blocking=False, **kwargs):
        pass

    @property
    def is_cuda(self):
        pass

    @property
    def dtype(self):
        pass

    def type(dtype=None, non_blocking=False, **kwargs):
        pass

    def is_floating_point(self):
        pass

    def bfloat16(self):
        pass

    def bool(self):
        pass

    def byte(self):
        pass

    def char(self):
        pass

    def half(self):
        pass

    def float(self):
        pass

    def double(self):
        pass

    def short(self):
        pass

    def int(self):
        pass

    def long(self):
        pass

    def __apply_index(self, func):
        pass

    def __apply_index_(self, func):
        self.__row = func(self.__row)
        self.__col = func(self.__col)
        self.__rowptr = func(self.__rowptr)
        self.__colptr = func(self.__colptr)
        self.__arg_csr_to_csc = func(self.__arg_csr_to_csc)
        self.__arg_csc_to_csr = func(self.__arg_csc_to_csr)

    def __apply_value(self, func):
        pass

    def __apply_value_(self, func):
        self.__value = func(self.__value) if self.has_value else None

    def __apply(self, func):
        pass

    def __apply_(self, func):
        self.__apply_index_(func)
        self.__apply_value_(func)


if __name__ == '__main__':
    from torch_geometric.datasets import Reddit  # noqa
    import time  # noqa

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = Reddit('/tmp/Reddit')
    data = dataset[0].to(device)
    edge_index = data.edge_index
    row, col = edge_index

    storage = SparseStorage(row, col)
    # idx = data.num_nodes * col + row
    # perm = idx.argsort()
    # row, col = row[perm], col[perm]
    # print(row[:20])
    # print(col[:20])
    # print('--------')

    # perm = perm.argsort()
    # row, col = row[perm], col[perm]
    # print(row[:20])
    # print(col[:20])
