from textwrap import indent

import torch
import scipy.sparse

from torch_sparse.storage import SparseStorage, get_layout

from torch_sparse.transpose import t
from torch_sparse.narrow import narrow
from torch_sparse.select import select
from torch_sparse.index_select import index_select, index_select_nnz
from torch_sparse.masked_select import masked_select, masked_select_nnz
import torch_sparse.reduce
from torch_sparse.diag import remove_diag


class SparseTensor(object):
    def __init__(self, index, value=None, sparse_size=None, is_sorted=False):
        self.storage = SparseStorage(index, value, sparse_size,
                                     is_sorted=is_sorted)

    @classmethod
    def from_storage(self, storage):
        self = SparseTensor.__new__(SparseTensor)
        self.storage = storage
        return self

    @classmethod
    def from_dense(self, mat):
        if mat.dim() > 2:
            index = mat.abs().sum([i for i in range(2, mat.dim())]).nonzero()
        else:
            index = mat.nonzero()

        index = index.t().contiguous()
        value = mat[index[0], index[1]]
        return SparseTensor(index, value, mat.size()[:2], is_sorted=True)

    @classmethod
    def from_torch_sparse_coo_tensor(self, mat, is_sorted=False):
        return SparseTensor(mat._indices(), mat._values(),
                            mat.size()[:2], is_sorted=is_sorted)

    @classmethod
    def from_scipy(self, mat):
        colptr = None
        if isinstance(mat, scipy.sparse.csc_matrix):
            colptr = torch.from_numpy(mat.indptr).to(torch.long)

        mat = mat.tocsr()
        rowptr = torch.from_numpy(mat.indptr).to(torch.long)
        mat = mat.tocoo()
        row = torch.from_numpy(mat.row).to(torch.long)
        col = torch.from_numpy(mat.col).to(torch.long)
        index = torch.stack([row, col], dim=0)
        value = torch.from_numpy(mat.data)
        size = mat.shape

        storage = SparseStorage(index, value, size, rowptr=rowptr,
                                colptr=colptr, is_sorted=True)

        return SparseTensor.from_storage(storage)

    @classmethod
    def eye(self, M, N=None, device=None, dtype=None, no_value=False,
            fill_cache=False):
        N = M if N is None else N

        index = torch.empty((2, min(M, N)), dtype=torch.long, device=device)
        torch.arange(index.size(1), out=index[0])
        torch.arange(index.size(1), out=index[1])

        value = None
        if not no_value:
            value = torch.ones(index.size(1), dtype=dtype, device=device)

        rowcount = rowptr = colcount = colptr = csr2csc = csc2csr = None
        if fill_cache:
            rowcount = index.new_ones(M)
            rowptr = torch.arange(M + 1, device=device)
            if M > N:
                rowcount[index.size(1):] = 0
                rowptr[index.size(1) + 1:] = index.size(1)
            colcount = index.new_ones(N)
            colptr = torch.arange(N + 1, device=device)
            if N > M:
                colcount[index.size(1):] = 0
                colptr[index.size(1) + 1:] = index.size(1)
            csr2csc = torch.arange(index.size(1), device=device)
            csc2csr = torch.arange(index.size(1), device=device)

        storage = SparseStorage(
            index,
            value,
            torch.Size([M, N]),
            rowcount=rowcount,
            rowptr=rowptr,
            colcount=colcount,
            colptr=colptr,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
        )
        return SparseTensor.from_storage(storage)

    def __copy__(self):
        return self.from_storage(self.storage)

    def clone(self):
        return self.from_storage(self.storage.clone())

    def __deepcopy__(self, memo):
        new_sparse_tensor = self.clone()
        memo[id(self)] = new_sparse_tensor
        return new_sparse_tensor

    # Formats #################################################################

    def coo(self):
        return self.storage.index, self.storage.value

    def csr(self):
        return self.storage.rowptr, self.storage.col, self.storage.value

    def csc(self):
        perm = self.storage.csr2csc  # Compute `csr2csc` first.
        return (self.storage.colptr, self.storage.row[perm],
                self.storage.value[perm] if self.has_value() else None)

    # Storage inheritance #####################################################

    def has_value(self):
        return self.storage.has_value()

    def set_value_(self, value, layout=None):
        self.storage.set_value_(value, layout)
        return self

    def set_value(self, value, layout=None):
        return self.from_storage(self.storage.set_value(value, layout))

    def sparse_size(self, dim=None):
        return self.storage.sparse_size(dim)

    def sparse_resize(self, *sizes):
        return self.from_storage(self.storage.sparse_resize(*sizes))

    def is_coalesced(self):
        return self.storage.is_coalesced()

    def coalesce(self, reduce='add'):
        return self.from_storage(self.storage.coalesce(reduce))

    def cached_keys(self):
        return self.storage.cached_keys()

    def fill_cache_(self, *args):
        self.storage.fill_cache_(*args)
        return self

    def clear_cache_(self, *args):
        self.storage.clear_cache_(*args)
        return self

    # Utility functions #######################################################

    def size(self, dim=None):
        size = self.sparse_size()
        size += self.storage.value.size()[1:] if self.has_value() else ()
        return size if dim is None else size[dim]

    def dim(self):
        return len(self.size())

    @property
    def shape(self):
        return self.size()

    def nnz(self):
        return self.storage.index.size(1)

    def density(self):
        return self.nnz() / (self.sparse_size(0) * self.sparse_size(1))

    def sparsity(self):
        return 1 - self.density()

    def avg_row_length(self):
        return self.nnz() / self.sparse_size(0)

    def avg_col_length(self):
        return self.nnz() / self.sparse_size(1)

    def numel(self):
        return self.value.numel() if self.has_value() else self.nnz()

    def is_quadratic(self):
        return self.sparse_size(0) == self.sparse_size(1)

    def is_symmetric(self):
        if not self.is_quadratic:
            return False

        rowptr, col, val1 = self.csr()
        colptr, row, val2 = self.csc()
        index_sym = (rowptr == colptr).all() and (col == row).all()
        value_sym = (val1 == val2).all().item() if self.has_value() else True
        return index_sym.item() and value_sym

    def detach_(self):
        self.storage.apply_(lambda x: x.detach_())
        return self

    def detach(self):
        return self.from_storage(self.storage.apply(lambda x: x.detach()))

    @property
    def requires_grad(self):
        return self.storage.value.requires_grad if self.has_value() else False

    def requires_grad_(self, requires_grad=True):
        if self.has_value():
            self.storage.value.requires_grad_(requires_grad)
        return self

    def pin_memory(self):
        return self.from_storage(self.storage.apply(lambda x: x.pin_memory()))

    def is_pinned(self):
        return all(self.storage.map(lambda x: x.is_pinned()))

    def share_memory_(self):
        self.storage.apply_(lambda x: x.share_memory_())
        return self

    def is_shared(self):
        return all(self.storage.map(lambda x: x.is_shared()))

    @property
    def device(self):
        return self.storage.index.device

    def cpu(self):
        return self.from_storage(self.storage.apply(lambda x: x.cpu()))

    def cuda(self, device=None, non_blocking=False, **kwargs):
        storage = self.storage.apply(
            lambda x: x.cuda(device, non_blocking, **kwargs))
        return self.from_storage(storage)

    @property
    def is_cuda(self):
        return self.storage.index.is_cuda

    @property
    def dtype(self):
        return self.storage.value.dtype if self.has_value() else None

    def is_floating_point(self):
        value = self.storage.value
        return self.has_value() and torch.is_floating_point(value)

    def type(self, dtype=None, non_blocking=False, **kwargs):
        if dtype is None:
            return self.dtype

        if dtype == self.dtype:
            return self

        storage = self.storage.apply_value(
            lambda x: x.type(dtype, non_blocking, **kwargs))

        return self.from_storage(storage)

    def to(self, *args, **kwargs):
        storage = None

        if 'device' in kwargs:
            device = kwargs['device']
            del kwargs['device']
            storage = self.storage.apply(lambda x: x.to(
                device, non_blocking=getattr(kwargs, 'non_blocking', False)))

        for arg in args[:]:
            if isinstance(arg, str) or isinstance(arg, torch.device):
                storage = self.storage.apply(lambda x: x.to(
                    arg, non_blocking=getattr(kwargs, 'non_blocking', False)))
                args.remove(arg)

        if storage is not None:
            self = self.from_storage(storage)

        if len(args) > 0 or len(kwargs) > 0:
            self = self.type(*args, **kwargs)

        return self

    def bfloat16(self):
        return self.type(torch.bfloat16)

    def bool(self):
        return self.type(torch.bool)

    def byte(self):
        return self.type(torch.byte)

    def char(self):
        return self.type(torch.char)

    def half(self):
        return self.type(torch.half)

    def float(self):
        return self.type(torch.float)

    def double(self):
        return self.type(torch.double)

    def short(self):
        return self.type(torch.short)

    def int(self):
        return self.type(torch.int)

    def long(self):
        return self.type(torch.long)

    # Conversions #############################################################

    def to_dense(self, dtype=None):
        dtype = dtype or self.dtype
        (row, col), value = self.coo()
        mat = torch.zeros(self.size(), dtype=dtype, device=self.device)
        mat[row, col] = value if self.has_value() else 1
        return mat

    def to_torch_sparse_coo_tensor(self, dtype=None, requires_grad=False):
        index, value = self.coo()
        return torch.sparse_coo_tensor(
            index, value if self.has_value() else torch.ones(
                self.nnz(), dtype=dtype, device=self.device), self.size(),
            device=self.device, requires_grad=requires_grad)

    def to_scipy(self, dtype=None, layout=None):
        assert self.dim() == 2
        layout = get_layout(layout)

        if not self.has_value():
            ones = torch.ones(self.nnz(), dtype=dtype).numpy()

        if layout == 'coo':
            (row, col), value = self.coo()
            row = row.detach().cpu().numpy()
            col = col.detach().cpu().numpy()
            value = value.detach().cpu().numpy() if self.has_value() else ones
            return scipy.sparse.coo_matrix((value, (row, col)), self.size())
        elif layout == 'csr':
            rowptr, col, value = self.csr()
            rowptr = rowptr.detach().cpu().numpy()
            col = col.detach().cpu().numpy()
            value = value.detach().cpu().numpy() if self.has_value() else ones
            return scipy.sparse.csr_matrix((value, col, rowptr), self.size())
        elif layout == 'csc':
            colptr, row, value = self.csc()
            colptr = colptr.detach().cpu().numpy()
            row = row.detach().cpu().numpy()
            value = value.detach().cpu().numpy() if self.has_value() else ones
            return scipy.sparse.csc_matrix((value, row, colptr), self.size())

    # Standard Operators ######################################################

    def __getitem__(self, index):
        index = list(index) if isinstance(index, tuple) else [index]
        # More than one `Ellipsis` is not allowed...
        if len([i for i in index if not torch.is_tensor(i) and i == ...]) > 1:
            raise SyntaxError()

        dim = 0
        out = self
        while len(index) > 0:
            item = index.pop(0)
            if isinstance(item, int):
                out = out.select(dim, item)
                dim += 1
            elif isinstance(item, slice):
                if item.step is not None:
                    raise ValueError('Step parameter not yet supported.')

                start = 0 if item.start is None else item.start
                start = self.size(dim) + start if start < 0 else start

                stop = self.size(dim) if item.stop is None else item.stop
                stop = self.size(dim) + stop if stop < 0 else stop

                out = out.narrow(dim, start, max(stop - start, 0))
                dim += 1
            elif torch.is_tensor(item):
                if item.dtype == torch.bool:
                    out = out.masked_select(dim, item)
                    dim += 1
                elif item.dtype == torch.long:
                    out = out.index_select(dim, item)
                    dim += 1
            elif item == Ellipsis:
                if self.dim() - len(index) < dim:
                    raise SyntaxError()
                dim = self.dim() - len(index)
            else:
                raise SyntaxError()

        return out

    # String Reputation #######################################################

    def __repr__(self):
        i = ' ' * 6
        index, value = self.coo()
        infos = [f'index={indent(index.__repr__(), i)[len(i):]}']

        if self.has_value():
            infos += [f'value={indent(value.__repr__(), i)[len(i):]}']

        infos += [
            f'size={tuple(self.size())}, '
            f'nnz={self.nnz()}, '
            f'density={100 * self.density():.02f}%'
        ]
        infos = ',\n'.join(infos)

        i = ' ' * (len(self.__class__.__name__) + 1)
        return f'{self.__class__.__name__}({indent(infos, i)[len(i):]})'


# Bindings ####################################################################

SparseTensor.t = t
SparseTensor.narrow = narrow
SparseTensor.select = select
SparseTensor.index_select = index_select
SparseTensor.index_select_nnz = index_select_nnz
SparseTensor.masked_select = masked_select
SparseTensor.masked_select_nnz = masked_select_nnz
SparseTensor.reduction = torch_sparse.reduce.reduction
SparseTensor.sum = torch_sparse.reduce.sum
SparseTensor.mean = torch_sparse.reduce.mean
SparseTensor.min = torch_sparse.reduce.min
SparseTensor.max = torch_sparse.reduce.max
SparseTensor.remove_diag = remove_diag
# SparseTensor.add = add
# SparseTensor.add_nnz = add_nnz

# def remove_diag(self):
#     raise NotImplementedError

#     def set_diag(self, value):
#         raise NotImplementedError

#     def __reduce(self, dim, reduce, only_nnz):
#         raise NotImplementedError

#     def sum(self, dim):
#         return self.__reduce(dim, reduce='add', only_nnz=True)

#     def prod(self, dim):
#         return self.__reduce(dim, reduce='mul', only_nnz=True)

#     def min(self, dim, only_nnz=False):
#         return self.__reduce(dim, reduce='min', only_nnz=only_nnz)

#     def max(self, dim, only_nnz=False):
#         return self.__reduce(dim, reduce='min', only_nnz=only_nnz)

#     def mean(self, dim, only_nnz=False):
#         return self.__reduce(dim, reduce='mean', only_nnz=only_nnz)

#     def matmul(self, mat, reduce='add'):
#         assert self.numel() == self.nnz()  # Disallow multi-dimensional value
#         if torch.is_tensor(mat):
#             raise NotImplementedError
#         elif isinstance(mat, self.__class__):
#             assert reduce == 'add'
#           assert mat.numel() == mat.nnz()  # Disallow multi-dimensional value
#             raise NotImplementedError
#         raise ValueError('Argument needs to be of type `torch.tensor` or '
#                          'type `torch_sparse.SparseTensor`.')

#     def __add__(self, other):
#         return self.add(other)

#     def __radd__(self, other):
#         return self.add(other)

#     def sub(self, layout=None):
#         raise NotImplementedError

#     def sub_(self, layout=None):
#         raise NotImplementedError

#     def mul(self, layout=None):
#         raise NotImplementedError

#     def mul_(self, layout=None):
#         raise NotImplementedError

#     def div(self, layout=None):
#         raise NotImplementedError

#     def div_(self, layout=None):
#         raise NotImplementedError
