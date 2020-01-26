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
from torch_sparse.diag import remove_diag, set_diag
from torch_sparse.matmul import matmul
from torch_sparse.add import add, add_, add_nnz, add_nnz_
from torch_sparse.mul import mul, mul_, mul_nnz, mul_nnz_


class SparseTensor(object):
    def __init__(self, row=None, rowptr=None, col=None, value=None,
                 sparse_size=None, is_sorted=False):
        self.storage = SparseStorage(row=row, rowptr=rowptr, col=col,
                                     value=value, sparse_size=sparse_size,
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

        row, col = index.t().contiguous()
        return SparseTensor(row=row, col=col, value=mat[row, col],
                            sparse_size=mat.size()[:2], is_sorted=True)

    @classmethod
    def from_torch_sparse_coo_tensor(self, mat, is_sorted=False):
        row, col = mat._indices()
        return SparseTensor(row=row, col=col, value=mat._values(),
                            sparse_size=mat.size()[:2], is_sorted=is_sorted)

    @classmethod
    def from_scipy(self, mat):
        colptr = None
        if isinstance(mat, scipy.sparse.csc_matrix):
            colptr = torch.from_numpy(mat.indptr).to(torch.long)

        mat = mat.tocsr()  # Pre-sort.
        rowptr = torch.from_numpy(mat.indptr).to(torch.long)
        mat = mat.tocoo()
        row = torch.from_numpy(mat.row).to(torch.long)
        col = torch.from_numpy(mat.col).to(torch.long)
        value = torch.from_numpy(mat.data)
        sparse_size = mat.shape[:2]

        storage = SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                                sparse_size=sparse_size, colptr=colptr,
                                is_sorted=True)

        return SparseTensor.from_storage(storage)

    @classmethod
    def eye(self, M, N=None, device=None, dtype=None, has_value=True,
            fill_cache=False):
        N = M if N is None else N

        row = torch.arange(min(M, N), device=device)
        rowptr = torch.arange(M + 1, device=device)
        if M > N:
            rowptr[row.size(0) + 1:] = row.size(0)
        col = row

        value = None
        if has_value:
            value = torch.ones(row.size(0), dtype=dtype, device=device)

        rowcount = colptr = colcount = csr2csc = csc2csr = None
        if fill_cache:
            rowcount = row.new_ones(M)
            if M > N:
                rowcount[row.size(0):] = 0
            colptr = torch.arange(N + 1, device=device)
            colcount = col.new_ones(N)
            if N > M:
                colptr[col.size(0) + 1:] = col.size(0)
                colcount[col.size(0):] = 0
            csr2csc = csc2csr = row

        storage = SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                                sparse_size=torch.Size([M, N]),
                                rowcount=rowcount, colptr=colptr,
                                colcount=colcount, csr2csc=csr2csc,
                                csc2csr=csc2csr, is_sorted=True)
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
        return self.storage.row, self.storage.col, self.storage.value

    def csr(self):
        return self.storage.rowptr, self.storage.col, self.storage.value

    def csc(self):
        perm = self.storage.csr2csc  # Compute `csr2csc` first.
        return (self.storage.colptr, self.storage.row[perm],
                self.storage.value[perm] if self.has_value() else None)

    # Storage inheritance #####################################################

    def has_value(self):
        return self.storage.has_value()

    def set_value_(self, value, layout=None, dtype=None):
        self.storage.set_value_(value, layout, dtype)
        return self

    def set_value(self, value, layout=None, dtype=None):
        return self.from_storage(self.storage.set_value(value, layout, dtype))

    def sparse_size(self, dim=None):
        sparse_size = self.storage.sparse_size
        return sparse_size if dim is None else sparse_size[dim]

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

    def dim(self):
        return len(self.size())

    def size(self, dim=None):
        size = self.sparse_size()
        size += self.storage.value.size()[1:] if self.has_value() else ()
        return size if dim is None else size[dim]

    @property
    def shape(self):
        return self.size()

    def nnz(self):
        return self.storage.col.numel()

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

        rowptr, col, value1 = self.csr()
        colptr, row, value2 = self.csc()

        if (rowptr != colptr).any() or (col != row).any():
            return False

        if not self.has_value():
            return True

        return (value1 == value2).all().item()

    def detach_(self):
        self.storage.apply_(lambda x: x.detach_())
        return self

    def detach(self):
        return self.from_storage(self.storage.apply(lambda x: x.detach()))

    @property
    def requires_grad(self):
        return self.storage.value.requires_grad if self.has_value() else False

    def requires_grad_(self, requires_grad=True, dtype=None):
        if requires_grad and not self.has_value():
            self.storage.set_value_(1, dtype=dtype)

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
        return self.storage.col.device

    def cpu(self):
        return self.from_storage(self.storage.apply(lambda x: x.cpu()))

    def cuda(self, device=None, non_blocking=False, **kwargs):
        storage = self.storage.apply(
            lambda x: x.cuda(device, non_blocking, **kwargs))
        return self.from_storage(storage)

    @property
    def is_cuda(self):
        return self.storage.col.is_cuda

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
        args = list(args)

        non_blocking = getattr(kwargs, 'non_blocking', False)

        storage = None
        if 'device' in kwargs:
            device = kwargs['device']
            del kwargs['device']
            storage = self.storage.apply(
                lambda x: x.to(device, non_blocking=non_blocking))
        else:
            for arg in args[:]:
                if isinstance(arg, str) or isinstance(arg, torch.device):
                    storage = self.storage.apply(
                        lambda x: x.to(arg, non_blocking=non_blocking))
                    args.remove(arg)

        storage = self.storage if storage is None else storage

        if len(args) > 0 or len(kwargs) > 0:
            storage = storage.apply_value(lambda x: x.type(*args, **kwargs))

        if storage == self.storage:  # Nothing has been changed...
            return self
        else:
            return self.from_storage(storage)

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
        row, col, value = self.coo()
        mat = torch.zeros(self.size(), dtype=dtype, device=self.device)
        mat[row, col] = value if self.has_value() else 1
        return mat

    def to_torch_sparse_coo_tensor(self, dtype=None, requires_grad=False):
        row, col, value = self.coo()
        index = torch.stack([row, col], dim=0)
        if value is None:
            value = torch.ones(self.nnz(), dtype=dtype, device=self.device)
        return torch.sparse_coo_tensor(index, value, self.size(),
                                       device=self.device,
                                       requires_grad=requires_grad)

    def to_scipy(self, layout=None, dtype=None):
        assert self.dim() == 2
        layout = get_layout(layout)

        if not self.has_value():
            ones = torch.ones(self.nnz(), dtype=dtype).numpy()

        if layout == 'coo':
            row, col, value = self.coo()
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
            raise SyntaxError

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

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add_(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __imul__(self, other):
        return self.mul_(other)

    def __matmul__(self, other):
        return matmul(self, other, reduce='sum')

    # String Reputation #######################################################

    def __repr__(self):
        i = ' ' * 6
        row, col, value = self.coo()
        infos = []
        infos += [f'row={indent(row.__repr__(), i)[len(i):]}']
        infos += [f'col={indent(col.__repr__(), i)[len(i):]}']

        if self.has_value():
            infos += [f'val={indent(value.__repr__(), i)[len(i):]}']

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
SparseTensor.set_diag = set_diag
SparseTensor.matmul = matmul
SparseTensor.add = add
SparseTensor.add_ = add_
SparseTensor.add_nnz = add_nnz
SparseTensor.add_nnz_ = add_nnz_
SparseTensor.mul = mul
SparseTensor.mul_ = mul_
SparseTensor.mul_nnz = mul_nnz
SparseTensor.mul_nnz_ = mul_nnz_

# Fix for PyTorch<=1.3 (https://github.com/pytorch/pytorch/pull/31769):
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if (TORCH_MAJOR <= 1) or (TORCH_MAJOR == 1 and TORCH_MINOR < 4):

    def add(self, other):
        return self.add(other) if torch.is_tensor(other) else NotImplemented

    def mul(self, other):
        return self.mul(other) if torch.is_tensor(other) else NotImplemented

    torch.Tensor.__add__ = add
    torch.Tensor.__mul__ = add
