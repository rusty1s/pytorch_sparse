# import warnings
# import inspect
# from textwrap import indent

# import torch

# from torch_sparse.storage import SparseStorage

# methods = list(zip(*inspect.getmembers(SparseStorage)))[0]
# methods = [name for name in methods if '__' not in name and name != 'clone']

# def __is_scalar__(x):
#     return isinstance(x, int) or isinstance(x, float)

# class SparseTensor(object):
#     def __init__(self, index, value=None, sparse_size=None, is_sorted=False):
#         assert index.dim() == 2 and index.size(0) == 2
#         self._storage = SparseStorage(index[0], index[1], value, sparse_size,
#                                       is_sorted=is_sorted)

#     @classmethod
#     def from_storage(self, storage):
#         self = SparseTensor.__new__(SparseTensor)
#         self._storage = storage
#         return self

#     @classmethod
#     def from_dense(self, mat):
#         if mat.dim() > 2:
#             index = mat.abs().sum([i for i in range(2, mat.dim())]).nonzero()
#         else:
#             index = mat.nonzero()

#         index = index.t().contiguous()
#         value = mat[index[0], index[1]]
#         return SparseTensor(index, value, mat.size()[:2], is_sorted=True)

#     @property
#     def _storage(self):
#         return self.__storage

#     @_storage.setter
#     def _storage(self, storage):
#         self.__storage = storage
#         for name in methods:
#             setattr(self, name, getattr(storage, name))

#     def clone(self):
#         return SparseTensor.from_storage(self._storage.clone())

#     def __copy__(self):
#         return self.clone()

#     def __deepcopy__(self, memo):
#         memo = memo.setdefault('SparseStorage', {})
#         if self._cdata in memo:
#             return memo[self._cdata]
#         new_sparse_tensor = self.clone()
#         memo[self._cdata] = new_sparse_tensor
#         return new_sparse_tensor

#     def coo(self):
#         return self._index, self._value

#     def csr(self):
#         return self._rowptr, self._col, self._value

#     def csc(self):
#         perm = self._arg_csr_to_csc
#         return self._colptr, self._row[perm], self._value[perm]

#     def is_quadratic(self):
#         return self.sparse_size[0] == self.sparse_size[1]

#     def is_symmetric(self):
#         if not self.is_quadratic:
#             return False

#         index1, value1 = self.coo()
#         index2, value2 = self.t().coo()
#         index_symmetric = (index1 == index2).all()
#         value_symmetric = (value1 == value2).all() if self.has_value else True
#         return index_symmetric and value_symmetric

#     def set_value(self, value, layout=None):
#         if layout is None:
#             layout = 'coo'
#             warnings.warn('`layout` argument unset, using default layout '
#                           '"coo". This may lead to unexpected behaviour.')
#         assert layout in ['coo', 'csr', 'csc']
#         if value is not None and layout == 'csc':
#             value = value[self._arg_csc_to_csr]
#         return self._apply_value(value)

#     def set_value_(self, value, layout=None):
#         if layout is None:
#             layout = 'coo'
#             warnings.warn('`layout` argument unset, using default layout '
#                           '"coo". This may lead to unexpected behaviour.')
#         assert layout in ['coo', 'csr', 'csc']
#         if value is not None and layout == 'csc':
#             value = value[self._arg_csc_to_csr]
#         return self._apply_value_(value)

#     def set_diag(self, value):
#         raise NotImplementedError

#     def t(self):
#         storage = SparseStorage(
#             self._col[self._arg_csr_to_csc],
#             self._row[self._arg_csr_to_csc],
#             self._value[self._arg_csr_to_csc] if self.has_value else None,
#             self.sparse_size()[::-1],
#             self._colptr,
#             self._rowptr,
#             self._arg_csc_to_csr,
#             self._arg_csr_to_csc,
#             is_sorted=True,
#         )
#         return self.__class__.from_storage(storage)

#     def coalesce(self, reduce='add'):
#         raise NotImplementedError

#     def is_coalesced(self):
#         raise NotImplementedError

#     def masked_select(self, mask):
#         raise NotImplementedError

#     def index_select(self, index):
#         raise NotImplementedError

#     def select(self, dim, index):
#         raise NotImplementedError

#     def filter(self, index):
#         assert self.is_symmetric
#         assert index.dtype == torch.long or index.dtype == torch.bool
#         raise NotImplementedError

#     def permute(self, index):
#         assert index.dtype == torch.long
#         return self.filter(index)

#     def __getitem__(self, idx):
#         # Convert int and slice to index tensor
#         # Filter list into edge and sparse slice
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
#             assert mat.numel() == mat.nnz()  # Disallow multi-dimensional value
#             raise NotImplementedError
#         raise ValueError('Argument needs to be of type `torch.tensor` or '
#                          'type `torch_sparse.SparseTensor`.')

#     def add(self, other, layout=None):
#         if __is_scalar__(other):
#             if self.has_value:
#                 return self.set_value(self._value + other, 'coo')
#             else:
#                 return self.set_value(torch.full((self.nnz(), ), other + 1),
#                                       'coo')
#         elif torch.is_tensor(other):
#             if layout is None:
#                 layout = 'coo'
#                 warnings.warn('`layout` argument unset, using default layout '
#                               '"coo". This may lead to unexpected behaviour.')
#             assert layout in ['coo', 'csr', 'csc']
#             if layout == 'csc':
#                 other = other[self._arg_csc_to_csr]
#             if self.has_value:
#                 return self.set_value(self._value + other, 'coo')
#             else:
#                 return self.set_value(other + 1, 'coo')
#         elif isinstance(other, self.__class__):
#             raise NotImplementedError
#         raise ValueError('Argument needs to be of type `int`, `float`, '
#                          '`torch.tensor` or `torch_sparse.SparseTensor`.')

#     def add_(self, other, layout=None):
#         if isinstance(other, int) or isinstance(other, float):
#             raise NotImplementedError
#         elif torch.is_tensor(other):
#             raise NotImplementedError
#         raise ValueError('Argument needs to be a scalar or of type '
#                          '`torch.tensor`.')

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

#     def to_dense(self, dtype=None):
#         dtype = dtype or self.dtype
#         mat = torch.zeros(self.size(), dtype=dtype, device=self.device)
#         mat[self._row, self._col] = self._value if self.has_value else 1
#         return mat

#     def to_scipy(self, layout):
#         raise NotImplementedError

#     def to_torch_sparse_coo_tensor(self, dtype=None, requires_grad=False):
#         index, value = self.coo()
#         return torch.sparse_coo_tensor(
#             index,
#             torch.ones_like(self._row, dtype) if value is None else value,
#             self.size(), device=self.device, requires_grad=requires_grad)

#     def __repr__(self):
#         i = ' ' * 6
#         index, value = self.coo()
#         infos = [f'index={indent(index.__repr__(), i)[len(i):]}']
#         if value is not None:
#             infos += [f'value={indent(value.__repr__(), i)[len(i):]}']
#         infos += [
#             f'size={tuple(self.size())}, '
#             f'nnz={self.nnz()}, '
#             f'density={100 * self.density():.02f}%'
#         ]
#         infos = ',\n'.join(infos)

#         i = ' ' * (len(self.__class__.__name__) + 1)
#         return f'{self.__class__.__name__}({indent(infos, i)[len(i):]})'

# def size(self, dim=None):
#     size = self.__sparse_size
#     size += () if self.__value is None else self.__value.size()[1:]
#     return size if dim is None else size[dim]

# def dim(self):
#     return len(self.size())

# @property
# def shape(self):
#     return self.size()

# def nnz(self):
#     return self.__row.size(0)

# def density(self):
#     return self.nnz() / (self.__sparse_size[0] * self.__sparse_size[1])

# def sparsity(self):
#     return 1 - self.density()

# def avg_row_length(self):
#     return self.nnz() / self.__sparse_size[0]

# def avg_col_length(self):
#     return self.nnz() / self.__sparse_size[1]

# def numel(self):
#     return self.nnz() if self.__value is None else self.__value.numel()

# def clone(self):
#     return self._apply(lambda x: x.clone())

# def __copy__(self):
#     return self.clone()

# def __deepcopy__(self, memo):
#     memo = memo.setdefault('SparseStorage', {})
#     if self._cdata in memo:
#         return memo[self._cdata]
#     new_storage = self.clone()
#     memo[self._cdata] = new_storage
#     return new_storage

# def pin_memory(self):
#     return self._apply(lambda x: x.pin_memory())

# def is_pinned(self):
#     return all([x.is_pinned for x in self.__attributes])

# def share_memory_(self):
#     return self._apply_(lambda x: x.share_memory_())

# def is_shared(self):
#     return all([x.is_shared for x in self.__attributes])

# @property
# def device(self):
#     return self.__row.device

# def cpu(self):
#     return self._apply(lambda x: x.cpu())

# def cuda(self, device=None, non_blocking=False, **kwargs):
#     return self._apply(lambda x: x.cuda(device, non_blocking, **kwargs))

# @property
# def is_cuda(self):
#     return self.__row.is_cuda

# @property
# def dtype(self):
#     return None if self.__value is None else self.__value.dtype

# def to(self, *args, **kwargs):
#     if 'device' in kwargs:
#         out = self._apply(lambda x: x.to(kwargs['device'], **kwargs))
#         del kwargs['device']

#     for arg in args[:]:
#         if isinstance(arg, str) or isinstance(arg, torch.device):
#             out = self._apply(lambda x: x.to(arg, **kwargs))
#             args.remove(arg)

#     if len(args) > 0 and len(kwargs) > 0:
#         out = self.type(*args, **kwargs)

#     return out

# def type(self, dtype=None, non_blocking=False, **kwargs):
#     return self.dtype if dtype is None else self._apply_value(
#         lambda x: x.type(dtype, non_blocking, **kwargs))

# def is_floating_point(self):
#     return self.__value is None or torch.is_floating_point(self.__value)

# def bfloat16(self):
#     return self._apply_value(lambda x: x.bfloat16())

# def bool(self):
#     return self._apply_value(lambda x: x.bool())

# def byte(self):
#     return self._apply_value(lambda x: x.byte())

# def char(self):
#     return self._apply_value(lambda x: x.char())

# def half(self):
#     return self._apply_value(lambda x: x.half())

# def float(self):
#     return self._apply_value(lambda x: x.float())

# def double(self):
#     return self._apply_value(lambda x: x.double())

# def short(self):
#     return self._apply_value(lambda x: x.short())

# def int(self):
#     return self._apply_value(lambda x: x.int())

# def long(self):
#     return self._apply_value(lambda x: x.long())

# if __name__ == '__main__':
#     from torch_geometric.datasets import Reddit, Planetoid  # noqa
#     import time  # noqa

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     device = 'cpu'

#     # dataset = Reddit('/tmp/Reddit')
#     dataset = Planetoid('/tmp/Cora', 'Cora')
#     # dataset = Planetoid('/tmp/PubMed', 'PubMed')
#     data = dataset[0].to(device)

#     _bytes = data.edge_index.numel() * 8
#     _kbytes = _bytes / 1024
#     _mbytes = _kbytes / 1024
#     _gbytes = _mbytes / 1024
#     print(f'Storage: {_gbytes:.04f} GB')

#     mat1 = SparseTensor(data.edge_index)
#     print(mat1)
#     mat1 = mat1.t()

#     mat2 = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges),
#                                    device=device)
#     mat2 = mat2.coalesce()
#     mat2 = mat2.t().coalesce()

#     index1, value1 = mat1.coo()
#     index2, value2 = mat2._indices(), mat2._values()
#     assert torch.allclose(index1, index2)

#     out1 = mat1.to_dense()
#     out2 = mat2.to_dense()
#     assert torch.allclose(out1, out2)

#     out = 2 + mat1
#     print(out)

#     # mat1[1]
#     # mat1[1, 1]
#     # mat1[..., -1]
#     # mat1[:, -1]
#     # mat1[1:4, 1:4]
#     # mat1[torch.tensor([0, 1, 2])]
