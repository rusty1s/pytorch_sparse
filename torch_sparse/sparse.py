import inspect
from textwrap import indent
import torch

from torch_sparse.storage import SparseStorage

methods = list(zip(*inspect.getmembers(SparseStorage)))[0]
methods = [name for name in methods if '__' not in name and name != 'clone']


class SparseTensor(object):
    def __init__(self, index, value=None, sparse_size=None, is_sorted=False):
        assert index.dim() == 2 and index.size(0) == 2
        self._storage = SparseStorage(index[0], index[1], value, sparse_size,
                                      is_sorted=is_sorted)

    @classmethod
    def from_storage(self, storage):
        self = SparseTensor.__new__(SparseTensor)
        self._storage = storage
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

    @property
    def _storage(self):
        return self.__storage

    @_storage.setter
    def _storage(self, storage):
        self.__storage = storage
        for name in methods:
            setattr(self, name, getattr(storage, name))

    def clone(self):
        return SparseTensor.from_storage(self._storage.clone())

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('SparseStorage', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_sparse_tensor = self.clone()
        memo[self._cdata] = new_sparse_tensor
        return new_sparse_tensor

    def coo(self):
        return self._index, self._value

    def csr(self):
        return self._rowptr, self._col, self._value

    def csc(self):
        perm = self._arg_csr_to_csc
        return self._colptr, self._row[perm], self._value[perm]

    def is_quadratic(self):
        return self.sparse_size[0] == self.sparse_size[1]

    def is_symmetric(self):
        if not self.is_quadratic:
            return False

        index1, value1 = self.coo()
        index2, value2 = self.t().coo()
        index_symmetric = (index1 == index2).all()
        value_symmetric = (value1 == value2).all() if self.has_value else True
        return index_symmetric and value_symmetric

    def set_value(self, value, layout):
        if value is not None and layout == 'csc':
            value = value[self._arg_csc_to_csr]
        return self._apply_value(value)

    def set_value_(self, value, layout):
        if value is not None and layout == 'csc':
            value = value[self._arg_csc_to_csr]
        return self._apply_value_(value)

    def t(self):
        storage = SparseStorage(
            self._col[self._arg_csr_to_csc],
            self._row[self._arg_csr_to_csc],
            self._value[self._arg_csr_to_csc] if self.has_value else None,
            self.sparse_size()[::-1],
            self._colptr,
            self._rowptr,
            self._arg_csc_to_csr,
            self._arg_csr_to_csc,
            is_sorted=True,
        )
        return self.__class__.from_storage(storage)

    def matmul(self, mat2):
        raise NotImplementedError

    def coalesce(self, reduce='add'):
        raise NotImplementedError

    def is_coalesced(self):
        raise NotImplementedError

    def add(self, layout=None):
        # sub, mul, div
        # can take scalars, tensors and other sparse matrices
        # inplace variants can only take scalars or tensors
        raise NotImplementedError

    # TODO: Slicing, (sum|max|min|prod|...), standard operators, masing, perm

    def to_dense(self, dtype=None):
        dtype = dtype or self.dtype
        mat = torch.zeros(self.size(), dtype=dtype, device=self.device)
        mat[self._row, self._col] = self._value if self.has_value else 1
        return mat

    def to_scipy(self):
        raise NotImplementedError

    def to_torch_sparse_coo_tensor(self):
        raise NotImplementedError

    def __repr__(self):
        i = ' ' * 6
        index, value = self.coo()
        infos = [f'index={indent(index.__repr__(), i)[len(i):]}']
        if value is not None:
            infos += [f'value={indent(value.__repr__(), i)[len(i):]}']
        infos += [
            f'size={tuple(self.size())}, '
            f'nnz={self.nnz()}, '
            f'density={100 * self.density():.02f}%'
        ]
        infos = ',\n'.join(infos)

        i = ' ' * (len(self.__class__.__name__) + 1)
        return f'{self.__class__.__name__}({indent(infos, i)[len(i):]})'


if __name__ == '__main__':
    from torch_geometric.datasets import Reddit, Planetoid  # noqa
    import time  # noqa

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    # dataset = Reddit('/tmp/Reddit')
    dataset = Planetoid('/tmp/PubMed', 'PubMed')
    data = dataset[0].to(device)

    _bytes = data.edge_index.numel() * 8
    _kbytes = _bytes / 1024
    _mbytes = _kbytes / 1024
    _gbytes = _mbytes / 1024
    print(f'Storage: {_gbytes:.04f} GB')

    mat1 = SparseTensor(data.edge_index)
    print(mat1)
    mat1 = mat1.t()

    mat2 = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges),
                                   device=device)
    mat2 = mat2.coalesce()
    mat2 = mat2.t().coalesce()

    index1, value1 = mat1.coo()
    index2, value2 = mat2._indices(), mat2._values()
    assert torch.allclose(index1, index2)

    out1 = mat1.to_dense()
    out2 = mat2.to_dense()
    assert torch.allclose(out1, out2)

    mat1 = SparseTensor.from_dense(out1)
    print(mat1)
