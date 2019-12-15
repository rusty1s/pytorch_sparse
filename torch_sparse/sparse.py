from textwrap import indent
import torch


class SparseTensor(object):
    def __init__(self, index, value=None, sparse_size=None, is_sorted=False):

        assert index.dtype == torch.long
        assert index.dim() == 2 and index.size(0) == 2
        index = index.contiguous()

        if value is not None:
            assert value.size(0) == index.size(1)
            assert index.device == value.device
            value = value.contiguous()

        if sparse_size is None:
            sparse_size = torch.Size((index.max(dim=-1)[0].cpu() + 1).tolist())

        self.__index__ = index
        self.__value__ = value
        self.__sparse_size__ = sparse_size

        if not is_sorted and not self.__is_sorted__():
            self.__sort__()

    def to(*args, **kwargs):
        # TODO
        pass

    def size(self, dim=None):
        size = self.__sparse_size__
        size += () if self.__value__ is None else self.__value__.size()[1:]
        return size if dim is None else size[dim]

    def storage(self):
        pass

    @property
    def shape(self):
        return self.size()

    def dim(self):
        return len(self.size())

    @property
    def dtype(self):
        return None if self.__value__ is None else self.__value__.dtype

    @property
    def device(self):
        return self.__index__.device

    def nnz(self):
        return self.__index__.size(1)

    def numel(self):
        return self.__value__.numel() if self.__value__ else self.nnz()

    def clone(self):
        return self.__class__(
            index=self.__index__.clone(),
            value=None if self.__value__ is None else self.__value__.clone(),
            sparse_size=self.__sparse_size__,
            is_sorted=True,
        )

    def sparse_resize_(self, *sizes):
        assert len(sizes) == 2
        self.__sparse_size__ = torch.Size(sizes)

    def __is_sorted__(self):
        idx1 = self.size(1) * index[0] + index[1]
        idx2 = torch.cat([idx1.new_zeros(1), idx1[:-1]], dim=0)
        return (idx1 >= idx2).all().item()

    def __sort__(self):
        idx = self.__sparse_size__(1) * self.__index__[0] + self.__index__[1]
        perm = idx.argsort()
        self.__index__ = index[:, perm]
        self.__value__ = None if self.__value__ is None else self.__value__[
            perm]

    def __repr__(self):
        i = ' ' * 6
        infos = [f'index={indent(self.__index__.__repr__(), i)[len(i):]}']
        if self.__value__ is not None:
            infos += [f'value={indent(self.__value__.__repr__(), i)[len(i):]}']
        infos += [f'size={tuple(self.size())}, nnz={self.nnz()}']
        infos = ',\n'.join(infos)

        i = ' ' * (len(self.__class__.__name__) + 1)
        return f'{self.__class__.__name__}({indent(infos, i)[len(i):]})'


if __name__ == '__main__':
    index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [2, 1, 2, 3, 0, 1],
    ])
    value = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)

    mat1 = SparseTensor(index, value)
    print(mat1)

    mat2 = torch.sparse_coo_tensor(index, value)
    # print(mat2)
