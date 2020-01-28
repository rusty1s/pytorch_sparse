import torch

from torch_sparse import SparseStorage, SparseTensor

from typing import Dict, Any

# class MyTensor(dict):
#     def __init__(self, rowptr, col):
#         self['rowptr'] = rowptr
#         self['col'] = col

# def rowptr(self: Dict[str, torch.Tensor]):
#     return self['rowptr']


@torch.jit.script
class Foo:
    rowptr: torch.Tensor
    col: torch.Tensor

    def __init__(self, rowptr: torch.Tensor, col: torch.Tensor):
        self.rowptr = rowptr
        self.col = col


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(2, 4)

    # def forward(self, x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor, adj: SparseTensor) -> torch.Tensor:
        out, _ = torch.ops.torch_sparse_cpu.spmm(adj.storage.rowptr(),
                                                 adj.storage.col(), None, x,
                                                 'sum')
        return out


#         ind = torch.ops.torch_sparse_cpu.ptr2ind(ptr, ptr[-1].item())
#         # ind = ptr2ind(ptr, E)
#         x_j = x[ind]
#         out = self.linear(x_j)
#         return out


def test_jit():
    my_cell = MyCell()

    # x = torch.rand(3, 2)
    # ptr = torch.tensor([0, 2, 4, 6])
    # out = my_cell(x, ptr)
    # print()
    # print(out)

    # traced_cell = torch.jit.trace(my_cell, (x, ptr))
    # print(traced_cell)
    # out = traced_cell(x, ptr)
    # print(out)

    x = torch.randn(3, 2)

    # adj = torch.randn(3, 3)
    # adj = SparseTensor.from_dense(adj)
    # adj = Foo(adj.storage.rowptr, adj.storage.col)
    # adj = adj.storage

    rowptr = torch.tensor([0, 1, 4, 7])
    col = torch.tensor([0, 0, 1, 2, 0, 1, 2])

    adj = SparseTensor(rowptr=rowptr, col=col)
    # scipy = adj.to_scipy(layout='csr')
    # mat = SparseTensor.from_scipy(scipy)
    print()
    print(adj)
    # adj = t(adj)
    adj = adj.t()
    print(adj)
    # print(adj.t)

    # adj = {'rowptr': mat.storage.rowptr, 'col': mat.storage.col}
    # foo = Foo(mat.storage.rowptr, mat.storage.col)
    # adj = MyTensor(mat.storage.rowptr, mat.storage.col)

    traced_cell = torch.jit.script(my_cell)
    print(traced_cell)
    out = traced_cell(x, adj)
    print(out)
    # # print(traced_cell.code)
