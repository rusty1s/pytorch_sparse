import torch
from torch_sparse.tensor import SparseTensor


def test_overload():
    row = torch.tensor([0, 1, 1, 2, 2])
    col = torch.tensor([1, 0, 2, 1, 2])
    mat = SparseTensor(row=row, col=col)

    other = torch.tensor([1, 2, 3]).view(3, 1)
    other + mat
    mat + other
    other * mat
    mat * other

    other = torch.tensor([1, 2, 3]).view(1, 3)
    other + mat
    mat + other
    other * mat
    mat * other
