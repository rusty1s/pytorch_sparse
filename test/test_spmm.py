import torch
from torch_sparse import spmm


def test_spmm():
    row = torch.tensor([0, 0, 1, 2, 2])
    col = torch.tensor([0, 2, 1, 0, 1])
    index = torch.stack([row, col], dim=0)
    value = torch.tensor([1, 2, 4, 1, 3])

    matrix = torch.tensor([[1, 4], [2, 5], [3, 6]])
    out = spmm(index, value, 3, matrix)
    assert out.tolist() == [[7, 16], [8, 20], [7, 19]]
