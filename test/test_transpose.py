import torch
from torch_sparse import transpose


def test_transpose():
    row = torch.tensor([1, 0, 1, 0, 2, 1])
    col = torch.tensor([0, 1, 1, 1, 0, 0])
    index = torch.stack([row, col], dim=0)
    value = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

    index, value = transpose(index, value, m=3, n=2)
    assert index.tolist() == [[0, 0, 1, 1], [1, 2, 0, 1]]
    assert value.tolist() == [[7, 9], [5, 6], [6, 8], [3, 4]]
