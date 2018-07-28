import torch
from torch_sparse import spspmm


def test_spspmm():
    e1 = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
    v1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float, requires_grad=True)
    matrix1 = (e1, v1, torch.Size([3, 3]))

    e2 = torch.tensor([[0, 2], [1, 0]])
    v2 = torch.tensor([2, 4], dtype=torch.float, requires_grad=True)
    matrix2 = (e2, v2, torch.Size([3, 2]))

    index, value = spspmm(*matrix1, *matrix2)
    out = torch.sparse.FloatTensor(index, value, torch.Size([3, 2])).to_dense()
    assert out.tolist() == [[8, 0], [0, 6], [0, 8]]

    value.sum().backward()
