import torch
from torch_sparse import to_scipy, from_scipy
from torch_sparse import to_torch_sparse, from_torch_sparse


def test_convert_scipy():
    index = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 1]])
    value = torch.Tensor([1, 2, 4, 1, 3])
    N = 3

    out = from_scipy(to_scipy(index, value, N, N))
    assert out[0].tolist() == index.tolist()
    assert out[1].tolist() == value.tolist()


def test_convert_torch_sparse():
    index = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 1]])
    value = torch.Tensor([1, 2, 4, 1, 3])
    N = 3

    out = from_torch_sparse(to_torch_sparse(index, value, N, N).coalesce())
    assert out[0].tolist() == index.tolist()
    assert out[1].tolist() == value.tolist()
