import torch
from torch_sparse import SparseTensor

neighbor_sample = torch.ops.torch_sparse.neighbor_sample


def test_neighbor_sample():
    adj = SparseTensor.from_edge_index(torch.tensor([[0], [1]]))
    colptr, row, _ = adj.csc()

    # Sampling in a non-directed way should not sample in wrong direction:
    out = neighbor_sample(colptr, row, torch.tensor([0]), [1], False, False)
    assert out[0].tolist() == [0]
    assert out[1].tolist() == []
    assert out[2].tolist() == []

    # Sampling should work:
    out = neighbor_sample(colptr, row, torch.tensor([1]), [1], False, False)
    assert out[0].tolist() == [1, 0]
    assert out[1].tolist() == [1]
    assert out[2].tolist() == [0]

    # Sampling with more hops:
    out = neighbor_sample(colptr, row, torch.tensor([1]), [1, 1], False, False)
    assert out[0].tolist() == [1, 0]
    assert out[1].tolist() == [1]
    assert out[2].tolist() == [0]
