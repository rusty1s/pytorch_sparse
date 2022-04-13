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


def test_neighbor_sample_seed():
    colptr = torch.tensor([0, 3, 6, 9])
    row = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    input_nodes = torch.tensor([0, 1])

    torch.manual_seed(42)
    out1 = neighbor_sample(colptr, row, input_nodes, [1, 1], True, False)

    torch.manual_seed(42)
    out2 = neighbor_sample(colptr, row, input_nodes, [1, 1], True, False)

    for data1, data2 in zip(out1, out2):
        assert data1.tolist() == data2.tolist()
