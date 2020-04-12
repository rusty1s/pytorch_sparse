import torch
from torch_sparse.tensor import SparseTensor


def test_saint_subgraph():
    row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 4])
    col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3])
    adj = SparseTensor(row=row, col=col)
    node_idx = torch.tensor([0, 1, 2])

    adj, edge_index = adj.saint_subgraph(node_idx)
