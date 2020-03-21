import pytest
import torch
from torch_sparse.tensor import SparseTensor

from .utils import devices


@pytest.mark.parametrize('device', devices)
def test_saint_subgraph(device):
    row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 4])
    col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3])
    adj = SparseTensor(row=row, col=col).to(device)
    node_idx = torch.tensor([0, 1, 2])

    adj, edge_index = adj.saint_subgraph(node_idx)
