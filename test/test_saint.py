import pytest
import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse.saint import subgraph

from .utils import devices


@pytest.mark.parametrize('device', devices)
def test_subgraph(device):
    row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 4])
    col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3])
    adj = SparseTensor(row=row, col=col).to(device)
    node_idx = torch.tensor([0, 1, 2])

    adj, edge_index = subgraph(adj, node_idx)


@pytest.mark.parametrize('device', devices)
def test_sample_node(device):
    row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 4])
    col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3])
    adj = SparseTensor(row=row, col=col).to(device)

    adj, perm = adj.sample_node(num_nodes=3)


@pytest.mark.parametrize('device', devices)
def test_sample_edge(device):
    row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 4])
    col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3])
    adj = SparseTensor(row=row, col=col).to(device)

    adj, perm = adj.sample_edge(num_edges=3)


@pytest.mark.parametrize('device', devices)
def test_sample_rw(device):
    row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 4])
    col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3])
    adj = SparseTensor(row=row, col=col).to(device)

    adj, perm = adj.sample_rw(num_root_nodes=3, walk_length=2)
