from itertools import product

import pytest
import torch
from torch_sparse import SparseTensor, padded_index_select

from .utils import grad_dtypes, tensor

devices = [torch.device('cuda')]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_padded_index_select(dtype, device):
    row = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    col = torch.tensor([0, 1, 2, 3, 0, 2, 3, 1, 3, 2])
    adj = SparseTensor(row=row, col=col).to(device)
    binptr = torch.tensor([0, 3, 5], device=device)

    data = adj.padded_index(binptr)
    node_perm, row_perm, col_perm, mask, node_size, edge_size = data

    assert node_perm.tolist() == [2, 3, 0, 1]
    assert row_perm.tolist() == [2, 2, 3, -1, 0, 0, 0, 0, 1, 1, 1, -1]
    assert col_perm.tolist() == [1, 3, 2, -1, 0, 1, 2, 3, 0, 2, 3, -1]
    assert mask.long().tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert node_size == [2, 2]
    assert edge_size == [4, 8]

    x = tensor([0, 1, 2, 3], dtype, device).view(-1, 1).requires_grad_()
    x_j = padded_index_select(x, col_perm)

    assert x_j.flatten().tolist() == [1, 3, 2, 0, 0, 1, 2, 3, 0, 2, 3, 0]

    grad_out = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype, device)
    x_j.backward(grad_out.view(-1, 1))

    assert x.grad.flatten().tolist() == [12, 5, 17, 18]


def test_padded_index_select_runtime():
    return
    from torch_geometric.datasets import Planetoid

    device = torch.device('cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    dataset = Planetoid('/tmp/Planetoid', name='PubMed')
    data = dataset[0]
    row, col = data.edge_index.to(device)

    adj = SparseTensor(row=row, col=col)
    rowcount = adj.storage.rowcount().to(device)
    rowptr = adj.storage.rowptr().to(device)
    binptr = torch.tensor([0, 4, 11, 30, 50, 80, 120, 140, 2000]).to(device)

    x = torch.randn(adj.size(0), 512).to(device)

    data = torch.ops.torch_sparse.padded_index(rowptr, col, rowcount, binptr)
    node_perm, row_perm, col_perm, mask, node_sizes, edge_sizes = data

    out = torch.ops.torch_sparse.padded_index_select(x, col_perm,
                                                     torch.tensor(0.))
    outs = out.split(edge_sizes)
    for out, size in zip(outs, node_sizes):
        print(out.view(size, -1, x.size(-1)).shape)

    for i in range(110):
        if i == 10:
            start.record()
        torch.ops.torch_sparse.padded_index(rowptr, col, rowcount, binptr)
    end.record()
    torch.cuda.synchronize()
    print('padded index', start.elapsed_time(end))

    for i in range(110):
        if i == 10:
            start.record()
        out = torch.ops.torch_sparse.padded_index_select(
            x, col_perm, torch.tensor(0.))
        out.split(edge_sizes)
    end.record()
    torch.cuda.synchronize()
    print('padded index select', start.elapsed_time(end))

    for i in range(110):
        if i == 10:
            start.record()
        x.index_select(0, col)
    end.record()
    torch.cuda.synchronize()
    print('index_select', start.elapsed_time(end))
