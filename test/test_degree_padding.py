import pytest
import torch
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid

devices = [torch.device('cuda')]


@pytest.mark.parametrize('device', devices)
def test_bin_assignment(device):
    rowcount = torch.tensor([2, 3, 6, 4, 5, 7, 8, 1], device=device)
    bin_strategy = torch.tensor([[1, 4], [5, 8]], device=device)

    perms = torch.ops.torch_sparse.bin_assignment(rowcount, bin_strategy)
    print()
    print(perms)

    dataset = Planetoid('/tmp/Planetoid', name='PubMed')
    row, col = dataset[0].edge_index
    adj = SparseTensor(row=row, col=col)
    rowcount = adj.storage.rowcount().to(device)
    # bin_strategy = torch.tensor([[1, 7], [8, 12]], device=device)
    bin_strategy = torch.tensor([[1, 4], [5, 13], [14, 22]], device=device)
    bin_count = [4, 13, 22]

    # src = torch.tensor([
    #     [1, 1],
    #     [2, 2],
    #     [3, 3],
    #     [4, 4],
    #     [5, 5],
    #     [6, 6],
    #     [7, 7],
    #     [8, 8],
    # ], dtype=torch.float, device=device)

    # rowptr = torch.tensor([0, 2, 5, 8, 10], device=device)
    # col = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 1], device=device)
    # index = torch.tensor([1, 2, 3], device=device)

    # out, mask = torch.ops.torch_sparse.padded_index_select(
    #     src, rowptr, col, index, 4)
    # print(out)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(102):
        if i == 2:
            start.record()
        perms = torch.ops.torch_sparse.bin_assignment(rowcount, bin_strategy)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    print('-------------')

    x = torch.randn(dataset[0].num_nodes, 512).to(device)
    col = col.to(device)
    for i in range(102):
        if i == 2:
            start.record()
        x = x.index_select(0, col)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    x = torch.randn(dataset[0].num_nodes, 512).to(device)
    rowptr = adj.storage.rowptr().to(device)
    col = col.to(device)
    for i in range(102):
        if i == 2:
            start.record()
        torch.ops.torch_sparse.padded_index_select(x, rowptr, col, perms[0],
                                                   bin_count[0])
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    for i in range(102):
        if i == 2:
            start.record()
        torch.ops.torch_sparse.padded_index_select(x, rowptr, col, perms[1],
                                                   bin_count[1])
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    for i in range(102):
        if i == 2:
            start.record()
        torch.ops.torch_sparse.padded_index_select(x, rowptr, col, perms[2],
                                                   bin_count[2])
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
