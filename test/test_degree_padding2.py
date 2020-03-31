import pytest
import torch
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

devices = [torch.device('cuda')]


@pytest.mark.parametrize('device', devices)
def test_padded_index_select(device):
    dataset = Planetoid('/tmp/Planetoid', name='PubMed')
    data = dataset[0]
    row, col = data.edge_index.to(device)

    row = torch.arange(data.num_nodes).view(-1, 1).repeat(1, 4).view(-1)
    col = torch.randint(0, data.num_nodes, (row.size(0), ))
    row, col = row.to(device), col.to(device)

    adj = SparseTensor(row=row, col=col)
    rowcount = adj.storage.rowcount().to(device)
    rowptr = adj.storage.rowptr().to(device)
    bin_strategy = torch.tensor([[1, 4]]).to(device)
    # bin_strategy = torch.tensor([[1, 5], [6, 12], [13, 19], [20, 30]],
    #                             device=device)
    perms = torch.ops.torch_sparse.bin_assignment(rowcount, bin_strategy)
    lengths = bin_strategy[:, 1].view(-1).tolist()
    print(lengths)

    deg = degree(row, dtype=torch.long)
    print(deg.size(), deg.min(), deg.float().mean(), deg.max())
    bins = torch.bincount(deg)
    print(bins)
    nonzero = bins.nonzero().flatten()
    print(nonzero)
    print(bins[nonzero])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for dim in [32, 64, 128, 256, 512, 1024]:
        print(f'--- Dim: {dim} ---')
        x = torch.randn(adj.size(0), dim).to(device)

        for i in range(110):
            if i == 10:
                start.record()
            for perm, length in zip(perms, lengths):
                out1, _ = torch.ops.torch_sparse.padded_index_select(
                    x, rowptr, col, perm, length, torch.tensor(0.))
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))

        for i in range(110):
            if i == 10:
                start.record()
            out2 = x.index_select(0, row)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))

        for i in range(110):
            if i == 10:
                start.record()
            out3 = x.index_select(0, col)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        print(torch.allclose(out1.view(-1, dim), out3))
