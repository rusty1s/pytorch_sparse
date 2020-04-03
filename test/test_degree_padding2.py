import pytest
import torch
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

devices = [torch.device('cuda')]


@pytest.mark.parametrize('device', devices)
def test_padded_index_select(device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    dataset = Planetoid('/tmp/Planetoid', name='PubMed')
    data = dataset[0]
    row, col = data.edge_index.to(device)

    adj = SparseTensor(row=row, col=col)
    rowcount = adj.storage.rowcount().to(device)
    rowptr = adj.storage.rowptr().to(device)

    bin_strategy = torch.tensor([[1, 4], [4, 11], [11, 30]]).to(device)
    binptr = torch.tensor([0, 4, 11, 30, 50, 80, 120, 140, 2000]).to(device)

    deg = degree(row, dtype=torch.long)
    bins = torch.bincount(deg)
    # print(bins.size())
    # print(bins[:200])
    # for i in range(110):
    #     if i == 10:
    #         start.record()
    #     perms, lengths = torch.ops.torch_sparse.bin_assignment(
    #         rowcount, binptr)
    # end.record()
    # torch.cuda.synchronize()
    # print('bin assignment', start.elapsed_time(end))
    idx, mask, size, length, offset = torch.ops.torch_sparse.padded_index(
        rowptr, rowcount, binptr)
    print(size)
    print(length)
    print(offset)
    print(mask[:10])
    print(idx[:10])

    x = torch.randn(data.num_nodes, 128).to(device)

    for i in range(110):
        if i == 10:
            start.record()
        torch.ops.torch_sparse.padded_index(rowptr, rowcount, binptr)
    end.record()
    torch.cuda.synchronize()
    print('padded index', start.elapsed_time(end))

    for i in range(110):
        if i == 10:
            start.record()
        torch.ops.torch_sparse.padded_index_select(x, col, idx,
                                                   torch.tensor(0.))
    end.record()
    torch.cuda.synchronize()
    print('padded index select', start.elapsed_time(end))

    for i in range(110):
        if i == 10:
            start.record()
        torch.repeat_interleave(rowcount, rowcount)
    end.record()
    torch.cuda.synchronize()
    print('repeat', start.elapsed_time(end))

    for i in range(110):
        if i == 10:
            start.record()
        rowcount.cumsum(0)
    end.record()
    torch.cuda.synchronize()
    print('cumsum', start.elapsed_time(end))

    rowcount2 = rowcount.unsqueeze(1).repeat(1, 5).contiguous()
    for i in range(110):
        if i == 10:
            start.record()
        rowcount2.cumsum(0)
    end.record()
    torch.cuda.synchronize()
    print('cumsum', start.elapsed_time(end))

    for i in range(110):
        if i == 10:
            start.record()
        rowcount.sort()
    end.record()
    torch.cuda.synchronize()
    print('sort', start.elapsed_time(end))

    for i in range(110):
        if i == 10:
            start.record()
        x.index_select(0, col)
    end.record()
    torch.cuda.synchronize()
    print('index_select', start.elapsed_time(end))
    return

    for i in range(110):
        if i == 10:
            start.record()
        for perm, length in zip(perms, lengths):
            torch.ops.torch_sparse.padded_index_select(x, rowptr, col,
                                                       perm, length,
                                                       torch.tensor(0.))
    end.record()
    torch.cuda.synchronize()
    print('padded_index_select', start.elapsed_time(end))

    for perm, length in zip(perms, lengths):
        out, mask = torch.ops.torch_sparse.padded_index_select(
            x, rowptr, col, perm, length, torch.tensor(0.))
        print(out.size(), mask.size(), out.numel(), (out != 0).sum().item())

    lengths = bin_strategy[:, 1].view(-1).tolist()

    for dim in [32, 64, 128, 256, 512, 1024]:
        print(f'--- Dim: {dim} ---')
        x = torch.randn(adj.size(0), dim).to(device)

        for i in range(110):
            if i == 10:
                start.record()
            perms = torch.ops.torch_sparse.bin_assignment(
                rowcount, bin_strategy)
            print(perms)
            return
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
