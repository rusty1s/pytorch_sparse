import time

import torch
import torch_sparse  # noqa
from ogb.nodeproppred import PygNodePropPredDataset


def test_sort():
    dataset = PygNodePropPredDataset('ogbn-products', '/tmp/OGB')
    data = dataset[0]
    row, col = data.edge_index
    row, col = row.contiguous(), col.contiguous()
    num_nodes = data.num_nodes

    print()
    t = time.perf_counter()
    out = torch.ops.torch_sparse.sort(row, col, num_nodes, False)
    print(time.perf_counter() - t)

    t = time.perf_counter()
    perm = (row * num_nodes).add_(col).argsort()
    out = row[perm], col[perm]
    print(time.perf_counter() - t)
