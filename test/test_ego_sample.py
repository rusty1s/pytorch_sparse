import torch
from torch_sparse import SparseTensor


def test_ego_k_hop_sample_adj():
    rowptr = torch.tensor([0, 3, 5, 9, 10, 12, 14])
    row = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 5])
    col = torch.tensor([1, 2, 3, 0, 2, 0, 1, 4, 5, 0, 2, 5, 2, 4])
    _ = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))

    idx = torch.tensor([2])

    fn = torch.ops.torch_sparse.ego_k_hop_sample_adj
    fn(rowptr, col, idx, 1, 3, False)
