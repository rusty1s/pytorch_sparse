import torch
from torch_sparse import SparseTensor


def test_ego_k_hop_sample_adj():
    rowptr = torch.tensor([0, 3, 5, 9, 10, 12, 14])
    row = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 5])
    col = torch.tensor([1, 2, 3, 0, 2, 0, 1, 4, 5, 0, 2, 5, 2, 4])
    _ = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))

    nid = torch.tensor([0, 1])
    fn = torch.ops.torch_sparse.ego_k_hop_sample_adj
    out = fn(rowptr, col, nid, 1, 3, False)
    rowptr, col, nid, eid, ptr, root_n_id = out

    assert nid.tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert rowptr.tolist() == [0, 3, 5, 7, 8, 10, 12, 14]
    #      row             [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6]
    assert col.tolist() == [1, 2, 3, 0, 2, 0, 1, 0, 5, 6, 4, 6, 4, 5]
    assert eid.tolist() == [0, 1, 2, 3, 4, 5, 6, 9, 0, 1, 3, 4, 5, 6]
    assert ptr.tolist() == [0, 4, 7]
    assert root_n_id.tolist() == [0, 5]
