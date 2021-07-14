# from typing import Dict, List

import torch
# from torch import Tensor
# from torch_sparse import SparseTensor


def test_hgt_sample():
    rowptr = torch.tensor([0, 1, 3, 4])
    # row = torch.tensor([0, 1, 1, 2])
    col = torch.tensor([1, 0, 2, 1])
    # _ = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))

    rowptr_dict = {'paper__to__paper': rowptr}
    col_dict = {'paper__to__paper': col}
    node_idx_dict = {'paper': torch.arange(rowptr.numel() - 1)}
    num_neighbors_dict = {'paper__to__paper': [5, 5]}
    num_hops = 2

    # nid = torch.tensor([0, 1])
    fn = torch.ops.torch_sparse.hgt_sample
    # print(fn)
    # fn(rowptr_dict, col_dict, node_idx_dict, num_neighbors_dict, num_hops)

    fn(rowptr_dict, col_dict, node_idx_dict, num_neighbors_dict, num_hops)

    # out = fn(rowptr, col, nid, 1, 3, False)
    # rowptr, col, nid, eid, ptr, root_n_id = out

    # assert nid.tolist() == [0, 1, 2, 3, 0, 1, 2]
    # assert rowptr.tolist() == [0, 3, 5, 7, 8, 10, 12, 14]
    # #      row             [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6]
    # assert col.tolist() == [1, 2, 3, 0, 2, 0, 1, 0, 5, 6, 4, 6, 4, 5]
    # assert eid.tolist() == [0, 1, 2, 3, 4, 5, 6, 9, 0, 1, 3, 4, 5, 6]
    # assert ptr.tolist() == [0, 4, 7]
    # assert root_n_id.tolist() == [0, 5]


# import timeit
# print('Timeit', timeit.timeit(test_hgt_sample))
