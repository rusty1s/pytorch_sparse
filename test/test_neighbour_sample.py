import torch
import torch_sparse


def test_neighbor_sample():

    adj = torch_sparse.SparseTensor(row=torch.Tensor([0]).type(torch.long),
                                    col=torch.Tensor([1]).type(torch.long),
                                    sparse_sizes=(2, 2))

    colptr, rowind, _ = adj.csc()

    # sampling in a non-directed way should not sample in wrong direction
    nodes = torch.Tensor([0]).type(torch.long)

    out = torch.ops.torch_sparse.neighbor_sample(colptr, rowind, nodes, [1],
                                                 False, False)
    node, row, col, _ = out

    assert node.numel() == 1
    assert row.numel() == 0
    assert col.numel() == 0

    # sampling should work
    nodes = torch.Tensor([1]).type(torch.long)

    out = torch.ops.torch_sparse.neighbor_sample(colptr, rowind, nodes, [1],
                                                 False, False)
    node, row, col, _ = out

    assert node.numel() == 2
    assert node[0] == row[0]
    assert node[1] == col[0]

    # sampling with repeated node should be the same as with no repeats
    nodes = torch.Tensor([1, 1]).type(torch.long)

    out = torch.ops.torch_sparse.neighbor_sample(colptr, rowind, nodes, [1],
                                                 False, False)
    node, row, col, _ = out

    assert node.numel() == 2
    assert 1 == row[0]
    assert 0 == col[0]

    adj = torch_sparse.SparseTensor(row=torch.Tensor([0, 1]).type(torch.long),
                                    col=torch.Tensor([1, 2]).type(torch.long),
                                    sparse_sizes=(3, 3))

    colptr, rowind, _ = adj.csc()

    # sampling with more edges shouldn't go further than one step
    nodes = torch.Tensor([2]).type(torch.long)

    out = torch.ops.torch_sparse.neighbor_sample(colptr, rowind, nodes, [2],
                                                 False, False)
    node, row, col, _ = out

    assert node.numel() == 2
    assert 1 == row[0]
    assert 0 == col[0]

    # sampling with more edges and multiple layers should go further
    out = torch.ops.torch_sparse.neighbor_sample(colptr, rowind, nodes, [2, 2],
                                                 False, False)
    node, row, col, _ = out

    assert node.numel() == 3
