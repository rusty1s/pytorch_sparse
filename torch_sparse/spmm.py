import torch
from torch_scatter import scatter_add

from torch_sparse.sparse import SparseTensor

if torch.cuda.is_available():
    import torch_sparse.spmm_cuda


def spmm_(sparse_mat, mat, reduce='add'):
    assert reduce in ['add', 'mean', 'min', 'max']
    assert sparse_mat.dim() == 2 and mat.dim() == 2
    assert sparse_mat.size(1) == mat.size(0)

    rowptr, col, value = sparse_mat.csr()
    mat = mat.contiguous()

    if reduce in ['add', 'mean']:
        return torch_sparse.spmm_cuda.spmm(rowptr, col, value, mat, reduce)
    else:
        return torch_sparse.spmm_cuda.spmm_arg(rowptr, col, value, mat, reduce)


def spmm(index, value, m, n, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of corresponding dense matrix.
        n (int): The second dimension of corresponding dense matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    assert n == matrix.size(0)

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col]
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row, dim=0, dim_size=m)

    return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    row = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    col = torch.tensor([0, 1, 2, 0, 1, 2], device=device)
    value = torch.ones_like(col, dtype=torch.float, device=device)
    value = None
    sparse_mat = SparseTensor(torch.stack([row, col], dim=0), value)
    mat = torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.float,
                       device=device)
    out1 = spmm_(sparse_mat, mat, reduce='add')
    out2 = sparse_mat.to_dense() @ mat
    assert torch.allclose(out1, out2)

    from torch_geometric.datasets import Reddit, Planetoid  # noqa
    import time  # noqa

    # Warmup
    x = torch.randn((1000, 1000), device=device)
    for _ in range(100):
        x.sum()

    # dataset = Reddit('/tmp/Reddit')
    dataset = Planetoid('/tmp/PubMed', 'PubMed')
    # dataset = Planetoid('/tmp/Cora', 'Cora')
    data = dataset[0].to(device)
    mat = torch.randn((data.num_nodes, 1024), device=device)
    value = torch.ones(data.num_edges, device=device)

    sparse_mat = SparseTensor(data.edge_index, value)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        out1 = spmm_(sparse_mat, mat, reduce='add')
        out1 = out1[0] if isinstance(out1, tuple) else out1
    torch.cuda.synchronize()
    print('My:   ', time.perf_counter() - t)

    sparse_mat = torch.sparse_coo_tensor(data.edge_index, value)
    sparse_mat = sparse_mat.coalesce()

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        out2 = sparse_mat @ mat
    torch.cuda.synchronize()
    print('Torch: ', time.perf_counter() - t)

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        spmm(data.edge_index, value, data.num_nodes, data.num_nodes, mat)
    torch.cuda.synchronize()
    print('Scatter:', time.perf_counter() - t)

    assert torch.allclose(out1, out2, atol=1e-2)
