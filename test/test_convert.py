import time
import torch
from torch_sparse import to_scipy, from_scipy
from torch_sparse import to_torch_sparse, from_torch_sparse
from torch_sparse.storage import SparseStorage
from scipy.io import loadmat


def test_convert_scipy():
    index = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 1]])
    value = torch.Tensor([1, 2, 4, 1, 3])
    N = 3

    out = from_scipy(to_scipy(index, value, N, N))
    assert out[0].tolist() == index.tolist()
    assert out[1].tolist() == value.tolist()


def test_convert_torch_sparse():
    index = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 1]])
    value = torch.Tensor([1, 2, 4, 1, 3])
    N = 3

    out = from_torch_sparse(to_torch_sparse(index, value, N, N).coalesce())
    assert out[0].tolist() == index.tolist()
    assert out[1].tolist() == value.tolist()


def test_ind2ptr():
    name = ('DIMACS10', 'citationCiteseer')[1]
    mat = loadmat(f'benchmark/{name}.mat')['Problem'][0][0][2]
    mat = mat.tocsr().tocoo()

    mat = mat.tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(torch.long).cuda()
    mat = mat.tocoo()
    row = torch.from_numpy(mat.row).to(torch.long).cuda()
    col = torch.from_numpy(mat.col).to(torch.long).cuda()

    storage = SparseStorage(row=row, col=col)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        storage.rowptr
        storage._rowptr = None
    torch.cuda.synchronize()
    print(time.perf_counter() - t)

    assert storage.rowptr.tolist() == rowptr.tolist()

    storage = SparseStorage(rowptr=rowptr, col=col)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        storage.row
        storage._row = None
    torch.cuda.synchronize()
    print(time.perf_counter() - t)

    assert storage.row.tolist() == row.tolist()
