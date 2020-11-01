import torch
import pytorch_indexing as pytorch_indexing
from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import matmul
from torch_sparse.coalesce import coalesce

def spspmm(indexA, valueA, indexB, valueB, m, k, n, autograd=True, data_split=1, coalesced=False):
    """Matrix product of two sparse tensors. Both input sparse matrices need to
    be coalesced (use the :obj:`coalesced` attribute to force).

    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of first corresponding dense matrix.
        k (int): The second dimension of first corresponding dense matrix and
            first dimension of second corresponding dense matrix.
        n (int): The second dimension of second corresponding dense matrix.
        coalesced (bool, optional): If set to :obj:`True`, will coalesce both
            input sparse matrices. (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if autograd == True:
        return pytorch_indexing.spspmm(indexA, valueA, indexB, valueB, m, k, n, data_split=data_split)
    else:
        A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
        B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)
        C = matmul(A, B)
        row, col, value = C.coo()
        return torch.stack([row, col], dim=0), value

def test_spspmm_autograd_setvals():
    indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
    valueA = torch.tensor([1, 2, 3, 4, 5])
    indexB = torch.tensor([[0, 2], [1, 0]])
    valueB = torch.tensor([2, 4])

    indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2, autograd=True, data_split=1)
    assert indexC.tolist() == [[0, 1, 2], [0, 1, 1]]
    assert valueC.tolist() == [8, 6, 8]

def test_spspmm_autograd_setvals_data_split21():
    indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
    valueA = torch.tensor([1, 2, 3, 4, 5])
    indexB = torch.tensor([[0, 2], [1, 0]])
    valueB = torch.tensor([2, 4])

    indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2, autograd=True, data_split=21)
    assert indexC.tolist() == [[0, 1, 2], [0, 1, 1]]
    assert valueC.tolist() == [8, 6, 8]

def test_spspmm_matches_cuda_vals_datasplit1():
    n = 7
    nz = 2**n
    vals1 = torch.rand(nz, requires_grad=True)
    inds1 = torch.LongTensor(2,nz).random_(0, 2**n)
    inds1, vals1 = coalesce(inds1, vals1, 2**n, 2**n)
    vals2 = torch.rand(nz, requires_grad=True)
    inds2 = torch.LongTensor(2,nz).random_(0, 2**n)
    inds2, vals2 = coalesce(inds2, vals2, 2**n, 2**n)
    my_prod_inds, my_prod_vals = spspmm(inds1, vals1, inds2, vals2, 2**n, 2**n, 2**n, autograd=True)
    prod_inds, prod_vals = spspmm(inds1, vals1, inds2, vals2, 2**n, 2**n, 2**n, autograd=False)
    assert torch.allclose(prod_vals, my_prod_vals) and torch.all(torch.eq(prod_inds, my_prod_inds))

def test_spspmm_matches_cuda_vals_datasplit17():
    n = 7
    nz = 2**n
    vals1 = torch.rand(nz, requires_grad=True)
    inds1 = torch.LongTensor(2,nz).random_(0, 2**n)
    inds1, vals1 = coalesce(inds1, vals1, 2**n, 2**n)
    vals2 = torch.rand(nz, requires_grad=True)
    inds2 = torch.LongTensor(2,nz).random_(0, 2**n)
    inds2, vals2 = coalesce(inds2, vals2, 2**n, 2**n)
    my_prod_inds, my_prod_vals = spspmm(inds1, vals1, inds2, vals2, 2**n, 2**n, 2**n, autograd=True, data_split=17)
    prod_inds, prod_vals = spspmm(inds1, vals1, inds2, vals2, 2**n, 2**n, 2**n, autograd=False)
    assert torch.allclose(prod_vals, my_prod_vals) and torch.all(torch.eq(prod_inds, my_prod_inds))
