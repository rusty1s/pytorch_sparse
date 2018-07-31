from itertools import product

import pytest
import torch
from torch.autograd import gradcheck
from torch_sparse import SparseTensor, spspmm, to_value

from .utils import dtypes, devices, tensor

devices = [torch.device('cpu')]
dtypes = [torch.double]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_coalesced_spspmm(dtype, device):
    indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], device=device)
    valueA = tensor([1, 2, 3, 4, 5], dtype, device)
    sizeA = torch.Size([3, 3])
    A = torch.sparse_coo_tensor(indexA, valueA, sizeA, device=device)

    indexB = torch.tensor([[0, 2], [1, 0]], device=device)
    valueB = tensor([2, 4], dtype, device)
    sizeB = torch.Size([3, 2])
    B = torch.sparse_coo_tensor(indexB, valueB, sizeB, device=device)

    assert spspmm(A, B).to_dense().tolist() == [[8, 0], [0, 6], [0, 8]]

    # A.requires_grad_()
    # B.requires_grad_()

    # A.requires_grad_()
    # B.requires_grad_()

    # to_value(C).sum().backward()
    # print(valueA)
    # print(valueA.grad)
    # print(valueB)
    # print(valueB.grad)

    # A_dense.requires_grad_()
    # B_dense.requires_grad_()

    # C_dense = torch.matmul(A_dense, B_dense)
    # C_dense[C_dense > 0].sum().backward()
    # print(A_dense)
    # print(A_dense.grad)
    # print(B_dense)
    # print(B_dense.grad)

    # A.requires_grad_()
    # B = B.to_dense()
    # B.requires_grad_()
    # torch.spmm(A, B).sum().backward()
    # print(B.grad)

    # valueA.requires_grad_()
    valueB.requires_grad_()

    def pipeline(valueA, valueB):
        A = SparseTensor(indexA, valueA, sizeA)
        B = SparseTensor(indexB, valueB, sizeB)
        C = spspmm(A, B)
        value = to_value(C)
        return value

    # out = pipeline(valueA, valueB).sum().backward()
    # print(valueA.grad)
    # print(valueB.grad)

    print(gradcheck(pipeline, (valueA, valueB), eps=1e-6, atol=1e-4))

    # A, B = Sparsetensor(SparseTensor(index, valueB, sizeB)
    # print(A.requires_grad)

    # to_value(C).sum().backward()
