from itertools import product

import pytest
import torch
from torch.autograd import gradcheck
from torch_sparse import spspmm
from torch_sparse.matmul import SpSpMM

from .utils import dtypes, devices, tensor

dtypes = [torch.double]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_coalesced_spspmm(dtype, device):
    indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], device=device)
    valueA = tensor([1, 2, 3, 4, 5], dtype, device, requires_grad=True)
    sizeA = torch.Size([3, 3])
    A = (indexA, valueA, sizeA)
    A_dense = torch.sparse_coo_tensor(indexA, valueA, sizeA).to_dense()
    A_dense = A_dense.requires_grad_()
    print('A', A_dense)

    indexB = torch.tensor([[0, 2], [1, 0]], device=device)
    valueB = tensor([2, 4], dtype, device, requires_grad=True)
    sizeB = torch.Size([3, 2])
    B = (indexB, valueB, sizeB)
    B_dense = torch.sparse_coo_tensor(indexB, valueB, sizeB).to_dense()
    B_dense = B_dense.requires_grad_()

    index, value, size = spspmm(*A, *B)
    # out = torch.sparse_coo_tensor(index, value, size)
    expected = torch.matmul(A_dense, B_dense)
    # assert out.to_dense().tolist() == expected.tolist()

    # valueA = valueA.requires_grad_()
    # valueB = valueB.requires_grad_()
    # data = (indexA, valueA, sizeA, indexB, valueB, sizeB)
    # assert gradcheck(SpSpMM.apply, data, eps=1e-6, atol=1e-4) is True

    # print(expected)

    value.sum().backward()
    expected.sum().backward()

    print(valueA.grad)
    print(A_dense.grad)

    # print(valueB.grad)
    # print(B_dense.grad)

    # # TODO TEST backward
    # # value.sum().backward()
