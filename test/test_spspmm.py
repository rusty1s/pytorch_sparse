from itertools import product

import pytest
import torch
from torch_sparse import spspmm

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_spspmm(dtype, device):
    indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], device=device)
    valueA = tensor([1, 2, 3, 4, 5], dtype, device)
    sizeA = torch.Size([3, 3])
    indexB = torch.tensor([[0, 2], [1, 0]], device=device)
    valueB = tensor([2, 4], dtype, device)
    sizeB = torch.Size([3, 2])

    indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
    assert indexC.tolist() == [[0, 1, 2], [0, 1, 1]]
    assert valueC.tolist() == [8, 6, 8]

    A = torch.sparse_coo_tensor(indexA, valueA, sizeA, device=device)
    A = A.to_dense().requires_grad_()
    B = torch.sparse_coo_tensor(indexB, valueB, sizeB, device=device)
    B = B.to_dense().requires_grad_()
    torch.matmul(A, B).sum().backward()

    valueA = valueA.requires_grad_()
    valueB = valueB.requires_grad_()
    indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
    valueC.sum().backward()

    assert valueA.grad.tolist() == A.grad[indexA[0], indexA[1]].tolist()
    assert valueB.grad.tolist() == B.grad[indexB[0], indexB[1]].tolist()
