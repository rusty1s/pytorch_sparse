from itertools import product

import pytest
import torch
from torch_sparse import sparse_coo_tensor, spspmm, to_value

from .utils import dtypes, devices, tensor

tests = [{
    'name': 'Test coalesced input',
    'indexA': [[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]],
    'valueA': [1, 2, 3, 4, 5],
    'sizeA': [3, 3],
    'indexB': [[0, 2], [1, 0]],
    'valueB': [2, 4],
    'sizeB': [3, 2],
}, {
    'name': 'Test uncoalesced input',
    'indexA': [[2, 2, 1, 0, 2, 0], [1, 1, 0, 2, 0, 1]],
    'valueA': [3, 2, 3, 2, 4, 1],
    'sizeA': [3, 3],
    'indexB': [[2, 0, 2], [0, 1, 0]],
    'valueB': [2, 2, 2],
    'sizeB': [3, 2],
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_spspmm(test, dtype, device):
    indexA = torch.tensor(test['indexA'], device=device)
    valueA = tensor(test['valueA'], dtype, device, requires_grad=True)
    sizeA = torch.Size(test['sizeA'])
    A = sparse_coo_tensor(indexA, valueA, sizeA)
    denseA = A.detach().to_dense().requires_grad_()

    indexB = torch.tensor(test['indexB'], device=device)
    valueB = tensor(test['valueB'], dtype, device, requires_grad=True)
    sizeB = torch.Size(test['sizeB'])
    B = sparse_coo_tensor(indexB, valueB, sizeB)
    denseB = B.detach().to_dense().requires_grad_()

    C = spspmm(A, B)
    denseC = torch.matmul(denseA, denseB)
    assert C.detach().to_dense().tolist() == denseC.tolist()

    to_value(C).sum().backward()
    denseC.sum().backward()
    assert valueA.grad.tolist() == denseA.grad[indexA[0], indexA[1]].tolist()
