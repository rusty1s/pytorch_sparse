from itertools import product

import pytest
import torch
from torch_sparse import spspmm

from .utils import dtypes, devices, tensor

devices = [torch.device('cuda')]
dtypes = [torch.float]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_spspmm(dtype, device):
    e1 = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], device=device)
    v1 = tensor([1, 2, 3, 4, 5], dtype, device)
    matrix1 = (e1, v1, torch.Size([3, 3]))

    e2 = torch.tensor([[0, 2], [1, 0]], device=device)
    v2 = tensor([2, 4], dtype, device)
    matrix2 = (e2, v2, torch.Size([3, 2]))

    index, value = spspmm(*matrix1, *matrix2)
    print(index)
    print(value)
    # out = torch.sparse_coo_tensor(index, value, torch.Size([3, 2]), dtype)
    # out = out.to_dense()
    # print(out)
    # assert out.tolist() == [[8, 0], [0, 6], [0, 8]]

    # value.sum().backward()
    # TODO TEST backward
