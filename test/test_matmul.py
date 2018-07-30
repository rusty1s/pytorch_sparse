from itertools import product

import pytest
import torch
from torch_sparse import spspmm

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_spspmm(dtype, device):
    index = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], device=device)
    value = tensor([1, 2, 3, 4, 5], dtype, device)
    A = (index, value, torch.Size([3, 3]))

    index = torch.tensor([[0, 2], [1, 0]], device=device)
    value = tensor([2, 4], dtype, device)
    B = (index, value, torch.Size([3, 2]))

    index, value, size = spspmm(*A, *B)
    print(index)
    print(value)
    print(size)

    # out = torch.sparse_coo_tensor(index, value, size)
    # assert out.to_dense().tolist() == [[8, 0], [0, 6], [0, 8]]

    # TODO TEST backward
    # value.sum().backward()
