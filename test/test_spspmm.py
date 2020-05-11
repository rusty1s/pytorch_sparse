from itertools import product

import pytest
import torch
from torch_sparse import spspmm, SparseTensor, transpose

from .utils import grad_dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_spspmm(dtype, device):
    indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], device=device)
    valueA = tensor([1, 2, 3, 4, 5], dtype, device)
    indexB = torch.tensor([[0, 2], [1, 0]], device=device)
    valueB = tensor([2, 4], dtype, device)

    indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
    assert indexC.tolist() == [[0, 1, 2], [0, 1, 1]]
    assert valueC.tolist() == [8, 6, 8]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_spspmm_2(dtype, device):
    row = torch.tensor(
        [0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9],
        device=device
    )
    col = torch.tensor(
        [0, 5, 10, 15, 1, 2, 3, 7, 13, 6, 9, 5, 10, 15, 11, 14, 5, 15],
        device=device
    )
    value = torch.tensor(
        [1, 3**-0.5, 3**-0.5, 3**-0.5, 1, 1, 1, -2**-0.5, -2**-0.5,
         -2**-0.5, -2**-0.5, 6**-0.5, -6**0.5 / 3, 6**-0.5, -2**-0.5,
         -2**-0.5, 2**-0.5, -2**-0.5],
        dtype=dtype, device=device
    )
    index = torch.stack([row, col])

    m = value.new_zeros(10, 16)
    m[index[0], index[1]] = value

    index_t, value_t = transpose(index, value, 10, 16)

    index, value = spspmm(index, value, index_t, value_t, 10, 16, 10)

    mask = value.abs() > 1e-4
    index, value = index[:, mask], value[mask]

    assert index.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    assert value.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_sparse_tensor_spspmm(dtype, device):
    x = SparseTensor(
        row=torch.tensor(
            [0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9],
            device=device
        ),
        col=torch.tensor(
            [0, 5, 10, 15, 1, 2, 3, 7, 13, 6, 9, 5, 10, 15, 11, 14, 5, 15],
            device=device
        ),
        value=torch.tensor(
            [1, 3**-0.5, 3**-0.5, 3**-0.5, 1, 1, 1, -2**-0.5, -2**-0.5,
             -2**-0.5, -2**-0.5, 6**-0.5, -6**0.5 / 3, 6**-0.5, -2**-0.5,
             -2**-0.5, 2**-0.5, -2**-0.5],
            dtype=dtype, device=device
        ),
    )

    i0 = torch.eye(10, dtype=dtype, device=device)

    i1 = x @ x.to_dense().t()
    assert torch.allclose(i0, i1)

    i1 = x @ x.t()
    i1 = i1.to_dense()
    assert torch.allclose(i0, i1)
