from itertools import product

import pytest
import torch

from torch_sparse import rowptr_cpu
from .utils import tensor, devices

if torch.cuda.is_available():
    from torch_sparse import rowptr_cuda

tests = [
    {
        'row': [0, 0, 1, 1, 1, 2, 2],
        'size': 5,
        'rowptr': [0, 2, 5, 7, 7, 7],
    },
    {
        'row': [0, 0, 1, 1, 1, 4, 4],
        'size': 5,
        'rowptr': [0, 2, 5, 5, 5, 7],
    },
    {
        'row': [2, 2, 4, 4],
        'size': 7,
        'rowptr': [0, 0, 0, 2, 2, 4, 4, 4],
    },
]


def rowptr(row, size):
    if row.is_cuda:
        return rowptr_cuda.rowptr(row, size)
    else:
        return rowptr_cpu.rowptr(row, size)


@pytest.mark.parametrize('test,device', product(tests, devices))
def test_rowptr(test, device):
    row = tensor(test['row'], torch.long, device)
    size = test['size']
    expected = tensor(test['rowptr'], torch.long, device)

    out = rowptr(row, size)
    assert torch.all(out == expected)
