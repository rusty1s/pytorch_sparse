import pytest
import torch

from torch_sparse.tensor import SparseTensor
from torch_sparse.testing import devices, tensor


@pytest.mark.parametrize('device', devices)
def test_permute(device):
    row, col = tensor([[0, 0, 1, 2, 2], [0, 1, 0, 1, 2]], torch.long, device)
    value = tensor([1, 2, 3, 4, 5], torch.float, device)
    adj = SparseTensor(row=row, col=col, value=value)

    row, col, value = adj.permute(torch.tensor([1, 0, 2])).coo()
    assert row.tolist() == [0, 1, 1, 2, 2]
    assert col.tolist() == [1, 0, 1, 0, 2]
    assert value.tolist() == [3, 2, 1, 4, 5]
