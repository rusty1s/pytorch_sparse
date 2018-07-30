import torch


def transpose(index, value, size):
    (row, col), (dim1, dim2) = index, size
    index, size = torch.stack([col, row], dim=0), torch.Size([dim2, dim1])
    return index, value, size
