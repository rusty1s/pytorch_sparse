import torch


def SparseTensor(index, value, size):
    t = torch.cuda if value.is_cuda else torch
    SparseTensor = getattr(t.sparse, value.type().split('.')[-1])
    return SparseTensor(index, value, size)
