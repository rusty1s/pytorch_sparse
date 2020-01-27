import torch

torch.ops.load_library('torch_sparse/convert_cpu.so')

try:
    torch.ops.load_library('torch_sparse/convert_cuda.so')
except OSError:
    pass


def ext(is_cuda):
    name = 'torch_sparse_cuda' if is_cuda else 'torch_sparse_cpu'
    return getattr(torch.ops, name)
