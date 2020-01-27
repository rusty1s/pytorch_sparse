from typing import Any

import torch

try:
    from typing_extensions import Final  # noqa
except ImportError:
    from torch.jit import Final  # noqa

torch.ops.load_library('torch_sparse/convert_cpu.so')
torch.ops.load_library('torch_sparse/diag_cpu.so')
torch.ops.load_library('torch_sparse/spmm_cpu.so')

try:
    torch.ops.load_library('torch_sparse/convert_cuda.so')
    torch.ops.load_library('torch_sparse/diag_cuda.so')
    torch.ops.load_library('torch_sparse/spmm_cuda.so')
    torch.ops.load_library('torch_sparse/spspmm_cuda.so')
except OSError as e:
    if torch.cuda.is_available():
        raise e


def is_scalar(other: Any) -> bool:
    return isinstance(other, int) or isinstance(other, float)
