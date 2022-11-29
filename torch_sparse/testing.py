from typing import Any

import torch
import torch_scatter
from packaging import version

reductions = ['sum', 'add', 'mean', 'min', 'max']

dtypes = [torch.half, torch.float, torch.double, torch.int, torch.long]
grad_dtypes = [torch.half, torch.float, torch.double]

if version.parse(torch_scatter.__version__) > version.parse("2.0.9"):
    dtypes.append(torch.bfloat16)
    grad_dtypes.append(torch.bfloat16)

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda:0')]


def tensor(x: Any, dtype: torch.dtype, device: torch.device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)
