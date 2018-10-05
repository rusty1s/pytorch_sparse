import torch
import numpy as np

if torch.cuda.is_available():
    import unique_cuda


def unique(src):
    src = src.contiguous().view(-1)

    if src.is_cuda:
        out, perm = unique_cuda.unique(src)
    else:
        out, perm = np.unique(src.numpy(), return_index=True)
        out, perm = torch.from_numpy(out), torch.from_numpy(perm)

    return out, perm
