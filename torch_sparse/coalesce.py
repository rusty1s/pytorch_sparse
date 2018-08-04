import torch
import torch_scatter


def coalesce(index, value, size, op='add', fill_value=0):
    m, n = size
    row, col = index

    index = row * n + col
    unique, inv = torch.unique(index, sorted=True, return_inverse=True)

    perm = torch.arange(index.size(0), dtype=index.dtype, device=index.device)
    perm = index.new_empty(inv.max().item() + 1).scatter_(0, inv, perm)
    index = torch.stack([row[perm], col[perm]], dim=0)

    if value is not None:
        scatter = getattr(torch_scatter, 'scatter_{}'.format(op))
        value = scatter(
            value, inv, dim=0, dim_size=perm.size(0), fill_value=fill_value)

    return index, value
