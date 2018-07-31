import torch

dtypes = [torch.float, torch.double]

devices = [torch.device('cpu')]
if torch.cuda.is_available():  # pragma: no cover
    devices += [torch.device('cuda:{}'.format(torch.cuda.current_device()))]


def tensor(x, dtype, device, requires_grad=False):
    return torch.tensor(
        x, dtype=dtype, device=device, requires_grad=requires_grad)
