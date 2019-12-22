import torch

dtypes = [torch.float]

devices = [torch.device('cpu')]
# if torch.cuda.is_available():
#     devices += [torch.device('cuda:{}'.format(torch.cuda.current_device()))]


def tensor(x, dtype, device):
    return torch.tensor(x, dtype=dtype, device=device)
