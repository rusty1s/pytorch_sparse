import torch
from torch.autograd import Variable

size = torch.Size([2, 2])

index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
value = torch.tensor([1, 1], dtype=torch.float).cuda()
A = torch.cuda.sparse.FloatTensor(index, value, size)

index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
value = torch.tensor([1, 1], dtype=torch.float)
B = torch.sparse.FloatTensor(index, value, size)
