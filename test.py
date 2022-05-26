import torch

import torch_sparse

device = "cuda:0"
src = torch.tensor([[10,2,3],[2,8,5]],dtype=torch.float32, device=device, requires_grad = True)
row = torch.tensor([0,1,1],device=device)
col = torch.tensor([0,0,1],device=device)


print(torch.ops.torch_sparse.spmm_coo_max(row, col, src, 2))