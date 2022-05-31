import torch

from torch_sparse._spmm_coo_cuda import spmm_coo_sum,spmm_coo_max,spmm_coo_mean
import time
device = "cuda:0"
src = torch.tensor([[10,2,3],[2,8,5]],dtype=torch.float32, device=device, requires_grad = True)
row = torch.tensor([0,1,1],device=device)
col = torch.tensor([0,0,1],device=device)


def speed_test(node_count = 10000, hidden_dim = 128, edge_count= 1000000):
    src = torch.rand((node_count,hidden_dim),dtype=torch.float64, device=device)
    row = torch.randint(node_count,(edge_count,),device=device)
    col = torch.randint(node_count,(edge_count,),device=device)
   
    start = time.time()
    res = spmm_coo_sum(row, col, src, src.shape[0])
    end = time.time()
    print(end - start)
    print(res)



speed_test()    