import pytest
from itertools import product
import torch
import time
from torch_sparse.spmm_coo import spmm_coo
from torch_sparse.matmul import matmul
from torch_sparse.tensor import SparseTensor

reductions = ['sum','max','min','mean']
dtypes = [ torch.float, torch.double]


@pytest.mark.parametrize('dtype,reduce', product(dtypes,reductions))
def test_spmm_coo(dtype, reduce):
    device = "cuda:0"
  
    mat = torch.randn((1000, 128), dtype=dtype, device=device, requires_grad=True)
    value = torch.randn((50000),device=device,dtype=dtype)
    row = torch.randint(1000,(50000,),device=device)
    col = torch.randint(1000,(50000,),device=device)

    #compute mat_mul
    source = SparseTensor(row=col,col=row,value=value)
    expected = matmul(source,mat,reduce=reduce)
    grad_out = torch.rand_like(expected)   
    expected.backward(grad_out)
    expected_grad_other = mat.grad
    mat.grad = None

    #compute spmm_coo
    out = spmm_coo(row,col,mat,mat.shape[0],value=value,reduce=reduce)
    out.backward(grad_out)
    out_grad = mat.grad

    assert torch.allclose(expected, out, atol=1e-2)
    #assert torch.allclose(expected_grad_value, value.grad, atol=1e-2)
    assert torch.allclose(expected_grad_other, out_grad, atol=1e-2)
