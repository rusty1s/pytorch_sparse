[pypi-image]: https://badge.fury.io/py/torch-sparse.svg
[pypi-url]: https://pypi.python.org/pypi/torch-sparse
[build-image]: https://travis-ci.org/rusty1s/pytorch_sparse.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_sparse
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_sparse/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_sparse?branch=master

# PyTorch Sparse

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

--------------------------------------------------------------------------------

This package consists of a small extension library of optimized sparse matrix operations for the use in [PyTorch](http://pytorch.org/), which are missing and or lack autograd support in the main package.
This package currently consists of the following methods:

* **[Autograd Sparse Tensor Creation](#autograd-sparse-tensor-creation)**
* **[Autograd Sparse Tensor Value Extraction](#autograd-sparse-tensor-value-extraction)**
* **[Sparse Sparse Matrix Multiplication](#sparse-sparse-matrix-multiplication)**

All included operations work on varying data types and are implemented both for CPU and GPU.

## Installation

Ensure that at least PyTorch 0.4.0 is installed and verify that `cuda/bin` and `cuda/install` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 0.4.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/install:...
```

Then run:

```
pip install torch-sparse
```

If you are running into any installation problems, please follow these [instructions](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) first before creating an [issue](https://github.com/rusty1s/pytorch_sparse/issues).

## Autograd Sparse Tensor Creation

```
torch_sparse.sparse_coo_tensor(torch.LongTensor, torch.Tensor, torch.Size) -> torch.SparseTensor
```

Constructs a [`torch.SparseTensor`](https://pytorch.org/docs/stable/sparse.html) with autograd capabilities w.r.t. `value`.

```python
from torch_sparse import sparse_coo_tensor

i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.Tensor([3, 4, 5], requires_grad=True)
A = sparse_coo_tensor(i, v, torch.Size([2,3]))
```

This method may become obsolete in future PyTorch releases (>= 0.4.1) as reported by this [issue](https://github.com/pytorch/pytorch/issues/9674).

## Autograd Sparse Tensor Value Extraction

```
torch_sparse.to_value(SparseTensor) --> Tensor
```

Wrapper method to support autograd on values of sparse tensors.

```python
from torch_sparse import to_value

i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.Tensor([3, 4, 5], requires_grad=True)
A = torch.sparse_coo_tensor(i, v, torch.Size([2,3]), requires_grad=True)
v = to_value(A)
```

This method may become obsolete in future PyTorch releases (>= 0.4.1) as reported by this [issue](https://github.com/pytorch/pytorch/issues/9674).

## Sparse Sparse Matrix Multiplication

```
torch_sparse.spspmm(SparseTensor, SparseTensor) --> SparseTensor
```

Sparse matrix product of two sparse tensors with autograd support.

```
from torch_sparse import spspmm

A = torch.sparse_coo_tensor(..., requries_grad=True)
B = torch.sparse_coo_tensor(..., requries_grad=True)

C = spspmm(A, B)
```

## Running tests

```
python setup.py test
```
