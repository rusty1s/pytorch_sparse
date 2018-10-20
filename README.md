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

[PyTorch](http://pytorch.org/) completely lacks autograd support and operations such as sparse sparse matrix multiplication, but is heavily working on improvement (*cf.* [this issue](https://github.com/pytorch/pytorch/issues/9674)).
In the meantime, this package consists of a small extension library of optimized sparse matrix operations with autograd support.
This package currently consists of the following methods:

* **[Coalesce](#coalesce)**
* **[Transpose](#transpose)**
* **[Sparse Dense Matrix Multiplication](#sparse-dense-matrix-multiplication)**
* **[Sparse Sparse Matrix Multiplication](#sparse-sparse-matrix-multiplication)**

All included operations work on varying data types and are implemented both for CPU and GPU.
To avoid the hazzle of creating [`torch.sparse_coo_tensor`](https://pytorch.org/docs/stable/torch.html?highlight=sparse_coo_tensor#torch.sparse_coo_tensor), this package defines operations on sparse tensors by simply passing `index` and `value` tensors as arguments ([with same shapes as defined in PyTorch](https://pytorch.org/docs/stable/sparse.html)).
Note that only `value` comes with autograd support, as `index` is discrete and therefore not differentiable.

## Installation

Ensure that at least PyTorch 0.4.1 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 0.4.1

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```

Then run:

```
pip install torch-scatter torch-sparse
```

If you are running into any installation problems, please create an [issue](https://github.com/rusty1s/pytorch_sparse/issues).
Be sure to import `torch` first before using this package to resolve symbols the dynamic linker must see.

## Coalesce

```
torch_sparse.coalesce(index, value, m, n, op="add", fill_value=0) -> (torch.LongTensor, torch.Tensor)
```

Row-wise sorts `value` and removes duplicate entries.
Duplicate entries are removed by scattering them together.
For scattering, any operation of [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) can be used.

### Parameters

* **index** *(LongTensor)* - The index tensor of sparse matrix.
* **value** *(Tensor)* - The value tensor of sparse matrix.
* **m** *(int)* - The first dimension of sparse matrix.
* **n** *(int)* - The second dimension of sparse matrix.
* **op** *(string, optional)* - The scatter operation to use. (default: `"add"`)
* **fill_value** *(int, optional)* - The initial fill value of scatter operation. (default: `0`)

### Returns

* **index** *(LongTensor)* - The coalesced index tensor of sparse matrix.
* **value** *(Tensor)* - The coalesced value tensor of sparse matrix.

### Example

```python
from torch_sparse import coalesce

index = torch.tensor([[1, 0, 1, 0, 2, 1],
                      [0, 1, 1, 1, 0, 0]])
value = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

index, value = coalesce(index, value, m=3, n=2)
```

```
print(index)
tensor([[0, 1, 1, 2],
        [1, 0, 1, 0]])
print(value)
tensor([[6, 8], [7, 9], [3, 4], [5, 6]])
```

## Transpose

```
torch_sparse.transpose(index, value, m, n) -> (torch.LongTensor, torch.Tensor)
```

Transposes dimensions 0 and 1 of a sparse matrix.

### Parameters

* **index** *(LongTensor)* - The index tensor of sparse matrix.
* **value** *(Tensor)* - The value tensor of sparse matrix.
* **m** *(int)* - The first dimension of sparse matrix.
* **n** *(int)* - The second dimension of sparse matrix.

### Returns

* **index** *(LongTensor)* - The transposed index tensor of sparse matrix.
* **value** *(Tensor)* - The transposed value tensor of sparse matrix.

### Example

```python
from torch_sparse import transpose

index = torch.tensor([[1, 0, 1, 0, 2, 1],
                      [0, 1, 1, 1, 0, 0]])
value = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

index, value = transpose(index, value, 3, 2)
```

```
print(index)
tensor([[0, 0, 1, 1],
        [1, 2, 0, 1]])
print(value)
tensor([[7, 9],
        [5, 6],
        [6, 8],
        [3, 4]])
```

## Sparse Dense Matrix Multiplication

```
torch_sparse.spmm(index, value, m, matrix) -> torch.Tensor
```

Matrix product of a sparse matrix with a dense matrix.

### Parameters

* **index** *(LongTensor)* - The index tensor of sparse matrix.
* **value** *(Tensor)* - The value tensor of sparse matrix.
* **m** *(int)* - The first dimension of sparse matrix.
* **matrix** *(Tensor)* - The dense matrix.

### Returns

* **out** *(Tensor)* - The dense output matrix.

### Example

```python
from torch_sparse import spmm

index = torch.tensor([[0, 0, 1, 2, 2],
                      [0, 2, 1, 0, 1]])
value = torch.tensor([1, 2, 4, 1, 3], dtype=torch.float)
matrix = torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.float)

out = spmm(index, value, 3, matrix)
```

```
print(out)
tensor([[7.0, 16.0],
        [8.0, 20.0],
        [7.0, 19.0]])
```

## Sparse Sparse Matrix Multiplication

```
torch_sparse.spspmm(indexA, valueA, indexB, valueB, m, k, n) -> (torch.LongTensor, torch.Tensor)
```

Matrix product of two sparse tensors.
Both input sparse matrices need to be **coalesced**.

### Parameters

* **indexA** *(LongTensor)* - The index tensor of first sparse matrix.
* **valueA** *(Tensor)* - The value tensor of first sparse matrix.
* **indexB** *(LongTensor)* - The index tensor of second sparse matrix.
* **valueB** *(Tensor)* - The value tensor of second sparse matrix.
* **m** *(int)* - The first dimension of first sparse matrix.
* **k** *(int)* - The second dimension of first sparse matrix and first dimension of second sparse matrix.
* **n** *(int)* - The second dimension of second sparse matrix.

### Returns

* **index** *(LongTensor)* - The output index tensor of sparse matrix.
* **value** *(Tensor)* - The output value tensor of sparse matrix.

### Example

```python
from torch_sparse import spspmm

indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
valueA = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)

indexB = torch.tensor([[0, 2], [1, 0]])
valueB = torch.tensor([2, 4], dtype=torch.float)

indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
```

```
print(index)
tensor([[0, 1, 2],
        [0, 1, 1]])
print(value)
tensor([8.0, 6.0, 8.0])
```

## Running tests

```
python setup.py test
```
