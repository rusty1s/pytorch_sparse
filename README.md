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

This package consists of a small extension library of optimized sparse matrix operations with autograd support.
This package currently consists of the following methods:

* **[Coalesce](#coalesce)**
* **[Transpose](#transpose)**
* **[Sparse Dense Matrix Multiplication](#sparse-dense-matrix-multiplication)**
* **[Sparse Sparse Matrix Multiplication](#sparse-sparse-matrix-multiplication)**

All included operations work on varying data types and are implemented both for CPU and GPU.
To avoid the hazzle of creating [`torch.sparse_coo_tensor`](https://pytorch.org/docs/stable/torch.html?highlight=sparse_coo_tensor#torch.sparse_coo_tensor), this package defines operations on sparse tensors by simply passing `index` and `value` tensors as arguments ([with same shapes as defined in PyTorch](https://pytorch.org/docs/stable/sparse.html)).
Note that only `value` comes with autograd support, as `index` is discrete and therefore not differentiable.

## Installation

### Binaries

We provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://pytorch-geometric.com/whl).

#### PyTorch 1.7.0

To install the binaries for PyTorch 1.7.0, simply run

```
pip install torch-scatter==latest+${CUDA} torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, or `cu110` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` | `cu110` |
|-------------|-------|--------|---------|---------|---------|
| **Linux**   | ✅    | ✅     | ✅      | ✅      | ✅      |
| **Windows** | ✅    | ❌     | ✅      | ✅      | ✅      |
| **macOS**   | ✅    |        |         |         |         |


#### PyTorch 1.6.0

To install the binaries for PyTorch 1.6.0, simply run

```
pip install torch-scatter==latest+${CUDA} torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` |
|-------------|-------|--------|---------|---------|
| **Linux**   | ✅    | ✅     | ✅      | ✅      |
| **Windows** | ✅    | ❌     | ✅      | ✅      |
| **macOS**   | ✅    |        |         |         |

**Note:** Binaries of older versions are also provided for PyTorch 1.4.0 and PyTorch 1.5.0 (following the same procedure).

### From source

Ensure that at least PyTorch 1.4.0 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 1.4.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```

If you want to additionally build `torch-sparse` with METIS support, *e.g.* for partioning, please download and install the [METIS library](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) by following the instructions in the `Install.txt` file.
Note that METIS needs to be installed with 64 bit `IDXTYPEWIDTH` by changing `include/metis.h`.
Afterwards, set the environment variable `WITH_METIS=1`.

Then run:

```
pip install torch-scatter torch-sparse
```

When running in a docker container without NVIDIA driver, PyTorch needs to evaluate the compute capabilities and may fail.
In this case, ensure that the compute capabilities are set via `TORCH_CUDA_ARCH_LIST`, *e.g.*:

```
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.2+PTX 7.5+PTX"
```

## Functions

### Coalesce

```
torch_sparse.coalesce(index, value, m, n, op="add") -> (torch.LongTensor, torch.Tensor)
```

Row-wise sorts `index` and removes duplicate entries.
Duplicate entries are removed by scattering them together.
For scattering, any operation of [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) can be used.

#### Parameters

* **index** *(LongTensor)* - The index tensor of sparse matrix.
* **value** *(Tensor)* - The value tensor of sparse matrix.
* **m** *(int)* - The first dimension of corresponding dense matrix.
* **n** *(int)* - The second dimension of corresponding dense matrix.
* **op** *(string, optional)* - The scatter operation to use. (default: `"add"`)

#### Returns

* **index** *(LongTensor)* - The coalesced index tensor of sparse matrix.
* **value** *(Tensor)* - The coalesced value tensor of sparse matrix.

#### Example

```python
import torch
from torch_sparse import coalesce

index = torch.tensor([[1, 0, 1, 0, 2, 1],
                      [0, 1, 1, 1, 0, 0]])
value = torch.Tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

index, value = coalesce(index, value, m=3, n=2)
```

```
print(index)
tensor([[0, 1, 1, 2],
        [1, 0, 1, 0]])
print(value)
tensor([[6.0, 8.0],
        [7.0, 9.0],
        [3.0, 4.0],
        [5.0, 6.0]])
```

### Transpose

```
torch_sparse.transpose(index, value, m, n) -> (torch.LongTensor, torch.Tensor)
```

Transposes dimensions 0 and 1 of a sparse matrix.

#### Parameters

* **index** *(LongTensor)* - The index tensor of sparse matrix.
* **value** *(Tensor)* - The value tensor of sparse matrix.
* **m** *(int)* - The first dimension of corresponding dense matrix.
* **n** *(int)* - The second dimension of corresponding dense matrix.
* **coalesced** *(bool, optional)* - If set to `False`, will not coalesce the output. (default: `True`)

#### Returns

* **index** *(LongTensor)* - The transposed index tensor of sparse matrix.
* **value** *(Tensor)* - The transposed value tensor of sparse matrix.

#### Example

```python
import torch
from torch_sparse import transpose

index = torch.tensor([[1, 0, 1, 0, 2, 1],
                      [0, 1, 1, 1, 0, 0]])
value = torch.Tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

index, value = transpose(index, value, 3, 2)
```

```
print(index)
tensor([[0, 0, 1, 1],
        [1, 2, 0, 1]])
print(value)
tensor([[7.0, 9.0],
        [5.0, 6.0],
        [6.0, 8.0],
        [3.0, 4.0]])
```

### Sparse Dense Matrix Multiplication

```
torch_sparse.spmm(index, value, m, n, matrix) -> torch.Tensor
```

Matrix product of a sparse matrix with a dense matrix.

#### Parameters

* **index** *(LongTensor)* - The index tensor of sparse matrix.
* **value** *(Tensor)* - The value tensor of sparse matrix.
* **m** *(int)* - The first dimension of corresponding dense matrix.
* **n** *(int)* - The second dimension of corresponding dense matrix.
* **matrix** *(Tensor)* - The dense matrix.

#### Returns

* **out** *(Tensor)* - The dense output matrix.

#### Example

```python
import torch
from torch_sparse import spmm

index = torch.tensor([[0, 0, 1, 2, 2],
                      [0, 2, 1, 0, 1]])
value = torch.Tensor([1, 2, 4, 1, 3])
matrix = torch.Tensor([[1, 4], [2, 5], [3, 6]])

out = spmm(index, value, 3, 3, matrix)
```

```
print(out)
tensor([[7.0, 16.0],
        [8.0, 20.0],
        [7.0, 19.0]])
```

### Sparse Sparse Matrix Multiplication

```
torch_sparse.spspmm(indexA, valueA, indexB, valueB, m, k, n) -> (torch.LongTensor, torch.Tensor)
```

Matrix product of two sparse tensors.
Both input sparse matrices need to be **coalesced** (use the `coalesced` attribute to force).

#### Parameters

* **indexA** *(LongTensor)* - The index tensor of first sparse matrix.
* **valueA** *(Tensor)* - The value tensor of first sparse matrix.
* **indexB** *(LongTensor)* - The index tensor of second sparse matrix.
* **valueB** *(Tensor)* - The value tensor of second sparse matrix.
* **m** *(int)* - The first dimension of first corresponding dense matrix.
* **k** *(int)* - The second dimension of first corresponding dense matrix and first dimension of second corresponding dense matrix.
* **n** *(int)* - The second dimension of second corresponding dense matrix.
* **coalesced** *(bool, optional)*: If set to `True`, will coalesce both input sparse matrices. (default: `False`)

#### Returns

* **index** *(LongTensor)* - The output index tensor of sparse matrix.
* **value** *(Tensor)* - The output value tensor of sparse matrix.

#### Example

```python
import torch
from torch_sparse import spspmm

indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
valueA = torch.Tensor([1, 2, 3, 4, 5])

indexB = torch.tensor([[0, 2], [1, 0]])
valueB = torch.Tensor([2, 4])

indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
```

```
print(indexC)
tensor([[0, 1, 2],
        [0, 1, 1]])
print(valueC)
tensor([8.0, 6.0, 8.0])
```

## Running tests

```
python setup.py test
```
