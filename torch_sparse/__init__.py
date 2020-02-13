# flake8: noqa

import importlib
import os.path as osp

import torch

__version__ = '0.5.1'
expected_torch_version = (1, 4)

try:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        '_version', [osp.dirname(__file__)]).origin)
except OSError as e:
    if 'undefined symbol' in str(e):
        major, minor = [int(x) for x in torch.__version__.split('.')[:2]]
        t_major, t_minor = expected_torch_version
        if major != t_major or (major == t_major and minor != t_minor):
            raise RuntimeError(
                f'Expected PyTorch version {t_major}.{t_minor} but found '
                f'version {major}.{minor}.')
    raise OSError(e)

cuda_version = torch.ops.torch_scatter.cuda_version()
if cuda_version != -1 and torch.version.cuda is not None:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]
    cuda_version = str(major) + '.' + str(minor)

    if t_major != major or t_minor != minor:
        raise RuntimeError(
            f'Detected that PyTorch and torch_sparse were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_sparse has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_sparse that '
            f'matches your PyTorch install.')

from .storage import SparseStorage
from .tensor import SparseTensor
from .transpose import t
from .narrow import narrow, __narrow_diag__
from .select import select
from .index_select import index_select, index_select_nnz
from .masked_select import masked_select, masked_select_nnz
from .diag import remove_diag, set_diag, fill_diag
from .add import add, add_, add_nnz, add_nnz_
from .mul import mul, mul_, mul_nnz, mul_nnz_
from .reduce import sum, mean, min, max
from .matmul import matmul
from .cat import cat, cat_diag

from .convert import to_torch_sparse, from_torch_sparse, to_scipy, from_scipy
from .coalesce import coalesce
from .transpose import transpose
from .eye import eye
from .spmm import spmm
from .spspmm import spspmm

__all__ = [
    'SparseStorage',
    'SparseTensor',
    't',
    'narrow',
    '__narrow_diag__',
    'select',
    'index_select',
    'index_select_nnz',
    'masked_select',
    'masked_select_nnz',
    'remove_diag',
    'set_diag',
    'fill_diag',
    'add',
    'add_',
    'add_nnz',
    'add_nnz_',
    'mul',
    'mul_',
    'mul_nnz',
    'mul_nnz_',
    'sum',
    'mean',
    'min',
    'max',
    'matmul',
    'cat',
    'cat_diag',
    'to_torch_sparse',
    'from_torch_sparse',
    'to_scipy',
    'from_scipy',
    'coalesce',
    'transpose',
    'eye',
    'spmm',
    'spspmm',
    '__version__',
]
