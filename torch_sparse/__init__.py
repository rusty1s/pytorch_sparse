import importlib
import os.path as osp

import torch

__version__ = '0.6.7'

for library in [
        '_version', '_convert', '_diag', '_spmm', '_spspmm', '_metis', '_rw',
        '_saint', '_sample', '_relabel'
]:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        library, [osp.dirname(__file__)]).origin)

if torch.cuda.is_available() and torch.version.cuda:  # pragma: no cover
    cuda_version = torch.ops.torch_sparse.cuda_version()

    if cuda_version == -1:
        major = minor = 0
    elif cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major or t_minor != minor:
        raise RuntimeError(
            f'Detected that PyTorch and torch_sparse were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_sparse has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_sparse that '
            f'matches your PyTorch install.')

from .storage import SparseStorage  # noqa
from .tensor import SparseTensor  # noqa
from .transpose import t  # noqa
from .narrow import narrow, __narrow_diag__  # noqa
from .select import select  # noqa
from .index_select import index_select, index_select_nnz  # noqa
from .masked_select import masked_select, masked_select_nnz  # noqa
from .permute import permute  # noqa
from .diag import remove_diag, set_diag, fill_diag  # noqa
from .add import add, add_, add_nnz, add_nnz_  # noqa
from .mul import mul, mul_, mul_nnz, mul_nnz_  # noqa
from .reduce import sum, mean, min, max  # noqa
from .matmul import matmul  # noqa
from .cat import cat  # noqa
from .rw import random_walk  # noqa
from .metis import partition  # noqa
from .bandwidth import reverse_cuthill_mckee  # noqa
from .saint import saint_subgraph  # noqa
from .padding import padded_index, padded_index_select  # noqa
from .sample import sample, sample_adj  # noqa

from .convert import to_torch_sparse, from_torch_sparse  # noqa
from .convert import to_scipy, from_scipy  # noqa
from .coalesce import coalesce  # noqa
from .transpose import transpose  # noqa
from .eye import eye  # noqa
from .spmm import spmm  # noqa
from .spspmm import spspmm  # noqa

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
    'permute',
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
    'random_walk',
    'partition',
    'reverse_cuthill_mckee',
    'saint_subgraph',
    'padded_index',
    'padded_index_select',
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
