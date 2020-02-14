import importlib
import os.path as osp

import torch

__version__ = '0.5.1'
expected_torch_version = (1, 4)

try:
    for library in ['_version', '_convert', '_diag', '_spmm', '_spspmm']:
        torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
            library, [osp.dirname(__file__)]).origin)
except OSError as e:
    if 'undefined symbol' in str(e):
        major, minor = [int(x) for x in torch.__version__.split('.')[:2]]
        t_major, t_minor = expected_torch_version
        if major != t_major or (major == t_major and minor != t_minor):
            raise RuntimeError(
                f'Expected PyTorch version {t_major}.{t_minor} but found '
                f'version {major}.{minor}.')
    raise OSError(e)

if torch.version.cuda is not None:  # pragma: no cover
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

from .storage import SparseStorage  # noqa: E4402
from .tensor import SparseTensor  # noqa: E4402
from .transpose import t  # noqa: E4402
from .narrow import narrow, __narrow_diag__  # noqa: E4402
from .select import select  # noqa: E4402
from .index_select import index_select, index_select_nnz  # noqa: E4402
from .masked_select import masked_select, masked_select_nnz  # noqa: E4402
from .diag import remove_diag, set_diag, fill_diag  # noqa: E4402
from .add import add, add_, add_nnz, add_nnz_  # noqa: E4402
from .mul import mul, mul_, mul_nnz, mul_nnz_  # noqa: E4402
from .reduce import sum, mean, min, max  # noqa: E4402
from .matmul import matmul  # noqa: E4402
from .cat import cat, cat_diag  # noqa: E4402

from .convert import to_torch_sparse, from_torch_sparse  # noqa: E4402
from .convert import to_scipy, from_scipy  # noqa: E4402
from .coalesce import coalesce  # noqa: E4402
from .transpose import transpose  # noqa: E4402
from .eye import eye  # noqa: E4402
from .spmm import spmm  # noqa: E4402
from .spspmm import spspmm  # noqa: E4402

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
