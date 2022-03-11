import glob
import os
import os.path as osp
import platform
import sys
from itertools import product

import torch
from setuptools import find_packages, setup
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

__version__ = '0.6.13'
URL = 'https://github.com/rusty1s/pytorch_sparse'

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
suffices = ['cpu', 'cuda'] if WITH_CUDA else ['cpu']
if os.getenv('FORCE_CUDA', '0') == '1':
    suffices = ['cuda', 'cpu']
if os.getenv('FORCE_ONLY_CUDA', '0') == '1':
    suffices = ['cuda']
if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    suffices = ['cpu']

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'

WITH_METIS = True if os.getenv('WITH_METIS', '0') == '1' else False
WITH_MTMETIS = True if os.getenv('WITH_MTMETIS', '0') == '1' else False

WITH_SYMBOLS = True if os.getenv('WITH_SYMBOLS', '0') == '1' else False


def get_extensions():
    extensions = []

    extensions_dir = osp.join('csrc')
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))

    for main, suffix in product(main_files, suffices):
        define_macros = [('WITH_PYTHON', None)]

        if sys.platform == 'win32':
            define_macros += [('torchsparse_EXPORTS', None)]

        libraries = []
        if WITH_METIS:
            define_macros += [('WITH_METIS', None)]
            libraries += ['metis']
        if WITH_MTMETIS:
            define_macros += [('WITH_MTMETIS', None)]
            define_macros += [('MTMETIS_64BIT_VERTICES', None)]
            define_macros += [('MTMETIS_64BIT_EDGES', None)]
            define_macros += [('MTMETIS_64BIT_WEIGHTS', None)]
            define_macros += [('MTMETIS_64BIT_PARTITIONS', None)]
            libraries += ['mtmetis', 'wildriver']

        extra_compile_args = {'cxx': ['-O2']}
        if not os.name == 'nt':  # Not on Windows:
            extra_compile_args['cxx'] += ['-Wno-sign-compare']
        extra_link_args = [] if WITH_SYMBOLS else ['-s']

        info = parallel_info()
        if ('backend: OpenMP' in info and 'OpenMP not found' not in info
                and sys.platform != 'darwin'):
            extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
            if sys.platform == 'win32':
                extra_compile_args['cxx'] += ['/openmp']
            else:
                extra_compile_args['cxx'] += ['-fopenmp']
        else:
            print('Compiling without OpenMP...')

        # Compile for mac arm64
        if (sys.platform == 'darwin' and platform.machine() == 'arm64'):
            extra_compile_args['cxx'] += ['-arch', 'arm64']
            extra_link_args += ['-arch', 'arm64']

        if suffix == 'cuda':
            define_macros += [('WITH_CUDA', None)]
            nvcc_flags = os.getenv('NVCC_FLAGS', '')
            nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
            nvcc_flags += ['--expt-relaxed-constexpr', '-O2']
            extra_compile_args['nvcc'] = nvcc_flags

            if sys.platform == 'win32':
                extra_link_args += ['cusparse.lib']
            else:
                extra_link_args += ['-lcusparse', '-l', 'cusparse']

        name = main.split(os.sep)[-1][:-4]
        sources = [main]

        path = osp.join(extensions_dir, 'cpu', f'{name}_cpu.cpp')
        if osp.exists(path):
            sources += [path]

        path = osp.join(extensions_dir, 'cuda', f'{name}_cuda.cu')
        if suffix == 'cuda' and osp.exists(path):
            sources += [path]

        Extension = CppExtension if suffix == 'cpu' else CUDAExtension
        extension = Extension(
            f'torch_sparse._{name}_{suffix}',
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
        )
        extensions += [extension]

    return extensions


install_requires = [
    'scipy',
]

test_requires = [
    'pytest',
    'pytest-cov',
]

setup(
    name='torch_sparse',
    version=__version__,
    description=('PyTorch Extension Library of Optimized Autograd Sparse '
                 'Matrix Operations'),
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'pytorch',
        'sparse',
        'sparse-matrices',
        'autograd',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'test': test_requires,
    },
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
    include_package_data=True,
)
