import platform
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

extra_compile_args = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    extra_compile_args += ['-DVERSION_GE_1_3']

ext_modules = [
    CppExtension('torch_sparse.spspmm_cpu', ['cpu/spspmm.cpp'],
                 extra_compile_args=extra_compile_args)
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if CUDA_HOME is not None:
    if platform.system() == 'Windows':
        extra_link_args = ['cusparse.lib']
    else:
        extra_link_args = ['-lcusparse', '-l', 'cusparse']

    ext_modules += [
        CUDAExtension('torch_sparse.spspmm_cuda',
                      ['cuda/spspmm.cpp', 'cuda/spspmm_kernel.cu'],
                      extra_link_args=extra_link_args,
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_sparse.unique_cuda',
                      ['cuda/unique.cpp', 'cuda/unique_kernel.cu'],
                      extra_compile_args=extra_compile_args),
    ]

__version__ = '0.4.3'
url = 'https://github.com/rusty1s/pytorch_sparse'

install_requires = ['scipy']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_sparse',
    version=__version__,
    description=('PyTorch Extension Library of Optimized Autograd Sparse '
                 'Matrix Operations'),
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'sparse',
        'sparse-matrices',
        'autograd',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
