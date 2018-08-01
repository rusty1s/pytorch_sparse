import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.1.0'
url = 'https://github.com/rusty1s/pytorch_sparse'

install_requires = ['numpy', 'scipy']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']
ext_modules = []
cmdclass = {}

if torch.cuda.is_available():
    ext_modules += [
        CUDAExtension('matmul_cuda',
                      ['cuda/matmul.cpp', 'cuda/matmul_cuda.cu'])
    ]
    cmdclass['build_ext'] = BuildExtension

setup(
    name='torch_sparse',
    version=__version__,
    description='PyTorch Extension Library of Optimized Sparse Matrix '
    'Operations',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'sparse', 'deep-learning'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
