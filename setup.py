from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.2.1'
url = 'https://github.com/rusty1s/pytorch_sparse'

install_requires = ['scipy']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']
ext_modules = []
cmdclass = {}

if torch.cuda.is_available():
    ext_modules += [
        CUDAExtension(
            'spspmm_cuda',
            ['cuda/spspmm.cpp', 'cuda/spspmm_kernel.cu'],
            extra_link_args=['-lcusparse', '-l', 'cusparse'],
        )
    ]
    cmdclass['build_ext'] = BuildExtension

setup(
    name='torch_sparse',
    version=__version__,
    description='PyTorch Extension Library of Optimized Autograd Sparse '
    'Matrix Operations',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'sparse', 'autograd', 'deep-learning'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
