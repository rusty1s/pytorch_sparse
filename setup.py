import platform
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

__version__ = '0.2.4'
url = 'https://github.com/rusty1s/pytorch_sparse'

install_requires = ['scipy']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']
ext_modules = []
cmdclass = {}

if CUDA_HOME is not None:
    if platform.system() == 'Windows':
        extra_link_args = ['cusparse.lib']
    else:
        extra_link_args = ['-lcusparse', '-l', 'cusparse']

    ext_modules += [
        CUDAExtension(
            'torch_sparse.spspmm_cuda',
            ['cuda/spspmm.cpp', 'cuda/spspmm_kernel.cu'],
            extra_link_args=extra_link_args),
        CUDAExtension('torch_sparse.unique_cuda',
                      ['cuda/unique.cpp', 'cuda/unique_kernel.cu']),
    ]
    cmdclass['build_ext'] = BuildExtension

setup(
    name='torch_sparse',
    version=__version__,
    description=('PyTorch Extension Library of Optimized Autograd Sparse '
                 'Matrix Operations'),
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
