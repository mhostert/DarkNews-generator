# boostrap install numpy and Cython
# from setuptools import dist
# dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10', 'wheel'])

# proceed as usual
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup_args = dict(
    ext_modules = cythonize(["src/Darknews/Cfourvec.pyx"]),
    include_dirs=np.get_include(),
    zip_safe=False,
)

if __name__ == "__main__":
    setup(**setup_args)