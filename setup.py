#!/usr/bin/env python3

# proceed as usual
from setuptools import setup, Extension
import numpy as np
import os

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


extensions = [
    Extension("DarkNews.Cfourvec", ["src/DarkNews/Cfourvec.pyx"], include_dirs=[np.get_include()]),
]
CYTHONIZE = cythonize is not None
if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

setup_args = dict(
    ext_modules=cythonize(["src/DarkNews/Cfourvec.pyx"]),
    # ext_modules=extensions,
    include_dirs=[np.get_include()],
)


if __name__ == "__main__":
    setup(**setup_args)
