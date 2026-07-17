#!/usr/bin/env python3
"""Build script for the DarkNews.Cfourvec Cython extension.

All project metadata lives in ``pyproject.toml`` (PEP 621). This file only
exists to declare the C extension, which setuptools cannot express purely
declaratively. Cython and numpy are guaranteed to be present at build time by
``[build-system].requires`` in pyproject.toml.
"""
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = cythonize(
    [
        Extension(
            "DarkNews.Cfourvec",
            ["src/DarkNews/Cfourvec.pyx"],
            include_dirs=[np.get_include()],
        ),
    ],
    compiler_directives={"language_level": "3", "embedsignature": True},
)

if __name__ == "__main__":
    setup(ext_modules=extensions, include_dirs=[np.get_include()])
