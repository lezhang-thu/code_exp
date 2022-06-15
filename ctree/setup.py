from setuptools import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize(Extension('cytree',
                                      sources=['cytree.pyx'],
                                      extra_compile_args=['-O3', '-std=c++11'],
                                      include_dirs=[]),
                            language_level="3"), )
