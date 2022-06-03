from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# URL: https://stackoverflow.com/a/49041815
# Avoid a gcc warning below:
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid
# for C/ObjC but not for C++
class BuildExt(build_ext):
    def build_extensions(self):
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()


from Cython.Build import cythonize

setup(cmdclass={'build_ext': BuildExt},
      ext_modules=cythonize(
          Extension('cytree',
                    sources=['cytree.pyx'],
                    language='c++',
                    extra_compile_args=['-O3'],
                    include_dirs=[])))
