from distutils.core import setup, Extension
from Cython.Distutils import build_ext

setup( name = "volterra",
       version = "0.0",
       packages = ["volterra"],
       cmdclass = {'build_ext': build_ext},
       ext_modules = [Extension("volterra.cython_solvers", ["volterra/cython_solvers.pyx"])]
      )
