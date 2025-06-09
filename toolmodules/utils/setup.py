from distutils.core import setup, Extension
from Cython.Build import cythonize

#ext = Extension(name="utils", sources=["classifier_base.py", "logger.py", "parallelism.py"])


#name = 'utils',
setup(
    ext_modules = cythonize(["__init__.py", "classifier_base.py", "logger.py", "parallelism.py"])
)

