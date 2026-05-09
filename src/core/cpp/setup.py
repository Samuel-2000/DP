from setuptools import setup, Extension
import pybind11
import platform

extra_compile_args = []
if platform.system() == 'Windows':
    extra_compile_args = ['/O2', '/std:c++17', '/fp:fast']
else:
    extra_compile_args = ['-O3', '-march=native', '-std=c++17', '-ffast-math']

ext = Extension(
    "maze_core",
    sources=["environment.cpp", "vector_environment.cpp", "bindings.cpp"],
    include_dirs=[".", pybind11.get_include()],
    language='c++',
    extra_compile_args=extra_compile_args,
)

setup(
    name="maze_core",
    version="1.0.0",
    description="Maze environment C++ core (single + vectorised)",
    ext_modules=[ext],
    zip_safe=False,
)