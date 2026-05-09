from setuptools import setup, Extension
import pybind11
import os
import sys
import platform
import glob

def find_opencv_windows():
    """Find OpenCV on Windows using OpenCV_DIR or typical paths."""
    opencv_root = None
    if 'OpenCV_DIR' in os.environ:
        opencv_root = os.environ['OpenCV_DIR']
    elif 'OPENCV_DIR' in os.environ:
        opencv_root = os.environ['OPENCV_DIR']
    else:
        candidates = [
            r"C:\opencv\build",
            r"C:\Program Files\opencv\build",
            r"C:\tools\opencv\build",
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "include", "opencv2", "opencv.hpp")):
                opencv_root = cand
                break

    if opencv_root is None:
        raise RuntimeError(
            "OpenCV not found. Set environment variable OpenCV_DIR to the root directory "
            "of OpenCV (e.g., C:\\opencv\\build) and try again.\n"
            "If you don't have OpenCV, download it from https://opencv.org/releases/"
        )

    inc_dir = os.path.join(opencv_root, "include")
    if not os.path.exists(os.path.join(inc_dir, "opencv2", "opencv.hpp")):
        raise RuntimeError(f"OpenCV headers not found in {inc_dir}")

    # Find library directory
    lib_candidates = [
        os.path.join(opencv_root, "x64", "vc15", "lib"),
        os.path.join(opencv_root, "x64", "vc16", "lib"),
        os.path.join(opencv_root, "lib"),
    ]
    lib_dir = None
    for cand in lib_candidates:
        if os.path.exists(cand):
            lib_dir = cand
            break
    if lib_dir is None:
        raise RuntimeError(f"OpenCV library directory not found in {opencv_root}")

    # Find all opencv_*.lib files
    lib_files = glob.glob(os.path.join(lib_dir, "opencv_*.lib"))
    if not lib_files:
        raise RuntimeError(f"No OpenCV .lib files found in {lib_dir}")

    # Return include flags and list of full library paths
    include_flags = [f"/I{inc_dir}"]
    return include_flags, lib_files

def get_opencv_flags():
    """Return include and link flags for OpenCV according to platform."""
    if platform.system() == 'Windows':
        return find_opencv_windows()
    else:
        import subprocess
        try:
            cflags = subprocess.check_output(["pkg-config", "--cflags", "opencv4"]).decode().strip().split()
            libs = subprocess.check_output(["pkg-config", "--libs", "opencv4"]).decode().strip().split()
            return cflags, libs
        except:
            # Fallback to common paths
            return ['-I/usr/include/opencv4'], ['-lopencv_core', '-lopencv_imgproc', '-lopencv_highgui']

def get_optimization_flags():
    """Optimization flags according to platform."""
    if platform.system() == 'Windows':
        return ['/O2', '/std:c++17', '/fp:fast', '/arch:AVX2', '/EHsc']
    else:
        flags = ['-O3', '-march=native', '-ffast-math', '-fopenmp',
                 '-funroll-loops', '-std=c++17']
        if platform.machine() in ['x86_64', 'amd64']:
            flags.extend(['-msse4.2', '-mavx', '-mfma'])
        return flags

# Get flags
opencv_cflags, opencv_libs = get_opencv_flags()
opt_flags = get_optimization_flags()

# Source files
sources = ["environment.cpp", "vector_environment.cpp", "bindings.cpp"]

# Include directories: current dir, pybind11, numpy, OpenCV
include_dirs = [
    ".",
    pybind11.get_include(),
]
# Add OpenCV include directories (strip -I prefix)
for flag in opencv_cflags:
    if flag.startswith('-I'):
        include_dirs.append(flag[2:])
    elif flag.startswith('/I'):
        include_dirs.append(flag[2:])

ext_modules = [
    Extension(
        "maze_core",
        sources=sources,
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=opt_flags + [flag for flag in opencv_cflags if not flag.startswith('-I') and not flag.startswith('/I')],
        extra_link_args=opencv_libs + (['/openmp'] if platform.system() == 'Windows' else ['-fopenmp']),
    )
]

setup(
    name="maze_core",
    version="1.0.0",
    description="Maze environment C++ core with OpenCV rendering",
    ext_modules=ext_modules,
    zip_safe=False,
)