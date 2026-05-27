# src/core/cpp_build.py
"""
Helper functions for building and importing the C++ maze_core module.

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path

def install_requirements():
    """Install Python requirements from requirements.txt."""
    req_file = Path(__file__).parent.parent.parent / "requirements.txt"
    if not req_file.exists():
        print("requirements.txt not found, skipping.")
        return True
    print("Installing Python requirements...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print("Requirements installed successfully!")
        return True
    else:
        print("Failed to install requirements!")
        print(result.stderr)
        return False

def add_opencv_dll_path():
    """Add OpenCV DLL directory to the DLL search path on Windows."""
    if platform.system() != 'Windows':
        return
    opencv_root = None
    # Check environment variables
    for var in ['OpenCV_DIR', 'OPENCV_DIR']:
        if var in os.environ:
            opencv_root = os.environ[var]
            break
    if not opencv_root:
        candidates = [
            r"C:\opencv\build",
            r"C:\Program Files\opencv\build",
            r"C:\tools\opencv\build",
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "include", "opencv2", "opencv.hpp")):
                opencv_root = cand
                break
    if opencv_root:
        bin_candidates = [
            os.path.join(opencv_root, "x64", "vc15", "bin"),
            os.path.join(opencv_root, "x64", "vc16", "bin"),
            os.path.join(opencv_root, "bin"),
        ]
        for bin_dir in bin_candidates:
            if os.path.exists(bin_dir):
                os.add_dll_directory(bin_dir)
                print(f"✅ Added OpenCV DLL path: {bin_dir}")
                return
    print("⚠️  OpenCV DLL directory not found; video saving may fail.")

def check_cpp_extension():
    """Check if C++ module (maze_core) is available."""
    try:
        import maze_core
        print("✓ C++ module (maze_core) is available")
        return True
    except ImportError as e:
        print(f"C++ module not available: {e}")
        return False

def build_cpp_extension():
    """Build the C++ maze_core module."""
    print("Building C++ maze core module...")
    cpp_dir = Path(__file__).parent / "cpp"
    if not cpp_dir.exists():
        print(f"Error: {cpp_dir} not found!")
        return False

    original_dir = Path.cwd()
    os.chdir(cpp_dir)
    try:
        # Clean previous binaries
        for ext in ["*.so", "*.pyd", "*.dll"]:
            for f in cpp_dir.glob(ext):
                f.unlink()

        result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("C++ module built successfully!")
            built = [f.name for f in cpp_dir.glob("*") if f.suffix in ('.so', '.pyd', '.dll')]
            print(f"Built files: {', '.join(built)}")
            # Move compiled module to project root
            for f in built:
                src = cpp_dir / f
                dst = original_dir / f
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
                print(f"Moved {f} → project root")
            return True
        else:
            print("Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"Build error: {e}")
        return False
    finally:
        os.chdir(original_dir)

def ensure_cpp_module():
    """Ensure the C++ module is built and importable."""
    if not install_requirements():
        print("Continuing anyway...")
    add_opencv_dll_path()
    if not check_cpp_extension():
        if not build_cpp_extension():
            sys.exit(1)
        add_opencv_dll_path()  # try again after build
        if not check_cpp_extension():
            print("C++ module still not available after build!")
            sys.exit(1)