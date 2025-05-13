from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import platform
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import subprocess
        import os

        build_dir = os.path.abspath(self.build_temp)
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        # Configure
        cmake_args = [
            '-DCMAKE_BUILD_TYPE=Release',
            '-DPYTHON_EXECUTABLE=' + sys.executable,
        ]
        
        # Add Windows-specific configuration
        if platform.system() == "Windows":
            cmake_args.extend([
                '-G', 'Visual Studio 17 2022',  # Use Visual Studio 2022
                '-A', 'x64',  # Use 64-bit architecture
            ])
        
        try:
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_dir)
        except subprocess.CalledProcessError as e:
            print(f"CMake configuration failed: {e}")
            raise
        
        # Build
        try:
            if platform.system() == "Windows":
                subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'], cwd=build_dir)
                lib_path = os.path.join(build_dir, 'lib', 'Release', 'cuda_ops.dll')
            else:
                subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)
                lib_path = os.path.join(build_dir, 'lib', 'libcuda_ops.so')
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed: {e}")
            raise
        
        if not os.path.exists(lib_path):
            raise RuntimeError(f"Could not find built library at {lib_path}")

        # Create the extension directory if it doesn't exist
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not os.path.exists(ext_dir):
            os.makedirs(ext_dir)

        # Copy the library to the extension directory
        import shutil
        if platform.system() == "Windows":
            shutil.copy2(lib_path, os.path.join(ext_dir, 'cuda_ops.pyd'))
        else:
            shutil.copy2(lib_path, os.path.join(ext_dir, 'cuda_ops.so'))

setup(
    name='cuda_ops',
    version='0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='CUDA matrix operations',
    long_description='',
    ext_modules=[CMakeExtension('cuda_ops')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
) 