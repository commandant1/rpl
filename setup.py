from setuptools import setup, find_packages
import os
import subprocess
from setuptools.command.build_py import build_py

class BuildCLib(build_py):
    def run(self):
        # Create build directory if it doesn't exist
        if not os.path.exists("build"):
            os.makedirs("build")
        
        # Run cmake and make
        subprocess.check_call(["cmake", ".."], cwd="build")
        subprocess.check_call(["make", "-j4"], cwd="build")
        
        # Call the original build_py
        super().run()

setup(
    name="rpl-learn",
    version="0.1.0",
    description="High-performance ML library for Raspberry Pi 4",
    author="RPL Team",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=[
        "numpy",
    ],
    cmdclass={
        'build_py': BuildCLib,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.7",
)
