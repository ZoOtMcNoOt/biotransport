from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import os
import sys
import subprocess


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            "-DPython_EXECUTABLE=" + sys.executable,
            "-DBUILD_PYTHON_BINDINGS=ON",
            # CMake 4+ removes compatibility with cmake_minimum_required(<3.5).
            # pybind11 v2.10 declares 3.4, so this keeps configuration working.
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ]

        # Configuration
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        # Multi-config generators (Visual Studio) place outputs under a per-config
        # subdirectory unless config-specific output dirs are set.
        cfg_upper = cfg.upper()
        cmake_args += [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg_upper}={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg_upper}={extdir}",
        ]
        # Use CMake's native parallel build flag (portable across generators).
        jobs = os.environ.get("JOBS", "2")
        build_args += ["--parallel", jobs]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        # Do not run `cmake --install` here: it can attempt to install the C++ library
        # into system locations on Windows, requiring admin privileges. The extension
        # module is already emitted into `extdir` via the output directory settings.


setup(
    name="biotransport",
    version="0.1.0",
    author="Grant McNatt",
    author_email="gmcnatt1@tamu.edu",
    description="A library for biotransport phenomena modeling",
    long_description="",
    packages=find_packages("python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("biotransport._core._core")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.17.0",
        "matplotlib>=3.0.0",
        "scipy>=1.9.0",
        "imageio>=2.19.0",
    ],
)
