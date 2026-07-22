"""Setuptools build hook for the CMake-backed Python extension.

Project metadata and dependency declarations live in ``pyproject.toml``. This
file intentionally contains only the custom native build integration.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """An extension whose sources are configured and built by CMake."""

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir or ".").resolve()


class CMakeBuild(build_ext):
    """Build the pybind11 extension without installing native files globally."""

    def build_extension(self, ext: CMakeExtension) -> None:
        extension_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_dir = Path(self.build_temp).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)

        configuration = "Debug" if self.debug else "Release"
        configuration_upper = configuration.upper()
        extension_output = f"{extension_dir}{os.sep}"
        pybind11_cmake_dir = Path(pybind11.get_cmake_dir()).resolve()
        eigen_cmake_dir = Path(
            distribution("cmeel-eigen").locate_file("cmeel.prefix/share/eigen3/cmake")
        ).resolve()
        if not (pybind11_cmake_dir / "pybind11Config.cmake").is_file():
            raise RuntimeError(
                f"pybind11 CMake package is missing from {pybind11_cmake_dir}"
            )
        if not (eigen_cmake_dir / "Eigen3Config.cmake").is_file():
            raise RuntimeError(
                f"Eigen3 CMake package is missing from {eigen_cmake_dir}"
            )

        # Parallel kernels must be explicitly enabled after their numerical
        # equivalence has been validated. CMAKE_ARGS remains the escape hatch
        # for controlled performance builds.
        cmake_args = [
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_BENCHMARKS=OFF",
            "-DBIOTRANSPORT_OPENMP=OFF",
            f"-Dpybind11_DIR={pybind11_cmake_dir.as_posix()}",
            f"-DEigen3_DIR={eigen_cmake_dir.as_posix()}",
            f"-DCMAKE_BUILD_TYPE={configuration}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_output}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extension_output}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{configuration_upper}={extension_output}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{configuration_upper}={extension_output}",
            # CMake 4 removed compatibility with pybind11's historical minimum.
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ]
        cmake_args.extend(shlex.split(os.environ.get("CMAKE_ARGS", "")))

        parallelism = os.environ.get(
            "CMAKE_BUILD_PARALLEL_LEVEL", os.environ.get("JOBS", "2")
        )
        build_args = ["--config", configuration, "--parallel", parallelism]

        subprocess.check_call(
            ["cmake", "-S", str(ext.sourcedir), "-B", str(build_dir), *cmake_args]
        )
        subprocess.check_call(["cmake", "--build", str(build_dir), *build_args])


setup(
    ext_modules=[CMakeExtension("biotransport._core._core")],
    cmdclass={"build_ext": CMakeBuild},
)
