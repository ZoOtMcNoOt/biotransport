cmake_minimum_required(VERSION 3.13)
project(
  biotransport
  VERSION 0.1.0
  LANGUAGES CXX)

include(CTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(BIOTRANSPORT_OPENMP "Enable OpenMP parallelization" OFF)

# Prefer the active virtual environment when present (but do not require one)
set(Python_FIND_VIRTUALENV FIRST)

# Add the C++ library
add_subdirectory(cpp)

# Add Python bindings if requested
if(BUILD_PYTHON_BINDINGS)
  # Find Python - be very explicit about version and location
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development)
  message(
    STATUS "Found Python: ${Python_EXECUTABLE} (Version: ${Python_VERSION})")
  message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")

  # Add pybind11 (download if not available)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.0)
  FetchContent_MakeAvailable(pybind11)

  # Add the bindings directory
  add_subdirectory(python/bindings)
endif()

# Add benchmarks if requested
if(BUILD_BENCHMARKS)
  add_subdirectory(cpp/benchmarks)
endif()

# Install targets
install(
  TARGETS biotransport
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
