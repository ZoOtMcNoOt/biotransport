cmake_minimum_required(VERSION 3.16)
project(
  biotransport
  VERSION 0.2.0
  LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(BIOTRANSPORT_OPENMP "Enable validated OpenMP parallelization" OFF)
option(BIOTRANSPORT_EIGEN "Enable Eigen for sparse matrices" ON)
option(BIOTRANSPORT_WARNINGS
       "Compile the library, bindings and tests with a strict warning set" ON)
option(BIOTRANSPORT_WERROR
       "Treat compiler warnings as errors (CI and development only)" OFF)

# Warning flags live on one INTERFACE target that is linked PRIVATE by the
# library, the Python extension and the tests, so consumers of the installed
# package never inherit them.
add_library(biotransport_warnings INTERFACE)
if(BIOTRANSPORT_WARNINGS)
  target_compile_options(
    biotransport_warnings
    INTERFACE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion;-Wnon-virtual-dtor;-Wold-style-cast;-Wcast-align;-Wunused;-Woverloaded-virtual;-Wnull-dereference;-Wdouble-promotion;-Wformat=2>
      $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/permissive->)
  if(BIOTRANSPORT_WERROR)
    target_compile_options(
      biotransport_warnings
      INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>
                $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX>)
  endif()
endif()

# OpenMP configuration
if(BIOTRANSPORT_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: version ${OpenMP_CXX_VERSION}")
  else()
    message(
      WARNING
        "BIOTRANSPORT_OPENMP was requested, but OpenMP is unavailable. Building the deterministic serial implementation."
    )
  endif()
endif()

# Eigen configuration (header-only library for sparse matrices)
if(BIOTRANSPORT_EIGEN)
  find_package(Eigen3 3.4 CONFIG REQUIRED)
  message(STATUS "Eigen sparse backend enabled")
endif()

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

  # Set variables that pybind11's find_package(PythonInterp/PythonLibs) might
  # use This ensures pybind11 uses the SAME Python we found
  set(PYTHON_EXECUTABLE
      ${Python_EXECUTABLE}
      CACHE PATH "Python executable" FORCE)
  set(PYBIND11_PYTHON_VERSION
      ${Python_VERSION}
      CACHE STRING "Python version" FORCE)

  # Python package builds declare pybind11 as a PEP 517 build dependency and
  # pass its CMake directory explicitly. Standalone native builds therefore fail
  # with an actionable dependency error instead of cloning undeclared source
  # code during configuration.
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 2.13.6 CONFIG REQUIRED)

  # Add the bindings directory
  add_subdirectory(python/bindings)
endif()

# Add benchmarks if requested
if(BUILD_BENCHMARKS)
  add_subdirectory(cpp/benchmarks)
endif()

# Install a consumable C++ package.
install(
  TARGETS biotransport
  EXPORT biotransportTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY cpp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE readme.md DESTINATION ${CMAKE_INSTALL_DOCDIR})

configure_package_config_file(
  cmake/biotransportConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/biotransportConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/biotransport")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/biotransportConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)

install(
  EXPORT biotransportTargets
  FILE biotransportTargets.cmake
  NAMESPACE biotransport::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/biotransport")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/biotransportConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/biotransportConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/biotransport")
