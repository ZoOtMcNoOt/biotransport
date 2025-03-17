cmake_minimum_required(VERSION 3.13)
project(biotransport VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Add the C++ library
add_subdirectory(cpp)

# Add Python bindings if requested
if(BUILD_PYTHON_BINDINGS)
    # Find Python
    find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
    
    # Add pybind11 (download if not available)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.10.0
    )
    FetchContent_MakeAvailable(pybind11)
    
    # Add the bindings directory
    add_subdirectory(python/bindings)
endif()

# Install targets
install(TARGETS biotransport
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)