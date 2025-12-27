# Benchmarks for biotransport performance tracking

# Check for OpenMP if enabled
if(BIOTRANSPORT_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found - benchmarks will include parallel variants")
    else()
        message(WARNING "OpenMP requested but not found")
    endif()
endif()

# Common benchmark utilities
add_library(bench_utils INTERFACE)
target_include_directories(bench_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# 2D Diffusion benchmark
add_executable(bench_diffusion_2d bench_diffusion_2d.cpp)
target_link_libraries(bench_diffusion_2d PRIVATE biotransport bench_utils)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_diffusion_2d PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(bench_diffusion_2d PRIVATE BIOTRANSPORT_HAS_OPENMP)
endif()

# Linear reaction-diffusion benchmark
add_executable(bench_linear_reaction_diffusion bench_linear_reaction_diffusion.cpp)
target_link_libraries(bench_linear_reaction_diffusion PRIVATE biotransport bench_utils)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_linear_reaction_diffusion PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(bench_linear_reaction_diffusion PRIVATE BIOTRANSPORT_HAS_OPENMP)
endif()

# Variable diffusion benchmark
add_executable(bench_variable_diffusion bench_variable_diffusion.cpp)
target_link_libraries(bench_variable_diffusion PRIVATE biotransport bench_utils)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_variable_diffusion PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(bench_variable_diffusion PRIVATE BIOTRANSPORT_HAS_OPENMP)
endif()
