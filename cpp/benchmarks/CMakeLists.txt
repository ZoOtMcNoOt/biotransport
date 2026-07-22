# Reproducible native performance evidence. These executables intentionally use
# only BioTransport's current public C++ headers and the standard library.

find_package(Git QUIET)
set(BIOTRANSPORT_BENCH_GIT_REVISION "unknown")
set(BIOTRANSPORT_BENCH_GIT_DIRTY 0)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --verify HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE BIOTRANSPORT_BENCH_GIT_RESULT
    OUTPUT_VARIABLE BIOTRANSPORT_BENCH_GIT_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(BIOTRANSPORT_BENCH_GIT_RESULT EQUAL 0)
    set(BIOTRANSPORT_BENCH_GIT_REVISION "${BIOTRANSPORT_BENCH_GIT_OUTPUT}")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" status --porcelain
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE BIOTRANSPORT_BENCH_GIT_STATUS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(NOT BIOTRANSPORT_BENCH_GIT_STATUS STREQUAL "")
      set(BIOTRANSPORT_BENCH_GIT_DIRTY 1)
    endif()
  endif()
endif()

function(biotransport_escape_cxx_string output value)
  string(REPLACE "\\" "\\\\" escaped "${value}")
  string(REPLACE "\"" "\\\"" escaped "${escaped}")
  string(REPLACE "\n" " " escaped "${escaped}")
  set(${output} "${escaped}" PARENT_SCOPE)
endfunction()

biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_CXX_FLAGS_DEBUG
                               "${CMAKE_CXX_FLAGS_DEBUG}")
biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_CXX_FLAGS_RELEASE
                               "${CMAKE_CXX_FLAGS_RELEASE}")
biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_CXX_FLAGS_RELWITHDEBINFO
                               "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_CXX_FLAGS_MINSIZEREL
                               "${CMAKE_CXX_FLAGS_MINSIZEREL}")
biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_OPENMP_FLAGS
                               "${OpenMP_CXX_FLAGS}")
set(BIOTRANSPORT_BENCH_TARGET_FLAGS "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(BIOTRANSPORT_BENCH_TARGET_FLAGS "/EHsc")
endif()
biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_GENERATOR
                               "${CMAKE_GENERATOR}")
biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_SYSTEM_NAME
                               "${CMAKE_SYSTEM_NAME}")
biotransport_escape_cxx_string(BIOTRANSPORT_BENCH_SYSTEM_PROCESSOR
                               "${CMAKE_SYSTEM_PROCESSOR}")

configure_file(bench_build_config.hpp.in bench_build_config.hpp @ONLY)

add_library(bench_utils INTERFACE)
target_include_directories(
  bench_utils
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

function(add_biotransport_benchmark target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE biotransport bench_utils)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${target} PRIVATE /EHsc)
  endif()
  target_compile_definitions(
    ${target} PRIVATE BIOTRANSPORT_BENCH_BUILD_TYPE="$<CONFIG>")
endfunction()

add_biotransport_benchmark(bench_diffusion_2d bench_diffusion_2d.cpp)
add_biotransport_benchmark(bench_linear_reaction_diffusion
                           bench_linear_reaction_diffusion.cpp)
add_biotransport_benchmark(bench_variable_diffusion
                           bench_variable_diffusion.cpp)
