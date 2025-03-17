# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/core/mesh.cpp
    src/core/utils.cpp
    src/solvers/diffusion.cpp
    src/solvers/reaction_diffusion.cpp
)

# Create the library
add_library(biotransport ${SOURCES})

# Set include directories for users of the library
target_include_directories(biotransport PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add tests if needed
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()