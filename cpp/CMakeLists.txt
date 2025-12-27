# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files - updated to match actual file locations
set(SOURCES
    # Core
    src/core/mesh/structured_mesh.cpp
    src/core/mesh/cylindrical_mesh.cpp
    src/core/utils.cpp
    # Mass transport (specialized solvers only - base solvers are header-only)
    src/physics/mass_transport/gray_scott.cpp
    src/physics/mass_transport/tumor_drug_delivery.cpp
    # Heat transfer
    src/physics/heat_transfer/bioheat_cryotherapy.cpp
    # Fluid dynamics
    src/physics/fluid_dynamics/darcy_flow.cpp
    src/physics/fluid_dynamics/stokes.cpp
    src/physics/fluid_dynamics/navier_stokes.cpp)

# Create the library
add_library(biotransport ${SOURCES})

# Set include directories for users of the library
target_include_directories(
  biotransport PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>)

# Link OpenMP if enabled
if(BIOTRANSPORT_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(biotransport PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add tests if needed
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
