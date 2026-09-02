# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files - updated to match actual file locations
set(SOURCES
    # Core
    src/core/mesh/structured_mesh.cpp
    src/core/mesh/structured_mesh_3d.cpp
    src/core/mesh/cylindrical_mesh.cpp
    src/core/mesh/nonuniform_mesh_1d.cpp
    src/core/utils.cpp
    # Solvers with compiled implementations
    src/solvers/nonuniform_diffusion_1d.cpp
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
add_library(biotransport::biotransport ALIAS biotransport)
set_target_properties(biotransport PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(biotransport PUBLIC cxx_std_17)
target_link_libraries(biotransport
                      PRIVATE $<BUILD_INTERFACE:biotransport_warnings>)

# Public headers and compiled sources both throw typed exceptions. MSVC needs
# /EHsc for correct C++ stack unwinding; propagate it to native consumers and to
# the Python extension that links this target.
target_compile_options(biotransport
                       PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>)

# Set include directories for users of the library
target_include_directories(
  biotransport PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>)

# Link OpenMP if enabled
if(BIOTRANSPORT_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(biotransport PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(biotransport PUBLIC BIOTRANSPORT_ENABLE_OPENMP)
endif()

# Add Eigen include directories if enabled
if(BIOTRANSPORT_EIGEN)
  target_link_libraries(biotransport PUBLIC Eigen3::Eigen)
  target_compile_definitions(biotransport PUBLIC BIOTRANSPORT_ENABLE_EIGEN)
endif()

# Add tests if needed
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
