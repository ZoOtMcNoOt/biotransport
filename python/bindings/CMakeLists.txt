# Modular Python bindings - split into focused modules for maintainability
set(BINDING_SOURCES
    biotransport_bindings_new.cpp
    balance_bindings.cpp
    metadata_bindings.cpp
    mesh_bindings.cpp
    nonuniform_bindings.cpp
    diffusion_bindings.cpp
    fluid_bindings.cpp
    high_order_bindings.cpp
    utilities_bindings.cpp
    io_bindings.cpp
    sparse_bindings.cpp)
list(APPEND BINDING_SOURCES transport_bindings.cpp)

pybind11_add_module(_core ${BINDING_SOURCES})
target_link_libraries(_core PRIVATE biotransport biotransport_warnings)

if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()

# Get Python site-packages directory
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import sysconfig; print(sysconfig.get_path('platlib'))"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Install the bindings to the Python package directory
install(
  TARGETS _core
  LIBRARY DESTINATION "biotransport/_core"
  RUNTIME DESTINATION "biotransport/_core"
  ARCHIVE DESTINATION "biotransport/_core")
