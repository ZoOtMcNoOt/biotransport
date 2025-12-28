# Modular Python bindings - split into focused modules for maintainability
set(BINDING_SOURCES
    biotransport_bindings_new.cpp
    mesh_bindings.cpp
    diffusion_bindings.cpp
    fluid_bindings.cpp
    utilities_bindings.cpp
    io_bindings.cpp
    sparse_bindings.cpp)

pybind11_add_module(_core ${BINDING_SOURCES})
target_link_libraries(_core PRIVATE biotransport)

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
  LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/biotransport/_core"
  RUNTIME DESTINATION "${PYTHON_SITE_PACKAGES}/biotransport/_core"
  ARCHIVE DESTINATION "${PYTHON_SITE_PACKAGES}/biotransport/_core")
