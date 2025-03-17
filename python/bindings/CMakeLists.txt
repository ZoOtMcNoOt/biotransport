pybind11_add_module(_core biotransport_bindings.cpp)
target_link_libraries(_core PRIVATE biotransport)

# Install the bindings to the Python package directory
install(TARGETS _core
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/python/biotransport/_core")