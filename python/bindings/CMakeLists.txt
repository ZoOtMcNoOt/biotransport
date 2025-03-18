pybind11_add_module(_core biotransport_bindings.cpp)
target_link_libraries(_core PRIVATE biotransport)

# Get Python site-packages directory
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Install the bindings to the Python package directory
install(TARGETS _core
        LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/biotransport/_core")