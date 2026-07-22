/**
 * @file biotransport_bindings.cpp
 * @brief Main pybind11 module registration
 *
 * This file defines the _core Python module and delegates binding registration
 * to specialized modules for better organization:
 *
 * - mesh_bindings.cpp      - StructuredMesh, CylindricalMesh, Boundary types
 * - diffusion_bindings.cpp - Diffusion, reaction-diffusion, ExplicitFD facade
 * - fluid_bindings.cpp     - Stokes, Navier-Stokes, Darcy, non-Newtonian
 * - utilities_bindings.cpp - dimensionless, analytical submodules
 *
 * This modular approach keeps each file under 500 lines and makes the codebase
 * easier to navigate and maintain.
 */

#include <pybind11/pybind11.h>

#include "balance_bindings.hpp"
#include "diffusion_bindings.hpp"
#include "fluid_bindings.hpp"
#include "high_order_bindings.hpp"
#include "io_bindings.hpp"
#include "mesh_bindings.hpp"
#include "metadata_bindings.hpp"
#include "nonuniform_bindings.hpp"
#include "sparse_bindings.hpp"
#include "transport_bindings.hpp"
#include "utilities_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "BioTransport library Python bindings - computational biotransport solvers";

    // Register all bindings from specialized modules
    biotransport::bindings::register_balance_bindings(m);
    biotransport::bindings::register_metadata_bindings(m);
    biotransport::bindings::register_mesh_bindings(m);
    biotransport::bindings::register_nonuniform_bindings(m);
    biotransport::bindings::register_diffusion_bindings(m);
    biotransport::bindings::register_fluid_bindings(m);
    biotransport::bindings::register_high_order_bindings(m);
    biotransport::bindings::register_sparse_bindings(m);
    biotransport::bindings::register_transport_bindings(m);
    biotransport::bindings::register_utilities_bindings(m);
    register_io_bindings(m);
}
