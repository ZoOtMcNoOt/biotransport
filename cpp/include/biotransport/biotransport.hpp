/**
 * @file biotransport.hpp
 * @brief Convenience header including all biotransport functionality.
 *
 * This is the main include file for the biotransport library.
 * Include this single header to get access to all solvers and utilities.
 */

#ifndef BIOTRANSPORT_HPP
#define BIOTRANSPORT_HPP

// Core components - mesh
#include <biotransport/core/mesh/cylindrical_mesh.hpp>
#include <biotransport/core/mesh/mesh_iterators.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>

// Core components - boundary and utilities
#include <biotransport/core/analytical.hpp>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/dimensionless.hpp>
#include <biotransport/core/utils.hpp>

// Core components - problems
#include <biotransport/core/problems/transport_problem.hpp>

// Numerics
#include <biotransport/core/numerics/linear_algebra/sparse_matrix.hpp>
#include <biotransport/core/numerics/linear_algebra/tridiagonal.hpp>
#include <biotransport/core/numerics/solvers/iterative.hpp>
#include <biotransport/core/numerics/stability.hpp>
#include <biotransport/core/numerics/time_integration/explicit_euler.hpp>

// Solver infrastructure
#include <biotransport/solvers/explicit_fd.hpp>
#include <biotransport/solvers/solver_base.hpp>

// Diffusion and reaction-diffusion (consolidated)
#include <biotransport/solvers/adi_solver.hpp>
#include <biotransport/solvers/advection_diffusion_solver.hpp>
#include <biotransport/solvers/crank_nicolson.hpp>
#include <biotransport/solvers/diffusion_solvers.hpp>
#include <biotransport/solvers/implicit_diffusion.hpp>

// Reactions library
#include <biotransport/physics/reactions.hpp>

// Fluid dynamics
#include <biotransport/physics/fluid_dynamics/darcy_flow.hpp>
#include <biotransport/physics/fluid_dynamics/navier_stokes.hpp>
#include <biotransport/physics/fluid_dynamics/non_newtonian.hpp>
#include <biotransport/physics/fluid_dynamics/stokes.hpp>

// Mass transport
#include <biotransport/physics/mass_transport/gray_scott.hpp>
#include <biotransport/physics/mass_transport/membrane_diffusion.hpp>
#include <biotransport/physics/mass_transport/tumor_drug_delivery.hpp>

// Heat transfer
#include <biotransport/physics/heat_transfer/bioheat_cryotherapy.hpp>

#endif  // BIOTRANSPORT_HPP
