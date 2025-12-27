/**
 * @file diffusion_bindings.hpp
 * @brief Python bindings for diffusion and reaction-diffusion solvers
 * 
 * This module provides bindings for:
 * - DiffusionSolver (base class)
 * - VariableDiffusionSolver
 * - ReactionDiffusionSolver (custom reaction function)
 * - LinearReactionDiffusionSolver (first-order decay)
 * - LogisticReactionDiffusionSolver
 * - MichaelisMentenReactionDiffusionSolver
 * - MaskedMichaelisMentenReactionDiffusionSolver
 * - ConstantSourceReactionDiffusionSolver
 * - GrayScottSolver (two-species)
 * - TumorDrugDeliverySolver
 * - BioheatCryotherapySolver
 * - MembraneDiffusion1DSolver, MultiLayerMembraneSolver
 * - AdvectionDiffusionSolver
 * - ExplicitFD facade and Problem classes
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

/**
 * @brief Register all diffusion-related bindings
 * @param m The pybind11 module to register with
 */
void register_diffusion_bindings(py::module_& m);

} // namespace bindings
} // namespace biotransport
