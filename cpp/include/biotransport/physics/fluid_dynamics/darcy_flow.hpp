#ifndef BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_DARCY_FLOW_HPP
#define BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_DARCY_FLOW_HPP

/**
 * @file darcy_flow.hpp
 * @brief Darcy flow solver for porous media transport.
 *
 * Solves Darcy's law for flow through porous media:
 *   v = -K/μ ∇p  or  v = -κ ∇p
 *
 * Where:
 *   - v is the Darcy velocity [m/s]
 *   - K is permeability [m²] or κ = K/μ is hydraulic conductivity [m²/(Pa·s)]
 *   - μ is dynamic viscosity [Pa·s]
 *   - p is pressure [Pa]
 *
 * The pressure field satisfies the continuity equation:
 *   ∇·(κ ∇p) = 0  (incompressible, steady-state)
 *
 * Applications in biotransport:
 *   - Interstitial fluid flow in tumors
 *   - Drug delivery through tissue
 *   - Flow through porous scaffolds
 */

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cstdint>
#include <vector>

namespace biotransport {

/**
 * @brief Result of Darcy flow pressure solve.
 */
struct DarcyFlowResult {
    std::vector<double> pressure;  ///< Pressure field [Pa]
    std::vector<double> vx;        ///< x-velocity field [m/s]
    std::vector<double> vy;        ///< y-velocity field [m/s]
    int iterations;                ///< Number of SOR iterations used
    double residual;               ///< Final residual
    bool converged;                ///< Whether solver converged
};

/**
 * @brief Solver for Darcy flow in porous media.
 *
 * Uses successive over-relaxation (SOR) to solve the elliptic pressure
 * equation, then computes velocity from Darcy's law.
 */
class DarcyFlowSolver {
public:
    /**
     * @brief Create solver with uniform hydraulic conductivity.
     */
    DarcyFlowSolver(const StructuredMesh& mesh, double kappa);

    /**
     * @brief Create solver with spatially-varying hydraulic conductivity.
     */
    DarcyFlowSolver(const StructuredMesh& mesh, const std::vector<double>& kappa);

    /**
     * @brief Set Dirichlet (fixed pressure) boundary condition.
     */
    DarcyFlowSolver& setDirichlet(Boundary side, double pressure);

    /**
     * @brief Set Neumann (fixed flux) boundary condition.
     */
    DarcyFlowSolver& setNeumann(Boundary side, double flux);

    /**
     * @brief Set internal pressure sources (e.g., tumor pressure).
     */
    DarcyFlowSolver& setInternalPressure(const std::vector<std::uint8_t>& mask, double pressure);

    /**
     * @brief Set SOR relaxation parameter.
     */
    DarcyFlowSolver& setOmega(double omega);

    /**
     * @brief Set convergence tolerance.
     */
    DarcyFlowSolver& setTolerance(double tol);

    /**
     * @brief Set maximum iterations.
     */
    DarcyFlowSolver& setMaxIterations(int max_iter);

    /**
     * @brief Set initial pressure guess.
     */
    DarcyFlowSolver& setInitialGuess(const std::vector<double>& pressure);

    /**
     * @brief Solve the Darcy flow problem.
     */
    [[nodiscard]] DarcyFlowResult solve() const;

    const StructuredMesh& mesh() const { return mesh_; }
    const std::vector<double>& kappa() const { return kappa_; }

private:
    const StructuredMesh& mesh_;
    std::vector<double> kappa_;

    std::array<BoundaryCondition, 4> boundaries_;

    std::vector<std::uint8_t> internal_mask_;
    double internal_pressure_ = 0.0;
    bool has_internal_pressure_ = false;

    double omega_ = 1.5;
    double tolerance_ = 1e-6;
    int max_iter_ = 10000;
    std::vector<double> initial_guess_;

    void applyBoundaryPressure(std::vector<double>& p) const;
    void computeVelocity(const std::vector<double>& pressure, std::vector<double>& vx,
                         std::vector<double>& vy) const;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_DARCY_FLOW_HPP
