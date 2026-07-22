/**
 * @file navier_stokes.hpp
 * @brief Bounded, incompressible two-dimensional Navier-Stokes solver.
 *
 * The solver advances
 *
 *   du/dt + (u . grad)u = -(1/rho) grad(p) + nu laplacian(u) + f/rho,
 *   div(u) = 0,
 *
 * on a uniform Cartesian mesh.  Velocity components are stored on a MAC
 * (staggered) grid and pressure is stored at cell centres.  All three arrays
 * use the mesh's `(nx + 1) * (ny + 1)` packed allocation so that they remain
 * easy to exchange through the existing API; unused padding entries are
 * documented in NavierStokesResult.
 *
 * This science-first implementation intentionally supports prescribed
 * (NOSLIP or DIRICHLET) velocity boundaries only.  Inflow/outflow, traction,
 * and profile boundary conditions require a distinct pressure-boundary model
 * and are rejected rather than silently approximated.
 */

#ifndef BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NAVIER_STOKES_HPP
#define BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NAVIER_STOKES_HPP

#include <array>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/fluid_dynamics/stokes.hpp>
#include <functional>
#include <limits>
#include <vector>

namespace biotransport {

/**
 * @brief Result and numerical diagnostics for a Navier-Stokes integration.
 *
 * Array layout for a mesh with `nx` by `ny` cells and stride `nx + 1`:
 * - `u[j*stride+i]` is x velocity on an x-normal face, for
 *   `0 <= i <= nx`, `0 <= j < ny`; the last row is display padding.
 * - `v[j*stride+i]` is y velocity on a y-normal face, for
 *   `0 <= i < nx`, `0 <= j <= ny`; the last column is display padding.
 * - `pressure[j*stride+i]` is cell-centred pressure for
 *   `0 <= i < nx`, `0 <= j < ny`; the last row and column repeat their
 *   nearest valid value for convenient reshaping.
 */
struct NavierStokesResult {
    std::vector<double> u;         ///< Staggered x-velocity field [m/s]
    std::vector<double> v;         ///< Staggered y-velocity field [m/s]
    std::vector<double> pressure;  ///< Cell-centred pressure field [Pa]
    double time = 0.0;             ///< Simulated time [s]
    int time_steps = 0;            ///< Number of completed time steps
    double max_velocity = 0.0;     ///< Maximum cell-centred speed [m/s]
    double reynolds = 0.0;         ///< Reynolds number based on domain length
    int pressure_iterations = 0;   ///< Iterations in the final pressure solve
    double pressure_residual =
        std::numeric_limits<double>::infinity();                  ///< Relative Poisson residual
    double divergence = std::numeric_limits<double>::infinity();  ///< Final max |div(u)| [1/s]
    bool stable = false;  ///< Finite and projection-converged
};

/**
 * @brief Spatial discretization for the explicit convective term.
 */
enum class ConvectionScheme {
    UPWIND,   ///< First-order donor-cell upwind
    CENTRAL,  ///< Second-order centred differences
    QUICK,    ///< Reserved: rejected because it is not implemented
    HYBRID    ///< Reserved: rejected because it is not implemented
};

/**
 * @brief Explicit finite-volume solver with a compatible MAC projection.
 *
 * Predictor convection and diffusion are explicit.  The pressure equation is
 * formed with exactly the same discrete divergence and pressure gradient used
 * in the velocity correction, so a converged pressure solve directly controls
 * the reported post-projection divergence.
 *
 * The mesh is owned by value.  Constructing the solver from a temporary mesh
 * is therefore safe.
 */
class NavierStokesSolver {
public:
    NavierStokesSolver(const StructuredMesh& mesh, double density, double viscosity);

    /**
     * @brief Set a prescribed velocity boundary.
     * @throws std::invalid_argument for INFLOW, OUTFLOW, or NEUMANN in this
     * bounded implementation.
     */
    NavierStokesSolver& setVelocityBC(Boundary side, VelocityBC bc);

    /**
     * @brief Profile inlets are not yet supported by the compatible projection.
     * @throws std::invalid_argument always; use flux-compatible DIRICHLET
     * boundaries for constant prescribed normal velocities.
     */
    NavierStokesSolver& setInlet(Boundary side, std::function<double(double x, double y)> u_profile,
                                 std::function<double(double x, double y)> v_profile = nullptr);

    NavierStokesSolver& setBodyForce(std::function<double(double x, double y)> fx,
                                     std::function<double(double x, double y)> fy);
    NavierStokesSolver& setBodyForce(double fx, double fy);

    /**
     * @brief Set staggered initial fields using the packed layout above.
     * @throws std::invalid_argument unless both fields have exactly numNodes()
     * finite entries.
     */
    NavierStokesSolver& setInitialVelocity(const std::vector<double>& u0,
                                           const std::vector<double>& v0);

    /** @throws std::invalid_argument for QUICK or HYBRID. */
    NavierStokesSolver& setConvectionScheme(ConvectionScheme scheme);

    /** @brief Set the explicit-step safety factor, 0 < cfl <= 1. */
    NavierStokesSolver& setCFL(double cfl);

    /** @brief Set a fixed step in seconds; zero restores adaptive stepping. */
    NavierStokesSolver& setTimeStep(double dt);

    /** @brief Set the dimensionless relative pressure residual tolerance. */
    NavierStokesSolver& setPressureTolerance(double tol);

    /** @brief Set the maximum SOR iterations per pressure projection. */
    NavierStokesSolver& setMaxPressureIterations(int max_iter);

    /**
     * @brief Integrate for exactly `duration` seconds.
     *
     * The final step is shortened to reach the requested time exactly.
     * Snapshot output is not implemented, so a nonzero output_interval is
     * rejected.
     */
    [[nodiscard]] NavierStokesResult solve(double duration, double output_interval = 0.0);

    /**
     * @brief Take exactly `num_steps` accepted time steps.
     *
     * Adaptive mode recomputes the stable step before every step.  Fixed-step
     * mode uses exactly the configured step.  Numerical failure throws instead
     * of returning fewer steps than requested.
     */
    [[nodiscard]] NavierStokesResult solveSteps(int num_steps);

    const StructuredMesh& mesh() const noexcept { return mesh_; }
    double density() const noexcept { return rho_; }
    double viscosity() const noexcept { return mu_; }
    double kinematicViscosity() const noexcept { return nu_; }

    /** @brief Compute the explicit stability bound for packed staggered fields. */
    double maxTimeStep(const std::vector<double>& u, const std::vector<double>& v) const;

    double reynolds(double L, double U) const;

private:
    struct PressureSolveInfo {
        int iterations = 0;
        double relative_residual = std::numeric_limits<double>::infinity();
        bool converged = false;
    };

    struct StepInfo {
        PressureSolveInfo pressure;
        double divergence = std::numeric_limits<double>::infinity();
    };

    StructuredMesh mesh_;
    double rho_;
    double mu_;
    double nu_;

    std::array<VelocityBC, 4> velocity_bcs_ = {VelocityBC::NoSlip(), VelocityBC::NoSlip(),
                                               VelocityBC::NoSlip(), VelocityBC::NoSlip()};

    std::function<double(double, double)> fx_ = [](double, double) {
        return 0.0;
    };
    std::function<double(double, double)> fy_ = [](double, double) {
        return 0.0;
    };

    std::vector<double> u0_;
    std::vector<double> v0_;
    bool has_initial_ = false;

    ConvectionScheme conv_scheme_ = ConvectionScheme::UPWIND;
    double cfl_ = 0.25;
    double dt_fixed_ = 0.0;
    double p_tolerance_ = 1e-8;
    int p_max_iter_ = 5000;

    void validateBoundaryConfiguration() const;
    void validatePackedField(const std::vector<double>& field, const char* name) const;
    double boundaryComponent(Boundary side, bool x_component) const;
    void applyVelocityBCs(std::vector<double>& u, std::vector<double>& v) const;
    void fillPressurePadding(std::vector<double>& pressure) const;

    void computeConvection(const std::vector<double>& u, const std::vector<double>& v,
                           std::vector<double>& conv_u, std::vector<double>& conv_v) const;
    void computeDiffusion(const std::vector<double>& u, const std::vector<double>& v,
                          std::vector<double>& diff_u, std::vector<double>& diff_v) const;
    PressureSolveInfo solvePressurePoisson(std::vector<double>& pressure,
                                           const std::vector<double>& u_star,
                                           const std::vector<double>& v_star, double dt) const;
    void projectVelocity(std::vector<double>& u, std::vector<double>& v,
                         const std::vector<double>& pressure, double dt) const;
    StepInfo takeStep(std::vector<double>& u, std::vector<double>& v, std::vector<double>& pressure,
                      double dt) const;

    double computeMaxVelocity(const std::vector<double>& u, const std::vector<double>& v) const;
    double computeDivergence(const std::vector<double>& u, const std::vector<double>& v) const;
    NavierStokesResult makeResult(std::vector<double> u, std::vector<double> v,
                                  std::vector<double> pressure, double time, int steps,
                                  const StepInfo& final_step) const;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NAVIER_STOKES_HPP
