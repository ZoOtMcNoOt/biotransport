/**
 * @file stokes.hpp
 * @brief Stokes flow solver for viscous incompressible flow.
 *
 * Solves the steady Stokes equations for creeping flow:
 *   -∇p + μ∇²v + f = 0   (momentum)
 *   ∇·v = 0          (continuity)
 *
 * Where:
 *   - v = (u, v) is velocity [m/s]
 *   - p is pressure [Pa]
 *   - μ is dynamic viscosity [Pa·s]
 *   - f is body force per unit volume [N/m³]
 *
 * Applications in biotransport:
 *   - Low Reynolds number flows (Re << 1)
 *   - Blood flow in microcirculation
 *   - Microfluidic devices
 *   - Cell motility
 *   - Flow around small particles/cells
 */

#ifndef BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_STOKES_HPP
#define BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_STOKES_HPP

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <functional>
#include <vector>

namespace biotransport {

/**
 * @brief Result of Stokes flow solve.
 */
struct StokesResult {
    std::vector<double> u;         ///< x-velocity field [m/s]
    std::vector<double> v;         ///< y-velocity field [m/s]
    std::vector<double> pressure;  ///< Pressure field [Pa]
    int iterations;                ///< Number of outer iterations
    double residual;               ///< Maximum discrete momentum defect [N/m^3]
    double divergence;             ///< Maximum discrete divergence [1/s]
    bool converged;                ///< True for every returned result; failures throw
};

/**
 * @brief Solver for steady Stokes flow.
 *
 * This implementation is collocated: u, v, and pressure are all stored at
 * StructuredMesh nodes. Centered pressure gradients are used without a
 * Rhie-Chow or equivalent checkerboard-pressure stabilization. It is not a
 * staggered MAC discretization and should not be described or interpreted as
 * one.
 *
 * The algorithm uses a pressure-correction (projection) method:
 * 1. Relax momentum equations using the current pressure
 * 2. Solve pressure Poisson equation from divergence constraint
 * 3. Correct velocities to be divergence-free
 * 4. Iterate until convergence
 *
 * Verified sealed-domain usage:
 * @code
 *   StructuredMesh mesh(20, 12, 0, 1, 0, 0.5);
 *   StokesSolver solver(mesh, 0.001);  // all four walls default to no slip
 *   solver.setBodyForce(3.0, -1.0);   // uniform force density [N/m^3]
 *
 *   auto result = solver.solve();
 *   // The exact sealed equilibrium has u = v = 0 and
 *   // p = 3*x - y minus its nodal mean (the deterministic pressure gauge).
 * @endcode
 */
class StokesSolver {
public:
    /**
     * @brief Create a Stokes flow solver.
     *
     * @param mesh The structured mesh (2D only)
     * @param viscosity Dynamic viscosity mu [Pa·s]
     */
    StokesSolver(const StructuredMesh& mesh, double viscosity);

    /**
     * @brief Set velocity boundary condition.
     *
     * NOSLIP, DIRICHLET, INFLOW, and OUTFLOW are supported. OUTFLOW means
     * zero outward-normal velocity gradient; it is not a traction or pressure
     * boundary. StressFree/NEUMANN is rejected because that traction condition
     * is not implemented.
     *
     * @param side Boundary side
     * @param bc Velocity boundary condition
     */
    StokesSolver& setVelocityBC(Boundary side, VelocityBC bc);

    /**
     * @brief Set body force function.
     *
     * @param fx x-component of body force per unit volume [N/m³]
     * @param fy y-component of body force per unit volume [N/m³]
     */
    StokesSolver& setBodyForce(std::function<double(double x, double y)> fx,
                               std::function<double(double x, double y)> fy);

    /**
     * @brief Set uniform body force.
     *
     * @param fx x-component of body force per unit volume [N/m³]
     * @param fy y-component of body force per unit volume [N/m³]
     */
    StokesSolver& setBodyForce(double fx, double fy);

    /**
     * @brief Set convergence tolerance.
     *
     * @param tol Positive finite threshold used for velocity-iterate change
     *        [m/s] and maximum discrete divergence [1/s].
     */
    StokesSolver& setTolerance(double tol);

    /**
     * @brief Set maximum outer iterations.
     *
     * @param max_iter Maximum pressure-correction iterations. Default 10000.
     */
    StokesSolver& setMaxIterations(int max_iter);

    /**
     * @brief Set pressure relaxation factor.
     *
     * @param omega_p Pressure relaxation in (0, 1]. Default 0.3.
     */
    StokesSolver& setPressureRelaxation(double omega_p);

    /**
     * @brief Set velocity relaxation factor.
     *
     * @param omega_v Velocity relaxation in (0, 1]. Default 0.7.
     */
    StokesSolver& setVelocityRelaxation(double omega_v);

    /**
     * @brief Solve the Stokes flow problem.
     *
     * @return A finite, converged StokesResult. Invalid callback values,
     *         numerical failure, and failure to meet the configured iteration
     *         criteria throw.
     */
    [[nodiscard]] StokesResult solve() const;

    /**
     * @brief Get the mesh.
     */
    const StructuredMesh& mesh() const { return mesh_; }

    /**
     * @brief Get the viscosity.
     */
    double viscosity() const { return mu_; }

    /**
     * @brief Compute the Reynolds number based on characteristic scales.
     *
     * @param L Characteristic length [m]
     * @param U Characteristic velocity [m/s]
     * @param rho Fluid density [kg/m³]
     * @return Reynolds number Re = rho*U*L/mu
     */
    double reynolds(double L, double U, double rho) const;

private:
    const StructuredMesh& mesh_;
    double mu_;  // Dynamic viscosity

    // Boundary conditions
    std::array<VelocityBC, 4> velocity_bcs_ = {VelocityBC::NoSlip(), VelocityBC::NoSlip(),
                                               VelocityBC::NoSlip(), VelocityBC::NoSlip()};

    // Body force
    std::function<double(double, double)> fx_ = [](double, double) {
        return 0.0;
    };
    std::function<double(double, double)> fy_ = [](double, double) {
        return 0.0;
    };
    // The scalar overload is distinguishable from arbitrary callbacks.  This
    // permits an exact hydrostatic solution for a sealed all-no-slip domain
    // without guessing whether a callback happens to be spatially uniform.
    bool has_uniform_body_force_ = true;
    double uniform_fx_ = 0.0;
    double uniform_fy_ = 0.0;

    // Solver parameters
    double tolerance_ = 1e-4;  // More practical tolerance for most problems
    int max_iter_ = 5000;      // Reasonable upper bound to prevent timeouts
    double omega_p_ = 0.3;     // Better pressure convergence rate
    double omega_v_ = 0.7;     // Better velocity convergence rate

    // Internal methods
    void applyVelocityBCs(std::vector<double>& u, std::vector<double>& v) const;
    void solveMomentum(std::vector<double>& u, std::vector<double>& v,
                       const std::vector<double>& p) const;
    void solvePressurePoisson(std::vector<double>& p, const std::vector<double>& u,
                              const std::vector<double>& v) const;
    void correctVelocities(std::vector<double>& u, std::vector<double>& v,
                           const std::vector<double>& p) const;
    double computeMomentumResidual(const std::vector<double>& u, const std::vector<double>& v,
                                   const std::vector<double>& p) const;
    double computeDivergence(const std::vector<double>& u, const std::vector<double>& v) const;
    double bodyForceX(double x, double y) const;
    double bodyForceY(double x, double y) const;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_STOKES_HPP
