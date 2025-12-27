/**
 * @file stokes.hpp
 * @brief Stokes flow solver for viscous incompressible flow.
 *
 * Solves the steady Stokes equations for creeping flow:
 *   -∇p + μ∇²v = f   (momentum)
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
    double residual;               ///< Final momentum residual
    double divergence;             ///< Final divergence residual
    bool converged;                ///< Whether solver converged
};

/**
 * @brief Solver for steady Stokes flow.
 *
 * Uses a staggered grid (MAC scheme) to avoid checkerboard pressure modes.
 * Velocities are stored at cell faces, pressure at cell centers.
 *
 * The algorithm uses a pressure-correction (projection) method:
 * 1. Solve momentum equations ignoring pressure (intermediate velocity)
 * 2. Solve pressure Poisson equation from divergence constraint
 * 3. Correct velocities to be divergence-free
 * 4. Iterate until convergence
 *
 * Example usage:
 * @code
 *   StructuredMesh mesh(50, 50, 0, 1, 0, 1);
 *   StokesSolver solver(mesh, 0.001);  // mu = 1 mPa·s (water-like)
 *
 *   solver.setVelocityBC(Boundary::Left, VelocityBC::Inflow(0.01));
 *   solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());
 *   solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
 *   solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
 *
 *   auto result = solver.solve();
 *   // result.u, result.v, result.pressure contain the solution
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
     * @param tol Maximum residual for convergence. Default 1e-6.
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
     * @param omega_p Pressure relaxation (0.1 to 0.8 typical). Default 0.3.
     */
    StokesSolver& setPressureRelaxation(double omega_p);

    /**
     * @brief Set velocity relaxation factor.
     *
     * @param omega_v Velocity relaxation (0.5 to 0.9 typical). Default 0.7.
     */
    StokesSolver& setVelocityRelaxation(double omega_v);

    /**
     * @brief Solve the Stokes flow problem.
     *
     * @return StokesResult containing velocity, pressure, and convergence info
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
    double reynolds(double L, double U, double rho) const { return rho * U * L / mu_; }

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
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_STOKES_HPP
