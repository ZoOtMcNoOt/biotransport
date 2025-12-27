/**
 * @file navier_stokes.hpp
 * @brief Incompressible Navier-Stokes solver for viscous flow.
 *
 * Solves the incompressible Navier-Stokes equations:
 *   rho*(dv/dt + v·nabla(v)) = -nabla(p) + mu*nabla^2(v) + f   (momentum)
 *   nabla·v = 0                                                 (continuity)
 *
 * Where:
 *   - v = (u, v) is velocity [m/s]
 *   - p is pressure [Pa]
 *   - rho is density [kg/m^3]
 *   - mu is dynamic viscosity [Pa·s]
 *   - f is body force per unit volume [N/m^3]
 *
 * Uses the projection method (Chorin's method):
 * 1. Compute intermediate velocity ignoring pressure
 * 2. Solve pressure Poisson equation
 * 3. Project velocity to divergence-free space
 *
 * Applications in biotransport:
 *   - Blood flow in vessels (Re ~ 100-1000)
 *   - Flow in bioreactors
 *   - Drug mixing in microchannels
 *   - Cardiovascular flows
 */

#ifndef BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NAVIER_STOKES_HPP
#define BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NAVIER_STOKES_HPP

#include <algorithm>
#include <array>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/fluid_dynamics/stokes.hpp>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace biotransport {

/**
 * @brief Result of Navier-Stokes solve.
 */
struct NavierStokesResult {
    std::vector<double> u;         ///< x-velocity field [m/s]
    std::vector<double> v;         ///< y-velocity field [m/s]
    std::vector<double> pressure;  ///< Pressure field [Pa]
    double time;                   ///< Current simulation time [s]
    int time_steps;                ///< Number of time steps taken
    double max_velocity;           ///< Maximum velocity magnitude
    double reynolds;               ///< Reynolds number based on max velocity
    bool stable;                   ///< Whether solution remained stable
};

/**
 * @brief Convection scheme for Navier-Stokes.
 */
enum class ConvectionScheme {
    UPWIND,   ///< First-order upwind (stable, diffusive)
    CENTRAL,  ///< Second-order central (may oscillate)
    QUICK,    ///< Quadratic upwind (third-order-ish)
    HYBRID    ///< Switch based on local cell Reynolds number
};

/**
 * @brief Solver for incompressible Navier-Stokes equations.
 *
 * Uses a fractional step (projection) method:
 *
 * Step 1 (Predictor): Solve for intermediate velocity u*
 *   (u* - u^n)/dt = -u·nabla(u) + nu*nabla^2(u) + f
 *
 * Step 2 (Pressure): Solve Poisson equation
 *   nabla^2(p) = (rho/dt) * nabla·u*
 *
 * Step 3 (Corrector): Project to divergence-free
 *   u^(n+1) = u* - (dt/rho) * nabla(p)
 *
 * Time stepping uses explicit Adams-Bashforth for convection and
 * implicit Crank-Nicolson for diffusion (stable for all dt).
 *
 * Example usage:
 * @code
 *   StructuredMesh mesh(100, 40, 0, 2.5, 0, 1);  // Channel
 *   NavierStokesSolver solver(mesh, 1000.0, 0.001);  // Water
 *
 *   // Poiseuille inlet profile
 *   auto parabolic = [](double y, double H) { return 6.0 * (y/H) * (1 - y/H); };
 *   solver.setInlet(Boundary::Left, [=](double x, double y) {
 *       return parabolic(y, 1.0) * 0.01;  // Max 1 cm/s
 *   });
 *   solver.setVelocityBC(Boundary::Top, VelocityBC::NoSlip());
 *   solver.setVelocityBC(Boundary::Bottom, VelocityBC::NoSlip());
 *   solver.setVelocityBC(Boundary::Right, VelocityBC::Outflow());
 *
 *   auto result = solver.solve(1.0);  // Solve for 1 second
 * @endcode
 */
class NavierStokesSolver {
public:
    /**
     * @brief Create a Navier-Stokes solver.
     *
     * @param mesh The structured mesh (2D only)
     * @param density Fluid density rho [kg/m^3]
     * @param viscosity Dynamic viscosity mu [Pa·s]
     */
    NavierStokesSolver(const StructuredMesh& mesh, double density, double viscosity);

    /**
     * @brief Set velocity boundary condition.
     *
     * @param side Boundary side
     * @param bc Velocity boundary condition
     */
    NavierStokesSolver& setVelocityBC(Boundary side, VelocityBC bc);

    /**
     * @brief Set inlet velocity profile.
     *
     * @param side Boundary side (usually Left)
     * @param u_profile Function returning u-velocity at (x, y)
     * @param v_profile Function returning v-velocity at (x, y) (optional)
     */
    NavierStokesSolver& setInlet(Boundary side, std::function<double(double x, double y)> u_profile,
                                 std::function<double(double x, double y)> v_profile = nullptr);

    /**
     * @brief Set body force function.
     *
     * @param fx x-component of body force per unit volume [N/m^3]
     * @param fy y-component of body force per unit volume [N/m^3]
     */
    NavierStokesSolver& setBodyForce(std::function<double(double x, double y)> fx,
                                     std::function<double(double x, double y)> fy);

    /**
     * @brief Set uniform body force (e.g., gravity).
     *
     * @param fx x-component of body force per unit volume [N/m^3]
     * @param fy y-component of body force per unit volume [N/m^3]
     */
    NavierStokesSolver& setBodyForce(double fx, double fy);

    /**
     * @brief Set initial velocity field.
     *
     * @param u0 Initial x-velocity at each node
     * @param v0 Initial y-velocity at each node
     */
    NavierStokesSolver& setInitialVelocity(const std::vector<double>& u0,
                                           const std::vector<double>& v0);

    /**
     * @brief Set convection scheme.
     *
     * @param scheme Discretization scheme for convective term
     */
    NavierStokesSolver& setConvectionScheme(ConvectionScheme scheme);

    /**
     * @brief Set CFL safety factor.
     *
     * @param cfl CFL number (0.1 to 0.5 typical). Default 0.25.
     */
    NavierStokesSolver& setCFL(double cfl);

    /**
     * @brief Set fixed time step (overrides adaptive CFL).
     *
     * @param dt Fixed time step [s]. Use 0 for adaptive.
     */
    NavierStokesSolver& setTimeStep(double dt);

    /**
     * @brief Set pressure solver tolerance.
     *
     * @param tol Maximum residual for pressure Poisson. Default 1e-6.
     */
    NavierStokesSolver& setPressureTolerance(double tol);

    /**
     * @brief Set maximum pressure solver iterations.
     *
     * @param max_iter Maximum iterations per pressure solve. Default 1000.
     */
    NavierStokesSolver& setMaxPressureIterations(int max_iter);

    /**
     * @brief Solve the Navier-Stokes equations for a given duration.
     *
     * @param duration Total simulation time [s]
     * @param output_interval Interval for storing snapshots (0 = final only)
     * @return NavierStokesResult containing final velocity, pressure fields
     */
    [[nodiscard]] NavierStokesResult solve(double duration, double output_interval = 0.0);

    /**
     * @brief Solve for a specified number of time steps.
     *
     * @param num_steps Number of time steps
     * @return NavierStokesResult containing final fields
     */
    [[nodiscard]] NavierStokesResult solveSteps(int num_steps);

    /**
     * @brief Get the mesh.
     */
    const StructuredMesh& mesh() const { return mesh_; }

    /**
     * @brief Get the density.
     */
    double density() const { return rho_; }

    /**
     * @brief Get the viscosity.
     */
    double viscosity() const { return mu_; }

    /**
     * @brief Get the kinematic viscosity.
     */
    double kinematicViscosity() const { return mu_ / rho_; }

    /**
     * @brief Compute maximum stable time step.
     *
     * Based on CFL condition for both convection and diffusion:
     *   dt <= min(dx/|u|, dy/|v|, dx^2/(2*nu), dy^2/(2*nu)) * CFL
     *
     * @param u Current x-velocity field
     * @param v Current y-velocity field
     * @return Maximum stable time step [s]
     */
    double maxTimeStep(const std::vector<double>& u, const std::vector<double>& v) const;

    /**
     * @brief Compute Reynolds number.
     *
     * @param L Characteristic length [m]
     * @param U Characteristic velocity [m/s]
     * @return Reynolds number Re = rho*U*L/mu
     */
    double reynolds(double L, double U) const { return rho_ * U * L / mu_; }

private:
    const StructuredMesh& mesh_;
    double rho_;  // Density [kg/m^3]
    double mu_;   // Dynamic viscosity [Pa·s]
    double nu_;   // Kinematic viscosity [m^2/s]

    // Boundary conditions
    std::array<VelocityBC, 4> velocity_bcs_ = {VelocityBC::NoSlip(), VelocityBC::NoSlip(),
                                               VelocityBC::NoSlip(), VelocityBC::NoSlip()};

    // Inlet profiles (nullptr means use VelocityBC values)
    std::array<std::function<double(double, double)>, 4> u_inlet_;
    std::array<std::function<double(double, double)>, 4> v_inlet_;

    // Body force
    std::function<double(double, double)> fx_ = [](double, double) {
        return 0.0;
    };
    std::function<double(double, double)> fy_ = [](double, double) {
        return 0.0;
    };

    // Initial conditions
    std::vector<double> u0_, v0_;
    bool has_initial_ = false;

    // Solver parameters
    ConvectionScheme conv_scheme_ = ConvectionScheme::HYBRID;
    double cfl_ = 0.25;
    double dt_fixed_ = 0.0;  // 0 = adaptive
    double p_tolerance_ = 1e-6;
    int p_max_iter_ = 1000;

    // Internal methods
    void applyVelocityBCs(std::vector<double>& u, std::vector<double>& v) const;
    void computeConvection(const std::vector<double>& u, const std::vector<double>& v,
                           std::vector<double>& conv_u, std::vector<double>& conv_v) const;
    void computeDiffusion(const std::vector<double>& u, const std::vector<double>& v,
                          std::vector<double>& diff_u, std::vector<double>& diff_v) const;
    void solvePressurePoisson(std::vector<double>& p, const std::vector<double>& u_star,
                              const std::vector<double>& v_star, double dt) const;
    void projectVelocity(std::vector<double>& u, std::vector<double>& v,
                         const std::vector<double>& p, double dt) const;
    double computeMaxVelocity(const std::vector<double>& u, const std::vector<double>& v) const;
    double computeDivergence(const std::vector<double>& u, const std::vector<double>& v) const;

    // Upwind helper
    double upwind(double phi_m, double phi_0, double phi_p, double vel) const {
        return (vel > 0) ? phi_0 - phi_m : phi_p - phi_0;
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NAVIER_STOKES_HPP
