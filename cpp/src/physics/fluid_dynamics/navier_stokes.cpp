/**
 * @file navier_stokes.cpp
 * @brief Implementation of NavierStokesSolver class.
 */

#include <algorithm>
#include <biotransport/physics/fluid_dynamics/navier_stokes.hpp>
#include <biotransport/physics/fluid_dynamics/velocity_bc_applicator.hpp>
#include <cmath>

namespace biotransport {

// =============================================================================
// Constructor
// =============================================================================

NavierStokesSolver::NavierStokesSolver(const StructuredMesh& mesh, double density, double viscosity)
    : mesh_(mesh), rho_(density), mu_(viscosity), nu_(viscosity / density) {
    if (mesh.is1D()) {
        throw std::invalid_argument("NavierStokesSolver requires a 2D mesh");
    }
    if (density <= 0.0) {
        throw std::invalid_argument("Density must be positive");
    }
    if (viscosity <= 0.0) {
        throw std::invalid_argument("Viscosity must be positive");
    }
}

// =============================================================================
// Configuration methods
// =============================================================================

NavierStokesSolver& NavierStokesSolver::setVelocityBC(Boundary side, VelocityBC bc) {
    velocity_bcs_[static_cast<int>(side)] = bc;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setInlet(
    Boundary side, std::function<double(double x, double y)> u_profile,
    std::function<double(double x, double y)> v_profile) {
    int idx = static_cast<int>(side);
    velocity_bcs_[idx] = VelocityBC::Inflow(0.0, 0.0);  // Mark as inflow
    u_inlet_[idx] = u_profile;
    v_inlet_[idx] = v_profile ? v_profile : [](double, double) {
        return 0.0;
    };
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setBodyForce(std::function<double(double x, double y)> fx,
                                                     std::function<double(double x, double y)> fy) {
    fx_ = fx;
    fy_ = fy;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setBodyForce(double fx, double fy) {
    double fx_val = fx;
    double fy_val = fy;
    fx_ = [fx_val](double, double) {
        return fx_val;
    };
    fy_ = [fy_val](double, double) {
        return fy_val;
    };
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setInitialVelocity(const std::vector<double>& u0,
                                                           const std::vector<double>& v0) {
    u0_ = u0;
    v0_ = v0;
    has_initial_ = true;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setConvectionScheme(ConvectionScheme scheme) {
    conv_scheme_ = scheme;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setCFL(double cfl) {
    cfl_ = cfl;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setTimeStep(double dt) {
    dt_fixed_ = dt;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setPressureTolerance(double tol) {
    p_tolerance_ = tol;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setMaxPressureIterations(int max_iter) {
    p_max_iter_ = max_iter;
    return *this;
}

// =============================================================================
// Time step calculation
// =============================================================================

double NavierStokesSolver::maxTimeStep(const std::vector<double>& u,
                                       const std::vector<double>& v) const {
    double dx = mesh_.dx();
    double dy = mesh_.dy();

    // Find maximum velocities
    double max_u = 1e-10, max_v = 1e-10;
    for (size_t i = 0; i < u.size(); ++i) {
        max_u = std::max(max_u, std::abs(u[i]));
        max_v = std::max(max_v, std::abs(v[i]));
    }

    // CFL conditions
    double dt_conv = std::min(dx / max_u, dy / max_v);
    double dt_diff = 0.5 / (nu_ * (1.0 / (dx * dx) + 1.0 / (dy * dy)));

    return cfl_ * std::min(dt_conv, dt_diff);
}

// =============================================================================
// Boundary conditions
// =============================================================================

void NavierStokesSolver::applyVelocityBCs(std::vector<double>& u, std::vector<double>& v) const {
    // Delegate to utility function with inlet profile support
    applyVelocityBoundaryConditions(mesh_, velocity_bcs_, u_inlet_, v_inlet_, u, v);
}

// =============================================================================
// Numerical methods
// =============================================================================

void NavierStokesSolver::computeConvection(const std::vector<double>& u,
                                           const std::vector<double>& v,
                                           std::vector<double>& conv_u,
                                           std::vector<double>& conv_v) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();

    conv_u.assign(u.size(), 0.0);
    conv_v.assign(v.size(), 0.0);

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            int idx = j * stride + i;

            double u_c = u[idx];
            double v_c = v[idx];

            // Cell Reynolds number for scheme selection
            double Re_cell = std::abs(u_c) * dx / nu_;

            bool use_upwind = (conv_scheme_ == ConvectionScheme::UPWIND) ||
                              (conv_scheme_ == ConvectionScheme::HYBRID && Re_cell > 2.0);

            double dudx, dudy, dvdx, dvdy;

            if (use_upwind) {
                // First-order upwind
                if (u_c > 0) {
                    dudx = (u[idx] - u[idx - 1]) / dx;
                    dvdx = (v[idx] - v[idx - 1]) / dx;
                } else {
                    dudx = (u[idx + 1] - u[idx]) / dx;
                    dvdx = (v[idx + 1] - v[idx]) / dx;
                }
                if (v_c > 0) {
                    dudy = (u[idx] - u[idx - stride]) / dy;
                    dvdy = (v[idx] - v[idx - stride]) / dy;
                } else {
                    dudy = (u[idx + stride] - u[idx]) / dy;
                    dvdy = (v[idx + stride] - v[idx]) / dy;
                }
            } else {
                // Central differencing
                dudx = (u[idx + 1] - u[idx - 1]) / (2.0 * dx);
                dudy = (u[idx + stride] - u[idx - stride]) / (2.0 * dy);
                dvdx = (v[idx + 1] - v[idx - 1]) / (2.0 * dx);
                dvdy = (v[idx + stride] - v[idx - stride]) / (2.0 * dy);
            }

            // Convection: v · grad(u)
            conv_u[idx] = u_c * dudx + v_c * dudy;
            conv_v[idx] = u_c * dvdx + v_c * dvdy;
        }
    }
}

void NavierStokesSolver::computeDiffusion(const std::vector<double>& u,
                                          const std::vector<double>& v, std::vector<double>& diff_u,
                                          std::vector<double>& diff_v) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();
    double dx2 = dx * dx;
    double dy2 = dy * dy;

    diff_u.assign(u.size(), 0.0);
    diff_v.assign(v.size(), 0.0);

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            int idx = j * stride + i;

            // Laplacian: d2u/dx2 + d2u/dy2
            diff_u[idx] = nu_ * ((u[idx - 1] - 2.0 * u[idx] + u[idx + 1]) / dx2 +
                                 (u[idx - stride] - 2.0 * u[idx] + u[idx + stride]) / dy2);
            diff_v[idx] = nu_ * ((v[idx - 1] - 2.0 * v[idx] + v[idx + 1]) / dx2 +
                                 (v[idx - stride] - 2.0 * v[idx] + v[idx + stride]) / dy2);
        }
    }
}

void NavierStokesSolver::solvePressurePoisson(std::vector<double>& p,
                                              const std::vector<double>& u_star,
                                              const std::vector<double>& v_star, double dt) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double coeff = 2.0 * (1.0 / dx2 + 1.0 / dy2);
    double omega = 1.5;  // SOR relaxation

    // RHS: (rho/dt) * div(u*)
    std::vector<double> rhs(p.size(), 0.0);
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            int idx = j * stride + i;
            double div = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx) +
                         (v_star[idx + stride] - v_star[idx - stride]) / (2.0 * dy);
            rhs[idx] = (rho_ / dt) * div;
        }
    }

    // SOR iteration
    for (int iter = 0; iter < p_max_iter_; ++iter) {
        double max_res = 0.0;

        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                int idx = j * stride + i;

                double p_lap =
                    (p[idx - 1] + p[idx + 1]) / dx2 + (p[idx - stride] + p[idx + stride]) / dy2;
                double p_new = (p_lap - rhs[idx]) / coeff;

                double diff = std::abs(p_new - p[idx]);
                max_res = std::max(max_res, diff);

                p[idx] = p[idx] + omega * (p_new - p[idx]);
            }
        }

        // Pressure BCs (Neumann)
        applyPressureNeumannBCs(mesh_, p);

        if (max_res < p_tolerance_)
            break;
    }
}

void NavierStokesSolver::projectVelocity(std::vector<double>& u, std::vector<double>& v,
                                         const std::vector<double>& p, double dt) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();
    double factor = dt / rho_;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            int idx = j * stride + i;

            double dpdx = (p[idx + 1] - p[idx - 1]) / (2.0 * dx);
            double dpdy = (p[idx + stride] - p[idx - stride]) / (2.0 * dy);

            u[idx] -= factor * dpdx;
            v[idx] -= factor * dpdy;
        }
    }
}

double NavierStokesSolver::computeMaxVelocity(const std::vector<double>& u,
                                              const std::vector<double>& v) const {
    double max_vel = 0.0;
    for (size_t i = 0; i < u.size(); ++i) {
        double vel = std::sqrt(u[i] * u[i] + v[i] * v[i]);
        max_vel = std::max(max_vel, vel);
    }
    return max_vel;
}

double NavierStokesSolver::computeDivergence(const std::vector<double>& u,
                                             const std::vector<double>& v) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();

    double max_div = 0.0;
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            int idx = j * stride + i;
            double div = (u[idx + 1] - u[idx - 1]) / (2.0 * dx) +
                         (v[idx + stride] - v[idx - stride]) / (2.0 * dy);
            max_div = std::max(max_div, std::abs(div));
        }
    }
    return max_div;
}

// =============================================================================
// Main solve methods
// =============================================================================

NavierStokesResult NavierStokesSolver::solve(double duration, double /*output_interval*/) {
    int num_nodes = mesh_.numNodes();

    // Initialize fields
    std::vector<double> u(num_nodes, 0.0);
    std::vector<double> v(num_nodes, 0.0);
    std::vector<double> p(num_nodes, 0.0);

    if (has_initial_) {
        u = u0_;
        v = v0_;
    }

    applyVelocityBCs(u, v);

    std::vector<double> conv_u, conv_v, diff_u, diff_v;
    std::vector<double> u_star(num_nodes), v_star(num_nodes);

    double t = 0.0;
    int step = 0;
    bool stable = true;

    while (t < duration && stable) {
        // Compute time step
        double dt = (dt_fixed_ > 0) ? dt_fixed_ : maxTimeStep(u, v);
        dt = std::min(dt, duration - t);

        // Step 1: Compute intermediate velocity (explicit Euler)
        computeConvection(u, v, conv_u, conv_v);
        computeDiffusion(u, v, diff_u, diff_v);

        int nx = mesh_.nx();
        int ny = mesh_.ny();
        int stride = nx + 1;

        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                int idx = j * stride + i;
                double x = mesh_.x(i);
                double y = mesh_.y(i, j);

                u_star[idx] = u[idx] + dt * (-conv_u[idx] + diff_u[idx] + fx_(x, y) / rho_);
                v_star[idx] = v[idx] + dt * (-conv_v[idx] + diff_v[idx] + fy_(x, y) / rho_);
            }
        }

        applyVelocityBCs(u_star, v_star);

        // Step 2: Solve pressure Poisson
        solvePressurePoisson(p, u_star, v_star, dt);

        // Step 3: Project velocity
        u = u_star;
        v = v_star;
        projectVelocity(u, v, p, dt);
        applyVelocityBCs(u, v);

        // Check for NaN (instability)
        double max_vel = computeMaxVelocity(u, v);
        if (std::isnan(max_vel) || std::isinf(max_vel) || max_vel > 1e10) {
            stable = false;
            break;
        }

        t += dt;
        ++step;
    }

    NavierStokesResult result;
    result.u = std::move(u);
    result.v = std::move(v);
    result.pressure = std::move(p);
    result.time = t;
    result.time_steps = step;
    result.max_velocity = computeMaxVelocity(result.u, result.v);
    result.reynolds = reynolds(mesh_.dx() * mesh_.nx(), result.max_velocity);
    result.stable = stable;

    return result;
}

NavierStokesResult NavierStokesSolver::solveSteps(int num_steps) {
    // Estimate duration based on typical timestep
    double dt_est = (dt_fixed_ > 0) ? dt_fixed_ : cfl_ * std::min(mesh_.dx(), mesh_.dy()) * 0.1;
    return solve(num_steps * dt_est);
}

}  // namespace biotransport
