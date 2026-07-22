/**
 * @file stokes.cpp
 * @brief Implementation of Stokes flow solver.
 */

#include <algorithm>
#include <biotransport/physics/fluid_dynamics/stokes.hpp>
#include <biotransport/physics/fluid_dynamics/velocity_bc_applicator.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace biotransport {

namespace {

int checkedBoundary(Boundary side) {
    const int index = to_index(side);
    if (index < to_index(Boundary::Left) || index > to_index(Boundary::Top)) {
        throw std::invalid_argument("Boundary identifier is outside [0, 3]");
    }
    return index;
}

void requireFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void requireFiniteNumerical(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " became non-finite");
    }
}

void requireFiniteField(const std::vector<double>& values, const char* name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error(std::string(name) + " contains a non-finite value at " +
                                     std::to_string(index));
        }
    }
}

void validateVelocityBC(const VelocityBC& bc) {
    requireFinite(bc.u_value, "Velocity boundary u value");
    requireFinite(bc.v_value, "Velocity boundary v value");
    switch (bc.type) {
        case VelocityBCType::NOSLIP:
        case VelocityBCType::DIRICHLET:
        case VelocityBCType::INFLOW:
        case VelocityBCType::OUTFLOW:
            return;
        case VelocityBCType::NEUMANN:
            throw std::invalid_argument(
                "StressFree/NEUMANN traction is not implemented by StokesSolver");
        default:
            throw std::invalid_argument("Unknown velocity boundary-condition type");
    }
}

}  // namespace

StokesSolver::StokesSolver(const StructuredMesh& mesh, double viscosity)
    : mesh_(mesh), mu_(viscosity) {
    if (mesh.is1D()) {
        throw std::invalid_argument("StokesSolver requires a 2D mesh");
    }
    if (mesh.nx() < 2 || mesh.ny() < 2) {
        throw std::invalid_argument("StokesSolver requires at least two cells in each direction");
    }
    if (!std::isfinite(viscosity) || viscosity <= 0.0) {
        throw std::invalid_argument("Viscosity must be finite and positive");
    }
}

StokesSolver& StokesSolver::setVelocityBC(Boundary side, VelocityBC bc) {
    validateVelocityBC(bc);
    velocity_bcs_[checkedBoundary(side)] = bc;
    return *this;
}

StokesSolver& StokesSolver::setBodyForce(std::function<double(double x, double y)> fx,
                                         std::function<double(double x, double y)> fy) {
    if (!fx || !fy) {
        throw std::invalid_argument("Body-force callbacks must both be callable");
    }
    fx_ = std::move(fx);
    fy_ = std::move(fy);
    has_uniform_body_force_ = false;
    uniform_fx_ = 0.0;
    uniform_fy_ = 0.0;
    return *this;
}

StokesSolver& StokesSolver::setBodyForce(double fx, double fy) {
    requireFinite(fx, "x body force");
    requireFinite(fy, "y body force");
    double fx_val = fx;
    double fy_val = fy;
    fx_ = [fx_val](double, double) {
        return fx_val;
    };
    fy_ = [fy_val](double, double) {
        return fy_val;
    };
    has_uniform_body_force_ = true;
    uniform_fx_ = fx_val;
    uniform_fy_ = fy_val;
    return *this;
}

StokesSolver& StokesSolver::setTolerance(double tol) {
    if (!std::isfinite(tol) || tol <= 0.0) {
        throw std::invalid_argument("Stokes tolerance must be finite and positive");
    }
    tolerance_ = tol;
    return *this;
}

StokesSolver& StokesSolver::setMaxIterations(int max_iter) {
    if (max_iter <= 0) {
        throw std::invalid_argument("Stokes maximum iterations must be positive");
    }
    max_iter_ = max_iter;
    return *this;
}

StokesSolver& StokesSolver::setPressureRelaxation(double omega_p) {
    if (!std::isfinite(omega_p) || omega_p <= 0.0 || omega_p > 1.0) {
        throw std::invalid_argument("Pressure relaxation must be finite and in (0, 1]");
    }
    omega_p_ = omega_p;
    return *this;
}

StokesSolver& StokesSolver::setVelocityRelaxation(double omega_v) {
    if (!std::isfinite(omega_v) || omega_v <= 0.0 || omega_v > 1.0) {
        throw std::invalid_argument("Velocity relaxation must be finite and in (0, 1]");
    }
    omega_v_ = omega_v;
    return *this;
}

double StokesSolver::reynolds(double L, double U, double rho) const {
    requireFinite(L, "Characteristic length");
    requireFinite(U, "Characteristic velocity");
    requireFinite(rho, "Fluid density");
    if (L <= 0.0 || U < 0.0 || rho <= 0.0) {
        throw std::invalid_argument("Reynolds scales require L > 0, U >= 0, and rho > 0");
    }
    const double value = rho * U * L / mu_;
    if (!std::isfinite(value)) {
        throw std::overflow_error("Reynolds number is not finite");
    }
    return value;
}

double StokesSolver::bodyForceX(double x, double y) const {
    const double value = fx_(x, y);
    if (!std::isfinite(value)) {
        throw std::domain_error("x body-force callback returned a non-finite value");
    }
    return value;
}

double StokesSolver::bodyForceY(double x, double y) const {
    const double value = fy_(x, y);
    if (!std::isfinite(value)) {
        throw std::domain_error("y body-force callback returned a non-finite value");
    }
    return value;
}

void StokesSolver::applyVelocityBCs(std::vector<double>& u, std::vector<double>& v) const {
    // Delegate to utility function (Stokes uses constant BC values, no inlet profiles)
    applyVelocityBoundaryConditions(mesh_, velocity_bcs_, u, v);
}

void StokesSolver::solveMomentum(std::vector<double>& u, std::vector<double>& v,
                                 const std::vector<double>& p) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();
    double dx2 = dx * dx;
    double dy2 = dy * dy;

    // Coefficient for the central node in the discrete Laplacian
    double a_ew = mu_ / dx2;                         // East-West coefficient
    double a_ns = mu_ / dy2;                         // North-South coefficient
    double a_p = 2.0 * mu_ / dx2 + 2.0 * mu_ / dy2;  // Central coefficient

    // Gauss-Seidel iterations for momentum equations
    for (int gs_iter = 0; gs_iter < 20; ++gs_iter) {
        // Solve for u-velocity (x-momentum): 0 = -dp/dx + mu*lap(u) + fx
        // This sweep is intentionally serial because neighboring updates are
        // data-dependent in an in-place Gauss-Seidel iteration.
        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                int idx = j * stride + i;
                double x = mesh_.x(i);
                double y_coord = mesh_.y(i, j);

                // Neighbor contributions (use current values for Gauss-Seidel)
                double u_west = u[idx - 1];
                double u_east = u[idx + 1];
                double u_south = u[idx - stride];
                double u_north = u[idx + stride];
                double u_neighbors = a_ew * (u_west + u_east) + a_ns * (u_south + u_north);

                // Pressure gradient: use central difference
                double dpdx = (p[idx + 1] - p[idx - 1]) / (2.0 * dx);

                // Body force
                double fx = bodyForceX(x, y_coord);

                // Solve: a_p * u = u_neighbors - dpdx + fx
                double u_new = (u_neighbors - dpdx + fx) / a_p;
                requireFiniteNumerical(u_new, "Stokes u-momentum iterate");

                // Under-relaxation update
                u[idx] = (1.0 - omega_v_) * u[idx] + omega_v_ * u_new;
                requireFiniteNumerical(u[idx], "Stokes relaxed u velocity");
            }
        }

        // Solve for v-velocity (y-momentum): 0 = -dp/dy + mu*lap(v) + fy
        // Keep the same deterministic Gauss-Seidel ordering as u-velocity.
        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                int idx = j * stride + i;
                double x = mesh_.x(i);
                double y_coord = mesh_.y(i, j);

                // Neighbor contributions
                double v_west = v[idx - 1];
                double v_east = v[idx + 1];
                double v_south = v[idx - stride];
                double v_north = v[idx + stride];
                double v_neighbors = a_ew * (v_west + v_east) + a_ns * (v_south + v_north);

                // Pressure gradient
                double dpdy = (p[idx + stride] - p[idx - stride]) / (2.0 * dy);

                // Body force
                double fy = bodyForceY(x, y_coord);

                // Solve: a_p * v = v_neighbors - dpdy + fy
                double v_new = (v_neighbors - dpdy + fy) / a_p;
                requireFiniteNumerical(v_new, "Stokes v-momentum iterate");

                // Under-relaxation update
                v[idx] = (1.0 - omega_v_) * v[idx] + omega_v_ * v_new;
                requireFiniteNumerical(v[idx], "Stokes relaxed v velocity");
            }
        }
    }
}

void StokesSolver::solvePressurePoisson(std::vector<double>& p, const std::vector<double>& u,
                                        const std::vector<double>& v) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();
    double dx2 = dx * dx;
    double dy2 = dy * dy;

    // For steady Stokes, we use a pressure Poisson formulation to enforce divergence-free velocity.
    // The pressure correction approach: lap(p) = scale * div(u)
    // where scale is chosen to properly couple pressure to velocity divergence.
    // A typical scaling for SIMPLE-like methods uses viscosity:
    double scale = mu_ / std::min(dx2, dy2);

    // Laplacian coefficients
    double a_ew = 1.0 / dx2;
    double a_ns = 1.0 / dy2;
    double a_p = 2.0 / dx2 + 2.0 / dy2;

    // SOR iterations for pressure Poisson
    double omega_sor = 1.5;  // Over-relaxation for faster convergence
    for (int iter = 0; iter < 200; ++iter) {
        double max_correction = 0.0;

        // Pressure SOR has the same in-place neighbor dependency as the
        // momentum sweeps, so it must not be parallelized by rows.
        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                int idx = j * stride + i;

                // Divergence of velocity at cell center (central difference)
                double div_u = (u[idx + 1] - u[idx - 1]) / (2.0 * dx) +
                               (v[idx + stride] - v[idx - stride]) / (2.0 * dy);

                // Neighbor pressure contributions
                double p_neighbors =
                    a_ew * (p[idx - 1] + p[idx + 1]) + a_ns * (p[idx - stride] + p[idx + stride]);

                // Solve: lap(p) = scale * div(u)
                // This drives pressure to enforce continuity
                double p_new = (p_neighbors - scale * div_u) / a_p;
                double correction = p_new - p[idx];
                requireFiniteNumerical(p_new, "Stokes pressure iterate");
                requireFiniteNumerical(correction, "Stokes pressure correction");
                p[idx] = p[idx] + omega_sor * correction;
                requireFiniteNumerical(p[idx], "Stokes relaxed pressure");
                max_correction = std::max(max_correction, std::abs(correction));
            }
        }

        // Neumann BCs for pressure (dp/dn = 0)
        applyPressureNeumannBCs(mesh_, p);

        // Early exit if converged
        if (max_correction < 1e-10)
            break;
    }

    // Subtract mean pressure to avoid drift (only Neumann BCs on p)
    subtractMeanPressure(mesh_, p);
}

void StokesSolver::correctVelocities(std::vector<double>& u, std::vector<double>& v,
                                     const std::vector<double>& p) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();
    double dx2 = dx * dx;
    double dy2 = dy * dy;

    // Velocity correction to enforce continuity.
    // From SIMPLE: u' = u* - (1/a_p) * dp/dx
    // For Stokes with our discretization, a_p = 2*mu*(1/dx^2 + 1/dy^2)
    // The correction factor is the reciprocal of the momentum equation central coefficient
    double a_p_mom = 2.0 * mu_ / dx2 + 2.0 * mu_ / dy2;
    double factor = 1.0 / a_p_mom;

    // Use the pressure relaxation parameter to control velocity correction strength
    double alpha = omega_p_;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            int idx = j * stride + i;

            // Pressure gradients (central difference)
            double dpdx = (p[idx + 1] - p[idx - 1]) / (2.0 * dx);
            double dpdy = (p[idx + stride] - p[idx - stride]) / (2.0 * dy);

            // Apply correction to make velocity field more divergence-free
            u[idx] -= alpha * factor * dpdx;
            v[idx] -= alpha * factor * dpdy;
        }
    }
}

double StokesSolver::computeMomentumResidual(const std::vector<double>& u,
                                             const std::vector<double>& v,
                                             const std::vector<double>& p) const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();
    double dx2 = dx * dx;
    double dy2 = dy * dy;

    double max_res = 0.0;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            int idx = j * stride + i;
            double x = mesh_.x(i);
            double y = mesh_.y(i, j);

            // x-momentum residual: -dp/dx + mu*lap(u) + fx = 0
            double lap_u = (u[idx - 1] - 2.0 * u[idx] + u[idx + 1]) / dx2 +
                           (u[idx - stride] - 2.0 * u[idx] + u[idx + stride]) / dy2;
            double dpdx = (p[idx + 1] - p[idx - 1]) / (2.0 * dx);
            double res_u = -dpdx + mu_ * lap_u + bodyForceX(x, y);

            // y-momentum residual
            double lap_v = (v[idx - 1] - 2.0 * v[idx] + v[idx + 1]) / dx2 +
                           (v[idx - stride] - 2.0 * v[idx] + v[idx + stride]) / dy2;
            double dpdy = (p[idx + stride] - p[idx - stride]) / (2.0 * dy);
            double res_v = -dpdy + mu_ * lap_v + bodyForceY(x, y);

            requireFiniteNumerical(res_u, "Stokes x-momentum residual");
            requireFiniteNumerical(res_v, "Stokes y-momentum residual");

            max_res = std::max(max_res, std::abs(res_u));
            max_res = std::max(max_res, std::abs(res_v));
        }
    }

    return max_res;
}

double StokesSolver::computeDivergence(const std::vector<double>& u,
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
            requireFiniteNumerical(div, "Stokes velocity divergence");
            max_div = std::max(max_div, std::abs(div));
        }
    }

    return max_div;
}

StokesResult StokesSolver::solve() const {
    int nx = mesh_.nx();
    int ny = mesh_.ny();
    int num_nodes = mesh_.numNodes();
    int stride = nx + 1;
    double dx = mesh_.dx();
    double dy = mesh_.dy();
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double Ly = ny * dy;        // Domain length in y
    double y0 = mesh_.y(0, 0);  // y at bottom boundary
    requireFiniteNumerical(Ly, "Stokes domain height");
    if (Ly <= 0.0) {
        throw std::invalid_argument("Stokes mesh must have a positive y extent");
    }

    // Validate callback-defined force fields over their declared nodal domain
    // before starting an iterative solve.
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            (void)bodyForceX(mesh_.x(i), mesh_.y(i, j));
            (void)bodyForceY(mesh_.x(i), mesh_.y(i, j));
        }
    }

    // Initialize fields
    std::vector<double> u(num_nodes, 0.0);
    std::vector<double> v(num_nodes, 0.0);
    std::vector<double> p(num_nodes, 0.0);

    // For SIMPLE algorithm, we need old values to compute corrections
    std::vector<double> u_old(num_nodes, 0.0);
    std::vector<double> v_old(num_nodes, 0.0);

    // Initialize velocity field with parabolic profile if we have inflow BC
    const auto& left_bc = velocity_bcs_[to_index(Boundary::Left)];
    if (left_bc.type == VelocityBCType::INFLOW) {
        double u_inlet = left_bc.u_value;
        for (int j = 0; j <= ny; ++j) {
            double y_coord = mesh_.y(0, j);
            double y_norm = (y_coord - y0) / Ly;  // 0 to 1
            // Parabolic profile: u = 6*u_avg*y*(1-y) where u_avg = (2/3)*u_max
            double u_parabolic = 6.0 * u_inlet * y_norm * (1.0 - y_norm);
            for (int i = 0; i <= nx; ++i) {
                int idx = j * stride + i;
                u[idx] = u_parabolic;
            }
        }
    }

    // Apply initial BCs
    applyVelocityBCs(u, v);
    requireFiniteField(u, "Initial Stokes u field");
    requireFiniteField(v, "Initial Stokes v field");

    StokesResult result{};
    result.converged = false;
    result.residual = std::numeric_limits<double>::infinity();
    result.divergence = std::numeric_limits<double>::infinity();

    const bool sealed_no_slip =
        std::all_of(velocity_bcs_.begin(), velocity_bcs_.end(),
                    [](const VelocityBC& bc) { return bc.type == VelocityBCType::NOSLIP; });
    if (sealed_no_slip && has_uniform_body_force_) {
        // A spatially uniform force is conservative: f = grad(f dot x).
        // In a sealed no-slip domain the exact steady solution is therefore
        // zero velocity with pressure p = f dot x.  Construct it directly
        // instead of forcing the generic pressure-correction iteration to use
        // an incompatible zero-normal-pressure-gradient wall approximation.
        long double pressure_sum = 0.0L;
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const int idx = j * stride + i;
                const long double pressure = static_cast<long double>(uniform_fx_) * mesh_.x(i) +
                                             static_cast<long double>(uniform_fy_) * mesh_.y(i, j);
                if (!std::isfinite(pressure) ||
                    std::abs(pressure) > std::numeric_limits<double>::max()) {
                    throw std::overflow_error("Hydrostatic Stokes pressure is not finite");
                }
                p[idx] = static_cast<double>(pressure);
                pressure_sum += pressure;
            }
        }

        const long double pressure_mean = pressure_sum / static_cast<long double>(num_nodes);
        if (!std::isfinite(pressure_mean) ||
            std::abs(pressure_mean) > std::numeric_limits<double>::max()) {
            throw std::overflow_error("Hydrostatic Stokes pressure gauge is not finite");
        }
        for (double& value : p) {
            value -= static_cast<double>(pressure_mean);
            requireFiniteNumerical(value, "Hydrostatic Stokes pressure");
        }

        result.u = std::move(u);
        result.v = std::move(v);
        result.pressure = std::move(p);
        result.iterations = 1;
        result.residual = computeMomentumResidual(result.u, result.v, result.pressure);
        result.divergence = computeDivergence(result.u, result.v);
        result.converged = true;
        return result;
    }

    // Coefficients for momentum equation (collocated grid)
    // Stokes: -dp/dx + mu*lap(u) + fx = 0
    // Discretized: a_p*u_p = sum(a_nb*u_nb) - dp/dx + fx
    double a_ew = mu_ / dx2;
    double a_ns = mu_ / dy2;
    double a_p = 2.0 * mu_ / dx2 + 2.0 * mu_ / dy2;

    // For pressure correction equation:
    // lap(p') = rho * div(u*) / dt, but for steady Stokes we use:
    // lap(p') = div(u*) / d_u where d_u = 1/a_p (velocity-pressure coupling)
    // This is the SIMPLE algorithm for steady flow
    double d_u = 1.0 / a_p;  // Velocity correction coefficient

    // Pressure Poisson coefficients
    double p_a_ew = d_u / dx2;
    double p_a_ns = d_u / dy2;
    double p_a_p = 2.0 * d_u / dx2 + 2.0 * d_u / dy2;

    // SIMPLE iteration
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Save old velocities for convergence check
        u_old = u;
        v_old = v;

        // Step 1: Solve momentum equations with current pressure (Gauss-Seidel)
        for (int gs_iter = 0; gs_iter < 5; ++gs_iter) {
            for (int j = 1; j < ny; ++j) {
                for (int i = 1; i < nx; ++i) {
                    int idx = j * stride + i;
                    double x = mesh_.x(i);
                    double y_coord = mesh_.y(i, j);

                    // u-momentum: a_p*u = a_nb*u_nb - dp/dx + fx
                    double u_neighbors = a_ew * (u[idx - 1] + u[idx + 1]) +
                                         a_ns * (u[idx - stride] + u[idx + stride]);
                    double dpdx = (p[idx + 1] - p[idx - 1]) / (2.0 * dx);
                    double fx = bodyForceX(x, y_coord);
                    double u_new = (u_neighbors - dpdx + fx) / a_p;
                    requireFiniteNumerical(u_new, "Stokes u-momentum iterate");
                    u[idx] = (1.0 - omega_v_) * u[idx] + omega_v_ * u_new;
                    requireFiniteNumerical(u[idx], "Stokes relaxed u velocity");

                    // v-momentum: a_p*v = a_nb*v_nb - dp/dy + fy
                    double v_neighbors = a_ew * (v[idx - 1] + v[idx + 1]) +
                                         a_ns * (v[idx - stride] + v[idx + stride]);
                    double dpdy = (p[idx + stride] - p[idx - stride]) / (2.0 * dy);
                    double fy = bodyForceY(x, y_coord);
                    double v_new = (v_neighbors - dpdy + fy) / a_p;
                    requireFiniteNumerical(v_new, "Stokes v-momentum iterate");
                    v[idx] = (1.0 - omega_v_) * v[idx] + omega_v_ * v_new;
                    requireFiniteNumerical(v[idx], "Stokes relaxed v velocity");
                }
            }
            applyVelocityBCs(u, v);
        }

        // Step 2: Solve pressure correction equation
        // d_u * lap(p') = div(u*)
        // Rearranged: lap(p') = div(u*) / d_u
        std::vector<double> p_prime(num_nodes, 0.0);

        for (int p_iter = 0; p_iter < 200; ++p_iter) {
            double max_corr = 0.0;
            for (int j = 1; j < ny; ++j) {
                for (int i = 1; i < nx; ++i) {
                    int idx = j * stride + i;

                    // Divergence of intermediate velocity (central difference)
                    double div_u = (u[idx + 1] - u[idx - 1]) / (2.0 * dx) +
                                   (v[idx + stride] - v[idx - stride]) / (2.0 * dy);

                    // Neighbor contributions
                    double p_neighbors = p_a_ew * (p_prime[idx - 1] + p_prime[idx + 1]) +
                                         p_a_ns * (p_prime[idx - stride] + p_prime[idx + stride]);

                    // Solve: p_a_p * p' = p_neighbors - div_u
                    double p_new = (p_neighbors - div_u) / p_a_p;
                    double corr = p_new - p_prime[idx];
                    requireFiniteNumerical(p_new, "Stokes pressure-correction iterate");
                    requireFiniteNumerical(corr, "Stokes pressure-correction update");
                    p_prime[idx] = p_prime[idx] + 0.8 * corr;  // SOR with omega < 1 for stability
                    requireFiniteNumerical(p_prime[idx], "Stokes relaxed pressure correction");
                    max_corr = std::max(max_corr, std::abs(corr));
                }
            }

            // Neumann BCs for p' (dp'/dn = 0)
            for (int j = 0; j <= ny; ++j) {
                p_prime[j * stride] = p_prime[j * stride + 1];
                p_prime[j * stride + nx] = p_prime[j * stride + nx - 1];
            }
            for (int i = 0; i <= nx; ++i) {
                p_prime[i] = p_prime[stride + i];
                p_prime[ny * stride + i] = p_prime[(ny - 1) * stride + i];
            }

            if (max_corr < 1e-10)
                break;
        }

        // Fix pressure reference (subtract mean to avoid drift)
        double p_mean = 0.0;
        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                p_mean += p_prime[j * stride + i];
            }
        }
        p_mean /= ((nx - 1) * (ny - 1));
        requireFiniteNumerical(p_mean, "Stokes pressure-correction mean");
        for (size_t i = 0; i < p_prime.size(); ++i) {
            p_prime[i] -= p_mean;
        }
        requireFiniteField(p_prime, "Stokes pressure-correction field");

        // Step 3: Update pressure with under-relaxation
        for (size_t i = 0; i < p.size(); ++i) {
            p[i] += omega_p_ * p_prime[i];
        }
        requireFiniteField(p, "Stokes pressure field");

        // Step 4: Correct velocities to satisfy continuity
        // u' = -d_u * dp'/dx, v' = -d_u * dp'/dy
        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                int idx = j * stride + i;
                double dp_dx = (p_prime[idx + 1] - p_prime[idx - 1]) / (2.0 * dx);
                double dp_dy = (p_prime[idx + stride] - p_prime[idx - stride]) / (2.0 * dy);
                u[idx] -= d_u * dp_dx;
                v[idx] -= d_u * dp_dy;
                requireFiniteNumerical(u[idx], "Corrected Stokes u velocity");
                requireFiniteNumerical(v[idx], "Corrected Stokes v velocity");
            }
        }
        applyVelocityBCs(u, v);

        // Check convergence
        double max_u_change = 0.0;
        for (size_t i = 0; i < u.size(); ++i) {
            max_u_change = std::max(max_u_change, std::abs(u[i] - u_old[i]));
            max_u_change = std::max(max_u_change, std::abs(v[i] - v_old[i]));
        }
        double div_res = computeDivergence(u, v);
        double momentum_res = computeMomentumResidual(u, v, p);
        requireFiniteNumerical(max_u_change, "Stokes velocity iteration change");
        requireFiniteNumerical(div_res, "Stokes divergence residual");
        requireFiniteNumerical(momentum_res, "Stokes momentum residual");

        result.iterations = iter + 1;
        // StokesResult::residual is the discrete momentum-equation defect,
        // not the change in velocity between nonlinear iterations.
        result.residual = momentum_res;
        result.divergence = div_res;

        // Check for divergence (blow-up detection)
        if (max_u_change > 1e10) {
            throw std::runtime_error("Stokes velocity iteration diverged");
        }

        if (max_u_change < tolerance_ && div_res < tolerance_) {
            result.converged = true;
            break;
        }
    }

    if (!result.converged) {
        throw std::runtime_error(
            "Stokes solve did not converge in " + std::to_string(max_iter_) +
            " iterations; momentum residual=" + std::to_string(result.residual) +
            ", divergence=" + std::to_string(result.divergence));
    }

    result.u = std::move(u);
    result.v = std::move(v);
    result.pressure = std::move(p);

    requireFiniteField(result.u, "Final Stokes u field");
    requireFiniteField(result.v, "Final Stokes v field");
    requireFiniteField(result.pressure, "Final Stokes pressure field");

    return result;
}

}  // namespace biotransport
