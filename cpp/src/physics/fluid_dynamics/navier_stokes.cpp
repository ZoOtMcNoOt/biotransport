/**
 * @file navier_stokes.cpp
 * @brief Compatible finite-volume implementation of NavierStokesSolver.
 */

#include <algorithm>
#include <biotransport/physics/fluid_dynamics/navier_stokes.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace biotransport {
namespace {

constexpr double kRoundoffFactor = 64.0;

bool finiteVector(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
}

int checkedBoundaryIndex(Boundary side) {
    const int index = static_cast<int>(side);
    if (index < 0 || index >= 4) {
        throw std::invalid_argument("Boundary value is outside the supported four sides");
    }
    return index;
}

}  // namespace

NavierStokesSolver::NavierStokesSolver(const StructuredMesh& mesh, double density, double viscosity)
    : mesh_(mesh), rho_(density), mu_(viscosity), nu_(viscosity / density) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("NavierStokesSolver requires a two-dimensional mesh");
    }
    if (mesh_.nx() < 2 || mesh_.ny() < 2) {
        throw std::invalid_argument(
            "NavierStokesSolver requires at least two cells in each direction");
    }
    if (!std::isfinite(density) || density <= 0.0) {
        throw std::invalid_argument("Density must be finite and positive");
    }
    if (!std::isfinite(viscosity) || viscosity <= 0.0) {
        throw std::invalid_argument("Viscosity must be finite and positive");
    }
}

NavierStokesSolver& NavierStokesSolver::setVelocityBC(Boundary side, VelocityBC bc) {
    const int index = checkedBoundaryIndex(side);
    if (bc.type != VelocityBCType::NOSLIP && bc.type != VelocityBCType::DIRICHLET) {
        throw std::invalid_argument(
            "The compatible Navier-Stokes projection currently supports NOSLIP and "
            "DIRICHLET velocity boundaries only; INFLOW, OUTFLOW, and NEUMANN need an "
            "explicit pressure-boundary model");
    }
    if (!std::isfinite(bc.u_value) || !std::isfinite(bc.v_value)) {
        throw std::invalid_argument("Velocity boundary values must be finite");
    }
    velocity_bcs_[static_cast<std::size_t>(index)] = bc;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setInlet(
    Boundary side, std::function<double(double x, double y)> u_profile,
    std::function<double(double x, double y)> v_profile) {
    (void)checkedBoundaryIndex(side);
    (void)u_profile;
    (void)v_profile;
    throw std::invalid_argument(
        "Profile inlets are not supported by the bounded compatible projection; use "
        "flux-compatible constant DIRICHLET boundaries or a dedicated open-boundary solver");
}

NavierStokesSolver& NavierStokesSolver::setBodyForce(std::function<double(double x, double y)> fx,
                                                     std::function<double(double x, double y)> fy) {
    if (!fx || !fy) {
        throw std::invalid_argument("Both body-force component functions must be callable");
    }
    fx_ = std::move(fx);
    fy_ = std::move(fy);
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setBodyForce(double fx, double fy) {
    if (!std::isfinite(fx) || !std::isfinite(fy)) {
        throw std::invalid_argument("Body-force components must be finite");
    }
    fx_ = [fx](double, double) {
        return fx;
    };
    fy_ = [fy](double, double) {
        return fy;
    };
    return *this;
}

void NavierStokesSolver::validatePackedField(const std::vector<double>& field,
                                             const char* name) const {
    const auto expected = static_cast<std::size_t>(mesh_.numNodes());
    if (field.size() != expected) {
        throw std::invalid_argument(std::string(name) + " must contain exactly " +
                                    std::to_string(expected) + " packed staggered values");
    }
    if (!finiteVector(field)) {
        throw std::invalid_argument(std::string(name) + " must contain finite values only");
    }
}

NavierStokesSolver& NavierStokesSolver::setInitialVelocity(const std::vector<double>& u0,
                                                           const std::vector<double>& v0) {
    validatePackedField(u0, "Initial x-velocity");
    validatePackedField(v0, "Initial y-velocity");
    u0_ = u0;
    v0_ = v0;
    has_initial_ = true;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setConvectionScheme(ConvectionScheme scheme) {
    if (scheme != ConvectionScheme::UPWIND && scheme != ConvectionScheme::CENTRAL) {
        throw std::invalid_argument(
            "QUICK and HYBRID convection are reserved but not implemented; choose UPWIND or "
            "CENTRAL explicitly");
    }
    conv_scheme_ = scheme;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setCFL(double cfl) {
    if (!std::isfinite(cfl) || cfl <= 0.0 || cfl > 1.0) {
        throw std::invalid_argument("CFL safety factor must be finite and in (0, 1]");
    }
    cfl_ = cfl;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setTimeStep(double dt) {
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("Time step must be finite and nonnegative (zero is adaptive)");
    }
    dt_fixed_ = dt;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setPressureTolerance(double tol) {
    if (!std::isfinite(tol) || tol <= 0.0) {
        throw std::invalid_argument("Pressure tolerance must be finite and positive");
    }
    p_tolerance_ = tol;
    return *this;
}

NavierStokesSolver& NavierStokesSolver::setMaxPressureIterations(int max_iter) {
    if (max_iter <= 0) {
        throw std::invalid_argument("Maximum pressure iterations must be positive");
    }
    p_max_iter_ = max_iter;
    return *this;
}

double NavierStokesSolver::reynolds(double length, double velocity) const {
    if (!std::isfinite(length) || length < 0.0 || !std::isfinite(velocity) || velocity < 0.0) {
        throw std::invalid_argument(
            "Reynolds-number length and velocity scales must be finite and nonnegative");
    }
    return rho_ * velocity * length / mu_;
}

void NavierStokesSolver::validateBoundaryConfiguration() const {
    for (const auto& bc : velocity_bcs_) {
        if (bc.type != VelocityBCType::NOSLIP && bc.type != VelocityBCType::DIRICHLET) {
            throw std::invalid_argument(
                "Unsupported open or traction velocity boundary in Navier-Stokes solve");
        }
        if (!std::isfinite(bc.u_value) || !std::isfinite(bc.v_value)) {
            throw std::invalid_argument("Velocity boundary values must be finite");
        }
    }
}

double NavierStokesSolver::boundaryComponent(Boundary side, bool x_component) const {
    const auto& bc = velocity_bcs_[static_cast<std::size_t>(checkedBoundaryIndex(side))];
    if (bc.type == VelocityBCType::NOSLIP) {
        return 0.0;
    }
    return x_component ? bc.u_value : bc.v_value;
}

void NavierStokesSolver::applyVelocityBCs(std::vector<double>& u, std::vector<double>& v) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;

    const double u_left = boundaryComponent(Boundary::Left, true);
    const double u_right = boundaryComponent(Boundary::Right, true);
    const double v_bottom = boundaryComponent(Boundary::Bottom, false);
    const double v_top = boundaryComponent(Boundary::Top, false);

    // Prescribed normal face velocities.
    for (int j = 0; j < ny; ++j) {
        u[j * stride] = u_left;
        u[j * stride + nx] = u_right;
    }
    for (int i = 0; i < nx; ++i) {
        v[i] = v_bottom;
        v[ny * stride + i] = v_top;
    }

    // The final u row and v column are padding.  Populate them with the
    // corresponding tangential wall values for predictable reshaping.
    const double u_top = boundaryComponent(Boundary::Top, true);
    const double v_right = boundaryComponent(Boundary::Right, false);
    for (int i = 0; i <= nx; ++i) {
        u[ny * stride + i] = u_top;
    }
    for (int j = 0; j <= ny; ++j) {
        v[j * stride + nx] = v_right;
    }
}

double NavierStokesSolver::maxTimeStep(const std::vector<double>& u,
                                       const std::vector<double>& v) const {
    validatePackedField(u, "x-velocity");
    validatePackedField(v, "y-velocity");

    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;
    const double dx = mesh_.dx();
    const double dy = mesh_.dy();

    double max_u = 0.0;
    double max_v = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            max_u = std::max(max_u, std::abs(u[j * stride + i]));
        }
    }
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            max_v = std::max(max_v, std::abs(v[j * stride + i]));
        }
    }

    const double inverse_convective_time = max_u / dx + max_v / dy;
    const double dt_convective = inverse_convective_time > 0.0
                                     ? 1.0 / inverse_convective_time
                                     : std::numeric_limits<double>::infinity();
    const double dt_diffusive = 0.5 / (nu_ * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
    const double result = cfl_ * std::min(dt_convective, dt_diffusive);
    if (!std::isfinite(result) || result <= 0.0) {
        throw std::runtime_error("Could not determine a finite positive explicit time step");
    }
    return result;
}

void NavierStokesSolver::computeConvection(const std::vector<double>& u,
                                           const std::vector<double>& v,
                                           std::vector<double>& conv_u,
                                           std::vector<double>& conv_v) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;
    const double dx = mesh_.dx();
    const double dy = mesh_.dy();

    conv_u.assign(u.size(), 0.0);
    conv_v.assign(v.size(), 0.0);

    const double u_bottom_wall = boundaryComponent(Boundary::Bottom, true);
    const double u_top_wall = boundaryComponent(Boundary::Top, true);
    const double v_left_wall = boundaryComponent(Boundary::Left, false);
    const double v_right_wall = boundaryComponent(Boundary::Right, false);

    const auto u_at = [&](int i, int j) {
        if (j < 0) {
            return 2.0 * u_bottom_wall - u[i];
        }
        if (j >= ny) {
            return 2.0 * u_top_wall - u[(ny - 1) * stride + i];
        }
        return u[j * stride + i];
    };
    const auto v_at = [&](int i, int j) {
        if (i < 0) {
            return 2.0 * v_left_wall - v[j * stride];
        }
        if (i >= nx) {
            return 2.0 * v_right_wall - v[j * stride + nx - 1];
        }
        return v[j * stride + i];
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int index = j * stride + i;
            const double u_face = u[index];
            const double v_face =
                0.25 * (v_at(i - 1, j) + v_at(i, j) + v_at(i - 1, j + 1) + v_at(i, j + 1));

            double du_dx = 0.0;
            double du_dy = 0.0;
            if (conv_scheme_ == ConvectionScheme::UPWIND) {
                du_dx = u_face >= 0.0 ? (u_at(i, j) - u_at(i - 1, j)) / dx
                                      : (u_at(i + 1, j) - u_at(i, j)) / dx;
                du_dy = v_face >= 0.0 ? (u_at(i, j) - u_at(i, j - 1)) / dy
                                      : (u_at(i, j + 1) - u_at(i, j)) / dy;
            } else {
                du_dx = (u_at(i + 1, j) - u_at(i - 1, j)) / (2.0 * dx);
                du_dy = (u_at(i, j + 1) - u_at(i, j - 1)) / (2.0 * dy);
            }
            conv_u[index] = u_face * du_dx + v_face * du_dy;
        }
    }

    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            const double u_face =
                0.25 * (u_at(i, j - 1) + u_at(i + 1, j - 1) + u_at(i, j) + u_at(i + 1, j));
            const double v_face = v[index];

            double dv_dx = 0.0;
            double dv_dy = 0.0;
            if (conv_scheme_ == ConvectionScheme::UPWIND) {
                dv_dx = u_face >= 0.0 ? (v_at(i, j) - v_at(i - 1, j)) / dx
                                      : (v_at(i + 1, j) - v_at(i, j)) / dx;
                dv_dy = v_face >= 0.0 ? (v_at(i, j) - v_at(i, j - 1)) / dy
                                      : (v_at(i, j + 1) - v_at(i, j)) / dy;
            } else {
                dv_dx = (v_at(i + 1, j) - v_at(i - 1, j)) / (2.0 * dx);
                dv_dy = (v_at(i, j + 1) - v_at(i, j - 1)) / (2.0 * dy);
            }
            conv_v[index] = u_face * dv_dx + v_face * dv_dy;
        }
    }
}

void NavierStokesSolver::computeDiffusion(const std::vector<double>& u,
                                          const std::vector<double>& v, std::vector<double>& diff_u,
                                          std::vector<double>& diff_v) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;
    const double dx2 = mesh_.dx() * mesh_.dx();
    const double dy2 = mesh_.dy() * mesh_.dy();

    diff_u.assign(u.size(), 0.0);
    diff_v.assign(v.size(), 0.0);

    const double u_bottom_wall = boundaryComponent(Boundary::Bottom, true);
    const double u_top_wall = boundaryComponent(Boundary::Top, true);
    const double v_left_wall = boundaryComponent(Boundary::Left, false);
    const double v_right_wall = boundaryComponent(Boundary::Right, false);

    const auto u_at = [&](int i, int j) {
        if (j < 0) {
            return 2.0 * u_bottom_wall - u[i];
        }
        if (j >= ny) {
            return 2.0 * u_top_wall - u[(ny - 1) * stride + i];
        }
        return u[j * stride + i];
    };
    const auto v_at = [&](int i, int j) {
        if (i < 0) {
            return 2.0 * v_left_wall - v[j * stride];
        }
        if (i >= nx) {
            return 2.0 * v_right_wall - v[j * stride + nx - 1];
        }
        return v[j * stride + i];
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int index = j * stride + i;
            diff_u[index] = nu_ * ((u_at(i - 1, j) - 2.0 * u_at(i, j) + u_at(i + 1, j)) / dx2 +
                                   (u_at(i, j - 1) - 2.0 * u_at(i, j) + u_at(i, j + 1)) / dy2);
        }
    }
    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            diff_v[index] = nu_ * ((v_at(i - 1, j) - 2.0 * v_at(i, j) + v_at(i + 1, j)) / dx2 +
                                   (v_at(i, j - 1) - 2.0 * v_at(i, j) + v_at(i, j + 1)) / dy2);
        }
    }
}

NavierStokesSolver::PressureSolveInfo NavierStokesSolver::solvePressurePoisson(
    std::vector<double>& pressure, const std::vector<double>& u_star,
    const std::vector<double>& v_star, double dt) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;
    const int cell_count = nx * ny;
    const double dx = mesh_.dx();
    const double dy = mesh_.dy();
    const double inverse_dx2 = 1.0 / (dx * dx);
    const double inverse_dy2 = 1.0 / (dy * dy);

    std::vector<double> rhs(pressure.size(), 0.0);
    double divergence_sum = 0.0;
    double max_abs_divergence = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            const double divergence = (u_star[index + 1] - u_star[index]) / dx +
                                      (v_star[index + stride] - v_star[index]) / dy;
            divergence_sum += divergence;
            max_abs_divergence = std::max(max_abs_divergence, std::abs(divergence));
            rhs[index] = (rho_ / dt) * divergence;
        }
    }

    const double mean_divergence = divergence_sum / static_cast<double>(cell_count);
    const double compatibility_tolerance = 1e-12 + kRoundoffFactor *
                                                       std::numeric_limits<double>::epsilon() *
                                                       std::max(1.0, max_abs_divergence);
    if (std::abs(mean_divergence) > compatibility_tolerance) {
        throw std::domain_error(
            "Prescribed normal velocities have nonzero net boundary flux; the closed-domain "
            "pressure Poisson problem is incompatible (mean divergence=" +
            std::to_string(mean_divergence) + ")");
    }

    // Remove only roundoff-level incompatibility from the pure-Neumann system.
    const double rhs_mean = (rho_ / dt) * mean_divergence;
    double rhs_norm = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            rhs[index] -= rhs_mean;
            rhs_norm = std::max(rhs_norm, std::abs(rhs[index]));
        }
    }

    if (rhs_norm == 0.0) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                pressure[j * stride + i] = 0.0;
            }
        }
        return PressureSolveInfo{0, 0.0, true};
    }

    // Gauge the initial iterate to zero mean.
    double pressure_mean = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            pressure_mean += pressure[j * stride + i];
        }
    }
    pressure_mean /= static_cast<double>(cell_count);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            pressure[j * stride + i] -= pressure_mean;
        }
    }

    const double pi = std::acos(-1.0);
    const double omega = 2.0 / (1.0 + std::sin(pi / static_cast<double>(std::max(nx, ny))));

    PressureSolveInfo info;
    for (int iteration = 1; iteration <= p_max_iter_; ++iteration) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int index = j * stride + i;
                double neighbor_sum = 0.0;
                double diagonal = 0.0;
                if (i > 0) {
                    neighbor_sum += pressure[index - 1] * inverse_dx2;
                    diagonal += inverse_dx2;
                }
                if (i + 1 < nx) {
                    neighbor_sum += pressure[index + 1] * inverse_dx2;
                    diagonal += inverse_dx2;
                }
                if (j > 0) {
                    neighbor_sum += pressure[index - stride] * inverse_dy2;
                    diagonal += inverse_dy2;
                }
                if (j + 1 < ny) {
                    neighbor_sum += pressure[index + stride] * inverse_dy2;
                    diagonal += inverse_dy2;
                }
                const double pressure_update = (neighbor_sum - rhs[index]) / diagonal;
                pressure[index] += omega * (pressure_update - pressure[index]);
            }
        }

        pressure_mean = 0.0;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                pressure_mean += pressure[j * stride + i];
            }
        }
        pressure_mean /= static_cast<double>(cell_count);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                pressure[j * stride + i] -= pressure_mean;
            }
        }

        double max_absolute_residual = 0.0;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int index = j * stride + i;
                double laplacian = 0.0;
                if (i > 0) {
                    laplacian += (pressure[index - 1] - pressure[index]) * inverse_dx2;
                }
                if (i + 1 < nx) {
                    laplacian += (pressure[index + 1] - pressure[index]) * inverse_dx2;
                }
                if (j > 0) {
                    laplacian += (pressure[index - stride] - pressure[index]) * inverse_dy2;
                }
                if (j + 1 < ny) {
                    laplacian += (pressure[index + stride] - pressure[index]) * inverse_dy2;
                }
                max_absolute_residual =
                    std::max(max_absolute_residual, std::abs(laplacian - rhs[index]));
            }
        }

        info.iterations = iteration;
        info.relative_residual = max_absolute_residual / rhs_norm;
        if (!std::isfinite(info.relative_residual)) {
            return info;
        }
        if (info.relative_residual <= p_tolerance_) {
            info.converged = true;
            return info;
        }
    }
    return info;
}

void NavierStokesSolver::projectVelocity(std::vector<double>& u, std::vector<double>& v,
                                         const std::vector<double>& pressure, double dt) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;
    const double x_factor = dt / (rho_ * mesh_.dx());
    const double y_factor = dt / (rho_ * mesh_.dy());

    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int index = j * stride + i;
            u[index] -= x_factor * (pressure[index] - pressure[index - 1]);
        }
    }
    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            v[index] -= y_factor * (pressure[index] - pressure[index - stride]);
        }
    }
}

double NavierStokesSolver::computeDivergence(const std::vector<double>& u,
                                             const std::vector<double>& v) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;
    const double dx = mesh_.dx();
    const double dy = mesh_.dy();
    double maximum = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            const double divergence =
                (u[index + 1] - u[index]) / dx + (v[index + stride] - v[index]) / dy;
            maximum = std::max(maximum, std::abs(divergence));
        }
    }
    return maximum;
}

double NavierStokesSolver::computeMaxVelocity(const std::vector<double>& u,
                                              const std::vector<double>& v) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;
    double maximum = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            const double u_cell = 0.5 * (u[index] + u[index + 1]);
            const double v_cell = 0.5 * (v[index] + v[index + stride]);
            maximum = std::max(maximum, std::hypot(u_cell, v_cell));
        }
    }
    return maximum;
}

NavierStokesSolver::StepInfo NavierStokesSolver::takeStep(std::vector<double>& u,
                                                          std::vector<double>& v,
                                                          std::vector<double>& pressure,
                                                          double dt) const {
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::runtime_error("Navier-Stokes integration produced a nonpositive time step");
    }
    const double stability_limit = maxTimeStep(u, v);
    if (dt > stability_limit) {
        throw std::domain_error("Configured time step " + std::to_string(dt) +
                                " exceeds the explicit stability bound " +
                                std::to_string(stability_limit));
    }

    std::vector<double> conv_u;
    std::vector<double> conv_v;
    std::vector<double> diff_u;
    std::vector<double> diff_v;
    computeConvection(u, v, conv_u, conv_v);
    computeDiffusion(u, v, diff_u, diff_v);

    std::vector<double> u_star = u;
    std::vector<double> v_star = v;
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;

    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int index = j * stride + i;
            const double x = mesh_.x(i);
            const double y = mesh_.y(0, j) + 0.5 * mesh_.dy();
            const double force = fx_(x, y);
            if (!std::isfinite(force)) {
                throw std::domain_error("x body-force callback returned a non-finite value");
            }
            u_star[index] += dt * (-conv_u[index] + diff_u[index] + force / rho_);
        }
    }
    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int index = j * stride + i;
            const double x = mesh_.x(i) + 0.5 * mesh_.dx();
            const double y = mesh_.y(0, j);
            const double force = fy_(x, y);
            if (!std::isfinite(force)) {
                throw std::domain_error("y body-force callback returned a non-finite value");
            }
            v_star[index] += dt * (-conv_v[index] + diff_v[index] + force / rho_);
        }
    }
    applyVelocityBCs(u_star, v_star);
    if (!finiteVector(u_star) || !finiteVector(v_star)) {
        throw std::runtime_error("Navier-Stokes predictor produced non-finite velocity values");
    }

    const double predictor_divergence = computeDivergence(u_star, v_star);
    StepInfo info;
    info.pressure = solvePressurePoisson(pressure, u_star, v_star, dt);
    if (!info.pressure.converged) {
        throw std::runtime_error(
            "Pressure projection did not reach the requested relative residual in " +
            std::to_string(p_max_iter_) +
            " iterations; final residual=" + std::to_string(info.pressure.relative_residual));
    }

    projectVelocity(u_star, v_star, pressure, dt);
    applyVelocityBCs(u_star, v_star);
    info.divergence = computeDivergence(u_star, v_star);

    const double divergence_allowance = 4.0 * p_tolerance_ * std::max(1.0, predictor_divergence) +
                                        kRoundoffFactor * std::numeric_limits<double>::epsilon() *
                                            std::max(1.0, predictor_divergence);
    if (!std::isfinite(info.divergence) || info.divergence > divergence_allowance) {
        throw std::runtime_error(
            "Pressure solve reported convergence but the compatible projection left excessive "
            "divergence: " +
            std::to_string(info.divergence));
    }
    if (!finiteVector(u_star) || !finiteVector(v_star) || !finiteVector(pressure)) {
        throw std::runtime_error("Navier-Stokes projection produced non-finite fields");
    }

    u.swap(u_star);
    v.swap(v_star);
    return info;
}

void NavierStokesSolver::fillPressurePadding(std::vector<double>& pressure) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const int stride = nx + 1;
    for (int j = 0; j < ny; ++j) {
        pressure[j * stride + nx] = pressure[j * stride + nx - 1];
    }
    for (int i = 0; i <= nx; ++i) {
        pressure[ny * stride + i] = pressure[(ny - 1) * stride + i];
    }
}

NavierStokesResult NavierStokesSolver::makeResult(std::vector<double> u, std::vector<double> v,
                                                  std::vector<double> pressure, double time,
                                                  int steps, const StepInfo& final_step) const {
    NavierStokesResult result;
    result.time = time;
    result.time_steps = steps;
    result.max_velocity = computeMaxVelocity(u, v);
    result.reynolds = reynolds(mesh_.dx() * static_cast<double>(mesh_.nx()), result.max_velocity);
    result.pressure_iterations = final_step.pressure.iterations;
    result.pressure_residual = final_step.pressure.relative_residual;
    result.divergence = computeDivergence(u, v);

    const double divergence_scale = result.max_velocity / std::min(mesh_.dx(), mesh_.dy());
    const double acceptable_divergence =
        4.0 * p_tolerance_ * std::max(1.0, divergence_scale) +
        kRoundoffFactor * std::numeric_limits<double>::epsilon() * std::max(1.0, divergence_scale);
    result.stable = final_step.pressure.converged && finiteVector(u) && finiteVector(v) &&
                    finiteVector(pressure) && std::isfinite(result.divergence) &&
                    result.divergence <= acceptable_divergence;

    fillPressurePadding(pressure);
    result.u = std::move(u);
    result.v = std::move(v);
    result.pressure = std::move(pressure);
    return result;
}

NavierStokesResult NavierStokesSolver::solve(double duration, double output_interval) {
    if (!std::isfinite(duration) || duration < 0.0) {
        throw std::invalid_argument("Solve duration must be finite and nonnegative");
    }
    if (!std::isfinite(output_interval) || output_interval < 0.0) {
        throw std::invalid_argument("Output interval must be finite and nonnegative");
    }
    if (output_interval != 0.0) {
        throw std::invalid_argument(
            "Snapshot output_interval is not implemented; pass zero and use solveSteps for "
            "explicit sampling");
    }
    validateBoundaryConfiguration();

    std::vector<double> u(static_cast<std::size_t>(mesh_.numNodes()), 0.0);
    std::vector<double> v(static_cast<std::size_t>(mesh_.numNodes()), 0.0);
    std::vector<double> pressure(static_cast<std::size_t>(mesh_.numNodes()), 0.0);
    if (has_initial_) {
        u = u0_;
        v = v0_;
    }
    applyVelocityBCs(u, v);

    StepInfo final_step;
    final_step.pressure = PressureSolveInfo{0, 0.0, true};
    final_step.divergence = computeDivergence(u, v);
    double time = 0.0;
    int steps = 0;
    constexpr int maximum_steps = 10000000;

    while (time < duration) {
        if (steps >= maximum_steps) {
            throw std::runtime_error("Navier-Stokes solve exceeded ten million time steps");
        }
        const double requested_dt = dt_fixed_ > 0.0 ? dt_fixed_ : maxTimeStep(u, v);
        const double remaining = duration - time;
        const double dt = std::min(requested_dt, remaining);
        final_step = takeStep(u, v, pressure, dt);
        ++steps;

        if (dt == remaining) {
            time = duration;
        } else {
            const double next_time = time + dt;
            if (!std::isfinite(next_time) || next_time <= time) {
                throw std::runtime_error(
                    "time step is too small to advance the Navier-Stokes clock");
            }
            if (next_time >= duration) {
                time = duration;
            } else {
                const double residual = duration - next_time;
                const double roundoff = kRoundoffFactor * std::numeric_limits<double>::epsilon() *
                                        std::max(std::abs(duration), std::abs(next_time));
                time = residual <= roundoff && residual < 0.5 * dt ? duration : next_time;
            }
        }
    }

    return makeResult(std::move(u), std::move(v), std::move(pressure), time, steps, final_step);
}

NavierStokesResult NavierStokesSolver::solveSteps(int num_steps) {
    if (num_steps < 0) {
        throw std::invalid_argument("Number of time steps must be nonnegative");
    }
    validateBoundaryConfiguration();

    std::vector<double> u(static_cast<std::size_t>(mesh_.numNodes()), 0.0);
    std::vector<double> v(static_cast<std::size_t>(mesh_.numNodes()), 0.0);
    std::vector<double> pressure(static_cast<std::size_t>(mesh_.numNodes()), 0.0);
    if (has_initial_) {
        u = u0_;
        v = v0_;
    }
    applyVelocityBCs(u, v);

    StepInfo final_step;
    final_step.pressure = PressureSolveInfo{0, 0.0, true};
    final_step.divergence = computeDivergence(u, v);
    double time = 0.0;
    for (int step = 0; step < num_steps; ++step) {
        const double dt = dt_fixed_ > 0.0 ? dt_fixed_ : maxTimeStep(u, v);
        final_step = takeStep(u, v, pressure, dt);
        time += dt;
        if (!std::isfinite(time)) {
            throw std::runtime_error("Accumulated Navier-Stokes time became non-finite");
        }
    }

    return makeResult(std::move(u), std::move(v), std::move(pressure), time, num_steps, final_step);
}

}  // namespace biotransport
