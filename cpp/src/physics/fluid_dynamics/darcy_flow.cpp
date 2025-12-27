#include <biotransport/physics/fluid_dynamics/darcy_flow.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace biotransport {

DarcyFlowSolver::DarcyFlowSolver(const StructuredMesh& mesh, double kappa)
    : mesh_(mesh), kappa_(mesh.numNodes(), kappa) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("DarcyFlowSolver requires a 2D mesh");
    }
    if (kappa <= 0.0 || !std::isfinite(kappa)) {
        throw std::invalid_argument("Hydraulic conductivity must be positive and finite");
    }

    // Default to Neumann (no-flux) on all boundaries
    for (int i = 0; i < 4; ++i) {
        boundaries_[i] = BoundaryCondition::Neumann(0.0);
    }
}

DarcyFlowSolver::DarcyFlowSolver(const StructuredMesh& mesh, const std::vector<double>& kappa)
    : mesh_(mesh), kappa_(kappa) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("DarcyFlowSolver requires a 2D mesh");
    }
    if (kappa.size() != static_cast<size_t>(mesh.numNodes())) {
        throw std::invalid_argument("kappa size must match mesh nodes");
    }
    for (double k : kappa) {
        if (k <= 0.0 || !std::isfinite(k)) {
            throw std::invalid_argument("All kappa values must be positive and finite");
        }
    }

    for (int i = 0; i < 4; ++i) {
        boundaries_[i] = BoundaryCondition::Neumann(0.0);
    }
}

DarcyFlowSolver& DarcyFlowSolver::setDirichlet(Boundary side, double pressure) {
    boundaries_[to_index(side)] = BoundaryCondition::Dirichlet(pressure);
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setNeumann(Boundary side, double flux) {
    boundaries_[to_index(side)] = BoundaryCondition::Neumann(flux);
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setInternalPressure(const std::vector<std::uint8_t>& mask,
                                                      double pressure) {
    if (mask.size() != static_cast<size_t>(mesh_.numNodes())) {
        throw std::invalid_argument("Internal mask size must match mesh nodes");
    }
    internal_mask_ = mask;
    internal_pressure_ = pressure;
    has_internal_pressure_ = true;
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setOmega(double omega) {
    if (omega <= 0.0 || omega >= 2.0) {
        throw std::invalid_argument("omega must be in (0, 2)");
    }
    omega_ = omega;
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setTolerance(double tol) {
    if (tol <= 0.0) {
        throw std::invalid_argument("tolerance must be positive");
    }
    tolerance_ = tol;
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setMaxIterations(int max_iter) {
    if (max_iter <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    max_iter_ = max_iter;
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setInitialGuess(const std::vector<double>& pressure) {
    if (pressure.size() != static_cast<size_t>(mesh_.numNodes())) {
        throw std::invalid_argument("Initial guess size must match mesh nodes");
    }
    initial_guess_ = pressure;
    return *this;
}

void DarcyFlowSolver::applyBoundaryPressure(std::vector<double>& p) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();

    // Left boundary
    if (boundaries_[to_index(Boundary::Left)].type == BoundaryType::DIRICHLET) {
        double val = boundaries_[to_index(Boundary::Left)].value;
        for (int j = 0; j <= ny; ++j) {
            p[mesh_.index(0, j)] = val;
        }
    }

    // Right boundary
    if (boundaries_[to_index(Boundary::Right)].type == BoundaryType::DIRICHLET) {
        double val = boundaries_[to_index(Boundary::Right)].value;
        for (int j = 0; j <= ny; ++j) {
            p[mesh_.index(nx, j)] = val;
        }
    }

    // Bottom boundary
    if (boundaries_[to_index(Boundary::Bottom)].type == BoundaryType::DIRICHLET) {
        double val = boundaries_[to_index(Boundary::Bottom)].value;
        for (int i = 0; i <= nx; ++i) {
            p[mesh_.index(i, 0)] = val;
        }
    }

    // Top boundary
    if (boundaries_[to_index(Boundary::Top)].type == BoundaryType::DIRICHLET) {
        double val = boundaries_[to_index(Boundary::Top)].value;
        for (int i = 0; i <= nx; ++i) {
            p[mesh_.index(i, ny)] = val;
        }
    }

    // Internal pressure sources
    if (has_internal_pressure_) {
        for (size_t i = 0; i < internal_mask_.size(); ++i) {
            if (internal_mask_[i] != 0) {
                p[i] = internal_pressure_;
            }
        }
    }
}

void DarcyFlowSolver::computeVelocity(const std::vector<double>& pressure, std::vector<double>& vx,
                                      std::vector<double>& vy) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const double dx = mesh_.dx();
    const double dy = mesh_.dy();
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    vx.assign(mesh_.numNodes(), 0.0);
    vy.assign(mesh_.numNodes(), 0.0);

    // Interior: central difference
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int c = mesh_.index(i, j);
            const double grad_x =
                (pressure[mesh_.index(i + 1, j)] - pressure[mesh_.index(i - 1, j)]) * inv_2dx;
            const double grad_y =
                (pressure[mesh_.index(i, j + 1)] - pressure[mesh_.index(i, j - 1)]) * inv_2dy;
            vx[c] = -kappa_[c] * grad_x;
            vy[c] = -kappa_[c] * grad_y;
        }
    }

    // Left boundary (i = 0)
    for (int j = 1; j < ny; ++j) {
        const int c = mesh_.index(0, j);
        const double grad_x = (pressure[mesh_.index(1, j)] - pressure[c]) * inv_dx;
        const double grad_y =
            (pressure[mesh_.index(0, j + 1)] - pressure[mesh_.index(0, j - 1)]) * inv_2dy;
        vx[c] = -kappa_[c] * grad_x;
        vy[c] = -kappa_[c] * grad_y;
    }

    // Right boundary (i = nx)
    for (int j = 1; j < ny; ++j) {
        const int c = mesh_.index(nx, j);
        const double grad_x = (pressure[c] - pressure[mesh_.index(nx - 1, j)]) * inv_dx;
        const double grad_y =
            (pressure[mesh_.index(nx, j + 1)] - pressure[mesh_.index(nx, j - 1)]) * inv_2dy;
        vx[c] = -kappa_[c] * grad_x;
        vy[c] = -kappa_[c] * grad_y;
    }

    // Bottom boundary (j = 0)
    for (int i = 1; i < nx; ++i) {
        const int c = mesh_.index(i, 0);
        const double grad_x =
            (pressure[mesh_.index(i + 1, 0)] - pressure[mesh_.index(i - 1, 0)]) * inv_2dx;
        const double grad_y = (pressure[mesh_.index(i, 1)] - pressure[c]) * inv_dy;
        vx[c] = -kappa_[c] * grad_x;
        vy[c] = -kappa_[c] * grad_y;
    }

    // Top boundary (j = ny)
    for (int i = 1; i < nx; ++i) {
        const int c = mesh_.index(i, ny);
        const double grad_x =
            (pressure[mesh_.index(i + 1, ny)] - pressure[mesh_.index(i - 1, ny)]) * inv_2dx;
        const double grad_y = (pressure[c] - pressure[mesh_.index(i, ny - 1)]) * inv_dy;
        vx[c] = -kappa_[c] * grad_x;
        vy[c] = -kappa_[c] * grad_y;
    }

    // Corners: copy from neighbors
    vx[mesh_.index(0, 0)] = vx[mesh_.index(1, 0)];
    vy[mesh_.index(0, 0)] = vy[mesh_.index(0, 1)];

    vx[mesh_.index(nx, 0)] = vx[mesh_.index(nx - 1, 0)];
    vy[mesh_.index(nx, 0)] = vy[mesh_.index(nx, 1)];

    vx[mesh_.index(0, ny)] = vx[mesh_.index(1, ny)];
    vy[mesh_.index(0, ny)] = vy[mesh_.index(0, ny - 1)];

    vx[mesh_.index(nx, ny)] = vx[mesh_.index(nx - 1, ny)];
    vy[mesh_.index(nx, ny)] = vy[mesh_.index(nx, ny - 1)];
}

DarcyFlowResult DarcyFlowSolver::solve() const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const double dx = mesh_.dx();
    const double dy = mesh_.dy();
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;

    // Initialize pressure field
    std::vector<double> p;
    if (!initial_guess_.empty()) {
        p = initial_guess_;
    } else {
        // Use average of Dirichlet values as initial guess
        double p_init = 0.0;
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            if (boundaries_[i].type == BoundaryType::DIRICHLET) {
                p_init += boundaries_[i].value;
                ++count;
            }
        }
        if (count > 0)
            p_init /= count;
        p.assign(mesh_.numNodes(), p_init);
    }

    applyBoundaryPressure(p);

    DarcyFlowResult result;
    result.converged = false;
    result.iterations = 0;
    result.residual = std::numeric_limits<double>::max();

    // SOR iteration
    for (int iter = 0; iter < max_iter_; ++iter) {
        double max_delta = 0.0;

        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                const int c = mesh_.index(i, j);

                if (has_internal_pressure_ && internal_mask_[c] != 0) {
                    continue;
                }

                const int e = mesh_.index(i + 1, j);
                const int w = mesh_.index(i - 1, j);
                const int n = mesh_.index(i, j + 1);
                const int s = mesh_.index(i, j - 1);

                // Harmonic mean of conductivities
                const double Kc = kappa_[c];
                const double Ke = 2.0 * Kc * kappa_[e] / (Kc + kappa_[e]);
                const double Kw = 2.0 * Kc * kappa_[w] / (Kc + kappa_[w]);
                const double Kn = 2.0 * Kc * kappa_[n] / (Kc + kappa_[n]);
                const double Ks = 2.0 * Kc * kappa_[s] / (Kc + kappa_[s]);

                const double a_center = (Ke + Kw) / dx2 + (Kn + Ks) / dy2;
                const double rhs = (Ke * p[e] + Kw * p[w]) / dx2 + (Kn * p[n] + Ks * p[s]) / dy2;

                const double p_gs = rhs / a_center;
                const double p_old = p[c];
                const double p_new = (1.0 - omega_) * p_old + omega_ * p_gs;

                max_delta = std::max(max_delta, std::abs(p_new - p_old));
                p[c] = p_new;
            }
        }

        // Handle Neumann boundaries
        if (boundaries_[to_index(Boundary::Left)].type == BoundaryType::NEUMANN) {
            for (int j = 0; j <= ny; ++j) {
                p[mesh_.index(0, j)] = p[mesh_.index(1, j)];
            }
        }
        if (boundaries_[to_index(Boundary::Right)].type == BoundaryType::NEUMANN) {
            for (int j = 0; j <= ny; ++j) {
                p[mesh_.index(nx, j)] = p[mesh_.index(nx - 1, j)];
            }
        }
        if (boundaries_[to_index(Boundary::Bottom)].type == BoundaryType::NEUMANN) {
            for (int i = 0; i <= nx; ++i) {
                p[mesh_.index(i, 0)] = p[mesh_.index(i, 1)];
            }
        }
        if (boundaries_[to_index(Boundary::Top)].type == BoundaryType::NEUMANN) {
            for (int i = 0; i <= nx; ++i) {
                p[mesh_.index(i, ny)] = p[mesh_.index(i, ny - 1)];
            }
        }

        applyBoundaryPressure(p);

        result.residual = max_delta;
        result.iterations = iter + 1;

        if (max_delta < tolerance_) {
            result.converged = true;
            break;
        }
    }

    result.pressure = std::move(p);
    computeVelocity(result.pressure, result.vx, result.vy);

    return result;
}

}  // namespace biotransport
