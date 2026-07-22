#include <algorithm>
#include <biotransport/physics/fluid_dynamics/darcy_flow.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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

void requireFiniteField(const std::vector<double>& values, const char* name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(std::string(name) + " contains a non-finite value at " +
                                        std::to_string(index));
        }
    }
}

double harmonicMean(double first, double second) {
    const double smaller = std::min(first, second);
    const double larger = std::max(first, second);
    const double value = smaller * (2.0 / (1.0 + smaller / larger));
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::overflow_error("Face hydraulic mobility is non-finite or non-positive");
    }
    return value;
}

void requireFiniteDerived(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " became non-finite");
    }
}

}  // namespace

DarcyFlowSolver::DarcyFlowSolver(const StructuredMesh& mesh, double kappa)
    : mesh_(mesh), kappa_(mesh.numNodes(), kappa) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("DarcyFlowSolver requires a 2D mesh");
    }
    if (mesh_.nx() < 2 || mesh_.ny() < 2) {
        throw std::invalid_argument(
            "DarcyFlowSolver requires at least two cells in each direction");
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
    if (mesh_.nx() < 2 || mesh_.ny() < 2) {
        throw std::invalid_argument(
            "DarcyFlowSolver requires at least two cells in each direction");
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
    requireFinite(pressure, "Dirichlet pressure");
    boundaries_[checkedBoundary(side)] = BoundaryCondition::Dirichlet(pressure);
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setNeumann(Boundary side, double outward_pressure_gradient) {
    requireFinite(outward_pressure_gradient, "Outward pressure gradient");
    boundaries_[checkedBoundary(side)] = BoundaryCondition::Neumann(outward_pressure_gradient);
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setInternalPressure(const std::vector<std::uint8_t>& mask,
                                                      double pressure) {
    if (mask.size() != static_cast<size_t>(mesh_.numNodes())) {
        throw std::invalid_argument("Internal mask size must match mesh nodes");
    }
    requireFinite(pressure, "Internal pressure");
    if (std::none_of(mask.begin(), mask.end(), [](std::uint8_t value) { return value != 0; })) {
        throw std::invalid_argument("Internal pressure mask must select at least one node");
    }
    internal_mask_ = mask;
    internal_pressure_ = pressure;
    has_internal_pressure_ = true;
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setOmega(double omega) {
    if (!std::isfinite(omega) || omega <= 0.0 || omega >= 2.0) {
        throw std::invalid_argument("omega must be finite and in (0, 2)");
    }
    omega_ = omega;
    return *this;
}

DarcyFlowSolver& DarcyFlowSolver::setTolerance(double tol) {
    if (!std::isfinite(tol) || tol <= 0.0) {
        throw std::invalid_argument("tolerance must be finite and positive");
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
    requireFiniteField(pressure, "Initial pressure guess");
    initial_guess_ = pressure;
    return *this;
}

void DarcyFlowSolver::applyBoundaryPressure(std::vector<double>& p) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const double dx = mesh_.dx();
    const double dy = mesh_.dy();

    // Neumann values are outward-normal pressure derivatives. Apply them
    // before Dirichlet values so a prescribed pressure wins at mixed corners.
    if (boundaries_[to_index(Boundary::Left)].type == BoundaryType::NEUMANN) {
        const double outward_gradient = boundaries_[to_index(Boundary::Left)].value;
        for (int j = 0; j <= ny; ++j) {
            p[mesh_.index(0, j)] = p[mesh_.index(1, j)] + outward_gradient * dx;
        }
    }
    if (boundaries_[to_index(Boundary::Right)].type == BoundaryType::NEUMANN) {
        const double outward_gradient = boundaries_[to_index(Boundary::Right)].value;
        for (int j = 0; j <= ny; ++j) {
            p[mesh_.index(nx, j)] = p[mesh_.index(nx - 1, j)] + outward_gradient * dx;
        }
    }
    if (boundaries_[to_index(Boundary::Bottom)].type == BoundaryType::NEUMANN) {
        const double outward_gradient = boundaries_[to_index(Boundary::Bottom)].value;
        for (int i = 0; i <= nx; ++i) {
            p[mesh_.index(i, 0)] = p[mesh_.index(i, 1)] + outward_gradient * dy;
        }
    }
    if (boundaries_[to_index(Boundary::Top)].type == BoundaryType::NEUMANN) {
        const double outward_gradient = boundaries_[to_index(Boundary::Top)].value;
        for (int i = 0; i <= nx; ++i) {
            p[mesh_.index(i, ny)] = p[mesh_.index(i, ny - 1)] + outward_gradient * dy;
        }
    }

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
    requireFiniteField(p, "Boundary-updated pressure field");
}

bool DarcyFlowSolver::hasPressureGauge() const {
    const bool boundary_gauge =
        std::any_of(boundaries_.begin(), boundaries_.end(),
                    [](const auto& bc) { return bc.type == BoundaryType::DIRICHLET; });
    const bool internal_gauge =
        has_internal_pressure_ && std::any_of(internal_mask_.begin(), internal_mask_.end(),
                                              [](std::uint8_t value) { return value != 0; });
    return boundary_gauge || internal_gauge;
}

double DarcyFlowSolver::computePressureResidual(const std::vector<double>& pressure) const {
    const int nx = mesh_.nx();
    const int ny = mesh_.ny();
    const double dx2 = mesh_.dx() * mesh_.dx();
    const double dy2 = mesh_.dy() * mesh_.dy();
    double maximum = 0.0;

    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            const int center = mesh_.index(i, j);
            if (has_internal_pressure_ && internal_mask_[center] != 0) {
                continue;
            }
            const int east = mesh_.index(i + 1, j);
            const int west = mesh_.index(i - 1, j);
            const int north = mesh_.index(i, j + 1);
            const int south = mesh_.index(i, j - 1);
            const double east_mobility = harmonicMean(kappa_[center], kappa_[east]);
            const double west_mobility = harmonicMean(kappa_[center], kappa_[west]);
            const double north_mobility = harmonicMean(kappa_[center], kappa_[north]);
            const double south_mobility = harmonicMean(kappa_[center], kappa_[south]);
            const double diagonal =
                (east_mobility + west_mobility) / dx2 + (north_mobility + south_mobility) / dy2;
            const double rhs =
                (east_mobility * pressure[east] + west_mobility * pressure[west]) / dx2 +
                (north_mobility * pressure[north] + south_mobility * pressure[south]) / dy2;
            requireFiniteDerived(diagonal, "Darcy diagonal coefficient");
            requireFiniteDerived(rhs, "Darcy pressure right-hand side");
            if (diagonal <= 0.0) {
                throw std::runtime_error("Darcy diagonal coefficient became non-positive");
            }
            const double defect = rhs / diagonal - pressure[center];
            requireFiniteDerived(defect, "Darcy pressure defect");
            maximum = std::max(maximum, std::abs(defect));
        }
    }
    return maximum;
}

void DarcyFlowSolver::computeVelocity(const std::vector<double>& pressure, std::vector<double>& vx,
                                      std::vector<double>& vy) const {
    requireFiniteField(pressure, "Pressure field");
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
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
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

    requireFiniteField(vx, "Darcy x-velocity field");
    requireFiniteField(vy, "Darcy y-velocity field");
}

DarcyFlowResult DarcyFlowSolver::solve() const {
    if (!hasPressureGauge()) {
        throw std::invalid_argument(
            "Darcy pressure is unanchored: provide a Dirichlet boundary or an active internal "
            "pressure mask; all-Neumann systems are singular");
    }
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

        // SOR is an in-place Gauss-Seidel sweep: each node deliberately uses
        // values updated earlier in this sweep. Parallel row updates race on
        // neighboring pressure values and change the iteration non-deterministically.
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
                const double Ke = harmonicMean(Kc, kappa_[e]);
                const double Kw = harmonicMean(Kc, kappa_[w]);
                const double Kn = harmonicMean(Kc, kappa_[n]);
                const double Ks = harmonicMean(Kc, kappa_[s]);

                const double a_center = (Ke + Kw) / dx2 + (Kn + Ks) / dy2;
                const double rhs = (Ke * p[e] + Kw * p[w]) / dx2 + (Kn * p[n] + Ks * p[s]) / dy2;
                requireFiniteDerived(a_center, "Darcy diagonal coefficient");
                requireFiniteDerived(rhs, "Darcy pressure right-hand side");
                if (a_center <= 0.0) {
                    throw std::runtime_error("Darcy diagonal coefficient became non-positive");
                }

                const double p_gs = rhs / a_center;
                const double p_old = p[c];
                const double p_new = (1.0 - omega_) * p_old + omega_ * p_gs;
                requireFiniteDerived(p_new, "Darcy pressure iterate");

                max_delta = std::max(max_delta, std::abs(p_new - p_old));
                p[c] = p_new;
            }
        }

        applyBoundaryPressure(p);

        requireFiniteDerived(max_delta, "Darcy iteration update");
        result.residual = computePressureResidual(p);
        result.iterations = iter + 1;

        if (result.residual <= tolerance_) {
            result.converged = true;
            break;
        }
    }

    if (!result.converged) {
        throw std::runtime_error(
            "Darcy pressure solve did not converge in " + std::to_string(max_iter_) +
            " iterations; final pressure defect=" + std::to_string(result.residual));
    }

    result.pressure = std::move(p);
    computeVelocity(result.pressure, result.vx, result.vy);

    return result;
}

}  // namespace biotransport
