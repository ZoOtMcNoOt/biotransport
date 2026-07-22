#include <algorithm>
#include <biotransport/solvers/nonuniform_diffusion_1d.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace biotransport {
namespace {

double compensatedIntegral(const std::vector<double>& values, const std::vector<double>& widths) {
    double sum = 0.0;
    double correction = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double term = values[i] * widths[i];
        const double adjusted = term - correction;
        const double next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    if (!std::isfinite(sum)) {
        throw std::overflow_error("Integrated concentration is not finite");
    }
    return sum;
}

}  // namespace

NonuniformDiffusion1D::NonuniformDiffusion1D(NonuniformMesh1D mesh, double diffusivity)
    : mesh_(std::move(mesh)), diffusivity_(mesh_.numNodes(), diffusivity) {
    initializeMaterialData();
}

NonuniformDiffusion1D::NonuniformDiffusion1D(NonuniformMesh1D mesh,
                                             std::vector<double> nodal_diffusivity)
    : mesh_(std::move(mesh)), diffusivity_(std::move(nodal_diffusivity)) {
    initializeMaterialData();
}

void NonuniformDiffusion1D::initializeMaterialData() {
    if (diffusivity_.size() != mesh_.numNodes()) {
        throw std::invalid_argument(
            "Nodal diffusivity length must equal the NonuniformMesh1D node count");
    }
    for (double value : diffusivity_) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("Nodal diffusivity values must be finite and non-negative");
        }
    }

    face_diffusivity_.resize(mesh_.numCells());
    conductance_.resize(mesh_.numCells());
    for (std::size_t face = 0; face < mesh_.numCells(); ++face) {
        face_diffusivity_[face] = harmonicMean(diffusivity_[face], diffusivity_[face + 1]);
        conductance_[face] = face_diffusivity_[face] / mesh_.spacing(face);
        if (!std::isfinite(conductance_[face])) {
            throw std::invalid_argument(
                "Diffusivity divided by local mesh spacing must remain finite");
        }
    }

    concentration_.assign(mesh_.numNodes(), 0.0);
    scratch_.assign(mesh_.numNodes(), 0.0);
    resetBalanceReference();
}

double NonuniformDiffusion1D::harmonicMean(double left, double right) {
    if (left == 0.0 || right == 0.0) {
        return 0.0;
    }
    if (left == right) {
        return left;
    }
    const double low = std::min(left, right);
    const double high = std::max(left, right);
    const double result = low * (2.0 / (1.0 + low / high));
    if (!std::isfinite(result) || result < 0.0) {
        throw std::invalid_argument("Harmonic face diffusivity must be finite");
    }
    return result;
}

void NonuniformDiffusion1D::validateBoundary(Boundary boundary) {
    if (boundary != Boundary::Left && boundary != Boundary::Right) {
        throw std::invalid_argument("NonuniformDiffusion1D only has Left and Right boundaries");
    }
}

void NonuniformDiffusion1D::validateConcentration(double value, const char* quantity) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(quantity) + " must be finite and non-negative");
    }
}

NonuniformDiffusion1D& NonuniformDiffusion1D::setInitialCondition(
    std::vector<double> concentration) {
    if (concentration.size() != mesh_.numNodes()) {
        throw std::invalid_argument(
            "Initial concentration length must equal the NonuniformMesh1D node count");
    }
    for (double value : concentration) {
        validateConcentration(value, "Initial concentration");
    }

    concentration_ = std::move(concentration);
    applyDirichletBoundaries();
    time_ = 0.0;
    steps_ = 0;
    resetBalanceReference();
    return *this;
}

NonuniformDiffusion1D& NonuniformDiffusion1D::setUniformInitialCondition(double concentration) {
    validateConcentration(concentration, "Initial concentration");
    concentration_.assign(mesh_.numNodes(), concentration);
    applyDirichletBoundaries();
    time_ = 0.0;
    steps_ = 0;
    resetBalanceReference();
    return *this;
}

NonuniformDiffusion1D& NonuniformDiffusion1D::setBoundaryCondition(
    Boundary boundary, const BoundaryCondition& condition) {
    validateBoundary(boundary);
    if (condition.type == BoundaryType::ROBIN) {
        throw std::invalid_argument(
            "NonuniformDiffusion1D does not yet implement Robin boundary conditions");
    }
    if (condition.type != BoundaryType::DIRICHLET && condition.type != BoundaryType::NEUMANN) {
        throw std::invalid_argument("Unsupported boundary-condition type");
    }
    if (!std::isfinite(condition.value)) {
        throw std::invalid_argument("Boundary-condition value must be finite");
    }
    if (condition.type == BoundaryType::DIRICHLET) {
        validateConcentration(condition.value, "Dirichlet concentration");
    }

    if (boundary == Boundary::Left) {
        left_boundary_ = condition;
    } else {
        right_boundary_ = condition;
    }
    applyDirichletBoundaries();
    resetBalanceReference();
    return *this;
}

NonuniformDiffusion1D& NonuniformDiffusion1D::setDirichletBoundary(Boundary boundary,
                                                                   double concentration) {
    return setBoundaryCondition(boundary, BoundaryCondition::Dirichlet(concentration));
}

NonuniformDiffusion1D& NonuniformDiffusion1D::setNeumannBoundary(Boundary boundary,
                                                                 double outward_normal_derivative) {
    return setBoundaryCondition(boundary, BoundaryCondition::Neumann(outward_normal_derivative));
}

const BoundaryCondition& NonuniformDiffusion1D::boundaryCondition(Boundary boundary) const {
    validateBoundary(boundary);
    return boundary == Boundary::Left ? left_boundary_ : right_boundary_;
}

void NonuniformDiffusion1D::applyDirichletBoundaries() noexcept {
    if (left_boundary_.type == BoundaryType::DIRICHLET) {
        concentration_.front() = left_boundary_.value;
    }
    if (right_boundary_.type == BoundaryType::DIRICHLET) {
        concentration_.back() = right_boundary_.value;
    }
}

double NonuniformDiffusion1D::maxStableTimeStep() const {
    double global_limit = std::numeric_limits<double>::infinity();
    bool has_active_node = false;

    for (std::size_t node = 0; node < mesh_.numNodes(); ++node) {
        const bool fixed =
            (node == 0 && left_boundary_.type == BoundaryType::DIRICHLET) ||
            (node + 1 == mesh_.numNodes() && right_boundary_.type == BoundaryType::DIRICHLET);
        if (fixed) {
            continue;
        }

        const double left = node > 0 ? conductance_[node - 1] : 0.0;
        const double right = node + 1 < mesh_.numNodes() ? conductance_[node] : 0.0;
        const double scale = std::max(left, right);
        if (scale == 0.0) {
            continue;
        }

        const double normalized_sum = left / scale + right / scale;
        const double local_limit = (mesh_.controlVolume(node) / scale) / normalized_sum;
        global_limit = std::min(global_limit, local_limit);
        has_active_node = true;
    }

    if (!has_active_node) {
        return std::numeric_limits<double>::infinity();
    }
    return global_limit;
}

bool NonuniformDiffusion1D::checkStability(double dt) const {
    return std::isfinite(dt) && dt > 0.0 && dt <= maxStableTimeStep();
}

double NonuniformDiffusion1D::boundaryInputRate(Boundary boundary) const {
    validateBoundary(boundary);

    double rate = 0.0;
    if (boundary == Boundary::Left) {
        if (left_boundary_.type == BoundaryType::NEUMANN) {
            rate = diffusivity_.front() * left_boundary_.value;
        } else {
            rate = conductance_.front() * (concentration_.front() - concentration_[1]);
        }
    } else if (right_boundary_.type == BoundaryType::NEUMANN) {
        rate = diffusivity_.back() * right_boundary_.value;
    } else {
        const std::size_t last = concentration_.size() - 1;
        rate = conductance_.back() * (concentration_[last] - concentration_[last - 1]);
    }

    if (!std::isfinite(rate)) {
        throw std::overflow_error("Boundary diffusive flux is not finite");
    }
    return rate;
}

void NonuniformDiffusion1D::step(double dt) {
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::invalid_argument("Diffusion time step must be finite and positive");
    }
    const double stability_limit = maxStableTimeStep();
    if (dt > stability_limit) {
        throw std::invalid_argument(
            "Diffusion time step exceeds the local nonuniform-grid CFL "
            "limit of " +
            std::to_string(stability_limit));
    }
    const double next_time = time_ + dt;
    if (!std::isfinite(next_time)) {
        throw std::overflow_error("Diffusion time would become non-finite");
    }
    if (next_time == time_) {
        throw std::invalid_argument(
            "Diffusion time step is too small to advance the current floating-point time");
    }
    if (steps_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Accepted diffusion step count would overflow");
    }

    const double left_input = boundaryInputRate(Boundary::Left);
    const double right_input = boundaryInputRate(Boundary::Right);
    const double net_boundary_input = left_input + right_input;
    if (!std::isfinite(net_boundary_input)) {
        throw std::overflow_error("Net boundary diffusive input is not finite");
    }
    const double next_cumulative_boundary_input =
        std::fma(dt, net_boundary_input, cumulative_boundary_input_);
    if (!std::isfinite(next_cumulative_boundary_input)) {
        throw std::overflow_error("Cumulative boundary diffusive input would become non-finite");
    }
    const std::size_t last = concentration_.size() - 1;
    const auto& control_volumes = mesh_.controlVolumes();

    for (std::size_t node = 0; node < concentration_.size(); ++node) {
        if ((node == 0 && left_boundary_.type == BoundaryType::DIRICHLET) ||
            (node == last && right_boundary_.type == BoundaryType::DIRICHLET)) {
            scratch_[node] = concentration_[node];
            continue;
        }

        double candidate = concentration_[node];
        const double dt_over_volume = dt / control_volumes[node];
        if (node > 0) {
            candidate = std::fma(dt_over_volume * conductance_[node - 1],
                                 concentration_[node - 1] - concentration_[node], candidate);
        }
        if (node < last) {
            candidate = std::fma(dt_over_volume * conductance_[node],
                                 concentration_[node + 1] - concentration_[node], candidate);
        }
        if (node == 0 && left_boundary_.type == BoundaryType::NEUMANN) {
            candidate =
                std::fma(dt_over_volume, diffusivity_.front() * left_boundary_.value, candidate);
        }
        if (node == last && right_boundary_.type == BoundaryType::NEUMANN) {
            candidate =
                std::fma(dt_over_volume, diffusivity_.back() * right_boundary_.value, candidate);
        }
        if (!std::isfinite(candidate)) {
            throw std::runtime_error(
                "Nonuniform diffusion produced a non-finite concentration; state unchanged");
        }
        scratch_[node] = candidate;
    }

    const double concentration_scale =
        *std::max_element(concentration_.begin(), concentration_.end());
    const double roundoff_tolerance =
        128.0 * std::numeric_limits<double>::epsilon() * concentration_scale;
    for (double& value : scratch_) {
        if (value < -roundoff_tolerance) {
            throw std::runtime_error(
                "Nonuniform diffusion produced a negative concentration; reduce the time step "
                "or boundary outflow");
        }
        if (value < 0.0) {
            value = 0.0;
        }
    }

    concentration_.swap(scratch_);
    applyDirichletBoundaries();
    cumulative_boundary_input_ = next_cumulative_boundary_input;
    time_ = next_time;
    ++steps_;
}

void NonuniformDiffusion1D::solve(double dt, std::size_t num_steps) {
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::invalid_argument("Diffusion time step must be finite and positive");
    }
    if (num_steps == 0) {
        return;
    }
    if (!checkStability(dt)) {
        throw std::invalid_argument(
            "Diffusion time step exceeds the local nonuniform-grid CFL "
            "limit of " +
            std::to_string(maxStableTimeStep()));
    }
    const double duration = dt * static_cast<double>(num_steps);
    if (!std::isfinite(duration) || !std::isfinite(time_ + duration)) {
        throw std::overflow_error("Requested diffusion solve duration is not finite");
    }
    for (std::size_t iteration = 0; iteration < num_steps; ++iteration) {
        step(dt);
    }
}

void NonuniformDiffusion1D::solveUntil(double final_time, double maximum_dt) {
    if (!std::isfinite(final_time) || final_time < time_) {
        throw std::invalid_argument(
            "final_time must be finite and no earlier than the current solver time");
    }
    if (!std::isfinite(maximum_dt) || maximum_dt <= 0.0) {
        throw std::invalid_argument("maximum_dt must be finite and positive");
    }
    if (final_time == time_) {
        return;
    }

    const double stable_cap = std::min(maximum_dt, maxStableTimeStep());
    if (!std::isfinite(stable_cap) || stable_cap <= 0.0) {
        if (std::isinf(stable_cap)) {
            solve(final_time - time_, 1);
            time_ = final_time;
            return;
        }
        throw std::runtime_error("No positive representable explicit time step exists");
    }

    const double remaining = final_time - time_;
    const double step_count_real = std::ceil(remaining / stable_cap);
    if (!std::isfinite(step_count_real) ||
        step_count_real > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Requested solve requires too many explicit time steps");
    }
    const auto step_count = static_cast<std::size_t>(std::max(1.0, step_count_real));
    const double dt = remaining / static_cast<double>(step_count);
    solve(dt, step_count);
    time_ = final_time;
}

std::vector<double> NonuniformDiffusion1D::faceFluxes() const {
    std::vector<double> flux(mesh_.numCells(), 0.0);
    for (std::size_t face = 0; face < mesh_.numCells(); ++face) {
        flux[face] = conductance_[face] * (concentration_[face] - concentration_[face + 1]);
        if (!std::isfinite(flux[face])) {
            throw std::overflow_error("Interior diffusive flux is not finite");
        }
    }
    return flux;
}

double NonuniformDiffusion1D::totalMass() const {
    return compensatedIntegral(concentration_, mesh_.controlVolumes());
}

double NonuniformDiffusion1D::boundaryOutwardFlux(Boundary boundary) const {
    return -boundaryInputRate(boundary);
}

void NonuniformDiffusion1D::resetBalanceReference() {
    const double mass = totalMass();
    reference_time_ = time_;
    reference_mass_ = mass;
    cumulative_boundary_input_ = 0.0;
}

NonuniformDiffusionDiagnostics NonuniformDiffusion1D::diagnostics() const {
    NonuniformDiffusionDiagnostics result;
    result.steps = steps_;
    result.reference_time = reference_time_;
    result.time = time_;
    result.stability_limit = maxStableTimeStep();
    result.reference_mass = reference_mass_;
    result.total_mass = totalMass();
    result.cumulative_boundary_input = cumulative_boundary_input_;
    result.mass_balance_error =
        result.total_mass - result.reference_mass - result.cumulative_boundary_input;
    if (!std::isfinite(result.mass_balance_error)) {
        throw std::overflow_error("Mass-balance residual is not finite");
    }
    const auto extrema = std::minmax_element(concentration_.begin(), concentration_.end());
    result.minimum_concentration = *extrema.first;
    result.maximum_concentration = *extrema.second;
    result.left_outward_flux = boundaryOutwardFlux(Boundary::Left);
    result.right_outward_flux = boundaryOutwardFlux(Boundary::Right);
    return result;
}

}  // namespace biotransport
