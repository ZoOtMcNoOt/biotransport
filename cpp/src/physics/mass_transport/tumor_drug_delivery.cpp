#include <algorithm>
#include <biotransport/physics/mass_transport/tumor_drug_delivery.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {
namespace {

double harmonicMean(double first, double second) noexcept {
    if (first == 0.0 || second == 0.0) {
        return 0.0;
    }
    const double smaller = std::min(first, second);
    const double larger = std::max(first, second);
    return 2.0 * smaller / (1.0 + smaller / larger);
}

void requireFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void requireNonNegativeField(const std::vector<double>& values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
        }
    }
}

double nodeWidth(int index, int last_index, double spacing) noexcept {
    return (index == 0 || index == last_index) ? 0.5 * spacing : spacing;
}

}  // namespace

TumorDrugDeliverySolver::TumorDrugDeliverySolver(const StructuredMesh& mesh,
                                                 std::vector<std::uint8_t> tumor_mask,
                                                 std::vector<double> hydraulic_conductivity,
                                                 double p_boundary, double p_tumor)
    : mesh_(mesh),
      tumor_mask_(std::move(tumor_mask)),
      K_(std::move(hydraulic_conductivity)),
      p_boundary_(p_boundary),
      p_tumor_(p_tumor) {
    if (mesh_.is1D() || mesh_.nx() < 2 || mesh_.ny() < 2) {
        throw std::invalid_argument(
            "TumorDrugDeliverySolver requires a 2D mesh with at least 2 cells per direction");
    }
    if (!(mesh_.dx() > 0.0) || !std::isfinite(mesh_.dx()) || !(mesh_.dy() > 0.0) ||
        !std::isfinite(mesh_.dy())) {
        throw std::invalid_argument("tumor transport mesh spacing must be positive and finite");
    }

    nx_ = mesh_.nx();
    ny_ = mesh_.ny();
    stride_ = nx_ + 1;

    const int nodes = mesh_.numNodes();
    if (nodes <= 0) {
        throw std::invalid_argument("tumor transport mesh node count overflowed");
    }
    if (tumor_mask_.size() != static_cast<std::size_t>(nodes)) {
        throw std::invalid_argument("tumor_mask size doesn't match mesh");
    }
    if (K_.size() != static_cast<std::size_t>(nodes)) {
        throw std::invalid_argument("hydraulic_conductivity size doesn't match mesh");
    }

    requireFinite(p_boundary_, "p_boundary");
    requireFinite(p_tumor_, "p_tumor");
    if (p_tumor_ < p_boundary_) {
        throw std::invalid_argument(
            "p_tumor must be >= p_boundary: inward Darcy flow needs an external solute "
            "concentration, which this model does not define");
    }

    for (double value : K_) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::invalid_argument("hydraulic_conductivity must be positive and finite");
        }
    }
    for (std::uint8_t value : tumor_mask_) {
        if (value > 1U) {
            throw std::invalid_argument("tumor_mask must contain only 0 or 1");
        }
    }

    // An outer node cannot simultaneously carry two contradictory pressures.
    for (int i = 0; i <= nx_; ++i) {
        if (tumor_mask_[idx(i, 0, stride_)] != 0U || tumor_mask_[idx(i, ny_, stride_)] != 0U) {
            throw std::invalid_argument("tumor_mask pressure clamps may not touch the boundary");
        }
    }
    for (int j = 0; j <= ny_; ++j) {
        if (tumor_mask_[idx(0, j, stride_)] != 0U || tumor_mask_[idx(nx_, j, stride_)] != 0U) {
            throw std::invalid_argument("tumor_mask pressure clamps may not touch the boundary");
        }
    }
}

std::vector<double> TumorDrugDeliverySolver::solvePressureSOR(int max_iter, double tol,
                                                              double omega) const {
    if (max_iter <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (!(tol > 0.0) || !std::isfinite(tol)) {
        throw std::invalid_argument("tol must be positive and finite");
    }
    if (!(omega > 0.0) || !(omega < 2.0) || !std::isfinite(omega)) {
        throw std::invalid_argument("omega must be finite and in (0,2)");
    }

    std::vector<double> pressure(static_cast<std::size_t>(mesh_.numNodes()), p_boundary_);
    for (std::size_t node = 0; node < tumor_mask_.size(); ++node) {
        if (tumor_mask_[node] != 0U) {
            pressure[node] = p_tumor_;
        }
    }

    const double inverse_dx_squared = 1.0 / (mesh_.dx() * mesh_.dx());
    const double inverse_dy_squared = 1.0 / (mesh_.dy() * mesh_.dy());
    bool converged = false;
    double maximum_defect = std::numeric_limits<double>::infinity();

    for (int iteration = 0; iteration < max_iter; ++iteration) {
        // SOR is an in-place Gauss-Seidel sweep.  Parallel row updates would
        // race on neighbouring pressure values and make the result thread-count dependent.
        for (int j = 1; j < ny_; ++j) {
            for (int i = 1; i < nx_; ++i) {
                const std::size_t center = idx(i, j, stride_);
                if (tumor_mask_[center] != 0U) {
                    continue;
                }

                const std::size_t east = idx(i + 1, j, stride_);
                const std::size_t west = idx(i - 1, j, stride_);
                const std::size_t north = idx(i, j + 1, stride_);
                const std::size_t south = idx(i, j - 1, stride_);

                const double mobility_east = harmonicMean(K_[center], K_[east]);
                const double mobility_west = harmonicMean(K_[center], K_[west]);
                const double mobility_north = harmonicMean(K_[center], K_[north]);
                const double mobility_south = harmonicMean(K_[center], K_[south]);

                const double coefficient_east = mobility_east * inverse_dx_squared;
                const double coefficient_west = mobility_west * inverse_dx_squared;
                const double coefficient_north = mobility_north * inverse_dy_squared;
                const double coefficient_south = mobility_south * inverse_dy_squared;
                const double diagonal =
                    coefficient_east + coefficient_west + coefficient_north + coefficient_south;
                if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
                    throw std::overflow_error(
                        "pressure operator is not representable for the supplied mesh and "
                        "hydraulic conductivity");
                }
                const double gauss_seidel =
                    (coefficient_east * pressure[east] + coefficient_west * pressure[west] +
                     coefficient_north * pressure[north] + coefficient_south * pressure[south]) /
                    diagonal;
                if (!std::isfinite(gauss_seidel)) {
                    throw std::overflow_error(
                        "pressure iteration became non-finite for the supplied parameters");
                }

                const double old_pressure = pressure[center];
                const double new_pressure = (1.0 - omega) * old_pressure + omega * gauss_seidel;
                if (!std::isfinite(new_pressure)) {
                    throw std::overflow_error(
                        "pressure iteration became non-finite for the supplied parameters");
                }
                pressure[center] = new_pressure;
            }
        }

        // Measure a fixed-point defect after the complete sweep.  The relaxed
        // update itself is not a convergence metric: omega -> 0 makes every
        // update arbitrarily small even when the elliptic equation is not met.
        maximum_defect = 0.0;
        for (int j = 1; j < ny_; ++j) {
            for (int i = 1; i < nx_; ++i) {
                const std::size_t center = idx(i, j, stride_);
                if (tumor_mask_[center] != 0U) {
                    continue;
                }

                const std::size_t east = idx(i + 1, j, stride_);
                const std::size_t west = idx(i - 1, j, stride_);
                const std::size_t north = idx(i, j + 1, stride_);
                const std::size_t south = idx(i, j - 1, stride_);
                const double coefficient_east =
                    harmonicMean(K_[center], K_[east]) * inverse_dx_squared;
                const double coefficient_west =
                    harmonicMean(K_[center], K_[west]) * inverse_dx_squared;
                const double coefficient_north =
                    harmonicMean(K_[center], K_[north]) * inverse_dy_squared;
                const double coefficient_south =
                    harmonicMean(K_[center], K_[south]) * inverse_dy_squared;
                const double diagonal =
                    coefficient_east + coefficient_west + coefficient_north + coefficient_south;
                const double target_pressure =
                    (coefficient_east * pressure[east] + coefficient_west * pressure[west] +
                     coefficient_north * pressure[north] + coefficient_south * pressure[south]) /
                    diagonal;
                if (!std::isfinite(target_pressure)) {
                    throw std::overflow_error(
                        "pressure defect became non-finite for the supplied parameters");
                }
                maximum_defect =
                    std::max(maximum_defect, std::abs(target_pressure - pressure[center]));
            }
        }

        if (maximum_defect <= tol) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        throw std::runtime_error(
            "pressure SOR did not converge within max_iter; last maximum "
            "discrete pressure defect was " +
            std::to_string(maximum_defect) + " Pa");
    }
    return pressure;
}

TumorDrugDeliverySaved TumorDrugDeliverySolver::simulate(
    const std::vector<double>& pressure, const std::vector<double>& diffusivity,
    const std::vector<double>& vessel_wall_solute_permeability,
    const std::vector<double>& vascular_surface_area_density, double k_binding, double k_uptake,
    double c_plasma, double dt, int num_steps, const std::vector<double>& times_to_save_s) const {
    const int node_count = mesh_.numNodes();
    const auto requireSize = [node_count](const std::vector<double>& values, const char* name) {
        if (values.size() != static_cast<std::size_t>(node_count)) {
            throw std::invalid_argument(std::string(name) + " size doesn't match mesh");
        }
    };
    requireSize(pressure, "pressure");
    requireSize(diffusivity, "diffusivity");
    requireSize(vessel_wall_solute_permeability, "vessel-wall solute permeability");
    requireSize(vascular_surface_area_density, "vascular surface area density");

    for (double value : pressure) {
        requireFinite(value, "pressure");
    }
    requireNonNegativeField(diffusivity, "diffusivity");
    requireNonNegativeField(vessel_wall_solute_permeability, "vessel-wall solute permeability");
    requireNonNegativeField(vascular_surface_area_density, "vascular surface area density");
    if (!(k_binding >= 0.0) || !std::isfinite(k_binding)) {
        throw std::invalid_argument("k_binding must be finite and non-negative");
    }
    if (!(k_uptake >= 0.0) || !std::isfinite(k_uptake)) {
        throw std::invalid_argument("k_uptake must be finite and non-negative");
    }
    if (!(c_plasma >= 0.0) || !std::isfinite(c_plasma)) {
        throw std::invalid_argument("c_plasma must be finite and non-negative");
    }
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        throw std::invalid_argument("dt must be positive and finite");
    }
    if (num_steps < 0) {
        throw std::invalid_argument("num_steps must be non-negative");
    }

    const double final_time = dt * static_cast<double>(num_steps);
    if (!std::isfinite(final_time)) {
        throw std::invalid_argument("dt*num_steps must be finite");
    }

    const auto matchesFixedPressure = [](double value, double target) {
        const double tolerance =
            128.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(target));
        return std::abs(value - target) <= tolerance;
    };
    for (int i = 0; i <= nx_; ++i) {
        if (!matchesFixedPressure(pressure[idx(i, 0, stride_)], p_boundary_) ||
            !matchesFixedPressure(pressure[idx(i, ny_, stride_)], p_boundary_)) {
            throw std::invalid_argument(
                "pressure must equal this solver's p_boundary on every outer node");
        }
    }
    for (int j = 0; j <= ny_; ++j) {
        if (!matchesFixedPressure(pressure[idx(0, j, stride_)], p_boundary_) ||
            !matchesFixedPressure(pressure[idx(nx_, j, stride_)], p_boundary_)) {
            throw std::invalid_argument(
                "pressure must equal this solver's p_boundary on every outer node");
        }
    }
    for (std::size_t node = 0; node < tumor_mask_.size(); ++node) {
        if (tumor_mask_[node] != 0U && !matchesFixedPressure(pressure[node], p_tumor_)) {
            throw std::invalid_argument(
                "pressure must equal this solver's p_tumor on every masked node");
        }
    }

    std::vector<double> save_times = times_to_save_s;
    const double time_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * final_time;
    for (double& save_time : save_times) {
        if (!std::isfinite(save_time) || save_time < -time_tolerance ||
            save_time > final_time + time_tolerance) {
            throw std::invalid_argument("save times must be finite and lie in [0, dt*num_steps]");
        }
        if (std::abs(save_time) <= time_tolerance) {
            save_time = 0.0;
        } else if (std::abs(save_time - final_time) <= time_tolerance) {
            save_time = final_time;
        }
    }
    std::sort(save_times.begin(), save_times.end());
    save_times.erase(std::unique(save_times.begin(), save_times.end()), save_times.end());

    const double dx = mesh_.dx();
    const double dy = mesh_.dy();
    const std::size_t x_face_count =
        static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_ + 1);
    const std::size_t y_face_count =
        static_cast<std::size_t>(nx_ + 1) * static_cast<std::size_t>(ny_);
    std::vector<double> velocity_x_face(x_face_count, 0.0);
    std::vector<double> velocity_y_face(y_face_count, 0.0);

    const auto xFace = [this](int i, int j) {
        return static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) +
               static_cast<std::size_t>(i);
    };
    const auto yFace = [this](int i, int j) {
        return static_cast<std::size_t>(j) * static_cast<std::size_t>(stride_) +
               static_cast<std::size_t>(i);
    };

    double maximum_face_speed = 0.0;
    for (int j = 0; j <= ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const std::size_t left = idx(i, j, stride_);
            const std::size_t right = idx(i + 1, j, stride_);
            const double velocity =
                -harmonicMean(K_[left], K_[right]) * (pressure[right] - pressure[left]) / dx;
            if (!std::isfinite(velocity)) {
                throw std::invalid_argument(
                    "Darcy velocity is not representable for the supplied pressure and "
                    "hydraulic conductivity");
            }
            velocity_x_face[xFace(i, j)] = velocity;
            maximum_face_speed = std::max(maximum_face_speed, std::abs(velocity));
        }
    }
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i <= nx_; ++i) {
            const std::size_t south = idx(i, j, stride_);
            const std::size_t north = idx(i, j + 1, stride_);
            const double velocity =
                -harmonicMean(K_[south], K_[north]) * (pressure[north] - pressure[south]) / dy;
            if (!std::isfinite(velocity)) {
                throw std::invalid_argument(
                    "Darcy velocity is not representable for the supplied pressure and "
                    "hydraulic conductivity");
            }
            velocity_y_face[yFace(i, j)] = velocity;
            maximum_face_speed = std::max(maximum_face_speed, std::abs(velocity));
        }
    }

    // The boundary velocity is extrapolated from its adjacent pressure edge.
    // This supports the elevated-tumor-pressure/outflow use case.  Inflow would
    // require an exterior concentration and is therefore rejected explicitly.
    const double velocity_tolerance =
        128.0 * std::numeric_limits<double>::epsilon() *
        std::max(maximum_face_speed, std::numeric_limits<double>::min());
    for (int j = 0; j <= ny_; ++j) {
        const double west_outward = -velocity_x_face[xFace(0, j)];
        const double east_outward = velocity_x_face[xFace(nx_ - 1, j)];
        if (west_outward < -velocity_tolerance || east_outward < -velocity_tolerance) {
            throw std::invalid_argument(
                "pressure produces Darcy inflow at the boundary; an external inflow "
                "concentration is required but is not part of this model");
        }
    }
    for (int i = 0; i <= nx_; ++i) {
        const double south_outward = -velocity_y_face[yFace(i, 0)];
        const double north_outward = velocity_y_face[yFace(i, ny_ - 1)];
        if (south_outward < -velocity_tolerance || north_outward < -velocity_tolerance) {
            throw std::invalid_argument(
                "pressure produces Darcy inflow at the boundary; an external inflow "
                "concentration is required but is not part of this model");
        }
    }

    std::vector<double> exchange_rate(static_cast<std::size_t>(node_count), 0.0);
    for (int node = 0; node < node_count; ++node) {
        const double rate = vessel_wall_solute_permeability[static_cast<std::size_t>(node)] *
                            vascular_surface_area_density[static_cast<std::size_t>(node)];
        if (!std::isfinite(rate)) {
            throw std::invalid_argument(
                "vessel-wall solute permeability times vascular surface area density "
                "overflowed");
        }
        exchange_rate[static_cast<std::size_t>(node)] = rate;
    }

    // Compute the sufficient monotonicity/positivity bound for explicit Euler.
    // Each row is written as positive neighbour/source coefficients minus a
    // diagonal loss; dt*loss <= 1 keeps every update a non-negative combination.
    double maximum_loss_rate = 0.0;
    for (int j = 0; j <= ny_; ++j) {
        const double width_y = nodeWidth(j, ny_, dy);
        for (int i = 0; i <= nx_; ++i) {
            const std::size_t center = idx(i, j, stride_);
            const double width_x = nodeWidth(i, nx_, dx);
            double loss = k_binding + k_uptake + exchange_rate[center];

            if (i > 0) {
                const std::size_t west = idx(i - 1, j, stride_);
                loss += harmonicMean(diffusivity[west], diffusivity[center]) / (width_x * dx);
                loss += std::max(-velocity_x_face[xFace(i - 1, j)], 0.0) / width_x;
            } else {
                loss += std::max(-velocity_x_face[xFace(0, j)], 0.0) / width_x;
            }
            if (i < nx_) {
                const std::size_t east = idx(i + 1, j, stride_);
                loss += harmonicMean(diffusivity[center], diffusivity[east]) / (width_x * dx);
                loss += std::max(velocity_x_face[xFace(i, j)], 0.0) / width_x;
            } else {
                loss += std::max(velocity_x_face[xFace(nx_ - 1, j)], 0.0) / width_x;
            }
            if (j > 0) {
                const std::size_t south = idx(i, j - 1, stride_);
                loss += harmonicMean(diffusivity[south], diffusivity[center]) / (width_y * dy);
                loss += std::max(-velocity_y_face[yFace(i, j - 1)], 0.0) / width_y;
            } else {
                loss += std::max(-velocity_y_face[yFace(i, 0)], 0.0) / width_y;
            }
            if (j < ny_) {
                const std::size_t north = idx(i, j + 1, stride_);
                loss += harmonicMean(diffusivity[center], diffusivity[north]) / (width_y * dy);
                loss += std::max(velocity_y_face[yFace(i, j)], 0.0) / width_y;
            } else {
                loss += std::max(velocity_y_face[yFace(i, ny_ - 1)], 0.0) / width_y;
            }
            if (!std::isfinite(loss)) {
                throw std::invalid_argument(
                    "the explicit transport loss rate is not representable for the supplied "
                    "mesh and fields");
            }
            maximum_loss_rate = std::max(maximum_loss_rate, loss);
        }
    }

    const double stability_limit =
        maximum_loss_rate > 0.0 ? 1.0 / maximum_loss_rate : std::numeric_limits<double>::infinity();
    if (num_steps > 0 && dt > stability_limit * (1.0 + 1.0e-12)) {
        throw std::invalid_argument("dt exceeds the monotonic explicit stability limit of " +
                                    std::to_string(stability_limit) + " s");
    }

    std::vector<double> free(static_cast<std::size_t>(node_count), 0.0);
    std::vector<double> bound(static_cast<std::size_t>(node_count), 0.0);
    std::vector<double> cellular(static_cast<std::size_t>(node_count), 0.0);
    std::vector<double> free_new(static_cast<std::size_t>(node_count), 0.0);
    std::vector<double> x_flux(x_face_count, 0.0);
    std::vector<double> y_flux(y_face_count, 0.0);
    std::vector<double> vascular_rate_by_node(static_cast<std::size_t>(node_count), 0.0);
    std::vector<double> boundary_outflow_rate_by_node(static_cast<std::size_t>(node_count), 0.0);
    std::vector<std::uint8_t> invalid_update(static_cast<std::size_t>(node_count), 0U);

    TumorDrugDeliverySaved saved;
    saved.nx = nx_ + 1;
    saved.ny = ny_ + 1;
    saved.final_time_s = final_time;
    saved.stability_limit_s = stability_limit;
    const std::size_t nodes_per_frame = static_cast<std::size_t>(node_count);
    double cumulative_vascular_exchange = 0.0;
    double cumulative_boundary_outflow = 0.0;
    const double initial_total_amount = 0.0;

    const auto amounts = [&]() {
        double free_amount = 0.0;
        double bound_amount = 0.0;
        double cellular_amount = 0.0;
        for (int j = 0; j <= ny_; ++j) {
            const double width_y = nodeWidth(j, ny_, dy);
            for (int i = 0; i <= nx_; ++i) {
                const std::size_t node = idx(i, j, stride_);
                const double area = nodeWidth(i, nx_, dx) * width_y;
                free_amount += free[node] * area;
                bound_amount += bound[node] * area;
                cellular_amount += cellular[node] * area;
            }
        }
        if (!std::isfinite(free_amount) || !std::isfinite(bound_amount) ||
            !std::isfinite(cellular_amount)) {
            throw std::overflow_error("tumor drug amount diagnostic overflowed");
        }
        return std::vector<double>{free_amount, bound_amount, cellular_amount};
    };

    const auto saveCurrentState = [&](double time) {
        saved.times_s.push_back(time);
        const std::size_t old_size = saved.free.size();
        saved.free.resize(old_size + nodes_per_frame);
        saved.bound.resize(old_size + nodes_per_frame);
        saved.cellular.resize(old_size + nodes_per_frame);
        saved.total.resize(old_size + nodes_per_frame);
        for (std::size_t node = 0; node < nodes_per_frame; ++node) {
            const double total_concentration = free[node] + bound[node] + cellular[node];
            if (!std::isfinite(total_concentration)) {
                throw std::overflow_error("total tumor drug concentration overflowed");
            }
            saved.free[old_size + node] = free[node];
            saved.bound[old_size + node] = bound[node];
            saved.cellular[old_size + node] = cellular[node];
            saved.total[old_size + node] = total_concentration;
        }

        const std::vector<double> integral = amounts();
        const double total_amount = integral[0] + integral[1] + integral[2];
        const double mass_balance_error = total_amount - initial_total_amount -
                                          cumulative_vascular_exchange +
                                          cumulative_boundary_outflow;
        if (!std::isfinite(total_amount) || !std::isfinite(mass_balance_error)) {
            throw std::overflow_error("tumor drug mass diagnostic overflowed");
        }
        saved.free_amount_per_depth.push_back(integral[0]);
        saved.bound_amount_per_depth.push_back(integral[1]);
        saved.cellular_amount_per_depth.push_back(integral[2]);
        saved.total_amount_per_depth.push_back(total_amount);
        saved.cumulative_net_vascular_exchange_per_depth.push_back(cumulative_vascular_exchange);
        saved.cumulative_boundary_outflow_per_depth.push_back(cumulative_boundary_outflow);
        saved.mass_balance_error_per_depth.push_back(mass_balance_error);
    };

    const auto advance = [&](double step_size) {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j <= ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const std::size_t left = idx(i, j, stride_);
                const std::size_t right = idx(i + 1, j, stride_);
                const double velocity = velocity_x_face[xFace(i, j)];
                const double advected = velocity >= 0.0 ? free[left] : free[right];
                const double diffusive = -harmonicMean(diffusivity[left], diffusivity[right]) *
                                         (free[right] - free[left]) / dx;
                x_flux[xFace(i, j)] = velocity * advected + diffusive;
            }
        }
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i <= nx_; ++i) {
                const std::size_t south = idx(i, j, stride_);
                const std::size_t north = idx(i, j + 1, stride_);
                const double velocity = velocity_y_face[yFace(i, j)];
                const double advected = velocity >= 0.0 ? free[south] : free[north];
                const double diffusive = -harmonicMean(diffusivity[south], diffusivity[north]) *
                                         (free[north] - free[south]) / dy;
                y_flux[yFace(i, j)] = velocity * advected + diffusive;
            }
        }

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j <= ny_; ++j) {
            const double width_y = nodeWidth(j, ny_, dy);
            for (int i = 0; i <= nx_; ++i) {
                const std::size_t center = idx(i, j, stride_);
                const double width_x = nodeWidth(i, nx_, dx);

                const double west_flux =
                    i > 0 ? x_flux[xFace(i - 1, j)]
                          : -std::max(-velocity_x_face[xFace(0, j)], 0.0) * free[center];
                const double east_flux =
                    i < nx_ ? x_flux[xFace(i, j)]
                            : std::max(velocity_x_face[xFace(nx_ - 1, j)], 0.0) * free[center];
                const double south_flux =
                    j > 0 ? y_flux[yFace(i, j - 1)]
                          : -std::max(-velocity_y_face[yFace(i, 0)], 0.0) * free[center];
                const double north_flux =
                    j < ny_ ? y_flux[yFace(i, j)]
                            : std::max(velocity_y_face[yFace(i, ny_ - 1)], 0.0) * free[center];

                const double exchange = exchange_rate[center] * (c_plasma - free[center]);
                const double derivative = (west_flux - east_flux) / width_x +
                                          (south_flux - north_flux) / width_y + exchange -
                                          (k_binding + k_uptake) * free[center];
                double updated = free[center] + step_size * derivative;
                const double negativity_tolerance = 256.0 * std::numeric_limits<double>::epsilon() *
                                                    std::max({1.0, c_plasma, free[center]});
                invalid_update[center] = static_cast<std::uint8_t>(
                    updated < -negativity_tolerance || !std::isfinite(updated));
                if (updated < 0.0) {
                    updated = 0.0;
                }
                free_new[center] = updated;
                const double area = width_x * width_y;
                vascular_rate_by_node[center] = exchange * area;
                double boundary_rate = 0.0;
                if (i == 0) {
                    boundary_rate +=
                        std::max(-velocity_x_face[xFace(0, j)], 0.0) * free[center] * width_y;
                }
                if (i == nx_) {
                    boundary_rate +=
                        std::max(velocity_x_face[xFace(nx_ - 1, j)], 0.0) * free[center] * width_y;
                }
                if (j == 0) {
                    boundary_rate +=
                        std::max(-velocity_y_face[yFace(i, 0)], 0.0) * free[center] * width_x;
                }
                if (j == ny_) {
                    boundary_rate +=
                        std::max(velocity_y_face[yFace(i, ny_ - 1)], 0.0) * free[center] * width_x;
                }
                boundary_outflow_rate_by_node[center] = boundary_rate;
            }
        }

        // Integrate in a fixed order so OpenMP thread count cannot change the
        // mass diagnostics or exception behavior.
        double vascular_rate_integral = 0.0;
        double boundary_outflow_rate_integral = 0.0;
        for (int node = 0; node < node_count; ++node) {
            const std::size_t position = static_cast<std::size_t>(node);
            if (invalid_update[position] != 0U) {
                throw std::runtime_error(
                    "free concentration became negative or non-finite despite the certified "
                    "time-step bound");
            }
            vascular_rate_integral += vascular_rate_by_node[position];
            boundary_outflow_rate_integral += boundary_outflow_rate_by_node[position];
        }
        if (!std::isfinite(vascular_rate_integral) ||
            !std::isfinite(boundary_outflow_rate_integral)) {
            throw std::overflow_error("tumor drug integral rate diagnostic overflowed");
        }

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int node = 0; node < node_count; ++node) {
            const std::size_t position = static_cast<std::size_t>(node);
            bound[position] += step_size * k_binding * free[position];
            cellular[position] += step_size * k_uptake * free[position];
        }
        for (int node = 0; node < node_count; ++node) {
            const std::size_t position = static_cast<std::size_t>(node);
            if (!std::isfinite(bound[position]) || !std::isfinite(cellular[position])) {
                throw std::overflow_error("tumor drug compartment concentration overflowed");
            }
        }
        free.swap(free_new);
        const double next_vascular_exchange =
            cumulative_vascular_exchange + step_size * vascular_rate_integral;
        const double next_boundary_outflow =
            cumulative_boundary_outflow + step_size * boundary_outflow_rate_integral;
        if (!std::isfinite(next_vascular_exchange) || !std::isfinite(next_boundary_outflow)) {
            throw std::overflow_error("cumulative tumor drug mass diagnostic overflowed");
        }
        cumulative_vascular_exchange = next_vascular_exchange;
        cumulative_boundary_outflow = next_boundary_outflow;
    };

    std::size_t next_save = 0;
    double time = 0.0;
    while (next_save < save_times.size() && save_times[next_save] == 0.0) {
        saveCurrentState(0.0);
        ++next_save;
    }

    while (time < final_time - time_tolerance) {
        double next_time = std::min(final_time, time + dt);
        if (next_save < save_times.size() && save_times[next_save] < next_time - time_tolerance) {
            next_time = save_times[next_save];
        }
        const double step_size = next_time - time;
        if (!(step_size > 0.0)) {
            throw std::logic_error("non-positive internal tumor-transport time step");
        }
        advance(step_size);
        time = next_time;

        while (next_save < save_times.size() &&
               std::abs(save_times[next_save] - time) <= time_tolerance) {
            saveCurrentState(save_times[next_save]);
            ++next_save;
        }
    }

    // final_time can be exactly zero, or roundoff can leave a final event unsaved.
    while (next_save < save_times.size() &&
           std::abs(save_times[next_save] - final_time) <= time_tolerance) {
        saveCurrentState(save_times[next_save]);
        ++next_save;
    }
    if (next_save != save_times.size()) {
        throw std::logic_error("failed to land on a requested tumor-transport save time");
    }

    saved.frames = static_cast<int>(saved.times_s.size());
    return saved;
}

}  // namespace biotransport
