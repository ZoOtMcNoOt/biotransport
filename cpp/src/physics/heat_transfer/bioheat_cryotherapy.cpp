#include <algorithm>
#include <biotransport/physics/heat_transfer/bioheat_cryotherapy.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace biotransport {
namespace {

constexpr double kSqrtTwo = 1.41421356237309504880168872420969808;
constexpr double kInvSqrtTwoPi = 0.39894228040143267793994605993438187;

void requireFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void requirePositive(double value, const char* name) {
    requireFinite(value, name);
    if (!(value > 0.0)) {
        throw std::invalid_argument(std::string(name) + " must be greater than zero");
    }
}

void requireNonnegative(double value, const char* name) {
    requireFinite(value, name);
    if (value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be nonnegative");
    }
}

void requireAbsoluteTemperature(double value, const char* name) {
    requirePositive(value, name);
}

double harmonicMean(double first, double second) noexcept {
    const double smaller = std::min(first, second);
    const double larger = std::max(first, second);
    return (2.0 * smaller) / (1.0 + smaller / larger);
}

}  // namespace

BioheatCryotherapySolver::BioheatCryotherapySolver(
    const StructuredMesh& mesh, std::vector<std::uint8_t> probe_mask,
    std::vector<double> perfusion_map, std::vector<double> q_met_map, double rho_tissue,
    double rho_blood, double c_blood, double k_unfrozen, double k_frozen, double c_unfrozen,
    double c_frozen, double T_body_K, double T_probe_K, double T_freeze_K, double T_freeze_range_K,
    double L_fusion, double A, double E_a, double R_gas)
    : mesh_(mesh),
      nx_(mesh_.nx()),
      ny_(mesh_.ny()),
      stride_(nx_ + 1),
      probe_mask_(std::move(probe_mask)),
      perfusion_map_(std::move(perfusion_map)),
      q_met_map_(std::move(q_met_map)),
      initial_temperature_K_(static_cast<std::size_t>(mesh_.numNodes()), T_body_K),
      rho_tissue_(rho_tissue),
      rho_blood_(rho_blood),
      c_blood_(c_blood),
      k_unfrozen_(k_unfrozen),
      k_frozen_(k_frozen),
      c_unfrozen_(c_unfrozen),
      c_frozen_(c_frozen),
      T_arterial_(T_body_K),
      T_boundary_(T_body_K),
      T_probe_(T_probe_K),
      T_freeze_(T_freeze_K),
      T_freeze_range_(T_freeze_range_K),
      L_fusion_(L_fusion),
      A_(A),
      E_a_(E_a),
      R_gas_(R_gas) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("BioheatCryotherapySolver requires a two-dimensional mesh");
    }
    if (nx_ < 2 || ny_ < 2) {
        throw std::invalid_argument(
            "BioheatCryotherapySolver requires at least two cells in each direction");
    }

    const auto node_count = static_cast<std::size_t>(mesh_.numNodes());
    if (probe_mask_.size() != node_count) {
        throw std::invalid_argument("probe_mask size must equal mesh.numNodes()");
    }
    if (perfusion_map_.size() != node_count) {
        throw std::invalid_argument("perfusion_map size must equal mesh.numNodes()");
    }
    if (q_met_map_.size() != node_count) {
        throw std::invalid_argument("q_met_map size must equal mesh.numNodes()");
    }

    requirePositive(rho_tissue_, "rho_tissue");
    requirePositive(rho_blood_, "rho_blood");
    requirePositive(c_blood_, "c_blood");
    requirePositive(k_unfrozen_, "k_unfrozen");
    requirePositive(k_frozen_, "k_frozen");
    requirePositive(c_unfrozen_, "c_unfrozen");
    requirePositive(c_frozen_, "c_frozen");
    requireAbsoluteTemperature(T_body_K, "T_body_K");
    requireAbsoluteTemperature(T_probe_, "T_probe_K");
    requireAbsoluteTemperature(T_freeze_, "T_freeze_K");
    requirePositive(T_freeze_range_, "T_freeze_range_K");
    requireNonnegative(L_fusion_, "L_fusion");
    requireNonnegative(A_, "A");
    requireNonnegative(E_a_, "E_a");
    requirePositive(R_gas_, "R_gas");

    if (!(T_probe_ < T_freeze_)) {
        throw std::invalid_argument(
            "T_probe_K must be below T_freeze_K for a cryotherapy simulation");
    }
    if (!(T_freeze_ < T_body_K)) {
        throw std::invalid_argument(
            "T_body_K must be above T_freeze_K for the default unfrozen initial state");
    }

    for (std::size_t node = 0; node < node_count; ++node) {
        if (probe_mask_[node] > 1U) {
            throw std::invalid_argument("probe_mask must contain only 0 or 1");
        }
        requireNonnegative(perfusion_map_[node], "perfusion_map entry");
        requireNonnegative(q_met_map_[node], "q_met_map entry");
    }

    if (!std::isfinite(effectiveSpecificHeatUnchecked(T_freeze_))) {
        throw std::invalid_argument(
            "phase-change parameters produce a non-finite apparent heat capacity");
    }
    const double stable_dt_s = maximumStableTimeStep();
    if (!std::isfinite(stable_dt_s) || !(stable_dt_s > 0.0)) {
        throw std::invalid_argument(
            "material properties and mesh must produce a positive finite stability bound");
    }
}

BioheatCryotherapySolver& BioheatCryotherapySolver::setInitialTemperatureK(double temperature_K) {
    requireAbsoluteTemperature(temperature_K, "initial temperature");
    std::fill(initial_temperature_K_.begin(), initial_temperature_K_.end(), temperature_K);
    return *this;
}

BioheatCryotherapySolver& BioheatCryotherapySolver::setInitialTemperatureFieldK(
    std::vector<double> temperature_K) {
    if (temperature_K.size() != initial_temperature_K_.size()) {
        throw std::invalid_argument("initial temperature field size must equal mesh.numNodes()");
    }
    for (double value : temperature_K) {
        requireAbsoluteTemperature(value, "initial temperature field entry");
    }
    initial_temperature_K_ = std::move(temperature_K);
    return *this;
}

BioheatCryotherapySolver& BioheatCryotherapySolver::setArterialTemperatureK(double temperature_K) {
    requireAbsoluteTemperature(temperature_K, "arterial temperature");
    T_arterial_ = temperature_K;
    return *this;
}

BioheatCryotherapySolver& BioheatCryotherapySolver::setBoundaryTemperatureK(double temperature_K) {
    requireAbsoluteTemperature(temperature_K, "boundary temperature");
    T_boundary_ = temperature_K;
    return *this;
}

double BioheatCryotherapySolver::frozenFractionUnchecked(double temperature_K) const noexcept {
    const double sigma_K = 0.5 * T_freeze_range_;
    const double standardized = (temperature_K - T_freeze_) / (kSqrtTwo * sigma_K);
    return 0.5 * std::erfc(standardized);
}

double BioheatCryotherapySolver::thermalConductivityUnchecked(double temperature_K) const noexcept {
    const double frozen = frozenFractionUnchecked(temperature_K);
    return k_unfrozen_ * (1.0 - frozen) + k_frozen_ * frozen;
}

double BioheatCryotherapySolver::effectiveSpecificHeatUnchecked(
    double temperature_K) const noexcept {
    const double frozen = frozenFractionUnchecked(temperature_K);
    const double sensible = c_unfrozen_ * (1.0 - frozen) + c_frozen_ * frozen;
    const double sigma_K = 0.5 * T_freeze_range_;
    const double z = (temperature_K - T_freeze_) / sigma_K;
    const double minus_dfrozen_dT = kInvSqrtTwoPi * std::exp(-0.5 * z * z) / sigma_K;
    return sensible + L_fusion_ * minus_dfrozen_dT;
}

double BioheatCryotherapySolver::arrheniusHeatInjuryRateUnchecked(
    double temperature_K) const noexcept {
    return A_ * std::exp(-E_a_ / (R_gas_ * temperature_K));
}

double BioheatCryotherapySolver::frozenFraction(double temperature_K) const {
    requireAbsoluteTemperature(temperature_K, "temperature_K");
    return frozenFractionUnchecked(temperature_K);
}

double BioheatCryotherapySolver::thermalConductivity(double temperature_K) const {
    requireAbsoluteTemperature(temperature_K, "temperature_K");
    return thermalConductivityUnchecked(temperature_K);
}

double BioheatCryotherapySolver::effectiveSpecificHeat(double temperature_K) const {
    requireAbsoluteTemperature(temperature_K, "temperature_K");
    const double value = effectiveSpecificHeatUnchecked(temperature_K);
    if (!std::isfinite(value) || !(value > 0.0)) {
        throw std::runtime_error("apparent heat capacity is non-finite or non-positive");
    }
    return value;
}

double BioheatCryotherapySolver::arrheniusHeatInjuryRate(double temperature_K) const {
    requireAbsoluteTemperature(temperature_K, "temperature_K");
    const double value = arrheniusHeatInjuryRateUnchecked(temperature_K);
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error("Arrhenius heat-injury rate is non-finite");
    }
    return value;
}

double BioheatCryotherapySolver::maximumStableTimeStep() const {
    const double maximum_conductivity = std::max(k_unfrozen_, k_frozen_);
    const double minimum_specific_heat = std::min(c_unfrozen_, c_frozen_);
    const double maximum_perfusion =
        *std::max_element(perfusion_map_.begin(), perfusion_map_.end());

    const double diffusion_diagonal =
        2.0 * maximum_conductivity *
        (1.0 / (mesh_.dx() * mesh_.dx()) + 1.0 / (mesh_.dy() * mesh_.dy()));
    const double perfusion_diagonal = rho_blood_ * c_blood_ * maximum_perfusion;
    return rho_tissue_ * minimum_specific_heat / (diffusion_diagonal + perfusion_diagonal);
}

BioheatSaved BioheatCryotherapySolver::simulate(double dt, int num_steps,
                                                const std::vector<double>& times_to_save_s) const {
    requirePositive(dt, "dt");
    if (num_steps <= 0) {
        throw std::invalid_argument("num_steps must be greater than zero");
    }

    const double total_time_s = dt * static_cast<double>(num_steps);
    if (!std::isfinite(total_time_s)) {
        throw std::invalid_argument("dt * num_steps must be finite");
    }

    const double stable_dt_s = maximumStableTimeStep();
    if (dt > stable_dt_s) {
        throw std::invalid_argument("dt exceeds the conservative explicit stability limit of " +
                                    std::to_string(stable_dt_s) + " s");
    }

    std::vector<double> save_times;
    save_times.reserve(times_to_save_s.size());
    for (double requested : times_to_save_s) {
        requireFinite(requested, "save time");
        if (requested < 0.0 || requested > total_time_s) {
            throw std::invalid_argument("save times must lie in [0, dt * num_steps]");
        }
        save_times.push_back(requested == 0.0 ? 0.0 : requested);
    }
    std::sort(save_times.begin(), save_times.end());
    save_times.erase(std::unique(save_times.begin(), save_times.end()), save_times.end());

    const auto node_count = static_cast<std::size_t>(mesh_.numNodes());
    std::vector<double> temperature = initial_temperature_K_;
    std::vector<double> next_temperature(node_count, 0.0);
    std::vector<double> damage(node_count, 0.0);

    const auto flatIndex = [this](int i, int j) {
        return static_cast<std::size_t>(j * stride_ + i);
    };

    // Enforce the fixed outer-boundary condition at t=0.
    for (int i = 0; i <= nx_; ++i) {
        temperature[flatIndex(i, 0)] = T_boundary_;
        temperature[flatIndex(i, ny_)] = T_boundary_;
    }
    for (int j = 0; j <= ny_; ++j) {
        temperature[flatIndex(0, j)] = T_boundary_;
        temperature[flatIndex(nx_, j)] = T_boundary_;
    }
    // An embedded cryoprobe is a stronger local Dirichlet constraint.
    for (std::size_t node = 0; node < node_count; ++node) {
        if (probe_mask_[node] != 0U) {
            temperature[node] = T_probe_;
        }
    }

    BioheatSaved result;
    result.nx = nx_ + 1;
    result.ny = ny_ + 1;
    result.maximum_stable_dt_s = stable_dt_s;

    const std::size_t nodes_per_frame = node_count;
    auto save = [&](double time_s) {
        result.times_s.push_back(time_s);
        result.temperature_K.insert(result.temperature_K.end(), temperature.begin(),
                                    temperature.end());
        result.damage.insert(result.damage.end(), damage.begin(), damage.end());
        for (double value : temperature) {
            result.frozen_fraction.push_back(frozenFractionUnchecked(value));
        }
        const auto extrema = std::minmax_element(temperature.begin(), temperature.end());
        result.minimum_temperature_K.push_back(*extrema.first);
        result.maximum_temperature_K.push_back(*extrema.second);
    };

    std::size_t next_save = 0;
    double time_s = 0.0;
    if (next_save < save_times.size() && save_times[next_save] == 0.0) {
        save(0.0);
        ++next_save;
    }

    const double inverse_dx_squared = 1.0 / (mesh_.dx() * mesh_.dx());
    const double inverse_dy_squared = 1.0 / (mesh_.dy() * mesh_.dy());

    auto advance = [&](double step_s) {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 1; j < ny_; ++j) {
            for (int i = 1; i < nx_; ++i) {
                const std::size_t center = flatIndex(i, j);
                if (probe_mask_[center] != 0U) {
                    next_temperature[center] = T_probe_;
                    continue;
                }

                const std::size_t east = flatIndex(i + 1, j);
                const std::size_t west = flatIndex(i - 1, j);
                const std::size_t north = flatIndex(i, j + 1);
                const std::size_t south = flatIndex(i, j - 1);
                const double center_temperature = temperature[center];

                const double center_conductivity = thermalConductivityUnchecked(center_temperature);
                const double east_conductivity = harmonicMean(
                    center_conductivity, thermalConductivityUnchecked(temperature[east]));
                const double west_conductivity = harmonicMean(
                    center_conductivity, thermalConductivityUnchecked(temperature[west]));
                const double north_conductivity = harmonicMean(
                    center_conductivity, thermalConductivityUnchecked(temperature[north]));
                const double south_conductivity = harmonicMean(
                    center_conductivity, thermalConductivityUnchecked(temperature[south]));

                const double conduction =
                    (east_conductivity * (temperature[east] - center_temperature) -
                     west_conductivity * (center_temperature - temperature[west])) *
                        inverse_dx_squared +
                    (north_conductivity * (temperature[north] - center_temperature) -
                     south_conductivity * (center_temperature - temperature[south])) *
                        inverse_dy_squared;

                const double liquid_fraction = 1.0 - frozenFractionUnchecked(center_temperature);
                const double perfusion = rho_blood_ * c_blood_ * perfusion_map_[center] *
                                         liquid_fraction * (T_arterial_ - center_temperature);
                const double metabolism = q_met_map_[center] * liquid_fraction;
                const double volumetric_heat_capacity =
                    rho_tissue_ * effectiveSpecificHeatUnchecked(center_temperature);

                next_temperature[center] =
                    center_temperature +
                    step_s * (conduction + perfusion + metabolism) / volumetric_heat_capacity;
            }
        }

        for (int i = 0; i <= nx_; ++i) {
            next_temperature[flatIndex(i, 0)] = T_boundary_;
            next_temperature[flatIndex(i, ny_)] = T_boundary_;
        }
        for (int j = 0; j <= ny_; ++j) {
            next_temperature[flatIndex(0, j)] = T_boundary_;
            next_temperature[flatIndex(nx_, j)] = T_boundary_;
        }
        for (std::size_t node = 0; node < node_count; ++node) {
            if (probe_mask_[node] != 0U) {
                next_temperature[node] = T_probe_;
            }
        }

        for (std::size_t node = 0; node < node_count; ++node) {
            const double next_value = next_temperature[node];
            if (!std::isfinite(next_value) || !(next_value > 0.0)) {
                throw std::runtime_error("temperature became non-finite or non-positive at node " +
                                         std::to_string(node));
            }
            if (probe_mask_[node] == 0U) {
                const double old_rate = arrheniusHeatInjuryRateUnchecked(temperature[node]);
                const double new_rate = arrheniusHeatInjuryRateUnchecked(next_value);
                damage[node] += 0.5 * step_s * (old_rate + new_rate);
                if (!std::isfinite(damage[node]) || damage[node] < 0.0) {
                    throw std::runtime_error("Arrhenius damage became non-finite at node " +
                                             std::to_string(node));
                }
            }
        }
        temperature.swap(next_temperature);
    };

    while (time_s < total_time_s) {
        while (next_save < save_times.size() && save_times[next_save] == time_s) {
            save(save_times[next_save]);
            ++next_save;
        }

        double target_time_s = total_time_s;
        if (next_save < save_times.size()) {
            target_time_s = std::min(target_time_s, save_times[next_save]);
        }
        const double remaining_s = target_time_s - time_s;
        const double step_s = std::min(dt, remaining_s);
        if (!(step_s > 0.0) || !std::isfinite(step_s)) {
            throw std::runtime_error("time integration failed to make positive finite progress");
        }

        advance(step_s);
        if (step_s == remaining_s) {
            time_s = target_time_s;
        } else {
            const double next_time_s = time_s + step_s;
            if (!std::isfinite(next_time_s) || next_time_s <= time_s) {
                throw std::runtime_error("time integration failed to advance the bioheat clock");
            }
            time_s = next_time_s;
        }
    }

    while (next_save < save_times.size() && save_times[next_save] == time_s) {
        save(save_times[next_save]);
        ++next_save;
    }
    if (next_save != save_times.size()) {
        throw std::runtime_error("failed to reach a requested save time exactly");
    }

    result.frames = static_cast<int>(result.times_s.size());
    const std::size_t expected_values = static_cast<std::size_t>(result.frames) * nodes_per_frame;
    if (result.temperature_K.size() != expected_values || result.damage.size() != expected_values ||
        result.frozen_fraction.size() != expected_values) {
        throw std::logic_error("internal bioheat result packing error");
    }
    return result;
}

}  // namespace biotransport
