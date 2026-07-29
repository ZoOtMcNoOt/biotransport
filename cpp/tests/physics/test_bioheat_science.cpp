#include "../test_support/science_test.hpp"
#include <algorithm>
#include <array>
#include <biotransport/physics/heat_transfer/bioheat_cryotherapy.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace biotransport;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename Exception, typename Callable>
void requireThrows(Callable&& callable, const std::string& context) {
    bool caught = false;
    try {
        callable();
    } catch (const Exception&) {
        caught = true;
    }
    SCIENCE_REQUIRE(caught, context);
}

struct Inputs {
    StructuredMesh mesh{4, 4, 0.0, 0.04, 0.0, 0.04};
    std::vector<std::uint8_t> probe_mask =
        std::vector<std::uint8_t>(static_cast<std::size_t>(mesh.numNodes()), 0U);
    std::vector<double> perfusion =
        std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    std::vector<double> metabolism =
        std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.0);
};

BioheatCryotherapySolver makeSolver(Inputs inputs, double latent_heat = 0.0,
                                    double arrhenius_A = 0.0, double arrhenius_E = 0.0) {
    return BioheatCryotherapySolver(
        inputs.mesh, std::move(inputs.probe_mask), std::move(inputs.perfusion),
        std::move(inputs.metabolism), 1000.0, 1000.0, 4000.0, 1.0, 1.0, 1000.0, 1000.0, 300.0,
        200.0, 250.0, 2.0, latent_heat, arrhenius_A, arrhenius_E, 8.31446261815324);
}

std::size_t centerIndex() {
    return 2U * 5U + 2U;
}

double linearDiffusionEigenmodeError(int cells, double final_time_s, double requested_dt_s,
                                     bool compare_to_semidiscrete_mode) {
    constexpr double domain_length_m = 0.04;
    constexpr double reference_temperature_K = 300.0;
    constexpr double amplitude_K = 4.0;
    constexpr double density_kg_m3 = 1000.0;
    constexpr double specific_heat_J_kg_K = 1000.0;
    constexpr double conductivity_W_m_K = 1.0;
    constexpr double thermal_diffusivity_m2_s =
        conductivity_W_m_K / (density_kg_m3 * specific_heat_J_kg_K);

    StructuredMesh mesh(cells, cells, 0.0, domain_length_m, 0.0, domain_length_m);
    const auto node_count = static_cast<std::size_t>(mesh.numNodes());
    std::vector<std::uint8_t> probe_mask(node_count, 0U);
    std::vector<double> perfusion(node_count, 0.0);
    std::vector<double> metabolism(node_count, 0.0);
    std::vector<double> initial_temperature(node_count);
    for (int j = 0; j <= cells; ++j) {
        for (int i = 0; i <= cells; ++i) {
            initial_temperature[static_cast<std::size_t>(mesh.index(i, j))] =
                reference_temperature_K + amplitude_K *
                                              std::sin(kPi * mesh.x(i) / domain_length_m) *
                                              std::sin(kPi * mesh.y(i, j) / domain_length_m);
        }
    }

    BioheatCryotherapySolver solver(mesh, std::move(probe_mask), std::move(perfusion),
                                    std::move(metabolism), density_kg_m3, 1000.0, 4000.0,
                                    conductivity_W_m_K, conductivity_W_m_K, specific_heat_J_kg_K,
                                    specific_heat_J_kg_K, reference_temperature_K, 200.0, 250.0,
                                    2.0, 0.0, 0.0, 0.0, 8.31446261815324);
    solver.setInitialTemperatureFieldK(std::move(initial_temperature));

    const int steps = static_cast<int>(std::ceil(final_time_s / requested_dt_s));
    const double dt_s = final_time_s / static_cast<double>(steps);
    SCIENCE_REQUIRE(dt_s <= solver.maximumStableTimeStep(),
                    "eigenmode verification step must satisfy the solver stability bound");
    const BioheatSaved saved = solver.simulate(dt_s, steps, {final_time_s});
    SCIENCE_REQUIRE(saved.frames == 1, "eigenmode run must save its final field");

    double decay_rate_s_inverse =
        -2.0 * thermal_diffusivity_m2_s * kPi * kPi / (domain_length_m * domain_length_m);
    if (compare_to_semidiscrete_mode) {
        const double half_phase = kPi / (2.0 * static_cast<double>(cells));
        decay_rate_s_inverse = -8.0 * thermal_diffusivity_m2_s * std::sin(half_phase) *
                               std::sin(half_phase) / (mesh.dx() * mesh.dx());
    }
    const double exact_amplitude_K = amplitude_K * std::exp(decay_rate_s_inverse * final_time_s);

    double squared_error = 0.0;
    for (int j = 0; j <= cells; ++j) {
        for (int i = 0; i <= cells; ++i) {
            const std::size_t node = static_cast<std::size_t>(mesh.index(i, j));
            const double exact_temperature_K =
                reference_temperature_K + exact_amplitude_K *
                                              std::sin(kPi * mesh.x(i) / domain_length_m) *
                                              std::sin(kPi * mesh.y(i, j) / domain_length_m);
            const double difference = saved.temperature_K[node] - exact_temperature_K;
            const double weight_x = (i == 0 || i == cells) ? 0.5 : 1.0;
            const double weight_y = (j == 0 || j == cells) ? 0.5 : 1.0;
            squared_error += weight_x * weight_y * difference * difference;
        }
    }
    return std::sqrt(squared_error / static_cast<double>(cells * cells));
}

double observedOrder(double coarse_error, double fine_error) {
    return std::log(coarse_error / fine_error) / std::log(2.0);
}

void apparentCapacityHasCorrectUnitsAndLatentIntegral() {
    constexpr double latent_heat = 333000.0;
    auto solver = makeSolver(Inputs{}, latent_heat);

    constexpr double inverse_sqrt_two_pi = 0.39894228040143267794;
    const double expected_peak = 1000.0 + latent_heat * inverse_sqrt_two_pi;
    SCIENCE_REQUIRE_NEAR(solver.effectiveSpecificHeat(250.0), expected_peak, 1.0e-9, 1.0e-14,
                         "mass-specific apparent heat capacity at freezing");

    // Integrate only the latent contribution across +/- 8 sigma. It must recover L [J/kg],
    // not rho*L. Midpoint quadrature is independent of the implementation expression.
    constexpr int intervals = 20000;
    constexpr double lower = 242.0;
    constexpr double upper = 258.0;
    const double spacing = (upper - lower) / static_cast<double>(intervals);
    double integrated_latent_heat = 0.0;
    for (int interval = 0; interval < intervals; ++interval) {
        const double temperature = lower + (static_cast<double>(interval) + 0.5) * spacing;
        integrated_latent_heat += (solver.effectiveSpecificHeat(temperature) - 1000.0) * spacing;
    }
    SCIENCE_REQUIRE_NEAR(integrated_latent_heat, latent_heat, 0.1, 1.0e-7,
                         "integrated latent heat");
}

void uniformPennesEquilibriumIsPreserved() {
    Inputs inputs;
    std::fill(inputs.perfusion.begin(), inputs.perfusion.end(), 0.01);
    auto solver = makeSolver(std::move(inputs));
    const BioheatSaved saved = solver.simulate(0.1, 10, {0.0, 1.0});

    SCIENCE_REQUIRE(saved.frames == 2, "two requested equilibrium frames must be saved");
    for (double temperature : saved.temperature_K) {
        SCIENCE_REQUIRE_NEAR(temperature, 300.0, 0.0, 0.0, "uniform equilibrium temperature");
    }
}

void perfusionSignAndUnitsRestoreTowardArterialTemperature() {
    Inputs inputs;
    std::fill(inputs.perfusion.begin(), inputs.perfusion.end(), 0.01);
    auto solver = makeSolver(std::move(inputs));
    solver.setInitialTemperatureK(290.0).setBoundaryTemperatureK(290.0).setArterialTemperatureK(
        300.0);

    const BioheatSaved saved = solver.simulate(0.1, 1, {0.1});
    // rho_b*c_b*w*(Ta-T)/(rho_t*c_t) = 0.4 K/s, hence +0.04 K in 0.1 s.
    SCIENCE_REQUIRE_NEAR(saved.temperature_K[centerIndex()], 290.04, 1.0e-12, 0.0,
                         "Pennes perfusion restoring increment");
}

void metabolicSourceUsesVolumetricPowerUnits() {
    Inputs inputs;
    std::fill(inputs.metabolism.begin(), inputs.metabolism.end(), 1000.0);
    auto solver = makeSolver(std::move(inputs));
    const BioheatSaved saved = solver.simulate(0.1, 1, {0.1});

    // q/(rho*c) = 1e-3 K/s and the step is 0.1 s.
    SCIENCE_REQUIRE_NEAR(saved.temperature_K[centerIndex()], 300.0001, 1.0e-12, 0.0,
                         "metabolic-source temperature increment");
    SCIENCE_REQUIRE_NEAR(saved.temperature_K.front(), 300.0, 0.0, 0.0,
                         "fixed outer-boundary temperature");
}

void arrheniusDiagnosticHasNoInventedFreezingMultiplier() {
    auto solver = makeSolver(Inputs{}, 0.0, 2.0, 0.0);
    SCIENCE_REQUIRE_NEAR(solver.arrheniusHeatInjuryRate(250.0), 2.0, 0.0, 0.0,
                         "canonical heat-injury rate query");
    const BioheatSaved saved = solver.simulate(0.1, 1, {0.1});

    // Ea=0 gives a constant 2/s rate at every tissue node: Omega=0.2 exactly.
    for (double damage : saved.damage) {
        SCIENCE_REQUIRE_NEAR(damage, 0.2, 1.0e-15, 0.0, "Arrhenius heat-injury integral");
    }
}

void saveTimesAreExactAndDiagnosticsAreConsistent() {
    auto solver = makeSolver(Inputs{});
    const BioheatSaved saved = solver.simulate(0.1, 1, {0.1, 0.035, 0.0, 0.035});

    SCIENCE_REQUIRE(saved.frames == 3, "duplicate save times must be coalesced");
    SCIENCE_REQUIRE_NEAR(saved.times_s[0], 0.0, 0.0, 0.0, "initial save time");
    SCIENCE_REQUIRE_NEAR(saved.times_s[1], 0.035, 0.0, 0.0, "off-grid save time");
    SCIENCE_REQUIRE_NEAR(saved.times_s[2], 0.1, 0.0, 0.0, "final save time");
    SCIENCE_REQUIRE(saved.maximum_stable_dt_s > 0.1,
                    "reported stability bound must exceed accepted dt");
    SCIENCE_REQUIRE(saved.frozen_fraction.size() == saved.temperature_K.size(),
                    "phase diagnostic must align with temperature fields");
    SCIENCE_REQUIRE(saved.minimum_temperature_K.size() == 3,
                    "one minimum-temperature diagnostic is required per frame");
    for (double fraction : saved.frozen_fraction) {
        SCIENCE_REQUIRE(fraction >= 0.0 && fraction <= 1.0, "frozen fraction must remain bounded");
    }
}

void tinyTimeScalePreservesTheFullHorizonAndDistinctSaves() {
    auto solver = makeSolver(Inputs{}, 0.0, 1.0, 0.0);
    const BioheatSaved final_only = solver.simulate(1.0e-16, 1000, {1.0e-13});
    SCIENCE_REQUIRE(final_only.frames == 1, "tiny final save was not retained");
    SCIENCE_REQUIRE_NEAR(final_only.times_s[0], 1.0e-13, 0.0, 0.0, "tiny final timestamp");
    SCIENCE_REQUIRE_NEAR(final_only.damage[centerIndex()], 1.0e-13, 1.0e-26, 1.0e-13,
                         "tiny-horizon Arrhenius integral");

    requireThrows<std::invalid_argument>([&] { (void)solver.simulate(1.0e-16, 10, {-1.0e-15}); },
                                         "a negative save time was accepted on a tiny horizon");
    requireThrows<std::invalid_argument>([&] { (void)solver.simulate(1.0e-16, 10, {2.0e-15}); },
                                         "a save time beyond a tiny horizon was accepted");

    const BioheatSaved distinct = solver.simulate(1.0e-15, 100, {2.0e-14, 3.0e-14});
    SCIENCE_REQUIRE(distinct.frames == 2,
                    "representably distinct femtosecond saves were coalesced");
    const std::size_t nodes =
        static_cast<std::size_t>(distinct.nx) * static_cast<std::size_t>(distinct.ny);
    SCIENCE_REQUIRE_NEAR(distinct.times_s[0], 2.0e-14, 0.0, 0.0, "first femtosecond timestamp");
    SCIENCE_REQUIRE_NEAR(distinct.times_s[1], 3.0e-14, 0.0, 0.0, "second femtosecond timestamp");
    SCIENCE_REQUIRE_NEAR(distinct.damage[centerIndex()], distinct.times_s[0], 1.0e-28, 1.0e-13,
                         "first femtosecond Arrhenius state");
    SCIENCE_REQUIRE_NEAR(distinct.damage[nodes + centerIndex()], distinct.times_s[1], 1.0e-28,
                         1.0e-13, "second femtosecond Arrhenius state");
}

void invalidParametersAndUnstableStepsFailLoudly() {
    Inputs invalid_map;
    invalid_map.perfusion[4] = std::numeric_limits<double>::quiet_NaN();
    requireThrows<std::invalid_argument>([&] { makeSolver(std::move(invalid_map)); },
                                         "non-finite perfusion must be rejected");

    Inputs invalid_mask;
    invalid_mask.probe_mask[4] = 2U;
    requireThrows<std::invalid_argument>([&] { makeSolver(std::move(invalid_mask)); },
                                         "non-binary probe mask must be rejected");

    auto solver = makeSolver(Inputs{});
    requireThrows<std::invalid_argument>([&] { (void)solver.simulate(100.0, 1, {}); },
                                         "unstable explicit step must be rejected");
    const double stable_dt_s = solver.maximumStableTimeStep();
    requireThrows<std::invalid_argument>(
        [&] {
            (void)solver.simulate(
                std::nextafter(stable_dt_s, std::numeric_limits<double>::infinity()), 1, {});
        },
        "the next representable step above the reported stability bound was accepted");
    requireThrows<std::invalid_argument>([&] { (void)solver.simulate(0.1, 1, {0.2}); },
                                         "out-of-interval save time must be rejected");
    requireThrows<std::invalid_argument>([&] { solver.setInitialTemperatureK(-1.0); },
                                         "non-absolute initial temperature must be rejected");
}

void smoothHeatEigenmodeHasSecondOrderSpatialConvergence() {
    constexpr double final_time_s = 20.0;
    constexpr double domain_length_m = 0.04;
    const std::array<int, 3> cell_counts{8, 16, 32};
    std::array<double, 3> errors{};

    for (std::size_t level = 0; level < cell_counts.size(); ++level) {
        const int cells = cell_counts[level];
        const double spacing_m = domain_length_m / static_cast<double>(cells);
        // The two-dimensional explicit stability limit is h^2/(4 alpha).
        // An additional h/L factor makes dt=O(h^3), suppressing Forward-Euler
        // error beneath the O(h^2) conduction error without approaching the
        // positivity boundary.
        const double requested_dt_s =
            0.1 * spacing_m * spacing_m / (4.0e-6) * (spacing_m / domain_length_m);
        errors[level] = linearDiffusionEigenmodeError(cells, final_time_s, requested_dt_s, false);
    }

    const double coarse_order = observedOrder(errors[0], errors[1]);
    const double fine_order = observedOrder(errors[1], errors[2]);
    science_test::report("bioheat spatial order (8 to 16)", coarse_order);
    science_test::report("bioheat spatial order (16 to 32)", fine_order);
    SCIENCE_REQUIRE(errors[2] < errors[1] && errors[1] < errors[0],
                    "continuum eigenmode error must decrease under mesh refinement");
    SCIENCE_REQUIRE(coarse_order > 1.8 && fine_order > 1.8,
                    "centered conduction with parabolic time refinement must approach order two");
}

void explicitEulerTimeIntegrationConvergesAtFirstOrder() {
    constexpr int cells = 16;
    constexpr double final_time_s = 16.0;
    const std::array<double, 3> requested_steps_s{0.8, 0.4, 0.2};
    std::array<double, 3> errors{};

    for (std::size_t level = 0; level < requested_steps_s.size(); ++level) {
        errors[level] =
            linearDiffusionEigenmodeError(cells, final_time_s, requested_steps_s[level], true);
    }

    const double coarse_order = observedOrder(errors[0], errors[1]);
    const double fine_order = observedOrder(errors[1], errors[2]);
    science_test::report("bioheat temporal order (0.8 s to 0.4 s)", coarse_order);
    science_test::report("bioheat temporal order (0.4 s to 0.2 s)", fine_order);
    SCIENCE_REQUIRE(errors[2] < errors[1] && errors[1] < errors[0],
                    "semi-discrete eigenmode error must decrease with the time step");
    SCIENCE_REQUIRE(coarse_order > 0.9 && fine_order > 0.9,
                    "explicit Euler bioheat integration must approach first order");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "bioheat cryotherapy",
        {{"apparent heat capacity units and latent integral",
          apparentCapacityHasCorrectUnitsAndLatentIntegral},
         {"uniform Pennes equilibrium", uniformPennesEquilibriumIsPreserved},
         {"perfusion restoring sign and units",
          perfusionSignAndUnitsRestoreTowardArterialTemperature},
         {"metabolic source units", metabolicSourceUsesVolumetricPowerUnits},
         {"Arrhenius diagnostic semantics", arrheniusDiagnosticHasNoInventedFreezingMultiplier},
         {"exact save times and diagnostics", saveTimesAreExactAndDiagnosticsAreConsistent},
         {"tiny-time exact integration and saves",
          tinyTimeScalePreservesTheFullHorizonAndDistinctSaves},
         {"invalid inputs fail loudly", invalidParametersAndUnstableStepsFailLoudly},
         {"smooth heat eigenmode has second-order spatial convergence",
          smoothHeatEigenmodeHasSecondOrderSpatialConvergence},
         {"explicit Euler time integration is first order",
          explicitEulerTimeIntegrationConvergesAtFirstOrder}});
}
