#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/physics/heat_transfer/bioheat_cryotherapy.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace biotransport;

namespace {

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
    requireThrows<std::invalid_argument>([&] { (void)solver.simulate(0.1, 1, {0.2}); },
                                         "out-of-interval save time must be rejected");
    requireThrows<std::invalid_argument>([&] { solver.setInitialTemperatureK(-1.0); },
                                         "non-absolute initial temperature must be rejected");
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
         {"invalid inputs fail loudly", invalidParametersAndUnstableStepsFailLoudly}});
}
