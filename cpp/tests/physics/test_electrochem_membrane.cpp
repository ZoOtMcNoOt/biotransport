#include "../test_support/science_test.hpp"
#include <biotransport/physics/mass_transport/membrane_diffusion.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace biotransport;

namespace {

void testSingleLayerResistanceAndApparentCoefficient() {
    const double thickness = 80.0e-6;
    const double diffusivity = 2.0e-10;
    const double partition = 0.25;
    const double left = 12.0;
    const double right = 3.0;
    const auto result = MembraneDiffusion1DSolver()
                            .setMembraneThickness(thickness)
                            .setDiffusivity(diffusivity)
                            .setPartitionCoefficient(partition)
                            .setLeftConcentration(left)
                            .setRightConcentration(right)
                            .solve();
    const double expected_permeability = diffusivity * partition / thickness;
    SCIENCE_REQUIRE_NEAR(result.permeability, expected_permeability, 0.0, 2.0e-15,
                         "single-layer permeability");
    SCIENCE_REQUIRE_NEAR(result.flux, expected_permeability * (left - right), 0.0, 2.0e-15,
                         "single-layer flux");
    SCIENCE_REQUIRE_NEAR(result.effective_diffusivity, diffusivity * partition, 0.0, 2.0e-15,
                         "external-gradient apparent diffusivity");
}

void testLayerInterfacesPreserveReferenceActivity() {
    const double L1 = 40.0e-6;
    const double D1 = 1.0e-10;
    const double K1 = 0.2;
    const double L2 = 70.0e-6;
    const double D2 = 4.0e-10;
    const double K2 = 1.5;
    const double left = 10.0;
    const double right = 1.0;
    const auto result = MultiLayerMembraneSolver()
                            .addLayer(L1, D1, K1)
                            .addLayer(L2, D2, K2)
                            .setLeftConcentration(left)
                            .setRightConcentration(right)
                            .solve();
    SCIENCE_REQUIRE(result.x.size() == 42,
                    "each layer must expose both one-sided interface concentrations");
    SCIENCE_REQUIRE_NEAR(result.x[20], result.x[21], 1.0e-18, 0.0,
                         "duplicate geometric interface coordinate");
    SCIENCE_REQUIRE_NEAR(result.concentration.front() / K1, left, 1.0e-13, 1.0e-14,
                         "left reference concentration");
    SCIENCE_REQUIRE_NEAR(result.concentration[20] / K1, result.concentration[21] / K2, 1.0e-13,
                         1.0e-14, "continuous ideal-dilute activity coordinate");
    SCIENCE_REQUIRE_NEAR(result.concentration.back() / K2, right, 1.0e-13, 1.0e-14,
                         "right reference concentration");
    const double resistance = L1 / (D1 * K1) + L2 / (D2 * K2);
    SCIENCE_REQUIRE_NEAR(result.flux, (left - right) / resistance, 0.0, 2.0e-15,
                         "multilayer series-resistance flux");
    const double layer_one_flux = -D1 * (result.concentration[20] - result.concentration[0]) / L1;
    const double layer_two_flux = -D2 * (result.concentration[41] - result.concentration[21]) / L2;
    SCIENCE_REQUIRE_NEAR(layer_one_flux, result.flux, 1.0e-18, 2.0e-14, "first-layer Fick flux");
    SCIENCE_REQUIRE_NEAR(layer_two_flux, result.flux, 1.0e-18, 2.0e-14, "second-layer Fick flux");
}

void testEqualReferenceConcentrationsAreEquilibrium() {
    const auto result = MultiLayerMembraneSolver()
                            .addLayer(20.0e-6, 2.0e-10, 0.1)
                            .addLayer(80.0e-6, 5.0e-10, 2.0)
                            .setLeftConcentration(3.0)
                            .setRightConcentration(3.0)
                            .solve();
    SCIENCE_REQUIRE_NEAR(result.flux, 0.0, 1.0e-30, 0.0,
                         "equal-reference-concentration equilibrium flux");
    SCIENCE_REQUIRE_NEAR(result.concentration[20] / 0.1, result.concentration[21] / 2.0, 1.0e-14,
                         1.0e-14, "partition jump at equilibrium");
}

void testReverseGradientReversesFlux() {
    const auto result =
        MembraneDiffusion1DSolver().setLeftConcentration(1.0).setRightConcentration(4.0).solve();
    SCIENCE_REQUIRE(result.flux < 0.0,
                    "flux sign must follow the declared positive left-to-right direction");
}

void testInvalidPhysicalParametersAreRejected() {
    bool negative_ratio_rejected = false;
    try {
        (void)renkin_hindrance(-0.1);
    } catch (const std::invalid_argument&) {
        negative_ratio_rejected = true;
    }
    SCIENCE_REQUIRE(negative_ratio_rejected, "negative radius ratios are not physical");

    bool nan_rejected = false;
    try {
        MembraneDiffusion1DSolver().setDiffusivity(std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument&) {
        nan_rejected = true;
    }
    SCIENCE_REQUIRE(nan_rejected, "non-finite membrane parameters must be rejected");

    bool negative_concentration_rejected = false;
    try {
        MultiLayerMembraneSolver().setLeftConcentration(-1.0);
    } catch (const std::invalid_argument&) {
        negative_concentration_rejected = true;
    }
    SCIENCE_REQUIRE(negative_concentration_rejected,
                    "negative dilute-solution concentrations must be rejected");

    bool overflow_rejected = false;
    try {
        (void)MembraneDiffusion1DSolver()
            .setDiffusivity(std::numeric_limits<double>::max())
            .setPartitionCoefficient(std::numeric_limits<double>::max())
            .solve();
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    SCIENCE_REQUIRE(overflow_rejected, "non-finite derived transport values must be rejected");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "steady membrane diffusion",
        {{"single-layer resistance and apparent coefficient",
          testSingleLayerResistanceAndApparentCoefficient},
         {"layer interfaces preserve reference activity",
          testLayerInterfacesPreserveReferenceActivity},
         {"equal reference concentrations are equilibrium",
          testEqualReferenceConcentrationsAreEquilibrium},
         {"reverse gradient reverses flux", testReverseGradientReversesFlux},
         {"invalid physical parameters are rejected", testInvalidPhysicalParametersAreRejected}});
}
