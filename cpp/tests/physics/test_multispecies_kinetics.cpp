#include "../test_support/science_test.hpp"
#include <biotransport/solvers/multi_species_solver.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace biotransport;

namespace {

void testLotkaVolterraRates() {
    LotkaVolterraReaction model(1.0, 0.1, 0.1, 0.02, 100.0);
    const std::vector<double> state = {40.0, 9.0};
    std::vector<double> rates(2, 0.0);

    model(rates, state, 0.0, 0.0, 0.0);

    SCIENCE_REQUIRE_NEAR(rates[0], -12.0, 2.0e-15, 2.0e-15, "logistic prey plus predation rate");
    SCIENCE_REQUIRE_NEAR(rates[1], 6.3, 2.0e-15, 2.0e-15,
                         "predator reproduction minus mortality rate");
}

void testSirAndSeirPopulationInvariants() {
    SIRReaction sir(0.3, 0.1, 1000.0);
    std::vector<double> sir_rates(3, 0.0);
    sir(sir_rates, {900.0, 50.0, 50.0}, 0.0, 0.0, 0.0);
    SCIENCE_REQUIRE_NEAR(sir_rates[0], -13.5, 1.0e-15, 1.0e-15, "SIR incidence rate");
    SCIENCE_REQUIRE_NEAR(sir_rates[0] + sir_rates[1] + sir_rates[2], 0.0, 2.0e-15, 0.0,
                         "SIR local population derivative");

    SEIRReaction seir(0.3, 0.2, 0.1, 1000.0);
    std::vector<double> seir_rates(4, 0.0);
    seir(seir_rates, {900.0, 30.0, 20.0, 50.0}, 0.0, 0.0, 0.0);
    SCIENCE_REQUIRE_NEAR(seir_rates[0], -5.4, 1.0e-15, 1.0e-15, "SEIR incidence rate");
    SCIENCE_REQUIRE_NEAR(seir_rates[0] + seir_rates[1] + seir_rates[2] + seir_rates[3], 0.0,
                         2.0e-15, 0.0, "SEIR local population derivative");
    SCIENCE_REQUIRE(std::isinf(SIRReaction(0.3, 0.0, 1000.0).R0()),
                    "zero recovery rate must report an infinite basic reproduction number");
}

void testEnzymeCascadeRates() {
    EnzymeCascadeReaction model({2.0, 3.0}, {1.0, 2.0}, {0.1, 0.2, 0.3});
    const std::vector<double> state = {4.0, 5.0, 6.0};
    std::vector<double> rates(3, 0.0);

    model(rates, state, 0.0, 0.0, 0.0);

    SCIENCE_REQUIRE_NEAR(rates[0], -0.4, 1.0e-15, 1.0e-15, "upstream signal degradation");
    SCIENCE_REQUIRE_NEAR(rates[1], 0.6, 1.0e-15, 1.0e-15, "first saturable activation link");
    SCIENCE_REQUIRE_NEAR(rates[2], 12.0 / 35.0, 1.0e-15, 1.0e-15,
                         "second saturable activation link");
}

void testCompetitiveInhibitionStoichiometry() {
    CompetitiveInhibitionReaction model(10.0, 2.0, 4.0, 0.1);
    const std::vector<double> state = {3.0, 4.0, 0.0};
    std::vector<double> rates(3, 0.0);

    model(rates, state, 0.0, 0.0, 0.0);

    SCIENCE_REQUIRE_NEAR(rates[0], -30.0 / 7.0, 1.0e-15, 1.0e-15,
                         "competitively inhibited substrate consumption");
    SCIENCE_REQUIRE_NEAR(rates[1], -0.4, 1.0e-15, 1.0e-15, "optional inhibitor decay");
    SCIENCE_REQUIRE_NEAR(rates[0] + rates[2], 0.0, 0.0, 0.0,
                         "one-to-one substrate-to-product stoichiometry");
}

void testBrusselatorSteadyStateAndThreshold() {
    BrusselatorReaction stable(1.0, 1.5);
    std::vector<double> rates(2, 0.0);
    stable(rates, {1.0, 1.5}, 0.0, 0.0, 0.0);
    SCIENCE_REQUIRE_NEAR(rates[0], 0.0, 0.0, 0.0, "Brusselator steady-state X rate");
    SCIENCE_REQUIRE_NEAR(rates[1], 0.0, 0.0, 0.0, "Brusselator steady-state Y rate");
    SCIENCE_REQUIRE(!stable.isOscillatory(), "B below 1+A^2 is below the Hopf threshold");
    SCIENCE_REQUIRE(BrusselatorReaction(1.0, 3.0).isOscillatory(),
                    "B above 1+A^2 is above the Hopf threshold");
}

void testNonfiniteAndUnphysicalParametersAreRejected() {
    bool rejected_nan = false;
    try {
        static_cast<void>(
            LotkaVolterraReaction(std::numeric_limits<double>::quiet_NaN(), 0.1, 0.1, 0.02));
    } catch (const std::invalid_argument&) {
        rejected_nan = true;
    }
    SCIENCE_REQUIRE(rejected_nan, "non-finite kinetic parameters must be rejected");

    bool rejected_negative_decay = false;
    try {
        static_cast<void>(CompetitiveInhibitionReaction(1.0, 1.0, 1.0, -0.1));
    } catch (const std::invalid_argument&) {
        rejected_negative_decay = true;
    }
    SCIENCE_REQUIRE(rejected_negative_decay, "negative inhibitor decay must be rejected");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "multi-species reaction kinetics",
        {
            {"Lotka-Volterra rates match the governing equations", testLotkaVolterraRates},
            {"SIR and SEIR preserve population", testSirAndSeirPopulationInvariants},
            {"enzyme cascade links use documented kinetics", testEnzymeCascadeRates},
            {"competitive inhibition preserves substrate-product stoichiometry",
             testCompetitiveInhibitionStoichiometry},
            {"Brusselator steady state and Hopf threshold are correct",
             testBrusselatorSteadyStateAndThreshold},
            {"invalid kinetic parameters fail loudly",
             testNonfiniteAndUnphysicalParametersAreRejected},
        });
}
