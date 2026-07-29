#include "../test_support/science_test.hpp"
#include <algorithm>
#include <array>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/nernst_planck_solver.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace biotransport;

namespace {

double trapezoidalAmount(const std::vector<double>& concentration, double dx) {
    double sum = 0.5 * (concentration.front() + concentration.back());
    for (std::size_t i = 1; i + 1 < concentration.size(); ++i) {
        sum += concentration[i];
    }
    return dx * sum;
}

double trapezoidalAmount2D(const std::vector<double>& concentration, const StructuredMesh& mesh) {
    double sum = 0.0;
    for (int j = 0; j <= mesh.ny(); ++j) {
        const double wy = (j == 0 || j == mesh.ny()) ? 0.5 : 1.0;
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double wx = (i == 0 || i == mesh.nx()) ? 0.5 : 1.0;
            sum += wx * wy * concentration[mesh.index(i, j)];
        }
    }
    return mesh.dx() * mesh.dy() * sum;
}

double centroid(const std::vector<double>& concentration, const StructuredMesh& mesh) {
    double amount = 0.0;
    double moment = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double weight = (i == 0 || i == mesh.nx()) ? 0.5 : 1.0;
        amount += weight * concentration[i];
        moment += weight * concentration[i] * mesh.x(i);
    }
    return moment / amount;
}

double diffusionEigenmodeError(int cells) {
    constexpr double domain_length_m = 1.0e-3;
    constexpr double diffusivity_m2_s = 1.0e-9;
    constexpr double final_time_s = 20.0;
    constexpr double mean_concentration = 2.0;
    constexpr double amplitude = 0.5;

    StructuredMesh mesh(cells, 0.0, domain_length_m);
    NernstPlanckSolver solver(mesh, IonSpecies("verification ion", 1, diffusivity_m2_s), 300.0);
    std::vector<double> initial(mesh.numNodes());
    for (int i = 0; i <= cells; ++i) {
        initial[static_cast<std::size_t>(i)] =
            mean_concentration + amplitude * std::cos(M_PI * mesh.x(i) / domain_length_m);
    }
    solver.setInitialCondition(initial);

    // Zero potential reduces Nernst-Planck to diffusion.  The cosine mode
    // independently satisfies the solver's default zero-total-flux boundaries.
    // The explicit ceiling scales as h^2.  The extra h/L factor makes
    // dt=O(h^3), so first-order time error is asymptotically smaller than the
    // fitted finite-volume spatial error measured here.
    const double requested_dt_s =
        0.1 * solver.maximumStableTimeStep() * (mesh.dx() / domain_length_m);
    const int steps = static_cast<int>(std::ceil(final_time_s / requested_dt_s));
    const double dt_s = final_time_s / static_cast<double>(steps);
    solver.solve(dt_s, steps);

    const double exact_decay = std::exp(-diffusivity_m2_s * M_PI * M_PI * final_time_s /
                                        (domain_length_m * domain_length_m));
    double weighted_squared_error = 0.0;
    for (int i = 0; i <= cells; ++i) {
        const double exact = mean_concentration +
                             amplitude * exact_decay * std::cos(M_PI * mesh.x(i) / domain_length_m);
        const double difference = solver.solution()[static_cast<std::size_t>(i)] - exact;
        const double weight = (i == 0 || i == cells) ? 0.5 : 1.0;
        weighted_squared_error += weight * difference * difference;
    }
    return std::sqrt(weighted_squared_error / static_cast<double>(cells));
}

void testDiffusionLimitConvergesAgainstNeumannEigenmode() {
    const std::array<int, 3> cell_counts{20, 40, 80};
    std::array<double, 3> errors{};
    for (std::size_t level = 0; level < cell_counts.size(); ++level) {
        errors[level] = diffusionEigenmodeError(cell_counts[level]);
    }

    const double coarse_order = std::log(errors[0] / errors[1]) / std::log(2.0);
    const double fine_order = std::log(errors[1] / errors[2]) / std::log(2.0);
    science_test::report("Nernst-Planck diffusion-limit order (20 to 40)", coarse_order);
    science_test::report("Nernst-Planck diffusion-limit order (40 to 80)", fine_order);
    SCIENCE_REQUIRE(errors[2] < errors[1] && errors[1] < errors[0],
                    "Neumann eigenmode error must decrease under spatial refinement");
    SCIENCE_REQUIRE(coarse_order > 1.8 && fine_order > 1.8,
                    "fitted finite-volume diffusion limit must approach second order in space "
                    "when temporal error is suppressed");
}

void testBoltzmannEquilibriumHasZeroFlux() {
    StructuredMesh mesh(80, 0.0, 1.0e-3);
    IonSpecies sodium("Na+", 1, 1.33e-9);
    NernstPlanckSolver solver(mesh, sodium, 310.0);

    const double thermal_voltage = solver.thermalVoltage();
    std::vector<double> potential(mesh.numNodes());
    std::vector<double> concentration(mesh.numNodes());
    for (int i = 0; i <= mesh.nx(); ++i) {
        potential[i] = 0.04 * mesh.x(i) / 1.0e-3;
        concentration[i] = 100.0 * std::exp(-potential[i] / thermal_voltage);
    }
    solver.setPotentialField(potential);
    solver.setInitialCondition(concentration);

    const auto current = solver.computeCurrentDensity();
    const double current_scale = constants::FARADAY * sodium.diffusivity * 100.0 / mesh.dx();
    double max_scaled_current = 0.0;
    for (std::size_t i = 0; i < concentration.size(); ++i) {
        max_scaled_current = std::max(max_scaled_current, std::abs(current[2 * i]) / current_scale);
    }
    SCIENCE_REQUIRE(max_scaled_current < 2.0e-13,
                    "Scharfetter-Gummel flux must exactly preserve Boltzmann equilibrium");

    solver.solve(1.0e-3, 20);
    double max_relative_change = 0.0;
    for (std::size_t i = 0; i < concentration.size(); ++i) {
        max_relative_change =
            std::max(max_relative_change,
                     std::abs(solver.solution()[i] - concentration[i]) / concentration[i]);
    }
    SCIENCE_REQUIRE(max_relative_change < 2.0e-13,
                    "zero-flux Boltzmann equilibrium must remain stationary");
}

void testNonuniformPotentialConservesAmount() {
    StructuredMesh mesh(64, 0.0, 1.0e-3);
    IonSpecies chloride("Cl-", -1, 2.03e-9);
    NernstPlanckSolver solver(mesh, chloride, 310.0);
    std::vector<double> potential(mesh.numNodes());
    std::vector<double> concentration(mesh.numNodes());
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double xi = mesh.x(i) / 1.0e-3;
        potential[i] = 0.03 * xi * xi;
        concentration[i] = 2.0 + 20.0 * std::exp(-std::pow((xi - 0.4) / 0.12, 2));
    }
    solver.setPotentialField(potential);
    solver.setInitialCondition(concentration);
    const double initial_amount = trapezoidalAmount(concentration, mesh.dx());

    const double dt = 2.0e-4;
    SCIENCE_REQUIRE(solver.checkStability(dt), "chosen conservation-test step must be stable");
    solver.solve(dt, 250);
    const double final_amount = trapezoidalAmount(solver.solution(), mesh.dx());
    SCIENCE_REQUIRE_NEAR(final_amount, initial_amount, 2.0e-15, 2.0e-13,
                         "zero-total-flux ionic amount");
    SCIENCE_REQUIRE(*std::min_element(solver.solution().begin(), solver.solution().end()) >= 0.0,
                    "stable conservative update must preserve non-negative concentration");
}

void testTwoDimensionalSealedDomainConservesAmount() {
    StructuredMesh mesh(24, 18, 0.0, 1.0e-3, 0.0, 8.0e-4);
    NernstPlanckSolver solver(mesh, IonSpecies("X2+", 2, 8.0e-10), 303.0);
    std::vector<double> potential(mesh.numNodes());
    std::vector<double> concentration(mesh.numNodes());
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double xi = mesh.x(i) / 1.0e-3;
            const double eta = mesh.y(i, j) / 8.0e-4;
            const int idx = mesh.index(i, j);
            potential[idx] = 0.012 * xi + 0.009 * eta * eta;
            concentration[idx] = 3.0 + 5.0 * std::exp(-std::pow((xi - 0.35) / 0.18, 2) -
                                                      std::pow((eta - 0.6) / 0.2, 2));
        }
    }
    solver.setPotentialField(potential);
    solver.setInitialCondition(concentration);
    const double initial_amount = trapezoidalAmount2D(concentration, mesh);
    const double dt = solver.recommendedTimeStep(0.3);
    solver.solve(dt, 60);
    SCIENCE_REQUIRE_NEAR(trapezoidalAmount2D(solver.solution(), mesh), initial_amount, 2.0e-18,
                         4.0e-13, "sealed 2D ionic amount");
    SCIENCE_REQUIRE(*std::min_element(solver.solution().begin(), solver.solution().end()) >= 0.0,
                    "2D fitted update must remain non-negative");
}

void testValenceControlsMigrationDirection() {
    StructuredMesh mesh(100, 0.0, 1.0e-3);
    IonSpecies cation("X+", 1, 1.0e-9);
    IonSpecies anion("Y-", -1, 1.0e-9);
    NernstPlanckSolver positive(mesh, cation, 310.0);
    NernstPlanckSolver negative(mesh, anion, 310.0);
    std::vector<double> initial(mesh.numNodes());
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double xi = (mesh.x(i) - 0.5e-3) / 0.08e-3;
        initial[i] = std::exp(-xi * xi);
    }
    positive.setInitialCondition(initial);
    negative.setInitialCondition(initial);
    positive.setUniformField(1000.0);
    negative.setUniformField(1000.0);
    const double before = centroid(initial, mesh);
    const double dt = 1.0e-4;
    SCIENCE_REQUIRE(positive.checkStability(dt) && negative.checkStability(dt),
                    "migration-direction step must be stable");
    positive.solve(dt, 500);
    negative.solve(dt, 500);
    SCIENCE_REQUIRE(centroid(positive.solution(), mesh) > before,
                    "positive ions must drift with a positive electric field");
    SCIENCE_REQUIRE(centroid(negative.solution(), mesh) < before,
                    "negative ions must drift against a positive electric field");
}

void testPrescribedOutwardFluxClosesMassBalance() {
    StructuredMesh mesh(100, 0.0, 1.0e-3);
    NernstPlanckSolver solver(mesh, IonSpecies("X+", 1, 1.0e-9), 300.0);
    std::vector<double> initial(mesh.numNodes(), 10.0);
    solver.setInitialCondition(initial);

    const double outward_flux = 1.0e-4;
    solver.setNeumannBoundary(Boundary::Right, outward_flux);
    const double initial_amount = trapezoidalAmount(initial, mesh.dx());
    const double dt = 1.0e-4;
    const int steps = 100;
    solver.solve(dt, steps);
    const double final_amount = trapezoidalAmount(solver.solution(), mesh.dx());
    const double expected = initial_amount - outward_flux * dt * steps;
    SCIENCE_REQUIRE_NEAR(final_amount, expected, 2.0e-15, 2.0e-13,
                         "integrated outward-flux mass balance");
}

void testMultiIonSpeciesConserveIndependently() {
    StructuredMesh mesh(48, 0.0, 8.0e-4);
    std::vector<IonSpecies> ions{IonSpecies("X+", 1, 1.2e-9), IonSpecies("Y-", -1, 1.8e-9)};
    MultiIonSolver solver(mesh, ions, 305.0);
    std::vector<double> potential(mesh.numNodes());
    std::vector<double> positive(mesh.numNodes());
    std::vector<double> negative(mesh.numNodes());
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double xi = mesh.x(i) / 8.0e-4;
        potential[i] = 0.02 * (xi - 0.3 * xi * xi);
        positive[i] = 4.0 + 3.0 * std::sin(M_PI * xi) * std::sin(M_PI * xi);
        negative[i] = 7.0 + 2.0 * std::exp(-std::pow((xi - 0.65) / 0.15, 2));
    }
    solver.setPotentialField(potential);
    solver.setInitialCondition(0, positive);
    solver.setInitialCondition(1, negative);
    const double amount_positive = trapezoidalAmount(positive, mesh.dx());
    const double amount_negative = trapezoidalAmount(negative, mesh.dx());

    const double dt = solver.recommendedTimeStep(0.35);
    SCIENCE_REQUIRE(solver.checkStability(dt), "recommended multi-ion step must be admissible");
    solver.solve(dt, 80);
    SCIENCE_REQUIRE_NEAR(trapezoidalAmount(solver.concentration(0), mesh.dx()), amount_positive,
                         2.0e-15, 3.0e-13, "sealed cation amount");
    SCIENCE_REQUIRE_NEAR(trapezoidalAmount(solver.concentration(1), mesh.dx()), amount_negative,
                         2.0e-15, 3.0e-13, "sealed anion amount");
    SCIENCE_REQUIRE(
        *std::min_element(solver.concentration(0).begin(), solver.concentration(0).end()) >= 0.0,
        "cation concentration must remain non-negative");
    SCIENCE_REQUIRE(
        *std::min_element(solver.concentration(1).begin(), solver.concentration(1).end()) >= 0.0,
        "anion concentration must remain non-negative");
}

void testNernstAndGhkLimitingCasesAgree() {
    const double potassium_nernst = ghk::nernstPotential(1, 140.0, 5.0, 310.0);
    const double potassium_ghk =
        ghk::ghkVoltage(1.0, 140.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 310.0);
    SCIENCE_REQUIRE_NEAR(potassium_ghk, potassium_nernst, 1.0e-15, 2.0e-15,
                         "potassium-only GHK/Nernst limit");
    SCIENCE_REQUIRE(potassium_nernst < -0.08 && potassium_nernst > -0.10,
                    "physiological potassium ratio should give about -89 mV at 310 K");

    const double chloride_nernst = ghk::nernstPotential(-1, 10.0, 110.0, 310.0);
    const double chloride_ghk =
        ghk::ghkVoltage(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0, 110.0, 310.0);
    SCIENCE_REQUIRE_NEAR(chloride_ghk, chloride_nernst, 1.0e-15, 2.0e-15,
                         "chloride-only GHK/Nernst limit");
}

void testMobilityUsesDeclaredTemperature() {
    IonSpecies sodium("Na+", 1, 1.33e-9, 310.0);
    StructuredMesh mesh(8, 0.0, 1.0e-3);
    NernstPlanckSolver solver(mesh, sodium, 298.0);
    SCIENCE_REQUIRE_NEAR(solver.electricalMobility(), sodium.mobilityAt(298.0), 0.0, 2.0e-15,
                         "solver-temperature mobility");
    SCIENCE_REQUIRE(solver.electricalMobility() > sodium.mobility,
                    "mobility at 298 K must exceed mobility at 310 K for fixed D");

    MultiIonSolver multi(mesh, {sodium}, 298.0);
    SCIENCE_REQUIRE_NEAR(multi.electricalMobility(0), sodium.mobilityAt(298.0), 0.0, 2.0e-15,
                         "multi-ion solver-temperature mobility");
}

void testUnsupportedCouplingAndInvalidInputsFailLoudly() {
    StructuredMesh mesh(8, 0.0, 1.0);
    std::vector<IonSpecies> ions{IonSpecies("Na+", 1, 1.0e-9), IonSpecies("Cl-", -1, 1.0e-9)};
    MultiIonSolver solver(mesh, ions);
    bool coupling_rejected = false;
    try {
        solver.setElectroneutralityMode(true);
    } catch (const std::logic_error&) {
        coupling_rejected = true;
    }
    SCIENCE_REQUIRE(coupling_rejected,
                    "unimplemented electroneutrality coupling must be rejected explicitly");

    bool ignored_background_rejected = false;
    try {
        solver.setElectroneutralityMode(false, 1.0);
    } catch (const std::logic_error&) {
        ignored_background_rejected = true;
    }
    SCIENCE_REQUIRE(ignored_background_rejected,
                    "unsupported background charge must not be silently ignored");

    bool negative_rejected = false;
    try {
        solver.setInitialCondition(0, std::vector<double>(mesh.numNodes(), -1.0));
    } catch (const std::invalid_argument&) {
        negative_rejected = true;
    }
    SCIENCE_REQUIRE(negative_rejected, "negative concentrations must be rejected");

    bool temperature_rejected = false;
    try {
        IonSpecies invalid("bad", 1, 1.0e-9, 0.0);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        temperature_rejected = true;
    }
    SCIENCE_REQUIRE(temperature_rejected, "non-positive absolute temperature must be rejected");

    NernstPlanckSolver single(mesh, ions.front());
    const double maximum_step = single.maximumStableTimeStep();
    SCIENCE_REQUIRE(single.checkStability(maximum_step),
                    "reported single-ion maximum step must satisfy its own bound");
    SCIENCE_REQUIRE(!single.checkStability(1.01 * maximum_step),
                    "a step above the reported maximum must be rejected");
    bool safety_rejected = false;
    try {
        (void)single.recommendedTimeStep(1.1);
    } catch (const std::invalid_argument&) {
        safety_rejected = true;
    }
    SCIENCE_REQUIRE(safety_rejected, "time-step safety factor above one must be rejected");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "Nernst-Planck electrochemistry",
        {{"diffusion limit has second-order spatial convergence",
          testDiffusionLimitConvergesAgainstNeumannEigenmode},
         {"Boltzmann equilibrium has zero fitted flux", testBoltzmannEquilibriumHasZeroFlux},
         {"nonuniform potential conserves ionic amount", testNonuniformPotentialConservesAmount},
         {"two-dimensional sealed domain conserves ionic amount",
          testTwoDimensionalSealedDomainConservesAmount},
         {"valence controls electromigration direction", testValenceControlsMigrationDirection},
         {"prescribed outward flux closes mass balance",
          testPrescribedOutwardFluxClosesMassBalance},
         {"multi-ion species conserve independently", testMultiIonSpeciesConserveIndependently},
         {"Nernst and GHK limiting cases agree", testNernstAndGhkLimitingCasesAgree},
         {"mobility uses declared temperature", testMobilityUsesDeclaredTemperature},
         {"unsupported coupling and invalid inputs fail loudly",
          testUnsupportedCouplingAndInvalidInputsFailLoudly}});
}
