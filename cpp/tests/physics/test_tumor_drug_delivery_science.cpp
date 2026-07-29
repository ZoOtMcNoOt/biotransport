/**
 * @file test_tumor_drug_delivery_science.cpp
 * @brief Limiting-case and conservation tests for tumor drug delivery.
 */

#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/physics/mass_transport/tumor_drug_delivery.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace biotransport;

namespace {

template <typename Exception, typename Function>
void requireThrows(Function&& function, const std::string& message) {
    bool threw_expected = false;
    try {
        function();
    } catch (const Exception&) {
        threw_expected = true;
    }
    SCIENCE_REQUIRE(threw_expected, message);
}

std::vector<std::uint8_t> emptyMask(const StructuredMesh& mesh) {
    return std::vector<std::uint8_t>(static_cast<std::size_t>(mesh.numNodes()), 0U);
}

std::vector<double> constantField(const StructuredMesh& mesh, double value) {
    return std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), value);
}

void constructionRejectsAmbiguousOrNonphysicalInputs() {
    StructuredMesh mesh(4, 4, 0.0, 1.0, 0.0, 1.0);
    auto mask = emptyMask(mesh);
    auto mobility = constantField(mesh, 1.0);

    requireThrows<std::invalid_argument>(
        [&] { TumorDrugDeliverySolver solver(mesh, mask, mobility, 1.0, 0.0); },
        "pressure-driven inflow was accepted without an exterior concentration");

    mask[static_cast<std::size_t>(mesh.index(0, 2))] = 1U;
    requireThrows<std::invalid_argument>(
        [&] { TumorDrugDeliverySolver solver(mesh, mask, mobility, 0.0, 1.0); },
        "a pressure clamp conflicting with the outer boundary was accepted");

    mask = emptyMask(mesh);
    mask[static_cast<std::size_t>(mesh.index(2, 2))] = 2U;
    requireThrows<std::invalid_argument>(
        [&] { TumorDrugDeliverySolver solver(mesh, mask, mobility, 0.0, 1.0); },
        "a non-binary pressure-clamp mask was accepted");

    StructuredMesh nonfinite_mesh(4, 4, 0.0, std::numeric_limits<double>::infinity(), 0.0, 1.0);
    requireThrows<std::invalid_argument>(
        [&] {
            TumorDrugDeliverySolver solver(nonfinite_mesh, emptyMask(nonfinite_mesh),
                                           constantField(nonfinite_mesh, 1.0), 0.0, 1.0);
        },
        "non-finite mesh spacing was accepted");

    mask = emptyMask(mesh);
    const auto center = static_cast<std::size_t>(mesh.index(2, 2));
    mask[center] = 1U;
    TumorDrugDeliverySolver high_contrast_solver(mesh, mask, mobility, 0.0, 1.0e12);
    auto wrong_boundary_pressure = constantField(mesh, 1.0);
    wrong_boundary_pressure[center] = 1.0e12;
    requireThrows<std::invalid_argument>(
        [&] {
            (void)high_contrast_solver.simulate(wrong_boundary_pressure, constantField(mesh, 0.0),
                                                constantField(mesh, 0.0), constantField(mesh, 0.0),
                                                0.0, 0.0, 0.0, 1.0, 0, {});
        },
        "tumor pressure improperly weakened the outer pressure-boundary check");

    StructuredMesh zero_pressure_mesh(2, 2, 0.0, 1.0, 0.0, 1.0);
    TumorDrugDeliverySolver zero_pressure_solver(zero_pressure_mesh, emptyMask(zero_pressure_mesh),
                                                 constantField(zero_pressure_mesh, 1.0), 0.0, 0.0);
    auto near_zero_boundary = constantField(zero_pressure_mesh, 0.0);
    for (int i = 0; i <= zero_pressure_mesh.nx(); ++i) {
        near_zero_boundary[static_cast<std::size_t>(zero_pressure_mesh.index(i, 0))] = -1.0e-14;
        near_zero_boundary[static_cast<std::size_t>(
            zero_pressure_mesh.index(i, zero_pressure_mesh.ny()))] = -1.0e-14;
    }
    for (int j = 0; j <= zero_pressure_mesh.ny(); ++j) {
        near_zero_boundary[static_cast<std::size_t>(zero_pressure_mesh.index(0, j))] = -1.0e-14;
        near_zero_boundary[static_cast<std::size_t>(
            zero_pressure_mesh.index(zero_pressure_mesh.nx(), j))] = -1.0e-14;
    }
    requireThrows<std::invalid_argument>(
        [&] {
            (void)zero_pressure_solver.simulate(
                near_zero_boundary, constantField(zero_pressure_mesh, 0.0),
                constantField(zero_pressure_mesh, 0.0), constantField(zero_pressure_mesh, 0.0), 0.0,
                0.0, 0.0, 1.0, 0, {});
        },
        "a physically different near-zero fixed-pressure boundary was accepted");
}

void pressureSolveRespectsBoundsSymmetryAndConvergence() {
    StructuredMesh mesh(12, 12, 0.0, 1.0, 0.0, 1.0);
    auto mask = emptyMask(mesh);
    mask[static_cast<std::size_t>(mesh.index(6, 6))] = 1U;
    TumorDrugDeliverySolver solver(mesh, mask, constantField(mesh, 2.0), 0.0, 10.0);

    const std::vector<double> pressure = solver.solvePressureSOR(20000, 1.0e-11, 1.6);
    SCIENCE_REQUIRE_NEAR(pressure[static_cast<std::size_t>(mesh.index(6, 6))], 10.0, 0.0, 0.0,
                         "clamped tumor pressure");
    for (int i = 0; i <= mesh.nx(); ++i) {
        SCIENCE_REQUIRE_NEAR(pressure[static_cast<std::size_t>(mesh.index(i, 0))], 0.0, 0.0, 0.0,
                             "south pressure boundary");
        SCIENCE_REQUIRE_NEAR(pressure[static_cast<std::size_t>(mesh.index(i, mesh.ny()))], 0.0, 0.0,
                             0.0, "north pressure boundary");
    }
    for (double value : pressure) {
        SCIENCE_REQUIRE_FINITE(value, "pressure");
        SCIENCE_REQUIRE(value >= -1.0e-10 && value <= 10.0 + 1.0e-10,
                        "pressure violated the elliptic maximum principle");
    }
    SCIENCE_REQUIRE_NEAR(pressure[static_cast<std::size_t>(mesh.index(4, 6))],
                         pressure[static_cast<std::size_t>(mesh.index(8, 6))], 2.0e-9, 0.0,
                         "left-right pressure symmetry");
    SCIENCE_REQUIRE_NEAR(pressure[static_cast<std::size_t>(mesh.index(6, 4))],
                         pressure[static_cast<std::size_t>(mesh.index(6, 8))], 2.0e-9, 0.0,
                         "bottom-top pressure symmetry");
    for (int j = 1; j < mesh.ny(); ++j) {
        for (int i = 1; i < mesh.nx(); ++i) {
            if (i == 6 && j == 6) {
                continue;
            }
            const double stencil_target =
                0.25 * (pressure[static_cast<std::size_t>(mesh.index(i - 1, j))] +
                        pressure[static_cast<std::size_t>(mesh.index(i + 1, j))] +
                        pressure[static_cast<std::size_t>(mesh.index(i, j - 1))] +
                        pressure[static_cast<std::size_t>(mesh.index(i, j + 1))]);
            SCIENCE_REQUIRE_NEAR(pressure[static_cast<std::size_t>(mesh.index(i, j))],
                                 stencil_target, 1.1e-11, 0.0,
                                 "post-sweep discrete pressure defect");
        }
    }

    requireThrows<std::runtime_error>([&] { (void)solver.solvePressureSOR(1, 1.0e-30, 1.0); },
                                      "an unconverged pressure field was returned silently");
    requireThrows<std::runtime_error>(
        [&] { (void)solver.solvePressureSOR(1, 1.0e-10, 1.0e-14); },
        "a tiny relaxation factor fooled the pressure convergence test");
}

TumorDrugDeliverySaved uniformExchange(double surface_area_density) {
    StructuredMesh mesh(2, 2, 0.0, 1.0, 0.0, 1.0);
    const auto mask = emptyMask(mesh);
    const auto mobility = constantField(mesh, 1.0);
    TumorDrugDeliverySolver solver(mesh, mask, mobility, 0.0, 0.0);
    const auto pressure = solver.solvePressureSOR(10, 1.0e-14, 1.0);
    return solver.simulate(pressure, constantField(mesh, 0.0), constantField(mesh, 0.1),
                           constantField(mesh, surface_area_density), 0.2, 0.3, 1.0, 0.01, 20,
                           {0.2, 0.0, 0.01, 0.155, 0.155});
}

void vascularExchangeUsesDimensionalSurfaceAreaAndExactSaveTimes() {
    const auto unit_surface = uniformExchange(1.0);
    const auto double_surface = uniformExchange(2.0);

    SCIENCE_REQUIRE(unit_surface.frames == 4, "duplicate save times were not collapsed");
    SCIENCE_REQUIRE_NEAR(unit_surface.times_s[0], 0.0, 0.0, 0.0, "initial save time");
    SCIENCE_REQUIRE_NEAR(unit_surface.times_s[1], 0.01, 0.0, 0.0, "first-step save time");
    SCIENCE_REQUIRE_NEAR(unit_surface.times_s[2], 0.155, 0.0, 0.0, "off-grid save time");
    SCIENCE_REQUIRE_NEAR(unit_surface.times_s[3], 0.2, 0.0, 0.0, "final save time");
    SCIENCE_REQUIRE(unit_surface.final_time_s == 0.2, "final time was not dt*num_steps");

    const std::size_t first_dynamic_frame =
        static_cast<std::size_t>(unit_surface.nx) * static_cast<std::size_t>(unit_surface.ny);
    const double concentration_one = unit_surface.free[first_dynamic_frame];
    const double concentration_two = double_surface.free[first_dynamic_frame];
    SCIENCE_REQUIRE(concentration_one > 0.0, "vascular exchange did not deliver drug");
    SCIENCE_REQUIRE(concentration_two > concentration_one,
                    "surface area density was normalized away");
    SCIENCE_REQUIRE_NEAR(concentration_two / concentration_one, 2.0, 5.0e-3, 0.0,
                         "small-time S_v source scaling");

    for (double concentration : unit_surface.free) {
        SCIENCE_REQUIRE_FINITE(concentration, "free concentration");
        SCIENCE_REQUIRE(concentration >= 0.0, "free concentration became negative");
    }
    for (double concentration : unit_surface.bound) {
        SCIENCE_REQUIRE(concentration >= 0.0, "bound concentration became negative");
    }
    for (double concentration : unit_surface.cellular) {
        SCIENCE_REQUIRE(concentration >= 0.0, "cellular concentration became negative");
    }
}

void compartmentTransfersAndVascularSourceCloseTheMassBalance() {
    const auto result = uniformExchange(1.0);
    SCIENCE_REQUIRE(result.bound_amount_per_depth.back() > 0.0,
                    "binding did not transfer free drug to bound drug");
    SCIENCE_REQUIRE(result.cellular_amount_per_depth.back() > 0.0,
                    "uptake did not transfer free drug to cellular drug");
    SCIENCE_REQUIRE_NEAR(
        result.bound_amount_per_depth.back() / result.cellular_amount_per_depth.back(), 2.0 / 3.0,
        2.0e-14, 0.0, "first-order binding-to-uptake transfer ratio");
    SCIENCE_REQUIRE_NEAR(result.total_amount_per_depth.back(),
                         result.cumulative_net_vascular_exchange_per_depth.back(), 5.0e-15, 0.0,
                         "closed-domain vascular mass balance");
    SCIENCE_REQUIRE_NEAR(result.mass_balance_error_per_depth.back(), 0.0, 5.0e-15, 0.0,
                         "reported mass-balance error");

    // With uniform fields and zero velocity/diffusion, every node obeys
    // dC_f/dt = a*C_p - (a+k_b+k_u)*C_f.  Check against that independent ODE.
    constexpr double exchange_rate = 0.1;
    constexpr double total_loss_rate = exchange_rate + 0.2 + 0.3;
    constexpr double final_time = 0.2;
    const double exact_free =
        exchange_rate / total_loss_rate * (1.0 - std::exp(-total_loss_rate * final_time));
    const std::size_t final_frame_offset = static_cast<std::size_t>(result.frames - 1) *
                                           static_cast<std::size_t>(result.nx) *
                                           static_cast<std::size_t>(result.ny);
    SCIENCE_REQUIRE_NEAR(result.free[final_frame_offset], exact_free, 7.0e-5, 0.0,
                         "uniform vascular-exchange analytical solution");
}

void pressureDrivenOutflowIsAccountedConservatively() {
    StructuredMesh mesh(10, 10, 0.0, 1.0, 0.0, 1.0);
    auto mask = emptyMask(mesh);
    mask[static_cast<std::size_t>(mesh.index(5, 5))] = 1U;
    TumorDrugDeliverySolver solver(mesh, mask, constantField(mesh, 2.0e-3), 0.0, 1.0);
    const auto pressure = solver.solvePressureSOR(30000, 1.0e-12, 1.5);
    const auto result =
        solver.simulate(pressure, constantField(mesh, 1.0e-3), constantField(mesh, 0.1),
                        constantField(mesh, 1.0), 0.02, 0.03, 1.0, 0.01, 100, {1.0});

    SCIENCE_REQUIRE(result.cumulative_boundary_outflow_per_depth.back() > 0.0,
                    "outward Darcy flow did not remove free drug");
    SCIENCE_REQUIRE_NEAR(result.total_amount_per_depth.back(),
                         result.cumulative_net_vascular_exchange_per_depth.back() -
                             result.cumulative_boundary_outflow_per_depth.back(),
                         5.0e-12, 0.0, "vascular-input minus boundary-outflow mass balance");
    SCIENCE_REQUIRE_NEAR(result.mass_balance_error_per_depth.back(), 0.0, 5.0e-12, 0.0,
                         "pressure-driven mass-balance diagnostic");
}

void unstableAndDimensionallyInvalidTransportFailsLoudly() {
    StructuredMesh mesh(2, 2, 0.0, 1.0, 0.0, 1.0);
    TumorDrugDeliverySolver solver(mesh, emptyMask(mesh), constantField(mesh, 1.0), 0.0, 0.0);
    const auto pressure = solver.solvePressureSOR(10, 1.0e-14, 1.0);

    const auto tiny_horizon =
        solver.simulate(pressure, constantField(mesh, 0.0), constantField(mesh, 0.0),
                        constantField(mesh, 0.0), 0.0, 0.0, 0.0, 1.0e-16, 1, {1.0e-16});
    SCIENCE_REQUIRE(tiny_horizon.frames == 1, "a small positive final time collapsed to zero");
    SCIENCE_REQUIRE_NEAR(tiny_horizon.times_s.front(), 1.0e-16, 0.0, 0.0, "small exact final time");

    requireThrows<std::invalid_argument>(
        [&] {
            (void)solver.simulate(pressure, constantField(mesh, 1.0), constantField(mesh, 0.0),
                                  constantField(mesh, 0.0), 0.0, 0.0, 1.0, 1.0, 1, {1.0});
        },
        "an explicitly unstable diffusion step was accepted");
    requireThrows<std::invalid_argument>(
        [&] {
            auto negative_surface = constantField(mesh, 1.0);
            negative_surface[0] = -1.0;
            (void)solver.simulate(pressure, constantField(mesh, 0.0), constantField(mesh, 0.1),
                                  negative_surface, 0.0, 0.0, 1.0, 0.1, 1, {0.1});
        },
        "negative vascular surface-area density was accepted");
    requireThrows<std::invalid_argument>(
        [&] {
            (void)solver.simulate(pressure, constantField(mesh, 0.0), constantField(mesh, 0.0),
                                  constantField(mesh, 0.0), 0.0, 0.0, 1.0, 0.1, 1, {0.2});
        },
        "a save time beyond the simulation end was accepted");
    requireThrows<std::invalid_argument>(
        [&] {
            (void)solver.simulate(pressure, constantField(mesh, 0.0), constantField(mesh, 0.0),
                                  constantField(mesh, 0.0), 0.0, 0.0, 1.0,
                                  std::numeric_limits<double>::max(), 2, {});
        },
        "an overflowing final time was accepted");
    requireThrows<std::overflow_error>(
        [&] {
            (void)solver.simulate(pressure, constantField(mesh, 0.0), constantField(mesh, 1.0),
                                  constantField(mesh, 1.0), 1.0, 0.0, 1.0e308, 0.4, 20, {8.0});
        },
        "overflowing compartment or mass accumulation was returned silently");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "tumor drug delivery",
        {{"physical input validation", constructionRejectsAmbiguousOrNonphysicalInputs},
         {"pressure bounds and convergence", pressureSolveRespectsBoundsSymmetryAndConvergence},
         {"dimensional vascular exchange and save times",
          vascularExchangeUsesDimensionalSurfaceAreaAndExactSaveTimes},
         {"compartment mass balance", compartmentTransfersAndVascularSourceCloseTheMassBalance},
         {"Darcy outflow mass balance", pressureDrivenOutflowIsAccountedConservatively},
         {"loud transport rejection", unstableAndDimensionallyInvalidTransportFailsLoudly}});
}
