#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/physics/fluid_dynamics/darcy_flow.hpp>
#include <cmath>
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

double harmonicMean(double first, double second) {
    return 2.0 * first * second / (first + second);
}

double layeredPressure(double x, double length, double interface_x, double pressure_left,
                       double pressure_right, double kappa_left, double kappa_right) {
    const double resistance = interface_x / kappa_left + (length - interface_x) / kappa_right;
    const double darcy_flux = (pressure_left - pressure_right) / resistance;
    if (x <= interface_x) {
        return pressure_left - darcy_flux * x / kappa_left;
    }
    const double interface_pressure = pressure_left - darcy_flux * interface_x / kappa_left;
    return interface_pressure - darcy_flux * (x - interface_x) / kappa_right;
}

std::vector<double> layeredMobility(const StructuredMesh& mesh, double interface_x,
                                    double kappa_left, double kappa_right,
                                    bool interface_node_is_left) {
    std::vector<double> kappa(static_cast<std::size_t>(mesh.numNodes()));
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double x = mesh.x(i);
            const bool left = x < interface_x || (interface_node_is_left && x == interface_x);
            kappa[static_cast<std::size_t>(mesh.index(i, j))] = left ? kappa_left : kappa_right;
        }
    }
    return kappa;
}

std::vector<double> layeredInitialGuess(const StructuredMesh& mesh, double length,
                                        double interface_x, double pressure_left,
                                        double pressure_right, double kappa_left,
                                        double kappa_right) {
    std::vector<double> pressure(static_cast<std::size_t>(mesh.numNodes()));
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            pressure[static_cast<std::size_t>(mesh.index(i, j))] =
                layeredPressure(mesh.x(i), length, interface_x, pressure_left, pressure_right,
                                kappa_left, kappa_right);
        }
    }
    return pressure;
}

void uniformPressureDropMatchesDarcyLaw() {
    constexpr double length = 0.04;                                                  // m
    constexpr double pressure_left = 3200.0;                                         // Pa
    constexpr double pressure_right = 800.0;                                         // Pa
    constexpr double kappa = 2.5e-10;                                                // m^2/(Pa s)
    constexpr double pressure_gradient = (pressure_right - pressure_left) / length;  // Pa/m
    constexpr double expected_velocity = -kappa * pressure_gradient;                 // m/s

    const StructuredMesh mesh(20, 6, 0.0, length, 0.0, 0.012);
    DarcyFlowSolver solver(mesh, kappa);
    solver.setDirichlet(Boundary::Left, pressure_left)
        .setDirichlet(Boundary::Right, pressure_right)
        .setNeumann(Boundary::Bottom, 0.0)
        .setNeumann(Boundary::Top, 0.0)
        .setOmega(1.65)
        .setTolerance(1.0e-10)
        .setMaxIterations(20000);

    const DarcyFlowResult result = solver.solve();
    double pressure_error = 0.0;
    double velocity_error = 0.0;
    double transverse_velocity = 0.0;
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const std::size_t index = static_cast<std::size_t>(mesh.index(i, j));
            const double exact_pressure = pressure_left + pressure_gradient * mesh.x(i);
            pressure_error =
                std::max(pressure_error, std::abs(result.pressure[index] - exact_pressure));
            velocity_error =
                std::max(velocity_error, std::abs(result.vx[index] - expected_velocity));
            transverse_velocity = std::max(transverse_velocity, std::abs(result.vy[index]));
        }
    }

    science_test::report("uniform pressure L_inf error", pressure_error, "Pa");
    science_test::report("uniform x-velocity L_inf error", velocity_error, "m/s");
    science_test::report("uniform max |vy|", transverse_velocity, "m/s");
    science_test::report("uniform iterations", static_cast<double>(result.iterations));
    science_test::report("uniform fixed-point defect", result.residual, "Pa");

    SCIENCE_REQUIRE(result.converged, "every returned Darcy result must be converged");
    SCIENCE_REQUIRE(result.residual <= 1.0e-10,
                    "reported pressure defect must satisfy the requested tolerance");
    SCIENCE_REQUIRE(pressure_error < 2.0e-8,
                    "linear pressure must match the analytical Dirichlet solution");
    SCIENCE_REQUIRE(velocity_error < 2.0e-15,
                    "v_x = -kappa dp/dx must hold in SI units on the linear field");
    SCIENCE_REQUIRE(transverse_velocity < 2.0e-15,
                    "zero outward top/bottom gradients must produce zero transverse flow");
}

void outwardGradientSignAndUnitsAreExplicit() {
    constexpr double length = 0.02;                        // m
    constexpr double fixed_pressure = 1200.0;              // Pa
    constexpr double gradient = 25000.0;                   // Pa/m
    constexpr double kappa = 4.0e-10;                      // m^2/(Pa s)
    constexpr double pressure_change = gradient * length;  // Pa
    constexpr double speed = kappa * gradient;             // m/s

    const StructuredMesh mesh(10, 4, 0.0, length, 0.0, 0.008);

    std::vector<double> right_guess(static_cast<std::size_t>(mesh.numNodes()));
    std::vector<double> left_guess(static_cast<std::size_t>(mesh.numNodes()));
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const std::size_t index = static_cast<std::size_t>(mesh.index(i, j));
            right_guess[index] = fixed_pressure + gradient * mesh.x(i);
            left_guess[index] = fixed_pressure + gradient * (length - mesh.x(i));
        }
    }

    DarcyFlowSolver right_gradient(mesh, kappa);
    right_gradient.setDirichlet(Boundary::Left, fixed_pressure)
        .setNeumann(Boundary::Right, gradient)
        .setNeumann(Boundary::Bottom, 0.0)
        .setNeumann(Boundary::Top, 0.0)
        .setInitialGuess(right_guess)
        .setTolerance(1.0e-11)
        .setMaxIterations(10);
    const DarcyFlowResult right_result = right_gradient.solve();

    DarcyFlowSolver left_gradient(mesh, kappa);
    left_gradient.setNeumann(Boundary::Left, gradient)
        .setDirichlet(Boundary::Right, fixed_pressure)
        .setNeumann(Boundary::Bottom, 0.0)
        .setNeumann(Boundary::Top, 0.0)
        .setInitialGuess(left_guess)
        .setTolerance(1.0e-11)
        .setMaxIterations(10);
    const DarcyFlowResult left_result = left_gradient.solve();

    const std::size_t right_mid = static_cast<std::size_t>(mesh.index(mesh.nx(), mesh.ny() / 2));
    const std::size_t left_mid = static_cast<std::size_t>(mesh.index(0, mesh.ny() / 2));
    SCIENCE_REQUIRE_NEAR(right_result.pressure[right_mid], fixed_pressure + pressure_change,
                         2.0e-11, 0.0, "right-boundary pressure from +dp/dn [Pa/m]");
    SCIENCE_REQUIRE_NEAR(left_result.pressure[left_mid], fixed_pressure + pressure_change, 2.0e-11,
                         0.0, "left-boundary pressure from +dp/dn [Pa/m]");
    SCIENCE_REQUIRE_NEAR(right_result.vx[right_mid], -speed, 2.0e-15, 0.0,
                         "right-boundary Darcy velocity from +dp/dn");
    SCIENCE_REQUIRE_NEAR(left_result.vx[left_mid], speed, 2.0e-15, 0.0,
                         "left-boundary x velocity from +outward dp/dn");

    science_test::report("prescribed outward gradient", gradient, "Pa/m");
    science_test::report("induced pressure change", pressure_change, "Pa");
    science_test::report("right outward Darcy velocity", right_result.vx[right_mid], "m/s");
    science_test::report("left outward Darcy velocity", -left_result.vx[left_mid], "m/s");
}

void layeredMediumConservesNormalFaceFlux() {
    constexpr double length = 1.0;             // m
    constexpr double interface_x = 0.5;        // m
    constexpr double pressure_left = 6000.0;   // Pa
    constexpr double pressure_right = 1000.0;  // Pa
    constexpr double kappa_left = 2.0e-10;     // m^2/(Pa s)
    constexpr double kappa_right = 8.0e-10;    // m^2/(Pa s)
    constexpr double resistance = interface_x / kappa_left + (length - interface_x) / kappa_right;
    constexpr double expected_flux = (pressure_left - pressure_right) / resistance;  // m/s
    constexpr double expected_interface_pressure =
        pressure_left - expected_flux * interface_x / kappa_left;

    // With 31 cells, x=0.5 lies exactly at the face between nodes 15 and 16.
    const StructuredMesh mesh(31, 4, 0.0, length, 0.0, 0.125);
    const std::vector<double> kappa =
        layeredMobility(mesh, interface_x, kappa_left, kappa_right, false);
    DarcyFlowSolver solver(mesh, kappa);
    solver.setDirichlet(Boundary::Left, pressure_left)
        .setDirichlet(Boundary::Right, pressure_right)
        .setNeumann(Boundary::Bottom, 0.0)
        .setNeumann(Boundary::Top, 0.0)
        .setInitialGuess(layeredInitialGuess(mesh, length, interface_x, pressure_left,
                                             pressure_right, kappa_left, kappa_right))
        .setTolerance(1.0e-10)
        .setMaxIterations(100);
    const DarcyFlowResult result = solver.solve();

    const int row = mesh.ny() / 2;
    double min_flux = expected_flux;
    double max_flux = expected_flux;
    double pressure_error = 0.0;
    for (int i = 0; i < mesh.nx(); ++i) {
        const std::size_t west = static_cast<std::size_t>(mesh.index(i, row));
        const std::size_t east = static_cast<std::size_t>(mesh.index(i + 1, row));
        const double face_kappa = harmonicMean(kappa[west], kappa[east]);
        const double face_flux =
            -face_kappa * (result.pressure[east] - result.pressure[west]) / mesh.dx();
        min_flux = std::min(min_flux, face_flux);
        max_flux = std::max(max_flux, face_flux);
    }
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const std::size_t index = static_cast<std::size_t>(mesh.index(i, j));
            const double exact = layeredPressure(mesh.x(i), length, interface_x, pressure_left,
                                                 pressure_right, kappa_left, kappa_right);
            pressure_error = std::max(pressure_error, std::abs(result.pressure[index] - exact));
        }
    }

    const int left_interface_i = (mesh.nx() - 1) / 2;
    const int right_interface_i = left_interface_i + 1;
    const double left_interface_pressure =
        result.pressure[static_cast<std::size_t>(mesh.index(left_interface_i, row))] -
        expected_flux * (interface_x - mesh.x(left_interface_i)) / kappa_left;
    const double right_interface_pressure =
        result.pressure[static_cast<std::size_t>(mesh.index(right_interface_i, row))] +
        expected_flux * (mesh.x(right_interface_i) - interface_x) / kappa_right;
    const double relative_flux_spread = (max_flux - min_flux) / expected_flux;

    science_test::report("layered analytical face flux", expected_flux, "m/s");
    science_test::report("layered relative face-flux spread", relative_flux_spread);
    science_test::report("layered pressure L_inf error", pressure_error, "Pa");
    science_test::report("layered interface pressure from left", left_interface_pressure, "Pa");
    science_test::report("layered interface pressure from right", right_interface_pressure, "Pa");

    SCIENCE_REQUIRE_NEAR(left_interface_pressure, expected_interface_pressure, 2.0e-7, 0.0,
                         "left piecewise pressure drop to the material interface");
    SCIENCE_REQUIRE_NEAR(right_interface_pressure, expected_interface_pressure, 2.0e-7, 0.0,
                         "right piecewise pressure drop to the material interface");
    SCIENCE_REQUIRE(pressure_error < 2.0e-7,
                    "face-aligned two-material pressure must match the series-resistance solution");
    SCIENCE_REQUIRE(relative_flux_spread < 2.0e-9,
                    "harmonic face mobility must conserve normal Darcy flux across the interface");
}

struct RefinementMeasurement {
    int cells;
    double spacing;
    double pressure_error;
};

RefinementMeasurement nodeAlignedInterfaceError(int cells) {
    constexpr double length = 1.0;
    constexpr double interface_x = 0.5;
    constexpr double pressure_left = 6000.0;
    constexpr double pressure_right = 1000.0;
    constexpr double kappa_left = 2.0e-10;
    constexpr double kappa_right = 8.0e-10;

    const StructuredMesh mesh(cells, 2, 0.0, length, 0.0, 0.125);
    const std::vector<double> kappa =
        layeredMobility(mesh, interface_x, kappa_left, kappa_right, true);
    DarcyFlowSolver solver(mesh, kappa);
    solver.setDirichlet(Boundary::Left, pressure_left)
        .setDirichlet(Boundary::Right, pressure_right)
        .setNeumann(Boundary::Bottom, 0.0)
        .setNeumann(Boundary::Top, 0.0)
        .setInitialGuess(layeredInitialGuess(mesh, length, interface_x, pressure_left,
                                             pressure_right, kappa_left, kappa_right))
        .setOmega(1.6)
        .setTolerance(1.0e-10)
        .setMaxIterations(20000);
    const DarcyFlowResult result = solver.solve();

    double error = 0.0;
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const std::size_t index = static_cast<std::size_t>(mesh.index(i, j));
            const double exact = layeredPressure(mesh.x(i), length, interface_x, pressure_left,
                                                 pressure_right, kappa_left, kappa_right);
            error = std::max(error, std::abs(result.pressure[index] - exact));
        }
    }
    return {cells, mesh.dx(), error};
}

void discontinuousInterfaceRefinesAtMeasuredFirstOrder() {
    const std::vector<RefinementMeasurement> measurements = {
        nodeAlignedInterfaceError(16),
        nodeAlignedInterfaceError(32),
        nodeAlignedInterfaceError(64),
    };
    const double order_16_to_32 =
        std::log(measurements[0].pressure_error / measurements[1].pressure_error) / std::log(2.0);
    const double order_32_to_64 =
        std::log(measurements[1].pressure_error / measurements[2].pressure_error) / std::log(2.0);

    for (const auto& measurement : measurements) {
        science_test::report("refinement h (N=" + std::to_string(measurement.cells) + ")",
                             measurement.spacing, "m");
        science_test::report(
            "refinement pressure L_inf error (N=" + std::to_string(measurement.cells) + ")",
            measurement.pressure_error, "Pa");
    }
    science_test::report("observed order N=16->32", order_16_to_32);
    science_test::report("observed order N=32->64", order_32_to_64);

    SCIENCE_REQUIRE(measurements[1].pressure_error < measurements[0].pressure_error &&
                        measurements[2].pressure_error < measurements[1].pressure_error,
                    "the represented interface error must decrease monotonically under refinement");
    SCIENCE_REQUIRE(order_16_to_32 > 0.8 && order_16_to_32 < 1.2,
                    "the measured discontinuous, node-aligned interface rate is first order");
    SCIENCE_REQUIRE(order_32_to_64 > 0.8 && order_32_to_64 < 1.2,
                    "the measured discontinuous, node-aligned interface rate is first order");
}

void singularAndUnconvergedProblemsFailLoudly() {
    const StructuredMesh mesh(8, 4, 0.0, 1.0, 0.0, 0.5);

    DarcyFlowSolver unanchored(mesh, 1.0e-10);
    requireThrows<std::invalid_argument>(
        [&] { (void)unanchored.solve(); },
        "an all-Neumann pressure system must reject its missing gauge");

    DarcyFlowSolver exhausted(mesh, 1.0e-10);
    exhausted.setDirichlet(Boundary::Left, 2000.0)
        .setDirichlet(Boundary::Right, 1000.0)
        .setNeumann(Boundary::Bottom, 0.0)
        .setNeumann(Boundary::Top, 0.0)
        .setTolerance(1.0e-15)
        .setMaxIterations(1);
    requireThrows<std::runtime_error>(
        [&] { (void)exhausted.solve(); },
        "iteration exhaustion must throw instead of returning a field");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "Darcy flow analytical and failure verification",
        {{"uniform analytical pressure and velocity", uniformPressureDropMatchesDarcyLaw},
         {"outward-gradient sign and SI units", outwardGradientSignAndUnitsAreExplicit},
         {"two-material pressure drop and flux continuity", layeredMediumConservesNormalFaceFlux},
         {"honest discontinuous-interface refinement",
          discontinuousInterfaceRefinesAtMeasuredFirstOrder},
         {"gauge and convergence failures", singularAndUnconvergedProblemsFailLoudly}});
}
