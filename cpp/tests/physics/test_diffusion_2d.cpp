#include "../test_support/science_test.hpp"
#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cmath>
#include <string>
#include <vector>

using namespace biotransport;

static double computeMass2D(const StructuredMesh& mesh, const std::vector<double>& u) {
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const double cell_area = mesh.dx() * mesh.dy();

    double mass = 0.0;
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            mass += u[mesh.index(i, j)] * cell_area;
        }
    }
    return mass;
}

void testDiffusion2DNeumannMassConservation() {
    StructuredMesh mesh(40, 25, 0.0, 1.0, 0.0, 1.0);

    const double D = 0.01;

    // Smooth deterministic initial field (non-uniform, positive)
    constexpr double pi = 3.14159265358979323846;
    std::vector<double> initial(mesh.numNodes(), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double x = mesh.x(i);
            const double y = mesh.y(i, j);
            const double value = 1.0 + 0.25 * std::sin(2.0 * pi * x) * std::cos(2.0 * pi * y);
            initial[mesh.index(i, j)] = value;
        }
    }

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .initialCondition(initial)
        .neumann(Boundary::Left, 0.0)
        .neumann(Boundary::Right, 0.0)
        .neumann(Boundary::Bottom, 0.0)
        .neumann(Boundary::Top, 0.0);

    const double initial_mass = computeMass2D(mesh, initial);

    // Explicit stability: dt <= min(dx^2, dy^2)/(4D)
    const double dt_estimate =
        0.2 * std::min(mesh.dx() * mesh.dx(), mesh.dy() * mesh.dy()) / (4.0 * D);
    const double t_end = dt_estimate * 250;

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    const double final_mass = computeMass2D(mesh, result.solution);

    // With zero flux, mass should be approximately conserved.
    const double rel_err =
        std::abs(final_mass - initial_mass) / std::max(1e-12, std::abs(initial_mass));
    science_test::report("relative mass change", rel_err);
    SCIENCE_REQUIRE(rel_err < 5e-3,
                    "zero-flux 2D diffusion must conserve mass to <0.5% relative error; actual=" +
                        science_test::number(rel_err));
}

void testDiffusion2DDirichletBoundaryPinned() {
    StructuredMesh mesh(20, 20, 0.0, 1.0, 0.0, 1.0);

    const double D = 0.05;

    std::vector<double> initial(mesh.numNodes(), 1.0);

    TransportProblem problem(mesh);
    problem.diffusivity(D)
        .initialCondition(initial)
        .dirichlet(Boundary::Left, 0.0)
        .dirichlet(Boundary::Right, 0.0)
        .dirichlet(Boundary::Bottom, 0.0)
        .dirichlet(Boundary::Top, 0.0);

    const double dt_estimate =
        0.2 * std::min(mesh.dx() * mesh.dx(), mesh.dy() * mesh.dy()) / (4.0 * D);
    const double t_end = dt_estimate * 5;

    ExplicitFD solver;
    auto result = solver.safetyFactor(0.4).run(problem, t_end);

    const auto& u = result.solution;

    for (int j = 0; j <= mesh.ny(); ++j) {
        SCIENCE_REQUIRE_NEAR(u[mesh.index(0, j)], 0.0, 1e-12, 0.0,
                             "left Dirichlet wall at row " + std::to_string(j));
        SCIENCE_REQUIRE_NEAR(u[mesh.index(mesh.nx(), j)], 0.0, 1e-12, 0.0,
                             "right Dirichlet wall at row " + std::to_string(j));
    }
    for (int i = 0; i <= mesh.nx(); ++i) {
        SCIENCE_REQUIRE_NEAR(u[mesh.index(i, 0)], 0.0, 1e-12, 0.0,
                             "bottom Dirichlet wall at column " + std::to_string(i));
        SCIENCE_REQUIRE_NEAR(u[mesh.index(i, mesh.ny())], 0.0, 1e-12, 0.0,
                             "top Dirichlet wall at column " + std::to_string(i));
    }
}

int main() {
    return science_test::runSuite(
        "2D diffusion", {{"zero-flux mass conservation", testDiffusion2DNeumannMassConservation},
                         {"Dirichlet boundary pinning", testDiffusion2DDirichletBoundaryPinned}});
}
