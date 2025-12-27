#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/solvers/explicit_fd.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
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
    std::cout << "Testing 2D diffusion (Neumann=0) mass conservation..." << std::endl;

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
    assert(rel_err < 5e-3);

    std::cout << "2D Neumann mass conservation test passed!" << std::endl;
}

void testDiffusion2DDirichletBoundaryPinned() {
    std::cout << "Testing 2D diffusion Dirichlet boundary pinning..." << std::endl;

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
        assert(std::abs(u[mesh.index(0, j)] - 0.0) < 1e-12);
        assert(std::abs(u[mesh.index(mesh.nx(), j)] - 0.0) < 1e-12);
    }
    for (int i = 0; i <= mesh.nx(); ++i) {
        assert(std::abs(u[mesh.index(i, 0)] - 0.0) < 1e-12);
        assert(std::abs(u[mesh.index(i, mesh.ny())] - 0.0) < 1e-12);
    }

    std::cout << "2D Dirichlet boundary pinning test passed!" << std::endl;
}

int main() {
    testDiffusion2DNeumannMassConservation();
    testDiffusion2DDirichletBoundaryPinned();

    std::cout << "All 2D diffusion tests passed!" << std::endl;
    return 0;
}
