/**
 * @file test_specialized_diffusion_science.cpp
 * @brief Scientific contract tests for the specialized diffusion solvers.
 *
 * The checks use an exception-based harness so they remain active in Release
 * builds.  They deliberately exercise conservation with trapezoidal control-
 * volume weights, the outward-normal Neumann convention, convergence order,
 * explicit stability rejection, and fail-loud iterative behavior.
 */

#include <algorithm>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <biotransport/solvers/adi_solver.hpp>
#include <biotransport/solvers/crank_nicolson.hpp>
#include <biotransport/solvers/diffusion_solver_3d.hpp>
#include <biotransport/solvers/implicit_diffusion.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using biotransport::ADIDiffusion2D;
using biotransport::ADIDiffusion3D;
using biotransport::Boundary;
using biotransport::Boundary3D;
using biotransport::CrankNicolsonDiffusion;
using biotransport::DiffusionSolver3D;
using biotransport::LinearReactionDiffusionSolver3D;
using biotransport::ReactionDiffusionSolver3D;
using biotransport::StructuredMesh;
using biotransport::StructuredMesh3D;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double tolerance, const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " +
                                 std::to_string(actual));
    }
}

template <typename Exception, typename Function>
void requireThrows(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

double maximumError(const std::vector<double>& actual, const std::vector<double>& expected) {
    require(actual.size() == expected.size(), "Cannot compare fields with different sizes");
    double error = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index)
        error = std::max(error, std::abs(actual[index] - expected[index]));
    return error;
}

double trapezoidalMass1D(const StructuredMesh& mesh, const std::vector<double>& values) {
    double mass = 0.0;
    for (int i = 0; i <= mesh.nx(); ++i) {
        const double weight = (i == 0 || i == mesh.nx()) ? 0.5 : 1.0;
        mass += weight * mesh.dx() * values[static_cast<std::size_t>(mesh.index(i))];
    }
    return mass;
}

double trapezoidalMass2D(const StructuredMesh& mesh, const std::vector<double>& values) {
    double mass = 0.0;
    for (int j = 0; j <= mesh.ny(); ++j) {
        const double wy = (j == 0 || j == mesh.ny()) ? 0.5 : 1.0;
        for (int i = 0; i <= mesh.nx(); ++i) {
            const double wx = (i == 0 || i == mesh.nx()) ? 0.5 : 1.0;
            mass += wx * wy * mesh.dx() * mesh.dy() *
                    values[static_cast<std::size_t>(mesh.index(i, j))];
        }
    }
    return mass;
}

double trapezoidalMass3D(const StructuredMesh3D& mesh, const std::vector<double>& values) {
    double mass = 0.0;
    for (int k = 0; k <= mesh.nz(); ++k) {
        const double wz = (k == 0 || k == mesh.nz()) ? 0.5 : 1.0;
        for (int j = 0; j <= mesh.ny(); ++j) {
            const double wy = (j == 0 || j == mesh.ny()) ? 0.5 : 1.0;
            for (int i = 0; i <= mesh.nx(); ++i) {
                const double wx = (i == 0 || i == mesh.nx()) ? 0.5 : 1.0;
                mass += wx * wy * wz * mesh.dx() * mesh.dy() * mesh.dz() *
                        values[static_cast<std::size_t>(mesh.index(i, j, k))];
            }
        }
    }
    return mass;
}

template <typename Solver>
void setHomogeneousNeumann2D(Solver& solver) {
    solver.setNeumannBoundary(Boundary::Left, 0.0);
    solver.setNeumannBoundary(Boundary::Right, 0.0);
    solver.setNeumannBoundary(Boundary::Bottom, 0.0);
    solver.setNeumannBoundary(Boundary::Top, 0.0);
}

template <typename Solver>
void setHomogeneousNeumann3D(Solver& solver) {
    solver.setNeumannBoundary(Boundary3D::XMin, 0.0);
    solver.setNeumannBoundary(Boundary3D::XMax, 0.0);
    solver.setNeumannBoundary(Boundary3D::YMin, 0.0);
    solver.setNeumannBoundary(Boundary3D::YMax, 0.0);
    solver.setNeumannBoundary(Boundary3D::ZMin, 0.0);
    solver.setNeumannBoundary(Boundary3D::ZMax, 0.0);
}

std::vector<double> smoothField3D(const StructuredMesh3D& mesh) {
    std::vector<double> field(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int j = 0; j <= mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                field[static_cast<std::size_t>(mesh.index(i, j, k))] =
                    1.0 + 0.2 * std::sin(1.3 * mesh.x(i)) + 0.1 * std::cos(0.8 * mesh.y(j)) +
                    0.05 * std::sin(1.7 * mesh.z(k));
            }
        }
    }
    return field;
}

void explicit3DPreservesLinearNeumannSolution() {
    StructuredMesh3D mesh(7, 5, 4, 0.0, 1.0, -0.5, 0.5, 0.0, 2.0);
    DiffusionSolver3D solver(mesh, 0.3);
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int j = 0; j <= mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                exact[static_cast<std::size_t>(mesh.index(i, j, k))] =
                    4.0 + mesh.x(i) + 2.0 * mesh.y(j) + 3.0 * mesh.z(k);
            }
        }
    }
    solver.setInitialCondition(exact);
    solver.setNeumannBoundary(Boundary3D::XMin, -1.0);
    solver.setNeumannBoundary(Boundary3D::XMax, 1.0);
    solver.setNeumannBoundary(Boundary3D::YMin, -2.0);
    solver.setNeumannBoundary(Boundary3D::YMax, 2.0);
    solver.setNeumannBoundary(Boundary3D::ZMin, -3.0);
    solver.setNeumannBoundary(Boundary3D::ZMax, 3.0);
    const double dt = 0.8 * solver.maxStableTimeStep();
    solver.solve(dt, 12);
    require(maximumError(solver.solution(), exact) < 2e-13,
            "Explicit 3D solver changed a discrete linear Neumann steady state");
    requireNear(solver.time(), 12.0 * dt, 1e-14, "Explicit 3D time did not advance exactly");
}

void explicit3DConservesControlVolumeMass() {
    StructuredMesh3D mesh(8, 6, 5, 0.0, 1.1, 0.0, 0.7, -0.2, 0.9);
    DiffusionSolver3D solver(mesh, 0.04);
    const auto initial = smoothField3D(mesh);
    solver.setInitialCondition(initial);
    setHomogeneousNeumann3D(solver);
    const double initial_mass = trapezoidalMass3D(mesh, initial);
    solver.solve(0.75 * solver.maxStableTimeStep(), 40);
    requireNear(trapezoidalMass3D(mesh, solver.solution()), initial_mass, 3e-13,
                "Explicit 3D diffusion did not conserve control-volume mass");
}

void explicit3DReactionContractsAreHonest() {
    StructuredMesh3D mesh(4, 3, 2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    LinearReactionDiffusionSolver3D decay(mesh, 0.01, 2.0);
    decay.setInitialCondition(std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 1.0));
    setHomogeneousNeumann3D(decay);
    decay.solve(0.1, 3);
    const double expected = 1.0 / (1.2 * 1.2 * 1.2);
    for (double value : decay.solution())
        requireNear(value, expected, 2e-14, "IMEX linear decay did not use Backward Euler");

    ReactionDiffusionSolver3D source(
        mesh, 0.01, [](double, double, double, double, double time) { return 2.0 + time; });
    source.setInitialCondition(std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.0));
    setHomogeneousNeumann3D(source);
    source.solve(0.01, 2);
    for (double value : source.solution())
        requireNear(value, 0.0401, 3e-15, "Reaction callback did not receive the step-start time");
}

void explicit3DRejectsUnstableAndInvalidInputs() {
    StructuredMesh3D mesh(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    requireThrows<std::invalid_argument>([&] { DiffusionSolver3D invalid(mesh, 0.0); },
                                         "Explicit 3D accepted zero diffusivity");
    DiffusionSolver3D solver(mesh, 0.1);
    const auto initial = smoothField3D(mesh);
    solver.setInitialCondition(initial);
    requireThrows<std::invalid_argument>(
        [&] { solver.setDirichletBoundary(6, 0.0); },
        "Explicit 3D accepted an out-of-range integer boundary identifier");
    requireThrows<std::invalid_argument>(
        [&] {
            solver.solve(
                std::nextafter(solver.maxStableTimeStep(), std::numeric_limits<double>::infinity()),
                1);
        },
        "Explicit 3D accepted a step above its documented CFL limit");
    require(maximumError(solver.solution(), initial) == 0.0 && solver.time() == 0.0,
            "Rejected explicit step mutated solver state");
}

void specializedExplicitAndCrankNicolsonRejectCornerConflicts() {
    StructuredMesh3D mesh3d(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    DiffusionSolver3D explicit_solver(mesh3d, 0.1);
    const std::vector<double> initial3d(static_cast<std::size_t>(mesh3d.numNodes()), 0.25);
    explicit_solver.setInitialCondition(initial3d);
    setHomogeneousNeumann3D(explicit_solver);
    explicit_solver.setDirichletBoundary(Boundary3D::XMin, 1.0);
    explicit_solver.setDirichletBoundary(Boundary3D::YMin, 2.0);
    requireThrows<std::invalid_argument>(
        [&] { explicit_solver.solve(0.5 * explicit_solver.maxStableTimeStep(), 1); },
        "Explicit 3D averaged contradictory Dirichlet edge data");
    require(
        maximumError(explicit_solver.solution(), initial3d) == 0.0 && explicit_solver.time() == 0.0,
        "Explicit 3D corner rejection mutated state");

    StructuredMesh mesh2d(4, 3, 0.0, 1.0, 0.0, 1.0);
    CrankNicolsonDiffusion crank_nicolson(mesh2d, 0.1);
    const std::vector<double> initial2d(static_cast<std::size_t>(mesh2d.numNodes()), 0.25);
    crank_nicolson.setInitialCondition(initial2d);
    setHomogeneousNeumann2D(crank_nicolson);
    crank_nicolson.setDirichletBoundary(Boundary::Left, 1.0);
    crank_nicolson.setDirichletBoundary(Boundary::Bottom, 2.0);
    requireThrows<std::invalid_argument>([&] { (void)crank_nicolson.step(0.1); },
                                         "Crank-Nicolson averaged conflicting corner data");
    require(
        maximumError(crank_nicolson.solution(), initial2d) == 0.0 && crank_nicolson.time() == 0.0,
        "Crank-Nicolson corner rejection mutated state");
}

void compatibleDirichletRoundoffIsAccepted() {
    StructuredMesh mesh(4, 3, 0.0, 1.0, 0.0, 1.0);
    CrankNicolsonDiffusion solver(mesh, 0.1);
    solver.setInitialCondition(std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 1.0));
    setHomogeneousNeumann2D(solver);
    solver.setDirichletBoundary(Boundary::Left, 1.0);
    solver.setDirichletBoundary(Boundary::Bottom,
                                1.0 + 32.0 * std::numeric_limits<double>::epsilon());
    const auto result = solver.step(0.01);
    require(result.converged,
            "Numerically equal Dirichlet traces inside the documented tolerance were rejected");
}

void crankNicolsonPreservesLinearNeumannSolution() {
    StructuredMesh mesh(13, 9, -0.2, 1.4, 0.0, 0.8);
    CrankNicolsonDiffusion solver(mesh, 0.25);
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            exact[static_cast<std::size_t>(mesh.index(i, j))] =
                3.0 + 1.5 * mesh.x(i) - 0.75 * mesh.y(i, j);
        }
    }
    solver.setInitialCondition(exact);
    solver.setNeumannBoundary(Boundary::Left, -1.5);
    solver.setNeumannBoundary(Boundary::Right, 1.5);
    solver.setNeumannBoundary(Boundary::Bottom, 0.75);
    solver.setNeumannBoundary(Boundary::Top, -0.75);
    const auto result = solver.step(0.9);
    require(result.converged, "Crank-Nicolson failed on a positive-definite diffusion system");
    require(maximumError(solver.solution(), exact) < 3e-11,
            "Crank-Nicolson changed a discrete linear Neumann steady state");
}

void crankNicolsonConservesControlVolumeMass() {
    StructuredMesh mesh(80, -0.3, 1.2);
    CrankNicolsonDiffusion solver(mesh, 0.08);
    std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int i = 0; i <= mesh.nx(); ++i)
        initial[static_cast<std::size_t>(mesh.index(i))] =
            1.0 + 0.2 * std::sin(4.0 * mesh.x(i)) + 0.1 * std::cos(7.0 * mesh.x(i));
    solver.setInitialCondition(initial);
    solver.setNeumannBoundary(Boundary::Left, 0.0);
    solver.setNeumannBoundary(Boundary::Right, 0.0);
    const double initial_mass = trapezoidalMass1D(mesh, initial);
    solver.solve(0.15, 8);
    requireNear(trapezoidalMass1D(mesh, solver.solution()), initial_mass, 2e-10,
                "Crank-Nicolson did not conserve 1D control-volume mass");
}

void crankNicolsonFailureDoesNotAdvanceState() {
    StructuredMesh mesh(30, 24, 0.0, 1.0, 0.0, 1.0);
    CrankNicolsonDiffusion solver(mesh, 0.2);
    std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            initial[static_cast<std::size_t>(mesh.index(i, j))] =
                std::sin(2.1 * mesh.x(i)) * std::cos(1.7 * mesh.y(i, j));
        }
    }
    solver.setInitialCondition(initial);
    setHomogeneousNeumann2D(solver);
    solver.setTolerance(1e-15).setMaxIterations(1);
    const auto result = solver.step(0.5);
    require(!result.converged && result.iterations == 1,
            "Crank-Nicolson did not expose its configured non-convergence");
    require(maximumError(solver.solution(), initial) == 0.0 && solver.time() == 0.0,
            "A failed Crank-Nicolson solve mutated public state");
}

std::vector<double> sineMode2D(const StructuredMesh& mesh) {
    const double pi = std::acos(-1.0);
    std::vector<double> values(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            values[static_cast<std::size_t>(mesh.index(i, j))] =
                std::sin(pi * mesh.x(i)) * std::sin(pi * mesh.y(i, j));
        }
    }
    return values;
}

std::vector<double> runADI2D(const StructuredMesh& mesh, double diffusivity, double final_time,
                             int steps) {
    ADIDiffusion2D solver(mesh, diffusivity);
    solver.setInitialCondition(sineMode2D(mesh));
    solver.solve(final_time / static_cast<double>(steps), steps);
    return solver.solution();
}

void adiIsSecondOrderInTime() {
    StructuredMesh mesh(32, 28, 0.0, 1.0, 0.0, 1.0);
    constexpr double diffusivity = 0.2;
    constexpr double final_time = 0.08;
    const auto coarse = runADI2D(mesh, diffusivity, final_time, 4);
    const auto medium = runADI2D(mesh, diffusivity, final_time, 8);
    const auto reference = runADI2D(mesh, diffusivity, final_time, 256);
    const double coarse_error = maximumError(coarse, reference);
    const double medium_error = maximumError(medium, reference);
    require(medium_error > 0.0 && coarse_error / medium_error > 3.8,
            "Symmetric ADI composition did not exhibit second-order temporal convergence");
}

void adi2DConservesMassAndPreservesLinearNeumannData() {
    StructuredMesh mesh(15, 11, -0.1, 1.2, 0.0, 0.9);
    ADIDiffusion2D solver(mesh, 0.15);
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i)
            exact[static_cast<std::size_t>(mesh.index(i, j))] =
                2.0 + 0.4 * mesh.x(i) - 0.7 * mesh.y(i, j);
    }
    solver.setInitialCondition(exact);
    solver.setNeumannBoundary(Boundary::Left, -0.4);
    solver.setNeumannBoundary(Boundary::Right, 0.4);
    solver.setNeumannBoundary(Boundary::Bottom, 0.7);
    solver.setNeumannBoundary(Boundary::Top, -0.7);
    solver.solve(0.17, 6);
    require(maximumError(solver.solution(), exact) < 2e-13,
            "2D ADI changed a discrete linear Neumann steady state");

    ADIDiffusion2D conservative(mesh, 0.15);
    const auto initial = sineMode2D(mesh);
    conservative.setInitialCondition(initial);
    setHomogeneousNeumann2D(conservative);
    const double initial_mass = trapezoidalMass2D(mesh, initial);
    conservative.solve(0.11, 9);
    requireNear(trapezoidalMass2D(mesh, conservative.solution()), initial_mass, 2e-13,
                "2D ADI did not conserve control-volume mass");
}

void adi3DConservesMassAndPreservesLinearNeumannData() {
    StructuredMesh3D mesh(7, 5, 4, 0.0, 1.0, -0.2, 0.7, 0.0, 1.3);
    ADIDiffusion3D solver(mesh, 0.12);
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int j = 0; j <= mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                exact[static_cast<std::size_t>(mesh.index(i, j, k))] =
                    3.0 + 0.2 * mesh.x(i) - 0.3 * mesh.y(j) + 0.5 * mesh.z(k);
            }
        }
    }
    solver.setInitialCondition(exact);
    solver.setNeumannBoundary(Boundary3D::XMin, -0.2);
    solver.setNeumannBoundary(Boundary3D::XMax, 0.2);
    solver.setNeumannBoundary(Boundary3D::YMin, 0.3);
    solver.setNeumannBoundary(Boundary3D::YMax, -0.3);
    solver.setNeumannBoundary(Boundary3D::ZMin, -0.5);
    solver.setNeumannBoundary(Boundary3D::ZMax, 0.5);
    solver.solve(0.13, 5);
    require(maximumError(solver.solution(), exact) < 3e-13,
            "3D ADI changed a discrete linear Neumann steady state");

    ADIDiffusion3D conservative(mesh, 0.12);
    const auto initial = smoothField3D(mesh);
    conservative.setInitialCondition(initial);
    setHomogeneousNeumann3D(conservative);
    const double initial_mass = trapezoidalMass3D(mesh, initial);
    conservative.solve(0.09, 7);
    requireNear(trapezoidalMass3D(mesh, conservative.solution()), initial_mass, 3e-13,
                "3D ADI did not conserve control-volume mass");
}

void adiRejectsInvalidInputsWithoutMutation() {
    StructuredMesh one_dimensional(8, 0.0, 1.0);
    requireThrows<std::invalid_argument>([&] { ADIDiffusion2D invalid(one_dimensional, 0.1); },
                                         "2D ADI accepted a 1D mesh");
    StructuredMesh mesh(4, 3, 0.0, 1.0, 0.0, 1.0);
    ADIDiffusion2D solver(mesh, 0.1);
    const auto initial = sineMode2D(mesh);
    solver.setInitialCondition(initial);
    requireThrows<std::invalid_argument>(
        [&] { solver.step(std::numeric_limits<double>::infinity()); },
        "ADI accepted a non-finite time step");
    requireThrows<std::invalid_argument>([&] { solver.solve(0.1, -1); },
                                         "ADI accepted a negative step count");
    require(maximumError(solver.solution(), initial) == 0.0 && solver.time() == 0.0,
            "Rejected ADI input mutated solver state");
}

void adiRejectsCornerAndEdgeConflictsAtomically() {
    StructuredMesh mesh2d(4, 3, 0.0, 1.0, 0.0, 1.0);
    ADIDiffusion2D solver2d(mesh2d, 0.1);
    const std::vector<double> initial2d(static_cast<std::size_t>(mesh2d.numNodes()), 0.25);
    solver2d.setInitialCondition(initial2d);
    setHomogeneousNeumann2D(solver2d);
    solver2d.setDirichletBoundary(Boundary::Left, 1.0);
    solver2d.setDirichletBoundary(Boundary::Bottom, 2.0);
    requireThrows<std::invalid_argument>([&] { (void)solver2d.step(0.1); },
                                         "2D ADI averaged conflicting corner data");
    require(maximumError(solver2d.solution(), initial2d) == 0.0 && solver2d.time() == 0.0,
            "2D ADI corner rejection mutated state");

    StructuredMesh3D mesh3d(3, 3, 2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    ADIDiffusion3D solver3d(mesh3d, 0.1);
    const std::vector<double> initial3d(static_cast<std::size_t>(mesh3d.numNodes()), 0.25);
    solver3d.setInitialCondition(initial3d);
    setHomogeneousNeumann3D(solver3d);
    solver3d.setDirichletBoundary(Boundary3D::XMin, 1.0);
    solver3d.setDirichletBoundary(Boundary3D::YMin, 1.0);
    solver3d.setDirichletBoundary(Boundary3D::ZMin, 2.0);
    requireThrows<std::invalid_argument>([&] { (void)solver3d.step(0.1); },
                                         "3D ADI averaged conflicting edge/corner data");
    require(maximumError(solver3d.solution(), initial3d) == 0.0 && solver3d.time() == 0.0,
            "3D ADI edge/corner rejection mutated state");
}

#ifdef BIOTRANSPORT_ENABLE_EIGEN

using biotransport::ImplicitDiffusion2D;
using biotransport::ImplicitDiffusion3D;

void implicit2DPreservesDiscreteVariableCoefficientFlux() {
    StructuredMesh mesh(10, 4, 0.0, 1.0, 0.0, 0.6);
    std::vector<double> diffusivity(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i)
            diffusivity[static_cast<std::size_t>(mesh.index(i, j))] = i <= 5 ? 1.0 : 4.0;
    }

    constexpr double flux = 0.7;
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 1.0);
    std::vector<double> profile(static_cast<std::size_t>(mesh.nx() + 1), 1.0);
    for (int i = 1; i <= mesh.nx(); ++i) {
        const double left = diffusivity[static_cast<std::size_t>(mesh.index(i - 1, 0))];
        const double right = diffusivity[static_cast<std::size_t>(mesh.index(i, 0))];
        const double face = 2.0 * left * right / (left + right);
        profile[static_cast<std::size_t>(i)] =
            profile[static_cast<std::size_t>(i - 1)] + flux * mesh.dx() / face;
    }
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i)
            exact[static_cast<std::size_t>(mesh.index(i, j))] =
                profile[static_cast<std::size_t>(i)];
    }

    ImplicitDiffusion2D solver(mesh, diffusivity);
    solver.setInitialCondition(exact);
    solver.setNeumannBoundary(Boundary::Left, -flux / 1.0);
    solver.setNeumannBoundary(Boundary::Right, flux / 4.0);
    solver.setNeumannBoundary(Boundary::Bottom, 0.0);
    solver.setNeumannBoundary(Boundary::Top, 0.0);
    const auto result = solver.step(0.8);
    require(result.success && result.residual < 1e-11,
            "Implicit 2D did not report a converged algebraic solve");
    require(maximumError(solver.solution(), exact) < 3e-11,
            "Implicit 2D did not preserve a harmonic-face flux equilibrium");
}

void implicit2DSourceAndNeumannMassBalance() {
    StructuredMesh mesh(8, 6, 0.0, 1.2, -0.1, 0.7);
    ImplicitDiffusion2D solver(mesh, 0.2);
    solver.setInitialCondition(std::vector<double>(static_cast<std::size_t>(mesh.numNodes()), 0.0));
    setHomogeneousNeumann2D(solver);
    solver.setSourceTerm([](double, double, double) { return 2.5; });
    const auto result = solver.solve(0.12, 3);
    require(result.steps == 3 && result.success, "Implicit 2D returned incorrect solve metadata");
    for (double value : solver.solution())
        requireNear(value, 0.9, 2e-11, "Uniform source did not produce the exact uniform balance");
    solver.clearSourceTerm();
    solver.step(0.12);
    for (double value : solver.solution())
        requireNear(value, 0.9, 2e-11, "Clearing the implicit source did not take effect");
}

void implicit3DPreservesLinearNeumannSolution() {
    StructuredMesh3D mesh(5, 4, 3, 0.0, 1.0, 0.0, 0.8, -0.2, 0.7);
    requireThrows<std::invalid_argument>(
        [&] { ImplicitDiffusion3D invalid(mesh, std::vector<double>{0.3}); },
        "Implicit 3D accepted a diffusivity field with the wrong size");
    std::vector<double> diffusivity(static_cast<std::size_t>(mesh.numNodes()), 0.3);
    ImplicitDiffusion3D solver(mesh, diffusivity);
    require(solver.diffusivity().size() == static_cast<std::size_t>(mesh.numNodes()),
            "Implicit 3D did not retain its nodal diffusivity field");
    std::vector<double> exact(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int j = 0; j <= mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                exact[static_cast<std::size_t>(mesh.index(i, j, k))] =
                    2.0 + mesh.x(i) - 0.5 * mesh.y(j) + 0.25 * mesh.z(k);
            }
        }
    }
    solver.setInitialCondition(exact);
    solver.setNeumannBoundary(Boundary3D::XMin, -1.0);
    solver.setNeumannBoundary(Boundary3D::XMax, 1.0);
    solver.setNeumannBoundary(Boundary3D::YMin, 0.5);
    solver.setNeumannBoundary(Boundary3D::YMax, -0.5);
    solver.setNeumannBoundary(Boundary3D::ZMin, -0.25);
    solver.setNeumannBoundary(Boundary3D::ZMax, 0.25);
    const auto result = solver.step(0.7);
    require(result.success && result.residual < 1e-9,
            "Implicit 3D did not report a converged algebraic solve");
    require(maximumError(solver.solution(), exact) < 2e-8,
            "Implicit 3D changed a discrete linear Neumann steady state");
}

void implicitSolversRejectCornerAndEdgeConflictsAtomically() {
    StructuredMesh mesh2d(4, 3, 0.0, 1.0, 0.0, 1.0);
    ImplicitDiffusion2D solver2d(mesh2d, 0.1);
    const std::vector<double> initial2d(static_cast<std::size_t>(mesh2d.numNodes()), 0.25);
    solver2d.setInitialCondition(initial2d);
    setHomogeneousNeumann2D(solver2d);
    solver2d.setDirichletBoundary(Boundary::Left, 1.0);
    solver2d.setDirichletBoundary(Boundary::Bottom, 2.0);
    requireThrows<std::invalid_argument>([&] { (void)solver2d.step(0.1); },
                                         "Implicit 2D averaged conflicting corner data");
    require(maximumError(solver2d.solution(), initial2d) == 0.0 && solver2d.time() == 0.0,
            "Implicit 2D corner rejection mutated state");

    StructuredMesh3D mesh3d(3, 3, 2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    ImplicitDiffusion3D solver3d(mesh3d, 0.1);
    const std::vector<double> initial3d(static_cast<std::size_t>(mesh3d.numNodes()), 0.25);
    solver3d.setInitialCondition(initial3d);
    setHomogeneousNeumann3D(solver3d);
    solver3d.setDirichletBoundary(Boundary3D::XMin, 1.0);
    solver3d.setDirichletBoundary(Boundary3D::YMin, 1.0);
    solver3d.setDirichletBoundary(Boundary3D::ZMin, 2.0);
    requireThrows<std::invalid_argument>([&] { (void)solver3d.step(0.1); },
                                         "Implicit 3D averaged conflicting edge/corner data");
    require(maximumError(solver3d.solution(), initial3d) == 0.0 && solver3d.time() == 0.0,
            "Implicit 3D edge/corner rejection mutated state");
}

#endif

}  // namespace

int main() {
    std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"explicit 3D linear Neumann steady state", explicit3DPreservesLinearNeumannSolution},
        {"explicit 3D mass conservation", explicit3DConservesControlVolumeMass},
        {"explicit 3D reaction contracts", explicit3DReactionContractsAreHonest},
        {"explicit 3D validation", explicit3DRejectsUnstableAndInvalidInputs},
        {"specialized explicit/CN corner conflicts",
         specializedExplicitAndCrankNicolsonRejectCornerConflicts},
        {"compatible Dirichlet roundoff", compatibleDirichletRoundoffIsAccepted},
        {"Crank-Nicolson linear Neumann steady state", crankNicolsonPreservesLinearNeumannSolution},
        {"Crank-Nicolson mass conservation", crankNicolsonConservesControlVolumeMass},
        {"Crank-Nicolson atomic non-convergence", crankNicolsonFailureDoesNotAdvanceState},
        {"ADI temporal order", adiIsSecondOrderInTime},
        {"ADI 2D conservation and Neumann convention",
         adi2DConservesMassAndPreservesLinearNeumannData},
        {"ADI 3D conservation and Neumann convention",
         adi3DConservesMassAndPreservesLinearNeumannData},
        {"ADI validation", adiRejectsInvalidInputsWithoutMutation},
        {"ADI corner/edge conflicts", adiRejectsCornerAndEdgeConflictsAtomically},
    };
#ifdef BIOTRANSPORT_ENABLE_EIGEN
    tests.emplace_back("implicit 2D variable-coefficient equilibrium",
                       implicit2DPreservesDiscreteVariableCoefficientFlux);
    tests.emplace_back("implicit 2D source balance", implicit2DSourceAndNeumannMassBalance);
    tests.emplace_back("implicit 3D linear Neumann steady state",
                       implicit3DPreservesLinearNeumannSolution);
    tests.emplace_back("implicit corner/edge conflicts",
                       implicitSolversRejectCornerAndEdgeConflictsAtomically);
#endif

    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }
    if (failures != 0)
        std::cerr << failures << " specialized diffusion contract(s) failed\n";
    return failures == 0 ? 0 : 1;
}
