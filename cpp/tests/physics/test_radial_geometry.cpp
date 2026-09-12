/**
 * @file test_radial_geometry.cpp
 * @brief Evidence for the cylindrical and spherical finite-volume balance.
 *
 * Five independent checks:
 *   1. Control volumes sum to the exact domain measure (R for a slab, R^2/2 for
 *      a cylinder, R^3/3 for a sphere), so the mass integral is exact rather
 *      than a quadrature approximation.
 *   2. A sealed radial domain conserves its contents to roundoff.
 *   3. A uniform field with matching boundary data stays uniform, which is the
 *      statement that the curved operator annihilates a constant.
 *   4. Transient decay converges at second order against Crank's spherical
 *      series. This is the strongest check, because the reference is completely
 *      independent of the discretization.
 *   5. Solvers that only implement slab geometry refuse a curved mesh, and a
 *      curved mesh refuses a negative inner radius.
 */

#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/problems/transport_problem.hpp>
#include <biotransport/solvers/transport_solver.hpp>

#include "../test_support/science_test.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace biotransport;

namespace {

constexpr double kPi = 3.14159265358979323846;

/** Crank's series for a sphere held at zero on its surface, uniform inside. */
double sphereSeries(double r, double t, double diffusivity, double radius, int terms) {
    const double fourier = diffusivity * t / (radius * radius);
    double theta = 0.0;
    for (int n = 1; n <= terms; ++n) {
        const double decay = std::exp(-static_cast<double>(n) * static_cast<double>(n) * kPi * kPi *
                                      fourier);
        const double sign = (n % 2 == 0) ? -1.0 : 1.0;
        if (r < 1.0e-12 * radius) {
            theta += 2.0 * sign * decay;
        } else {
            theta += (2.0 * radius / (kPi * r)) * (sign / static_cast<double>(n)) *
                     std::sin(static_cast<double>(n) * kPi * r / radius) * decay;
        }
    }
    return theta;
}

void controlVolumesAreExact() {
    const int cells = 64;
    const double radius = 2.0;
    const StructuredMesh slab(cells, 0.0, radius, Geometry::CARTESIAN);
    const StructuredMesh tube(cells, 0.0, radius, Geometry::CYLINDRICAL);
    const StructuredMesh ball(cells, 0.0, radius, Geometry::SPHERICAL);

    double slab_total = 0.0;
    double tube_total = 0.0;
    double ball_total = 0.0;
    for (int i = 0; i <= cells; ++i) {
        slab_total += slab.controlVolume(i);
        tube_total += tube.controlVolume(i);
        ball_total += ball.controlVolume(i);
    }

    SCIENCE_REQUIRE_NEAR(slab_total, radius, 1.0e-13, 0.0, "slab measure");
    SCIENCE_REQUIRE_NEAR(tube_total, radius * radius / 2.0, 1.0e-13, 0.0, "cylinder measure");
    SCIENCE_REQUIRE_NEAR(ball_total, radius * radius * radius / 3.0, 1.0e-13, 0.0,
                         "sphere measure");

    // The centre has no area, which is what makes symmetry automatic there.
    SCIENCE_REQUIRE(tube.lowerFaceArea(0) == 0.0, "cylinder centre face area must vanish");
    SCIENCE_REQUIRE(ball.lowerFaceArea(0) == 0.0, "sphere centre face area must vanish");
    science_test::report("sphere measure", ball_total);
}

void sealedRadialDomainsConserve() {
    const int cells = 80;
    const double radius = 1.0e-3;
    const double diffusivity = 1.0e-9;

    for (Geometry geometry : {Geometry::CYLINDRICAL, Geometry::SPHERICAL}) {
        StructuredMesh mesh(cells, 0.0, radius, geometry);
        std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()));
        for (int i = 0; i <= cells; ++i) {
            const double r = mesh.x(i);
            initial[static_cast<std::size_t>(i)] = std::exp(-std::pow(r / (0.3 * radius), 2.0));
        }

        TransportProblem problem(mesh);
        problem.diffusivity(diffusivity)
            .initialCondition(initial)
            .neumann(Boundary::Left, 0.0)
            .neumann(Boundary::Right, 0.0);

        const TransportResult result = solve(problem, SolveOptions::until(20.0));
        const double relative = std::abs(result.diagnostics.mass_change) /
                                std::max(std::abs(result.diagnostics.initial_mass), 1.0e-300);
        science_test::report("relative mass drift", relative);
        SCIENCE_REQUIRE(relative < 1.0e-12, "a sealed radial domain must conserve its contents");
    }
}

void curvedOperatorAnnihilatesAConstant() {
    const int cells = 40;
    const double radius = 1.0;

    for (Geometry geometry : {Geometry::CYLINDRICAL, Geometry::SPHERICAL}) {
        StructuredMesh mesh(cells, 0.0, radius, geometry);
        std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()), 3.0);

        TransportProblem problem(mesh);
        problem.diffusivity(1.0e-3).initialCondition(initial).dirichlet(Boundary::Right, 3.0);

        const TransportResult result = solve(problem, SolveOptions::until(50.0));
        double worst = 0.0;
        for (double value : result.concentration) {
            worst = std::max(worst, std::abs(value - 3.0));
        }
        science_test::report("departure from uniform", worst);
        SCIENCE_REQUIRE(worst < 1.0e-12, "a curved operator must annihilate a constant");
    }
}

void sphericalTransientIsSecondOrder() {
    const double radius = 1.0e-3;
    const double diffusivity = 1.0e-9;
    const double final_time = 100.0;

    double previous_error = 0.0;
    double finest_order = 0.0;
    for (int cells : {100, 200, 400}) {
        StructuredMesh mesh(cells, 0.0, radius, Geometry::SPHERICAL);
        std::vector<double> initial(static_cast<std::size_t>(mesh.numNodes()), 1.0);

        TransportProblem problem(mesh);
        problem.diffusivity(diffusivity).initialCondition(initial).dirichlet(Boundary::Right, 0.0);

        const std::vector<double> numeric =
            solve(problem, SolveOptions::until(final_time)).concentration;

        double error = 0.0;
        for (int i = 0; i <= cells; ++i) {
            const double reference = sphereSeries(mesh.x(i), final_time, diffusivity, radius, 200);
            error = std::max(error, std::abs(numeric[static_cast<std::size_t>(i)] - reference));
        }
        science_test::report("max error", error);
        if (previous_error > 0.0) {
            finest_order = std::log2(previous_error / error);
        }
        previous_error = error;
    }

    science_test::report("observed spatial order", finest_order);
    SCIENCE_REQUIRE(finest_order > 1.8 && finest_order < 2.2,
                    "spherical transient must converge at second order");
}

void axisymmetricReducesToTheRadialProblem() {
    // With nothing driving the axial direction, an (r, z) mesh must reproduce
    // the 1D radial answer. That is the sharpest available check that the axial
    // weighting cancels the way the algebra says it does.
    //
    // Both runs are pinned to the same explicit step. Left to choose, the 2D
    // mesh would take a smaller one -- it carries an extra axial stability
    // constraint -- and first-order time integration would then separate the two
    // answers by O(dt), masking whether the spatial operators agree.
    const int radial_cells = 60;
    const int axial_cells = 7;
    const double radius = 1.0e-3;
    const double height = 4.0e-4;
    const double diffusivity = 1.0e-9;
    const double final_time = 60.0;
    const double time_step = 0.01;

    SolveOptions options = SolveOptions::until(final_time);
    options.time_step = time_step;

    StructuredMesh line(radial_cells, 0.0, radius, Geometry::CYLINDRICAL);
    std::vector<double> line_initial(static_cast<std::size_t>(line.numNodes()), 1.0);
    TransportProblem line_problem(line);
    line_problem.diffusivity(diffusivity)
        .initialCondition(line_initial)
        .dirichlet(Boundary::Right, 0.0);
    const std::vector<double> line_result = solve(line_problem, options).concentration;

    StructuredMesh ring(radial_cells, axial_cells, 0.0, radius, 0.0, height,
                        Geometry::CYLINDRICAL);
    std::vector<double> ring_initial(static_cast<std::size_t>(ring.numNodes()), 1.0);
    TransportProblem ring_problem(ring);
    ring_problem.diffusivity(diffusivity)
        .initialCondition(ring_initial)
        .dirichlet(Boundary::Right, 0.0)
        .neumann(Boundary::Bottom, 0.0)
        .neumann(Boundary::Top, 0.0);
    const std::vector<double> ring_result = solve(ring_problem, options).concentration;

    double worst = 0.0;
    for (int j = 0; j <= axial_cells; ++j) {
        for (int i = 0; i <= radial_cells; ++i) {
            const double axisymmetric = ring_result[static_cast<std::size_t>(ring.index(i, j))];
            const double reference = line_result[static_cast<std::size_t>(i)];
            worst = std::max(worst, std::abs(axisymmetric - reference));
        }
    }
    science_test::report("axisymmetric vs 1D radial", worst);
    SCIENCE_REQUIRE(worst < 1.0e-12,
                    "an axially uniform axisymmetric solve must match the 1D radial one");

    // The ring volume must be the true annulus, not a rectangle.
    double volume = 0.0;
    for (int j = 0; j <= axial_cells; ++j) {
        for (int i = 0; i <= radial_cells; ++i) {
            volume += ring.controlVolume(i) * ring.axialHeight(j);
        }
    }
    SCIENCE_REQUIRE_NEAR(volume, radius * radius / 2.0 * height, 1.0e-18, 1.0e-13,
                         "axisymmetric domain measure");

    // And a 2D spherical mesh is (r, theta), which is not this operator.
    bool refused = false;
    try {
        const StructuredMesh invalid(4, 4, 0.0, 1.0, 0.0, 1.0, Geometry::SPHERICAL);
        (void)invalid.nx();
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    SCIENCE_REQUIRE(refused, "a 2D spherical mesh must be refused");
}

void curvedMeshesAreRefusedWhereTheyWouldBeIgnored() {
    const StructuredMesh ball(16, 0.0, 1.0, Geometry::SPHERICAL);

    bool refused = false;
    try {
        requireCartesian(ball, "test solver");
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    SCIENCE_REQUIRE(refused, "slab-only solvers must refuse a curved mesh");

    bool rejected_negative = false;
    try {
        const StructuredMesh invalid(8, -1.0, 1.0, Geometry::SPHERICAL);
        (void)invalid.nx();
    } catch (const std::invalid_argument&) {
        rejected_negative = true;
    }
    SCIENCE_REQUIRE(rejected_negative, "a curved mesh must reject a negative inner radius");

    // A Cartesian mesh must be completely unaffected by any of this.
    const StructuredMesh slab(10, 0.0, 1.0);
    SCIENCE_REQUIRE(!slab.isRadial(), "the default geometry must remain Cartesian");
    SCIENCE_REQUIRE_NEAR(slab.controlVolume(5), 0.1, 1.0e-15, 0.0, "interior slab control volume");
    SCIENCE_REQUIRE_NEAR(slab.controlVolume(0), 0.05, 1.0e-15, 0.0, "boundary slab control volume");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "radial_geometry",
        {
            {"control volumes are exact", controlVolumesAreExact},
            {"sealed radial domains conserve", sealedRadialDomainsConserve},
            {"curved operator annihilates a constant", curvedOperatorAnnihilatesAConstant},
            {"spherical transient is second order", sphericalTransientIsSecondOrder},
            {"axisymmetric reduces to the radial problem", axisymmetricReducesToTheRadialProblem},
            {"curved meshes are refused where ignored",
             curvedMeshesAreRefusedWhereTheyWouldBeIgnored},
        });
}
