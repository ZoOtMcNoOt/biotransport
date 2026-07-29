#include "../test_support/science_test.hpp"
#include <algorithm>
#include <array>
#include <biotransport/core/mesh/cylindrical_mesh.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

bool near(double actual, double expected, double tolerance = 1e-11) {
    return std::abs(actual - expected) <=
           tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

template <typename Exception, typename Callable>
void expectThrows(Callable callable) {
    bool threw = false;
    try {
        callable();
    } catch (const Exception&) {
        threw = true;
    }
    SCIENCE_REQUIRE(threw, "the invalid cylindrical operation must throw");
}

void testValidationAndIndexing() {
    using biotransport::CylindricalMesh;

    expectThrows<std::invalid_argument>([] { CylindricalMesh mesh(0, 0.0, 1.0); });
    expectThrows<std::invalid_argument>([] { CylindricalMesh mesh(4, -0.1, 1.0); });
    expectThrows<std::invalid_argument>(
        [] { CylindricalMesh mesh(4, 8, 3, 0.1, 1.0, 0.0, kPi, 0.0, 1.0); });
    expectThrows<std::invalid_argument>(
        [] { CylindricalMesh mesh(4, 2, 3, 0.1, 1.0, 0.0, 2.0 * kPi, 0.0, 1.0); });
    expectThrows<std::invalid_argument>(
        [] { CylindricalMesh mesh(4, 0.0, std::numeric_limits<double>::infinity()); });

    CylindricalMesh axisymmetric(3, 4, 0.0, 1.0, -2.0, 2.0);
    SCIENCE_REQUIRE(axisymmetric.numNodes() == 20, "axisymmetric node count");
    SCIENCE_REQUIRE(axisymmetric.numCells() == 12, "axisymmetric cell count");
    SCIENCE_REQUIRE(axisymmetric.index(2, 3) == axisymmetric.index(2, 0, 3),
                    "axisymmetric indexing overloads must agree");
    expectThrows<std::invalid_argument>([&] { axisymmetric.index(2, 1, 2); });
    expectThrows<std::out_of_range>([&] { axisymmetric.r(4); });

    for (int linear = 0; linear < axisymmetric.numNodes(); ++linear) {
        int i = -1;
        int j = -1;
        int k = -1;
        axisymmetric.ijk(linear, i, j, k);
        SCIENCE_REQUIRE(j == 0, "axisymmetric inverse index must have zero theta index");
        SCIENCE_REQUIRE(axisymmetric.index(i, 0, k) == linear, "axisymmetric index round trip");
    }
    expectThrows<std::out_of_range>([&] {
        int i = 0;
        int j = 0;
        int k = 0;
        axisymmetric.ijk(axisymmetric.numNodes(), i, j, k);
    });

    CylindricalMesh full(2, 12, 3, 0.25, 1.0, -kPi, kPi, -1.0, 2.0);
    SCIENCE_REQUIRE(full.thetaPeriodic(), "full cylindrical theta coordinate must be periodic");
    SCIENCE_REQUIRE(full.thetaNodeCount() == 12, "periodic theta node count");
    SCIENCE_REQUIRE(full.numNodes() == 3 * 12 * 4, "full cylindrical node count");
    SCIENCE_REQUIRE(full.numCells() == 2 * 12 * 3, "full cylindrical cell count");
    SCIENCE_REQUIRE(near(full.theta(0), -kPi), "first periodic theta coordinate");
    SCIENCE_REQUIRE(near(full.theta(11), -kPi + 11.0 * 2.0 * kPi / 12.0),
                    "last periodic theta coordinate must not duplicate the endpoint");
    expectThrows<std::out_of_range>([&] { full.theta(12); });

    for (int linear = 0; linear < full.numNodes(); ++linear) {
        int i = -1;
        int j = -1;
        int k = -1;
        full.ijk(linear, i, j, k);
        SCIENCE_REQUIRE(full.index(i, j, k) == linear, "full cylindrical index round trip");
    }
}

void testExactNodalMeasures() {
    using biotransport::CylindricalMesh;

    CylindricalMesh radial(17, 0.0, 2.5);
    double area = 0.0;
    for (int i = 0; i <= radial.nr(); ++i) {
        area += radial.cellArea(i);
        SCIENCE_REQUIRE(near(radial.cellArea(i), radial.cellVolume(i)),
                        "radial unit-depth volume must equal annular area");
    }
    SCIENCE_REQUIRE(near(area, kPi * 2.5 * 2.5), "radial nodal areas integrate a disk exactly");
    SCIENCE_REQUIRE(near(area, radial.crossSectionArea()),
                    "radial nodal areas recover reported cross-section area");

    CylindricalMesh annularRz(9, 11, 0.4, 1.7, -0.3, 2.2);
    double volume = 0.0;
    for (int k = 0; k <= annularRz.nz(); ++k) {
        for (int i = 0; i <= annularRz.nr(); ++i) {
            volume += annularRz.cellVolume(i, 0, k);
        }
    }
    const double exactVolume = kPi * (1.7 * 1.7 - 0.4 * 0.4) * 2.5;
    SCIENCE_REQUIRE(near(volume, exactVolume),
                    "axisymmetric nodal volumes integrate an annular cylinder exactly");

    CylindricalMesh full(7, 16, 8, 0.2, 1.3, 0.4, 0.4 + 2.0 * kPi, -0.5, 1.5);
    volume = 0.0;
    for (int k = 0; k <= full.nz(); ++k) {
        for (int j = 0; j < full.ntheta(); ++j) {
            for (int i = 0; i <= full.nr(); ++i) {
                volume += full.cellVolume(i, j, k);
            }
        }
    }
    SCIENCE_REQUIRE(near(volume, kPi * (1.3 * 1.3 - 0.2 * 0.2) * 2.0, 2e-12),
                    "full cylindrical nodal volumes integrate an annular cylinder");

    CylindricalMesh fullIncludingAxis(5, 12, 6, 0.0, 0.8, 0.0, 2.0 * kPi, -0.2, 0.9);
    volume = 0.0;
    for (int k = 0; k <= fullIncludingAxis.nz(); ++k) {
        for (int j = 0; j < fullIncludingAxis.ntheta(); ++j) {
            for (int i = 0; i <= fullIncludingAxis.nr(); ++i) {
                volume += fullIncludingAxis.cellVolume(i, j, k);
            }
        }
    }
    SCIENCE_REQUIRE(near(volume, kPi * 0.8 * 0.8 * 1.1, 2e-12),
                    "full cylindrical nodal volumes integrate a cylinder including the axis");
}

void testAxisymmetricPolynomialOperators() {
    using biotransport::CylindricalMesh;

    CylindricalMesh radial(12, 0.0, 2.0);
    std::vector<double> radialField(static_cast<std::size_t>(radial.numNodes()));
    for (int i = 0; i <= radial.nr(); ++i) {
        radialField[static_cast<std::size_t>(i)] = radial.r(i) * radial.r(i);
    }
    const auto radialGradient = radial.gradientR(radialField);
    const auto radialLaplacian = radial.laplacian(radialField);
    for (int i = 0; i <= radial.nr(); ++i) {
        SCIENCE_REQUIRE(near(radialGradient[static_cast<std::size_t>(i)], 2.0 * radial.r(i)),
                        "radial gradient of r squared");
        SCIENCE_REQUIRE(near(radialLaplacian[static_cast<std::size_t>(i)], 4.0),
                        "radial cylindrical Laplacian of r squared");
    }

    CylindricalMesh mesh(10, 9, 0.0, 2.0, -1.0, 2.0);
    std::vector<double> phi(static_cast<std::size_t>(mesh.numNodes()));
    std::vector<double> vr(static_cast<std::size_t>(mesh.numNodes()));
    std::vector<double> vz(static_cast<std::size_t>(mesh.numNodes()));
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int i = 0; i <= mesh.nr(); ++i) {
            const int idx = mesh.index(i, 0, k);
            phi[static_cast<std::size_t>(idx)] =
                mesh.r(i) * mesh.r(i) + 3.0 * mesh.z(k) * mesh.z(k);
            vr[static_cast<std::size_t>(idx)] = 2.0 * mesh.r(i);
            vz[static_cast<std::size_t>(idx)] = -0.5 * mesh.z(k);
        }
    }

    const auto gradientR = mesh.gradientR(phi);
    const auto gradientZ = mesh.gradientZ(phi);
    const auto laplacian = mesh.laplacian(phi);
    const auto divergence = mesh.divergence(vr, vz);
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int i = 0; i <= mesh.nr(); ++i) {
            const std::size_t idx = static_cast<std::size_t>(mesh.index(i, 0, k));
            SCIENCE_REQUIRE(near(gradientR[idx], 2.0 * mesh.r(i)),
                            "axisymmetric radial polynomial gradient");
            SCIENCE_REQUIRE(near(gradientZ[idx], 6.0 * mesh.z(k)),
                            "axisymmetric axial polynomial gradient");
            SCIENCE_REQUIRE(near(laplacian[idx], 10.0), "axisymmetric polynomial Laplacian");
            SCIENCE_REQUIRE(near(divergence[idx], 3.5), "axisymmetric polynomial divergence");
        }
    }

    vr[0] = 1.0;
    expectThrows<std::domain_error>([&] { mesh.divergence(vr, vz); });
    expectThrows<std::invalid_argument>([&] { mesh.laplacian(std::vector<double>(3)); });
}

void testPeriodicThreeDimensionalOperators() {
    using biotransport::CylindricalMesh;

    CylindricalMesh mesh(6, 256, 6, 0.75, 1.5, -kPi, kPi, -0.4, 0.8);
    std::vector<double> phi(static_cast<std::size_t>(mesh.numNodes()));
    std::vector<double> vr(static_cast<std::size_t>(mesh.numNodes()));
    std::vector<double> vtheta(static_cast<std::size_t>(mesh.numNodes()));
    std::vector<double> vz(static_cast<std::size_t>(mesh.numNodes()));
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ntheta(); ++j) {
            for (int i = 0; i <= mesh.nr(); ++i) {
                const std::size_t idx = static_cast<std::size_t>(mesh.index(i, j, k));
                phi[idx] = mesh.r(i) * mesh.r(i) + mesh.z(k) * mesh.z(k) + std::cos(mesh.theta(j));
                vr[idx] = mesh.r(i);
                vtheta[idx] = std::sin(mesh.theta(j));
                vz[idx] = mesh.z(k);
            }
        }
    }

    const auto gradientTheta = mesh.gradientTheta(phi);
    const auto laplacian = mesh.laplacian(phi);
    const auto divergence = mesh.divergence(vr, vtheta, vz);
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ntheta(); ++j) {
            for (int i = 0; i <= mesh.nr(); ++i) {
                const std::size_t idx = static_cast<std::size_t>(mesh.index(i, j, k));
                const double radius = mesh.r(i);
                SCIENCE_REQUIRE(near(gradientTheta[idx], -std::sin(mesh.theta(j)) / radius, 2e-4),
                                "periodic azimuthal gradient");
                SCIENCE_REQUIRE(
                    near(laplacian[idx], 6.0 - std::cos(mesh.theta(j)) / (radius * radius), 3e-4),
                    "full cylindrical manufactured Laplacian");
                SCIENCE_REQUIRE(near(divergence[idx], 3.0 + std::cos(mesh.theta(j)) / radius, 2e-4),
                                "full cylindrical manufactured divergence");
            }
        }
    }

    CylindricalMesh includesAxis(4, 16, 4, 0.0, 1.0, 0.0, 2.0 * kPi, 0.0, 1.0);
    std::vector<double> axisField(static_cast<std::size_t>(includesAxis.numNodes()), 1.0);
    expectThrows<std::domain_error>([&] { includesAxis.laplacian(axisField); });
}

double radialLaplacianRelativeError(int cells) {
    using biotransport::CylindricalMesh;

    CylindricalMesh mesh(cells, 0.5, 1.5);
    std::vector<double> phi(static_cast<std::size_t>(mesh.numNodes()));
    for (int i = 0; i <= cells; ++i) {
        phi[static_cast<std::size_t>(i)] = std::exp(mesh.r(i));
    }
    const auto laplacian = mesh.laplacian(phi);

    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (int i = 0; i <= cells; ++i) {
        // For phi(r)=exp(r), (1/r)d_r(r d_r phi)=exp(r)(1+1/r).
        const double exact = std::exp(mesh.r(i)) * (1.0 + 1.0 / mesh.r(i));
        const double difference = laplacian[static_cast<std::size_t>(i)] - exact;
        squared_error += difference * difference;
        squared_reference += exact * exact;
    }
    return std::sqrt(squared_error / squared_reference);
}

struct AngularOperatorErrors {
    double gradient = 0.0;
    double laplacian = 0.0;
};

AngularOperatorErrors angularOperatorRelativeErrors(int angular_cells) {
    using biotransport::CylindricalMesh;

    constexpr int azimuthal_mode = 3;
    CylindricalMesh mesh(4, angular_cells, 2, 0.8, 1.4, 0.0, 2.0 * kPi, -0.2, 0.2);
    std::vector<double> phi(static_cast<std::size_t>(mesh.numNodes()));
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ntheta(); ++j) {
            for (int i = 0; i <= mesh.nr(); ++i) {
                phi[static_cast<std::size_t>(mesh.index(i, j, k))] =
                    std::cos(static_cast<double>(azimuthal_mode) * mesh.theta(j));
            }
        }
    }

    const auto gradient = mesh.gradientTheta(phi);
    const auto laplacian = mesh.laplacian(phi);
    double gradient_error = 0.0;
    double gradient_reference = 0.0;
    double laplacian_error = 0.0;
    double laplacian_reference = 0.0;
    for (int k = 0; k <= mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ntheta(); ++j) {
            for (int i = 0; i <= mesh.nr(); ++i) {
                const std::size_t node = static_cast<std::size_t>(mesh.index(i, j, k));
                const double radius = mesh.r(i);
                const double phase = static_cast<double>(azimuthal_mode) * mesh.theta(j);
                const double exact_gradient =
                    -static_cast<double>(azimuthal_mode) * std::sin(phase) / radius;
                const double exact_laplacian =
                    -static_cast<double>(azimuthal_mode * azimuthal_mode) * std::cos(phase) /
                    (radius * radius);
                gradient_error += std::pow(gradient[node] - exact_gradient, 2);
                gradient_reference += exact_gradient * exact_gradient;
                laplacian_error += std::pow(laplacian[node] - exact_laplacian, 2);
                laplacian_reference += exact_laplacian * exact_laplacian;
            }
        }
    }
    return {std::sqrt(gradient_error / gradient_reference),
            std::sqrt(laplacian_error / laplacian_reference)};
}

double observedOrder(double coarse_error, double fine_error) {
    return std::log(coarse_error / fine_error) / std::log(2.0);
}

void testCylindricalOperatorsConvergeAtSecondOrder() {
    const std::array<int, 3> radial_cells{16, 32, 64};
    std::array<double, 3> radial_errors{};
    for (std::size_t level = 0; level < radial_cells.size(); ++level) {
        radial_errors[level] = radialLaplacianRelativeError(radial_cells[level]);
    }
    const double radial_coarse_order = observedOrder(radial_errors[0], radial_errors[1]);
    const double radial_fine_order = observedOrder(radial_errors[1], radial_errors[2]);
    std::cout << "  radial Laplacian orders: " << radial_coarse_order << ", " << radial_fine_order
              << '\n';
    SCIENCE_REQUIRE(radial_errors[2] < radial_errors[1] && radial_errors[1] < radial_errors[0],
                    "radial Laplacian error must decrease under refinement");
    SCIENCE_REQUIRE(radial_coarse_order > 1.8 && radial_fine_order > 1.8,
                    "radial Laplacian must approach second order");

    const std::array<int, 3> angular_cells{32, 64, 128};
    std::array<AngularOperatorErrors, 3> angular_errors{};
    for (std::size_t level = 0; level < angular_cells.size(); ++level) {
        angular_errors[level] = angularOperatorRelativeErrors(angular_cells[level]);
    }
    const double gradient_coarse_order =
        observedOrder(angular_errors[0].gradient, angular_errors[1].gradient);
    const double gradient_fine_order =
        observedOrder(angular_errors[1].gradient, angular_errors[2].gradient);
    const double laplacian_coarse_order =
        observedOrder(angular_errors[0].laplacian, angular_errors[1].laplacian);
    const double laplacian_fine_order =
        observedOrder(angular_errors[1].laplacian, angular_errors[2].laplacian);
    std::cout << "  angular gradient orders: " << gradient_coarse_order << ", "
              << gradient_fine_order << '\n';
    std::cout << "  angular Laplacian orders: " << laplacian_coarse_order << ", "
              << laplacian_fine_order << '\n';
    SCIENCE_REQUIRE(gradient_coarse_order > 1.8 && gradient_fine_order > 1.8,
                    "periodic azimuthal gradient must approach second order");
    SCIENCE_REQUIRE(laplacian_coarse_order > 1.8 && laplacian_fine_order > 1.8,
                    "periodic azimuthal Laplacian must approach second order");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "cylindrical mesh geometry and operators",
        {{"validation and indexing", testValidationAndIndexing},
         {"exact nodal measures", testExactNodalMeasures},
         {"axisymmetric polynomial operators", testAxisymmetricPolynomialOperators},
         {"periodic three-dimensional operators", testPeriodicThreeDimensionalOperators},
         {"cylindrical operators converge at second order",
          testCylindricalOperatorsConvergeAtSecondOrder}});
}
