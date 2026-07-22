#include <algorithm>
#include <biotransport/core/mesh/cylindrical_mesh.hpp>
#include <cassert>
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
    assert(threw);
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
    assert(axisymmetric.numNodes() == 20);
    assert(axisymmetric.numCells() == 12);
    assert(axisymmetric.index(2, 3) == axisymmetric.index(2, 0, 3));
    expectThrows<std::invalid_argument>([&] { axisymmetric.index(2, 1, 2); });
    expectThrows<std::out_of_range>([&] { axisymmetric.r(4); });

    for (int linear = 0; linear < axisymmetric.numNodes(); ++linear) {
        int i = -1;
        int j = -1;
        int k = -1;
        axisymmetric.ijk(linear, i, j, k);
        assert(j == 0);
        assert(axisymmetric.index(i, 0, k) == linear);
    }
    expectThrows<std::out_of_range>([&] {
        int i = 0;
        int j = 0;
        int k = 0;
        axisymmetric.ijk(axisymmetric.numNodes(), i, j, k);
    });

    CylindricalMesh full(2, 12, 3, 0.25, 1.0, -kPi, kPi, -1.0, 2.0);
    assert(full.thetaPeriodic());
    assert(full.thetaNodeCount() == 12);
    assert(full.numNodes() == 3 * 12 * 4);
    assert(full.numCells() == 2 * 12 * 3);
    assert(near(full.theta(0), -kPi));
    assert(near(full.theta(11), -kPi + 11.0 * 2.0 * kPi / 12.0));
    expectThrows<std::out_of_range>([&] { full.theta(12); });

    for (int linear = 0; linear < full.numNodes(); ++linear) {
        int i = -1;
        int j = -1;
        int k = -1;
        full.ijk(linear, i, j, k);
        assert(full.index(i, j, k) == linear);
    }
}

void testExactNodalMeasures() {
    using biotransport::CylindricalMesh;

    CylindricalMesh radial(17, 0.0, 2.5);
    double area = 0.0;
    for (int i = 0; i <= radial.nr(); ++i) {
        area += radial.cellArea(i);
        assert(near(radial.cellArea(i), radial.cellVolume(i)));
    }
    assert(near(area, kPi * 2.5 * 2.5));
    assert(near(area, radial.crossSectionArea()));

    CylindricalMesh annularRz(9, 11, 0.4, 1.7, -0.3, 2.2);
    double volume = 0.0;
    for (int k = 0; k <= annularRz.nz(); ++k) {
        for (int i = 0; i <= annularRz.nr(); ++i) {
            volume += annularRz.cellVolume(i, 0, k);
        }
    }
    const double exactVolume = kPi * (1.7 * 1.7 - 0.4 * 0.4) * 2.5;
    assert(near(volume, exactVolume));

    CylindricalMesh full(7, 16, 8, 0.2, 1.3, 0.4, 0.4 + 2.0 * kPi, -0.5, 1.5);
    volume = 0.0;
    for (int k = 0; k <= full.nz(); ++k) {
        for (int j = 0; j < full.ntheta(); ++j) {
            for (int i = 0; i <= full.nr(); ++i) {
                volume += full.cellVolume(i, j, k);
            }
        }
    }
    assert(near(volume, kPi * (1.3 * 1.3 - 0.2 * 0.2) * 2.0, 2e-12));

    CylindricalMesh fullIncludingAxis(5, 12, 6, 0.0, 0.8, 0.0, 2.0 * kPi, -0.2, 0.9);
    volume = 0.0;
    for (int k = 0; k <= fullIncludingAxis.nz(); ++k) {
        for (int j = 0; j < fullIncludingAxis.ntheta(); ++j) {
            for (int i = 0; i <= fullIncludingAxis.nr(); ++i) {
                volume += fullIncludingAxis.cellVolume(i, j, k);
            }
        }
    }
    assert(near(volume, kPi * 0.8 * 0.8 * 1.1, 2e-12));
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
        assert(near(radialGradient[static_cast<std::size_t>(i)], 2.0 * radial.r(i)));
        assert(near(radialLaplacian[static_cast<std::size_t>(i)], 4.0));
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
            assert(near(gradientR[idx], 2.0 * mesh.r(i)));
            assert(near(gradientZ[idx], 6.0 * mesh.z(k)));
            assert(near(laplacian[idx], 10.0));
            assert(near(divergence[idx], 3.5));
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
                assert(near(gradientTheta[idx], -std::sin(mesh.theta(j)) / radius, 2e-4));
                assert(
                    near(laplacian[idx], 6.0 - std::cos(mesh.theta(j)) / (radius * radius), 3e-4));
                assert(near(divergence[idx], 3.0 + std::cos(mesh.theta(j)) / radius, 2e-4));
            }
        }
    }

    CylindricalMesh includesAxis(4, 16, 4, 0.0, 1.0, 0.0, 2.0 * kPi, 0.0, 1.0);
    std::vector<double> axisField(static_cast<std::size_t>(includesAxis.numNodes()), 1.0);
    expectThrows<std::domain_error>([&] { includesAxis.laplacian(axisField); });
}

}  // namespace

int main() {
    testValidationAndIndexing();
    testExactNodalMeasures();
    testAxisymmetricPolynomialOperators();
    testPeriodicThreeDimensionalOperators();
    std::cout << "All cylindrical science tests passed.\n";
    return 0;
}
