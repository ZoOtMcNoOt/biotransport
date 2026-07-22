/**
 * @file cylindrical_mesh.cpp
 * @brief Implementation of metric-aware cylindrical meshes.
 */

#include <algorithm>
#include <biotransport/core/mesh/cylindrical_mesh.hpp>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace biotransport {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

void requireFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

int checkedProduct(std::initializer_list<long long> factors, const char* quantity) {
    long long product = 1;
    for (const long long factor : factors) {
        if (factor <= 0 || product > std::numeric_limits<int>::max() / factor) {
            throw std::invalid_argument(std::string(quantity) + " exceeds the supported int range");
        }
        product *= factor;
    }
    return static_cast<int>(product);
}

void validateField(const std::vector<double>& field, int expected, const char* name) {
    if (field.size() != static_cast<std::size_t>(expected)) {
        throw std::invalid_argument(std::string(name) + " must contain exactly " +
                                    std::to_string(expected) + " nodal values");
    }
    if (!std::all_of(field.begin(), field.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
    }
}

template <typename ValueAt>
double firstDerivative(int q, int intervals, double spacing, ValueAt valueAt) {
    if (intervals == 1) {
        return (valueAt(1) - valueAt(0)) / spacing;
    }
    if (q == 0) {
        return (-3.0 * valueAt(0) + 4.0 * valueAt(1) - valueAt(2)) / (2.0 * spacing);
    }
    if (q == intervals) {
        return (3.0 * valueAt(intervals) - 4.0 * valueAt(intervals - 1) + valueAt(intervals - 2)) /
               (2.0 * spacing);
    }
    return (valueAt(q + 1) - valueAt(q - 1)) / (2.0 * spacing);
}

template <typename ValueAt>
double secondDerivative(int q, int intervals, double spacing, ValueAt valueAt,
                        const char* direction) {
    if (intervals < 2) {
        throw std::domain_error(std::string("at least two ") + direction +
                                " cells are required for a second derivative");
    }
    const double spacingSquared = spacing * spacing;
    if (intervals == 2) {
        return (valueAt(0) - 2.0 * valueAt(1) + valueAt(2)) / spacingSquared;
    }
    if (q == 0) {
        return (2.0 * valueAt(0) - 5.0 * valueAt(1) + 4.0 * valueAt(2) - valueAt(3)) /
               spacingSquared;
    }
    if (q == intervals) {
        return (2.0 * valueAt(intervals) - 5.0 * valueAt(intervals - 1) +
                4.0 * valueAt(intervals - 2) - valueAt(intervals - 3)) /
               spacingSquared;
    }
    return (valueAt(q + 1) - 2.0 * valueAt(q) + valueAt(q - 1)) / spacingSquared;
}

}  // namespace

CylindricalMesh::CylindricalMesh(int nr, double rmin, double rmax)
    : type_(CylindricalMeshType::RADIAL_R),
      nr_(nr),
      ntheta_(0),
      nz_(0),
      rmin_(rmin),
      rmax_(rmax),
      thetamin_(0.0),
      thetamax_(0.0),
      zmin_(0.0),
      zmax_(0.0),
      dr_(0.0),
      dtheta_(0.0),
      dz_(0.0),
      num_nodes_(0) {
    requireFinite(rmin, "rmin");
    requireFinite(rmax, "rmax");
    if (nr < 1) {
        throw std::invalid_argument("nr must be at least 1");
    }
    if (rmin < 0.0) {
        throw std::invalid_argument("rmin must be non-negative in cylindrical coordinates");
    }
    if (!(rmax > rmin)) {
        throw std::invalid_argument("rmax must be greater than rmin");
    }
    num_nodes_ = checkedProduct({static_cast<long long>(nr) + 1}, "node count");
    checkedProduct({nr}, "cell count");
    dr_ = (rmax - rmin) / static_cast<double>(nr);
}

CylindricalMesh::CylindricalMesh(int nr, int nz, double rmin, double rmax, double zmin, double zmax)
    : type_(CylindricalMeshType::AXISYMMETRIC_RZ),
      nr_(nr),
      ntheta_(0),
      nz_(nz),
      rmin_(rmin),
      rmax_(rmax),
      thetamin_(0.0),
      thetamax_(kTwoPi),
      zmin_(zmin),
      zmax_(zmax),
      dr_(0.0),
      dtheta_(0.0),
      dz_(0.0),
      num_nodes_(0) {
    requireFinite(rmin, "rmin");
    requireFinite(rmax, "rmax");
    requireFinite(zmin, "zmin");
    requireFinite(zmax, "zmax");
    if (nr < 1 || nz < 1) {
        throw std::invalid_argument("nr and nz must each be at least 1");
    }
    if (rmin < 0.0) {
        throw std::invalid_argument("rmin must be non-negative in cylindrical coordinates");
    }
    if (!(rmax > rmin)) {
        throw std::invalid_argument("rmax must be greater than rmin");
    }
    if (!(zmax > zmin)) {
        throw std::invalid_argument("zmax must be greater than zmin");
    }
    num_nodes_ = checkedProduct({static_cast<long long>(nr) + 1, static_cast<long long>(nz) + 1},
                                "node count");
    checkedProduct({nr, nz}, "cell count");
    dr_ = (rmax - rmin) / static_cast<double>(nr);
    dz_ = (zmax - zmin) / static_cast<double>(nz);
}

CylindricalMesh::CylindricalMesh(int nr, int ntheta, int nz, double rmin, double rmax,
                                 double thetamin, double thetamax, double zmin, double zmax)
    : type_(CylindricalMeshType::FULL_3D),
      nr_(nr),
      ntheta_(ntheta),
      nz_(nz),
      rmin_(rmin),
      rmax_(rmax),
      thetamin_(thetamin),
      thetamax_(thetamax),
      zmin_(zmin),
      zmax_(zmax),
      dr_(0.0),
      dtheta_(0.0),
      dz_(0.0),
      num_nodes_(0) {
    requireFinite(rmin, "rmin");
    requireFinite(rmax, "rmax");
    requireFinite(thetamin, "thetamin");
    requireFinite(thetamax, "thetamax");
    requireFinite(zmin, "zmin");
    requireFinite(zmax, "zmax");
    if (nr < 1 || nz < 1) {
        throw std::invalid_argument("nr and nz must each be at least 1");
    }
    if (ntheta < 3) {
        throw std::invalid_argument("ntheta must be at least 3 for a periodic angular mesh");
    }
    if (rmin < 0.0) {
        throw std::invalid_argument("rmin must be non-negative in cylindrical coordinates");
    }
    if (!(rmax > rmin)) {
        throw std::invalid_argument("rmax must be greater than rmin");
    }
    if (!(zmax > zmin)) {
        throw std::invalid_argument("zmax must be greater than zmin");
    }
    const double thetaSpan = thetamax - thetamin;
    const double thetaTolerance = 128.0 * std::numeric_limits<double>::epsilon() * kTwoPi;
    if (std::abs(thetaSpan - kTwoPi) > thetaTolerance) {
        throw std::invalid_argument(
            "full 3D cylindrical meshes must span exactly one complete turn (2*pi)");
    }
    num_nodes_ = checkedProduct(
        {static_cast<long long>(nr) + 1, ntheta, static_cast<long long>(nz) + 1}, "node count");
    checkedProduct({nr, ntheta, nz}, "cell count");
    dr_ = (rmax - rmin) / static_cast<double>(nr);
    dtheta_ = thetaSpan / static_cast<double>(ntheta);
    dz_ = (zmax - zmin) / static_cast<double>(nz);
}

int CylindricalMesh::numCells() const noexcept {
    if (isRadial()) {
        return nr_;
    }
    if (isAxisymmetric()) {
        return nr_ * nz_;
    }
    return nr_ * ntheta_ * nz_;
}

double CylindricalMesh::r(int i) const {
    if (i < 0 || i > nr_) {
        throw std::out_of_range("radial index must lie in [0, nr]");
    }
    return i == nr_ ? rmax_ : rmin_ + static_cast<double>(i) * dr_;
}

double CylindricalMesh::theta(int j) const {
    if (!is3D()) {
        if (j != 0) {
            throw std::out_of_range("radial and axisymmetric meshes have only theta index 0");
        }
        return 0.0;
    }
    if (j < 0 || j >= ntheta_) {
        throw std::out_of_range(
            "theta index must lie in [0, ntheta); the periodic endpoint is not duplicated");
    }
    return thetamin_ + static_cast<double>(j) * dtheta_;
}

double CylindricalMesh::z(int k) const {
    if (isRadial()) {
        if (k != 0) {
            throw std::out_of_range("radial meshes have only axial index 0");
        }
        return 0.0;
    }
    if (k < 0 || k > nz_) {
        throw std::out_of_range("axial index must lie in [0, nz]");
    }
    return k == nz_ ? zmax_ : zmin_ + static_cast<double>(k) * dz_;
}

int CylindricalMesh::index(int i, int j, int k) const {
    if (i < 0 || i > nr_) {
        throw std::out_of_range("radial index must lie in [0, nr]");
    }
    if (isRadial()) {
        if (j != 0 || k != 0) {
            throw std::out_of_range("radial meshes have no theta or axial index");
        }
        return i;
    }
    if (isAxisymmetric()) {
        if (j != 0 && k != 0) {
            throw std::invalid_argument(
                "for an axisymmetric mesh use index(i, k) or index(i, 0, k), not both");
        }
        const int axialIndex = j != 0 ? j : k;
        if (axialIndex < 0 || axialIndex > nz_) {
            throw std::out_of_range("axial index must lie in [0, nz]");
        }
        return i + axialIndex * (nr_ + 1);
    }
    if (j < 0 || j >= ntheta_) {
        throw std::out_of_range("theta index must lie in [0, ntheta)");
    }
    if (k < 0 || k > nz_) {
        throw std::out_of_range("axial index must lie in [0, nz]");
    }
    return i + j * (nr_ + 1) + k * (nr_ + 1) * ntheta_;
}

void CylindricalMesh::ijk(int linearIndex, int& i, int& j, int& k) const {
    if (linearIndex < 0 || linearIndex >= num_nodes_) {
        throw std::out_of_range("linear index must lie in [0, numNodes)");
    }
    if (isRadial()) {
        i = linearIndex;
        j = 0;
        k = 0;
        return;
    }
    if (isAxisymmetric()) {
        i = linearIndex % (nr_ + 1);
        j = 0;
        k = linearIndex / (nr_ + 1);
        return;
    }
    const int planeSize = (nr_ + 1) * ntheta_;
    k = linearIndex / planeSize;
    const int withinPlane = linearIndex % planeSize;
    j = withinPlane / (nr_ + 1);
    i = withinPlane % (nr_ + 1);
}

double CylindricalMesh::x(int i, int j) const {
    return r(i) * std::cos(theta(j));
}

double CylindricalMesh::y(int i, int j) const {
    return r(i) * std::sin(theta(j));
}

double CylindricalMesh::cellArea(int i) const {
    const double radius = r(i);
    const double lower = std::max(rmin_, radius - 0.5 * dr_);
    const double upper = std::min(rmax_, radius + 0.5 * dr_);
    return kPi * (upper * upper - lower * lower);
}

double CylindricalMesh::cellVolume(int i, int j, int k) const {
    if (isRadial()) {
        index(i, j, k);
        return cellArea(i);
    }

    int axialIndex = k;
    if (isAxisymmetric()) {
        if (j != 0 && k != 0) {
            throw std::invalid_argument(
                "for an axisymmetric mesh use cellVolume(i, k) or cellVolume(i, 0, k)");
        }
        axialIndex = j != 0 ? j : k;
        index(i, 0, axialIndex);
    } else {
        index(i, j, k);
    }

    const double axialWidth = (axialIndex == 0 || axialIndex == nz_) ? 0.5 * dz_ : dz_;
    if (isAxisymmetric()) {
        return cellArea(i) * axialWidth;
    }
    return cellArea(i) * (dtheta_ / kTwoPi) * axialWidth;
}

double CylindricalMesh::crossSectionArea() const noexcept {
    return kPi * (rmax_ * rmax_ - rmin_ * rmin_);
}

std::vector<double> CylindricalMesh::gradientR(const std::vector<double>& phi) const {
    validateField(phi, num_nodes_, "phi");
    if (is3D() && hasAxisSingularity()) {
        throw std::domain_error("full 3D cylindrical differential operators require rmin > 0");
    }

    std::vector<double> gradient(static_cast<std::size_t>(num_nodes_), 0.0);
    const int thetaCount = is3D() ? ntheta_ : 1;
    const int axialCount = isRadial() ? 1 : nz_ + 1;
    const auto linear = [this](int i, int j, int k) {
        if (isRadial()) {
            return i;
        }
        if (isAxisymmetric()) {
            return i + k * (nr_ + 1);
        }
        return i + j * (nr_ + 1) + k * (nr_ + 1) * ntheta_;
    };

    for (int k = 0; k < axialCount; ++k) {
        for (int j = 0; j < thetaCount; ++j) {
            for (int i = 0; i <= nr_; ++i) {
                const auto at = [&](int radialIndex) {
                    return phi[static_cast<std::size_t>(linear(radialIndex, j, k))];
                };
                // A regular axisymmetric scalar is even in r, so its physical
                // radial component is exactly zero at the axis.
                gradient[static_cast<std::size_t>(linear(i, j, k))] =
                    hasAxisSingularity() && i == 0 ? 0.0 : firstDerivative(i, nr_, dr_, at);
            }
        }
    }
    return gradient;
}

std::vector<double> CylindricalMesh::gradientTheta(const std::vector<double>& phi) const {
    validateField(phi, num_nodes_, "phi");
    std::vector<double> gradient(static_cast<std::size_t>(num_nodes_), 0.0);
    if (!is3D()) {
        return gradient;
    }
    if (hasAxisSingularity()) {
        throw std::domain_error("full 3D cylindrical differential operators require rmin > 0");
    }

    for (int k = 0; k <= nz_; ++k) {
        for (int j = 0; j < ntheta_; ++j) {
            const int previous = (j + ntheta_ - 1) % ntheta_;
            const int next = (j + 1) % ntheta_;
            for (int i = 0; i <= nr_; ++i) {
                const int idx = index(i, j, k);
                gradient[static_cast<std::size_t>(idx)] =
                    (phi[static_cast<std::size_t>(index(i, next, k))] -
                     phi[static_cast<std::size_t>(index(i, previous, k))]) /
                    (2.0 * dtheta_ * r(i));
            }
        }
    }
    return gradient;
}

std::vector<double> CylindricalMesh::gradientZ(const std::vector<double>& phi) const {
    validateField(phi, num_nodes_, "phi");
    std::vector<double> gradient(static_cast<std::size_t>(num_nodes_), 0.0);
    if (isRadial()) {
        return gradient;
    }
    if (is3D() && hasAxisSingularity()) {
        throw std::domain_error("full 3D cylindrical differential operators require rmin > 0");
    }

    const int thetaCount = is3D() ? ntheta_ : 1;
    const auto linear = [this](int i, int j, int k) {
        if (isAxisymmetric()) {
            return i + k * (nr_ + 1);
        }
        return i + j * (nr_ + 1) + k * (nr_ + 1) * ntheta_;
    };
    for (int k = 0; k <= nz_; ++k) {
        for (int j = 0; j < thetaCount; ++j) {
            for (int i = 0; i <= nr_; ++i) {
                const auto at = [&](int axialIndex) {
                    return phi[static_cast<std::size_t>(linear(i, j, axialIndex))];
                };
                gradient[static_cast<std::size_t>(linear(i, j, k))] =
                    firstDerivative(k, nz_, dz_, at);
            }
        }
    }
    return gradient;
}

std::vector<double> CylindricalMesh::laplacian(const std::vector<double>& phi) const {
    validateField(phi, num_nodes_, "phi");
    if (is3D() && hasAxisSingularity()) {
        throw std::domain_error("full 3D cylindrical Laplacians require an annulus (rmin > 0)");
    }

    std::vector<double> result(static_cast<std::size_t>(num_nodes_), 0.0);
    const int thetaCount = is3D() ? ntheta_ : 1;
    const int axialCount = isRadial() ? 1 : nz_ + 1;
    const auto linear = [this](int i, int j, int k) {
        if (isRadial()) {
            return i;
        }
        if (isAxisymmetric()) {
            return i + k * (nr_ + 1);
        }
        return i + j * (nr_ + 1) + k * (nr_ + 1) * ntheta_;
    };

    for (int k = 0; k < axialCount; ++k) {
        for (int j = 0; j < thetaCount; ++j) {
            for (int i = 0; i <= nr_; ++i) {
                const auto radialAt = [&](int radialIndex) {
                    return phi[static_cast<std::size_t>(linear(radialIndex, j, k))];
                };
                const double radialSecond = secondDerivative(i, nr_, dr_, radialAt, "radial");
                double value = 0.0;
                if (hasAxisSingularity() && i == 0) {
                    value = 2.0 * radialSecond;
                } else {
                    value = radialSecond + firstDerivative(i, nr_, dr_, radialAt) / r(i);
                }

                if (!isRadial()) {
                    const auto axialAt = [&](int axialIndex) {
                        return phi[static_cast<std::size_t>(linear(i, j, axialIndex))];
                    };
                    value += secondDerivative(k, nz_, dz_, axialAt, "axial");
                }
                if (is3D()) {
                    const int previous = (j + ntheta_ - 1) % ntheta_;
                    const int next = (j + 1) % ntheta_;
                    const double angularSecond =
                        (phi[static_cast<std::size_t>(linear(i, next, k))] -
                         2.0 * phi[static_cast<std::size_t>(linear(i, j, k))] +
                         phi[static_cast<std::size_t>(linear(i, previous, k))]) /
                        (dtheta_ * dtheta_);
                    value += angularSecond / (r(i) * r(i));
                }
                result[static_cast<std::size_t>(linear(i, j, k))] = value;
            }
        }
    }
    return result;
}

std::vector<double> CylindricalMesh::divergence(const std::vector<double>& vr,
                                                const std::vector<double>& vz) const {
    validateField(vr, num_nodes_, "vr");
    validateField(vz, num_nodes_, "vz");
    if (!isAxisymmetric()) {
        throw std::domain_error(
            "the two-component divergence is defined only on axisymmetric (r,z) meshes");
    }

    std::vector<double> result(static_cast<std::size_t>(num_nodes_), 0.0);
    const auto linear = [this](int i, int k) {
        return i + k * (nr_ + 1);
    };
    if (hasAxisSingularity()) {
        for (int k = 0; k <= nz_; ++k) {
            const double scale =
                std::max(1.0, std::abs(vr[static_cast<std::size_t>(linear(1, k))]));
            if (std::abs(vr[static_cast<std::size_t>(linear(0, k))]) >
                64.0 * std::numeric_limits<double>::epsilon() * scale) {
                throw std::domain_error(
                    "a regular axisymmetric radial vector field must satisfy vr=0 at r=0");
            }
        }
    }

    for (int k = 0; k <= nz_; ++k) {
        for (int i = 0; i <= nr_; ++i) {
            const auto axialAt = [&](int axialIndex) {
                return vz[static_cast<std::size_t>(linear(i, axialIndex))];
            };
            const double axialPart = firstDerivative(k, nz_, dz_, axialAt);
            double radialPart = 0.0;
            if (hasAxisSingularity() && i == 0) {
                const auto radialVelocityAt = [&](int radialIndex) {
                    return vr[static_cast<std::size_t>(linear(radialIndex, k))];
                };
                radialPart = 2.0 * firstDerivative(0, nr_, dr_, radialVelocityAt);
            } else {
                const auto radialFluxAt = [&](int radialIndex) {
                    return r(radialIndex) * vr[static_cast<std::size_t>(linear(radialIndex, k))];
                };
                radialPart = firstDerivative(i, nr_, dr_, radialFluxAt) / r(i);
            }
            result[static_cast<std::size_t>(linear(i, k))] = radialPart + axialPart;
        }
    }
    return result;
}

std::vector<double> CylindricalMesh::divergence(const std::vector<double>& vr,
                                                const std::vector<double>& vtheta,
                                                const std::vector<double>& vz) const {
    validateField(vr, num_nodes_, "vr");
    validateField(vtheta, num_nodes_, "vtheta");
    validateField(vz, num_nodes_, "vz");
    if (!is3D()) {
        throw std::domain_error(
            "the three-component divergence is defined only on full 3D cylindrical meshes");
    }
    if (hasAxisSingularity()) {
        throw std::domain_error("full 3D cylindrical differential operators require rmin > 0");
    }

    std::vector<double> result(static_cast<std::size_t>(num_nodes_), 0.0);
    const auto linear = [this](int i, int j, int k) {
        return i + j * (nr_ + 1) + k * (nr_ + 1) * ntheta_;
    };
    for (int k = 0; k <= nz_; ++k) {
        for (int j = 0; j < ntheta_; ++j) {
            const int previous = (j + ntheta_ - 1) % ntheta_;
            const int next = (j + 1) % ntheta_;
            for (int i = 0; i <= nr_; ++i) {
                const auto radialFluxAt = [&](int radialIndex) {
                    return r(radialIndex) * vr[static_cast<std::size_t>(linear(radialIndex, j, k))];
                };
                const auto axialAt = [&](int axialIndex) {
                    return vz[static_cast<std::size_t>(linear(i, j, axialIndex))];
                };
                const double radialPart = firstDerivative(i, nr_, dr_, radialFluxAt) / r(i);
                const double angularPart =
                    (vtheta[static_cast<std::size_t>(linear(i, next, k))] -
                     vtheta[static_cast<std::size_t>(linear(i, previous, k))]) /
                    (2.0 * dtheta_ * r(i));
                const double axialPart = firstDerivative(k, nz_, dz_, axialAt);
                result[static_cast<std::size_t>(linear(i, j, k))] =
                    radialPart + angularPart + axialPart;
            }
        }
    }
    return result;
}

}  // namespace biotransport
