/**
 * @file cylindrical_mesh.cpp
 * @brief Implementation of CylindricalMesh class.
 */

#define _USE_MATH_DEFINES
#include <algorithm>
#include <biotransport/core/mesh/cylindrical_mesh.hpp>
#include <cmath>

namespace biotransport {

// =============================================================================
// Constructors
// =============================================================================

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
      zmax_(0.0) {
    if (nr < 1)
        throw std::invalid_argument("nr must be >= 1");
    if (rmax <= rmin)
        throw std::invalid_argument("rmax must be > rmin");

    dr_ = (rmax - rmin) / nr;
    dtheta_ = 0.0;
    dz_ = 0.0;
    num_nodes_ = nr + 1;
}

CylindricalMesh::CylindricalMesh(int nr, int nz, double rmin, double rmax, double zmin, double zmax)
    : type_(CylindricalMeshType::AXISYMMETRIC_RZ),
      nr_(nr),
      ntheta_(0),
      nz_(nz),
      rmin_(rmin),
      rmax_(rmax),
      thetamin_(0.0),
      thetamax_(2.0 * M_PI),
      zmin_(zmin),
      zmax_(zmax) {
    if (nr < 1)
        throw std::invalid_argument("nr must be >= 1");
    if (nz < 1)
        throw std::invalid_argument("nz must be >= 1");
    if (rmax <= rmin)
        throw std::invalid_argument("rmax must be > rmin");
    if (zmax <= zmin)
        throw std::invalid_argument("zmax must be > zmin");

    dr_ = (rmax - rmin) / nr;
    dtheta_ = 0.0;  // Not used for axisymmetric
    dz_ = (zmax - zmin) / nz;
    num_nodes_ = (nr + 1) * (nz + 1);
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
      zmax_(zmax) {
    if (nr < 1)
        throw std::invalid_argument("nr must be >= 1");
    if (ntheta < 1)
        throw std::invalid_argument("ntheta must be >= 1");
    if (nz < 1)
        throw std::invalid_argument("nz must be >= 1");
    if (rmax <= rmin)
        throw std::invalid_argument("rmax must be > rmin");
    if (thetamax <= thetamin)
        throw std::invalid_argument("thetamax must be > thetamin");
    if (zmax <= zmin)
        throw std::invalid_argument("zmax must be > zmin");

    dr_ = (rmax - rmin) / nr;
    dtheta_ = (thetamax - thetamin) / ntheta;
    dz_ = (zmax - zmin) / nz;
    num_nodes_ = (nr + 1) * (ntheta + 1) * (nz + 1);
}

// =============================================================================
// Mesh dimensions
// =============================================================================

int CylindricalMesh::numCells() const {
    switch (type_) {
        case CylindricalMeshType::RADIAL_R:
            return nr_;
        case CylindricalMeshType::AXISYMMETRIC_RZ:
            return nr_ * nz_;
        case CylindricalMeshType::FULL_3D:
            return nr_ * ntheta_ * nz_;
    }
    return 0;
}

// =============================================================================
// Index conversion
// =============================================================================

int CylindricalMesh::index(int i, int j, int k) const {
    switch (type_) {
        case CylindricalMeshType::RADIAL_R:
            return i;
        case CylindricalMeshType::AXISYMMETRIC_RZ:
            return i + k * (nr_ + 1);
        case CylindricalMeshType::FULL_3D:
            return i + j * (nr_ + 1) + k * (nr_ + 1) * (ntheta_ + 1);
    }
    return 0;
}

void CylindricalMesh::ijk(int linear_idx, int& i, int& j, int& k) const {
    switch (type_) {
        case CylindricalMeshType::RADIAL_R:
            i = linear_idx;
            j = 0;
            k = 0;
            break;
        case CylindricalMeshType::AXISYMMETRIC_RZ:
            i = linear_idx % (nr_ + 1);
            j = 0;
            k = linear_idx / (nr_ + 1);
            break;
        case CylindricalMeshType::FULL_3D: {
            int plane_size = (nr_ + 1) * (ntheta_ + 1);
            k = linear_idx / plane_size;
            int remainder = linear_idx % plane_size;
            j = remainder / (nr_ + 1);
            i = remainder % (nr_ + 1);
            break;
        }
    }
}

// =============================================================================
// Geometry helpers
// =============================================================================

double CylindricalMesh::cellVolume(int i, int j, int k) const {
    double r_c = r(i);

    switch (type_) {
        case CylindricalMeshType::RADIAL_R:
            // Integrating over theta and assuming unit z-length
            // Volume element for shell: 2*pi*r*dr * 1
            if (r_c < 1e-14) {
                // At r = 0, use half the first cell
                return M_PI * dr_ * dr_ / 4.0;
            }
            return 2.0 * M_PI * r_c * dr_;

        case CylindricalMeshType::AXISYMMETRIC_RZ:
            // Volume = 2*pi*r*dr*dz (full rotation)
            if (r_c < 1e-14) {
                return M_PI * dr_ * dr_ / 4.0 * dz_;
            }
            return 2.0 * M_PI * r_c * dr_ * dz_;

        case CylindricalMeshType::FULL_3D:
            // Volume = r*dr*dtheta*dz
            if (r_c < 1e-14) {
                return dr_ * dr_ / 4.0 * dtheta_ * dz_;
            }
            return r_c * dr_ * dtheta_ * dz_;
    }
    return 0.0;
}

double CylindricalMesh::cellArea(int i) const {
    double r_c = r(i);
    if (r_c < 1e-14) {
        return M_PI * dr_ * dr_ / 4.0;
    }
    return 2.0 * M_PI * r_c * dr_;
}

double CylindricalMesh::crossSectionArea() const {
    return M_PI * (rmax_ * rmax_ - rmin_ * rmin_);
}

// =============================================================================
// Differential operators
// =============================================================================

std::vector<double> CylindricalMesh::gradientR(const std::vector<double>& phi) const {
    std::vector<double> grad(num_nodes_, 0.0);

    if (type_ == CylindricalMeshType::RADIAL_R) {
        // 1D radial
        for (int i = 1; i < nr_; ++i) {
            grad[i] = (phi[i + 1] - phi[i - 1]) / (2.0 * dr_);
        }
        // Boundaries: one-sided
        grad[0] = (phi[1] - phi[0]) / dr_;
        grad[nr_] = (phi[nr_] - phi[nr_ - 1]) / dr_;
    } else if (type_ == CylindricalMeshType::AXISYMMETRIC_RZ) {
        // 2D axisymmetric
        for (int k = 0; k <= nz_; ++k) {
            for (int i = 1; i < nr_; ++i) {
                int idx = index(i, 0, k);
                grad[idx] = (phi[idx + 1] - phi[idx - 1]) / (2.0 * dr_);
            }
            // Boundaries
            int idx0 = index(0, 0, k);
            int idx_n = index(nr_, 0, k);
            grad[idx0] = (phi[idx0 + 1] - phi[idx0]) / dr_;
            grad[idx_n] = (phi[idx_n] - phi[idx_n - 1]) / dr_;
        }
    }

    return grad;
}

std::vector<double> CylindricalMesh::gradientZ(const std::vector<double>& phi) const {
    std::vector<double> grad(num_nodes_, 0.0);

    if (type_ != CylindricalMeshType::AXISYMMETRIC_RZ && type_ != CylindricalMeshType::FULL_3D) {
        return grad;  // No z for 1D radial
    }

    int stride =
        (type_ == CylindricalMeshType::AXISYMMETRIC_RZ) ? (nr_ + 1) : (nr_ + 1) * (ntheta_ + 1);

    for (int k = 1; k < nz_; ++k) {
        for (int i = 0; i <= nr_; ++i) {
            int idx = index(i, 0, k);
            grad[idx] = (phi[idx + stride] - phi[idx - stride]) / (2.0 * dz_);
        }
    }
    // Boundaries
    for (int i = 0; i <= nr_; ++i) {
        int idx0 = index(i, 0, 0);
        int idx_n = index(i, 0, nz_);
        grad[idx0] = (phi[idx0 + stride] - phi[idx0]) / dz_;
        grad[idx_n] = (phi[idx_n] - phi[idx_n - stride]) / dz_;
    }

    return grad;
}

std::vector<double> CylindricalMesh::laplacian(const std::vector<double>& phi) const {
    std::vector<double> lap(num_nodes_, 0.0);
    double dr2 = dr_ * dr_;

    if (type_ == CylindricalMeshType::RADIAL_R) {
        // 1D: lap = d2phi/dr2 + (1/r)*dphi/dr
        for (int i = 1; i < nr_; ++i) {
            double r_i = r(i);
            double d2phi_dr2 = (phi[i + 1] - 2.0 * phi[i] + phi[i - 1]) / dr2;
            double dphi_dr = (phi[i + 1] - phi[i - 1]) / (2.0 * dr_);

            if (r_i > 1e-14) {
                lap[i] = d2phi_dr2 + dphi_dr / r_i;
            } else {
                // At r = 0: use L'Hopital's rule, 1/r * dphi/dr -> d2phi/dr2
                lap[i] = 2.0 * d2phi_dr2;
            }
        }
    } else if (type_ == CylindricalMeshType::AXISYMMETRIC_RZ) {
        // 2D: lap = d2phi/dr2 + (1/r)*dphi/dr + d2phi/dz2
        double dz2 = dz_ * dz_;
        int stride = nr_ + 1;

        for (int k = 1; k < nz_; ++k) {
            for (int i = 1; i < nr_; ++i) {
                int idx = index(i, 0, k);
                double r_i = r(i);

                double d2phi_dr2 = (phi[idx + 1] - 2.0 * phi[idx] + phi[idx - 1]) / dr2;
                double dphi_dr = (phi[idx + 1] - phi[idx - 1]) / (2.0 * dr_);
                double d2phi_dz2 = (phi[idx + stride] - 2.0 * phi[idx] + phi[idx - stride]) / dz2;

                if (r_i > 1e-14) {
                    lap[idx] = d2phi_dr2 + dphi_dr / r_i + d2phi_dz2;
                } else {
                    lap[idx] = 2.0 * d2phi_dr2 + d2phi_dz2;
                }
            }
        }
    }

    return lap;
}

std::vector<double> CylindricalMesh::divergence(const std::vector<double>& vr,
                                                const std::vector<double>& vz) const {
    std::vector<double> div(num_nodes_, 0.0);

    if (type_ != CylindricalMeshType::AXISYMMETRIC_RZ) {
        return div;  // Only implemented for axisymmetric
    }

    int stride = nr_ + 1;

    // div(v) = (1/r) * d(r*vr)/dr + dvz/dz
    for (int k = 1; k < nz_; ++k) {
        for (int i = 1; i < nr_; ++i) {
            int idx = index(i, 0, k);
            double r_i = r(i);
            double r_p = r(i + 1);
            double r_m = r(i - 1);

            // d(r*vr)/dr using central differencing
            double d_rvr_dr = (r_p * vr[idx + 1] - r_m * vr[idx - 1]) / (2.0 * dr_);
            double dvz_dz = (vz[idx + stride] - vz[idx - stride]) / (2.0 * dz_);

            if (r_i > 1e-14) {
                div[idx] = d_rvr_dr / r_i + dvz_dz;
            } else {
                // At r = 0: use L'Hopital
                div[idx] = 2.0 * (vr[idx + 1] - vr[idx]) / dr_ + dvz_dz;
            }
        }
    }

    return div;
}

}  // namespace biotransport
