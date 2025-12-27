/**
 * @file cylindrical_mesh.hpp
 * @brief Cylindrical coordinate mesh for axisymmetric and 3D problems.
 *
 * Supports meshes in cylindrical coordinates (r, theta, z) for:
 *   - Axisymmetric problems (r, z) with theta = 0 or 2*pi
 *   - Full 3D cylindrical problems (r, theta, z)
 *   - Pipe flow simulations
 *   - Blood vessel modeling
 *   - Bioreactor design
 *
 * The mesh handles the r = 0 axis singularity by either:
 *   - Avoiding r = 0 (annular meshes with r_min > 0)
 *   - Special treatment at r = 0 for solid cylinder problems
 *
 * Coordinate system:
 *   - r: radial coordinate [0, R] or [r_min, r_max]
 *   - theta: azimuthal angle [0, 2*pi] (or subset)
 *   - z: axial coordinate [z_min, z_max]
 *
 * For 2D axisymmetric problems, theta is ignored and we solve in (r, z).
 */

#ifndef BIOTRANSPORT_CORE_MESH_CYLINDRICAL_MESH_HPP
#define BIOTRANSPORT_CORE_MESH_CYLINDRICAL_MESH_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

namespace biotransport {

/**
 * @brief Mesh type for cylindrical coordinates.
 */
enum class CylindricalMeshType {
    AXISYMMETRIC_RZ,  ///< 2D (r, z) for axisymmetric problems
    RADIAL_R,         ///< 1D radial (r only)
    FULL_3D           ///< Full 3D (r, theta, z)
};

/**
 * @brief A structured mesh in cylindrical coordinates.
 *
 * Supports three configurations:
 * 1. RADIAL_R: 1D radial problems (r only)
 * 2. AXISYMMETRIC_RZ: 2D axisymmetric problems (r, z)
 * 3. FULL_3D: Complete 3D cylindrical mesh (r, theta, z)
 *
 * The mesh is uniform in each coordinate direction.
 * Node indexing: i = radial, j = azimuthal (theta), k = axial (z)
 * Linear index = i + j*(nr+1) + k*(nr+1)*(ntheta+1)
 *
 * Example usage:
 * @code
 *   // Pipe cross-section (2D axisymmetric)
 *   CylindricalMesh mesh(20, 50, 0.0, 0.005, 0.0, 0.1);  // r: 0-5mm, z: 0-10cm
 *
 *   // Blood vessel with wall (annular)
 *   CylindricalMesh mesh(10, 1, 30, 0.003, 0.004, 0.0, 2*M_PI, 0.0, 0.05);
 *
 *   for (int k = 0; k <= mesh.nz(); ++k) {
 *       for (int i = 0; i <= mesh.nr(); ++i) {
 *           double r = mesh.r(i);
 *           double z = mesh.z(k);
 *           // Use (r, z) coordinates
 *       }
 *   }
 * @endcode
 */
class CylindricalMesh {
public:
    /**
     * @brief Create a 1D radial mesh.
     *
     * For purely radial problems (e.g., spherical diffusion, pipe flow profile).
     *
     * @param nr Number of cells in radial direction
     * @param rmin Minimum radius [m] (0 for solid cylinder)
     * @param rmax Maximum radius [m]
     */
    CylindricalMesh(int nr, double rmin, double rmax);

    /**
     * @brief Create a 2D axisymmetric (r, z) mesh.
     *
     * For axisymmetric problems like pipe flow, blood vessel transport.
     *
     * @param nr Number of cells in radial direction
     * @param nz Number of cells in axial direction
     * @param rmin Minimum radius [m] (0 for solid cylinder)
     * @param rmax Maximum radius [m]
     * @param zmin Minimum z coordinate [m]
     * @param zmax Maximum z coordinate [m]
     */
    CylindricalMesh(int nr, int nz, double rmin, double rmax, double zmin, double zmax);

    /**
     * @brief Create a 3D cylindrical mesh.
     *
     * For full 3D problems with azimuthal variation.
     *
     * @param nr Number of cells in radial direction
     * @param ntheta Number of cells in azimuthal direction
     * @param nz Number of cells in axial direction
     * @param rmin Minimum radius [m]
     * @param rmax Maximum radius [m]
     * @param thetamin Minimum angle [rad]
     * @param thetamax Maximum angle [rad] (2*pi for full cylinder)
     * @param zmin Minimum z coordinate [m]
     * @param zmax Maximum z coordinate [m]
     */
    CylindricalMesh(int nr, int ntheta, int nz, double rmin, double rmax, double thetamin,
                    double thetamax, double zmin, double zmax);

    // =========================================================================
    // Mesh dimensions
    // =========================================================================

    /** @brief Get number of nodes in mesh. */
    int numNodes() const { return num_nodes_; }

    /** @brief Get number of cells in mesh. */
    int numCells() const;

    /** @brief Get mesh type. */
    CylindricalMeshType type() const { return type_; }

    /** @brief Check if mesh is 1D radial. */
    bool isRadial() const { return type_ == CylindricalMeshType::RADIAL_R; }

    /** @brief Check if mesh is 2D axisymmetric. */
    bool isAxisymmetric() const { return type_ == CylindricalMeshType::AXISYMMETRIC_RZ; }

    /** @brief Check if mesh is full 3D. */
    bool is3D() const { return type_ == CylindricalMeshType::FULL_3D; }

    // =========================================================================
    // Cell counts and sizes
    // =========================================================================

    /** @brief Number of cells in radial direction. */
    int nr() const { return nr_; }

    /** @brief Number of cells in azimuthal direction. */
    int ntheta() const { return ntheta_; }

    /** @brief Number of cells in axial direction. */
    int nz() const { return nz_; }

    /** @brief Cell size in radial direction. */
    double dr() const { return dr_; }

    /** @brief Cell size in azimuthal direction. */
    double dtheta() const { return dtheta_; }

    /** @brief Cell size in axial direction. */
    double dz() const { return dz_; }

    // =========================================================================
    // Coordinate ranges
    // =========================================================================

    double rmin() const { return rmin_; }
    double rmax() const { return rmax_; }
    double thetamin() const { return thetamin_; }
    double thetamax() const { return thetamax_; }
    double zmin() const { return zmin_; }
    double zmax() const { return zmax_; }

    // =========================================================================
    // Coordinate access
    // =========================================================================

    /**
     * @brief Get r coordinate of node i.
     * @param i Radial index (0 to nr)
     */
    double r(int i) const { return rmin_ + i * dr_; }

    /**
     * @brief Get theta coordinate of node j.
     * @param j Azimuthal index (0 to ntheta)
     */
    double theta(int j) const { return thetamin_ + j * dtheta_; }

    /**
     * @brief Get z coordinate of node k.
     * @param k Axial index (0 to nz)
     */
    double z(int k) const { return zmin_ + k * dz_; }

    // =========================================================================
    // Index conversion
    // =========================================================================

    /**
     * @brief Get linear index from (i, j, k) indices.
     *
     * For 1D radial: index(i)
     * For 2D axisym: index(i, k) where j is ignored
     * For 3D: index(i, j, k)
     */
    int index(int i, int j = 0, int k = 0) const;

    /**
     * @brief Get (i, j, k) indices from linear index.
     */
    void ijk(int linear_idx, int& i, int& j, int& k) const;

    // =========================================================================
    // Geometry helpers
    // =========================================================================

    /**
     * @brief Check if mesh includes r = 0 (axis singularity).
     */
    bool hasAxisSingularity() const { return rmin_ < 1e-14; }

    /**
     * @brief Get Cartesian x coordinate from cylindrical.
     * @param i Radial index
     * @param j Azimuthal index
     */
    double x(int i, int j = 0) const { return r(i) * std::cos(theta(j)); }

    /**
     * @brief Get Cartesian y coordinate from cylindrical.
     * @param i Radial index
     * @param j Azimuthal index
     */
    double y(int i, int j = 0) const { return r(i) * std::sin(theta(j)); }

    /**
     * @brief Get cell volume at node (i, j, k).
     *
     * In cylindrical coordinates, volume element = r * dr * dtheta * dz
     * For axisymmetric, integrating over theta gives 2*pi*r*dr*dz
     */
    double cellVolume(int i, int j = 0, int k = 0) const;

    /**
     * @brief Get cell area at node i (radial 1D).
     *
     * For radial problems, area = 2*pi*r*dr
     */
    double cellArea(int i) const;

    /**
     * @brief Get cross-sectional area at z (axisymmetric).
     *
     * Area = pi * (rmax^2 - rmin^2)
     */
    double crossSectionArea() const;

    // =========================================================================
    // Differential operators in cylindrical coordinates
    // =========================================================================

    /**
     * @brief Compute gradient of scalar field (radial component).
     *
     * grad_r(phi) = dphi/dr
     *
     * @param phi Scalar field values
     * @return Radial gradient at each node
     */
    std::vector<double> gradientR(const std::vector<double>& phi) const;

    /**
     * @brief Compute gradient of scalar field (axial component).
     *
     * grad_z(phi) = dphi/dz
     *
     * @param phi Scalar field values
     * @return Axial gradient at each node
     */
    std::vector<double> gradientZ(const std::vector<double>& phi) const;

    /**
     * @brief Compute Laplacian in cylindrical coordinates.
     *
     * For axisymmetric (no theta dependence):
     *   lap(phi) = (1/r) * d/dr(r * dphi/dr) + d^2(phi)/dz^2
     *            = d^2(phi)/dr^2 + (1/r)*dphi/dr + d^2(phi)/dz^2
     *
     * For 1D radial:
     *   lap(phi) = (1/r) * d/dr(r * dphi/dr)
     *            = d^2(phi)/dr^2 + (1/r)*dphi/dr
     *
     * @param phi Scalar field values
     * @return Laplacian at each node
     */
    std::vector<double> laplacian(const std::vector<double>& phi) const;

    /**
     * @brief Compute divergence of vector field (axisymmetric).
     *
     * div(v) = (1/r) * d(r*v_r)/dr + dv_z/dz
     *
     * @param vr Radial velocity component
     * @param vz Axial velocity component
     * @return Divergence at each node
     */
    std::vector<double> divergence(const std::vector<double>& vr,
                                   const std::vector<double>& vz) const;

private:
    CylindricalMeshType type_;
    int nr_, ntheta_, nz_;
    double rmin_, rmax_;
    double thetamin_, thetamax_;
    double zmin_, zmax_;
    double dr_, dtheta_, dz_;
    int num_nodes_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_CYLINDRICAL_MESH_HPP
