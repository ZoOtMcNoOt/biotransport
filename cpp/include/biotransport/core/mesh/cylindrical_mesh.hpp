/**
 * @file cylindrical_mesh.hpp
 * @brief Uniform meshes and metric-aware operators in cylindrical coordinates.
 */

#ifndef BIOTRANSPORT_CORE_MESH_CYLINDRICAL_MESH_HPP
#define BIOTRANSPORT_CORE_MESH_CYLINDRICAL_MESH_HPP

#include <vector>

namespace biotransport {

/** @brief Dimensionality represented by a cylindrical mesh. */
enum class CylindricalMeshType {
    AXISYMMETRIC_RZ,  ///< Axisymmetric two-dimensional (r,z) mesh
    RADIAL_R,         ///< Axisymmetric one-dimensional radial mesh
    FULL_3D           ///< Periodic three-dimensional (r,theta,z) mesh
};

/**
 * @brief A uniform structured mesh in cylindrical coordinates.
 *
 * Radial and axisymmetric meshes store ``nr+1`` radial nodes.  Axisymmetric
 * meshes additionally store ``nz+1`` axial nodes.  A FULL_3D mesh represents
 * one complete periodic turn and stores ``ntheta`` unique azimuthal nodes;
 * the duplicate endpoint at ``thetamin+2*pi`` is deliberately omitted.
 *
 * Linear storage is radial-fastest.  For FULL_3D,
 * ``index = i + j*(nr+1) + k*(nr+1)*ntheta``.  For AXISYMMETRIC_RZ,
 * ``index = i + k*(nr+1)``.
 *
 * Geometry and integration are valid for a FULL_3D mesh that includes
 * ``r=0``, but its non-axisymmetric cylindrical basis is not single-valued at
 * the axis.  Differential operators therefore reject that case; use an
 * annulus or an AXISYMMETRIC_RZ mesh.
 */
class CylindricalMesh {
public:
    /** @brief Construct a radial mesh on ``rmin <= r <= rmax``. */
    CylindricalMesh(int nr, double rmin, double rmax);

    /** @brief Construct an axisymmetric mesh on ``(r,z)``. */
    CylindricalMesh(int nr, int nz, double rmin, double rmax, double zmin, double zmax);

    /**
     * @brief Construct a periodic full cylindrical mesh.
     *
     * ``thetamax-thetamin`` must be one complete turn (2*pi), and
     * ``ntheta >= 3``.  Partial angular wedges require boundary conditions and
     * are rejected rather than silently treated as periodic.
     */
    CylindricalMesh(int nr, int ntheta, int nz, double rmin, double rmax, double thetamin,
                    double thetamax, double zmin, double zmax);

    int numNodes() const noexcept { return num_nodes_; }
    int numCells() const noexcept;

    CylindricalMeshType type() const noexcept { return type_; }
    bool isRadial() const noexcept { return type_ == CylindricalMeshType::RADIAL_R; }
    bool isAxisymmetric() const noexcept { return type_ == CylindricalMeshType::AXISYMMETRIC_RZ; }
    bool is3D() const noexcept { return type_ == CylindricalMeshType::FULL_3D; }

    int nr() const noexcept { return nr_; }
    int ntheta() const noexcept { return ntheta_; }
    int nz() const noexcept { return nz_; }
    int thetaNodeCount() const noexcept { return is3D() ? ntheta_ : 1; }
    bool thetaPeriodic() const noexcept { return is3D(); }

    double dr() const noexcept { return dr_; }
    double dtheta() const noexcept { return dtheta_; }
    double dz() const noexcept { return dz_; }

    double rmin() const noexcept { return rmin_; }
    double rmax() const noexcept { return rmax_; }
    double thetamin() const noexcept { return thetamin_; }
    double thetamax() const noexcept { return thetamax_; }
    double zmin() const noexcept { return zmin_; }
    double zmax() const noexcept { return zmax_; }

    /** @brief Coordinate accessors with range validation. */
    double r(int i) const;
    double theta(int j) const;
    double z(int k) const;

    /**
     * @brief Convert coordinate indices to a linear index.
     *
     * For AXISYMMETRIC_RZ, both ``index(i,k)`` and ``index(i,0,k)`` are
     * accepted.  Supplying both a non-zero second and third index is rejected.
     */
    int index(int i, int j = 0, int k = 0) const;

    /** @brief Convert a validated linear index to ``(i,j,k)``. */
    void ijk(int linear_idx, int& i, int& j, int& k) const;

    bool hasAxisSingularity() const noexcept { return rmin_ == 0.0; }
    double x(int i, int j = 0) const;
    double y(int i, int j = 0) const;

    /**
     * @brief Exact nodal control-volume measure.
     *
     * RADIAL_R assumes unit axial length.  AXISYMMETRIC_RZ integrates a full
     * turn.  FULL_3D returns the periodic angular-sector control volume.
     * Summing this value over every stored node gives the exact domain volume.
     */
    double cellVolume(int i, int j = 0, int k = 0) const;

    /**
     * @brief Exact annular nodal control area integrated through a full turn.
     *
     * Summing over ``i=0..nr`` gives ``crossSectionArea()`` exactly.
     */
    double cellArea(int i) const;

    double crossSectionArea() const noexcept;

    /**
     * @brief Physical radial component ``d(phi)/dr``.
     *
     * A regular radial or axisymmetric scalar has an exactly zero radial
     * gradient at ``r=0``.  Other boundaries use second-order one-sided
     * differences when at least two cells are available.
     */
    std::vector<double> gradientR(const std::vector<double>& phi) const;

    /** @brief Physical azimuthal component ``(1/r)d(phi)/dtheta``. */
    std::vector<double> gradientTheta(const std::vector<double>& phi) const;

    /** @brief Physical axial component ``d(phi)/dz``. */
    std::vector<double> gradientZ(const std::vector<double>& phi) const;

    /**
     * @brief Metric-aware scalar Laplacian.
     *
     * Radial and axisymmetric meshes use the regular axis limit at ``r=0``.
     * FULL_3D operators currently require an annulus (``rmin>0``), because a
     * non-axisymmetric cylindrical basis is undefined at the axis.
     * At least two cells are required in every non-periodic active direction.
     */
    std::vector<double> laplacian(const std::vector<double>& phi) const;

    /**
     * @brief Axisymmetric divergence ``(1/r)d(r vr)/dr + d(vz)/dz``.
     *
     * Calling this overload for FULL_3D throws; use the three-component
     * overload so the azimuthal metric term cannot be lost.
     * A mesh containing the axis requires the regularity condition ``vr=0``
     * there; a field that violates it is rejected.
     */
    std::vector<double> divergence(const std::vector<double>& vr,
                                   const std::vector<double>& vz) const;

    /**
     * @brief Full divergence
     * ``(1/r)d(r vr)/dr + (1/r)d(vtheta)/dtheta + d(vz)/dz``.
     */
    std::vector<double> divergence(const std::vector<double>& vr, const std::vector<double>& vtheta,
                                   const std::vector<double>& vz) const;

private:
    CylindricalMeshType type_;
    int nr_;
    int ntheta_;
    int nz_;
    double rmin_;
    double rmax_;
    double thetamin_;
    double thetamax_;
    double zmin_;
    double zmax_;
    double dr_;
    double dtheta_;
    double dz_;
    int num_nodes_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_CYLINDRICAL_MESH_HPP
