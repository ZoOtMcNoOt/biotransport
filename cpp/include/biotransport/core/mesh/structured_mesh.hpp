/**
 * @file structured_mesh.hpp
 * @brief Uniform structured mesh for 1D and 2D finite difference simulations.
 *
 * Provides a simple Cartesian mesh with uniform cell spacing. The mesh stores:
 *   - Grid dimensions (nx, ny)
 *   - Domain bounds (xmin, xmax, ymin, ymax)
 *   - Derived quantities (dx, dy, numNodes)
 *
 * Indexing uses row-major order: index = j * (nx+1) + i
 *
 * This is the primary mesh class used throughout the library. For cylindrical
 * coordinate problems, see CylindricalMesh.
 *
 * @see CylindricalMesh for cylindrical coordinate meshes
 * @see indexing.hpp for grid_index() utility function
 */

#ifndef BIOTRANSPORT_CORE_MESH_STRUCTURED_MESH_HPP
#define BIOTRANSPORT_CORE_MESH_STRUCTURED_MESH_HPP

#include <stdexcept>
#include <string>
#include <vector>

namespace biotransport {

/**
 * @brief Coordinate system a 1D mesh represents.
 *
 * The finite-volume balance is identical in all three; only the face area and
 * the control-volume measure change. Writing @f$A(r)@f$ for the area factor,
 * the conservative statement is
 *
 * @f[ \frac{\partial c}{\partial t} = \frac{1}{A}\frac{\partial}{\partial r}(A q) + R @f]
 *
 * with @f$A = 1@f$ for a slab, @f$A = r@f$ for a cylinder and @f$A = r^2@f$ for
 * a sphere. Because @f$A(0) = 0@f$ in both curved cases, the flux through the
 * centre vanishes on its own: symmetry at @f$r = 0@f$ is enforced by the
 * geometry rather than by a boundary condition.
 *
 * A 2D mesh may be CARTESIAN or CYLINDRICAL. Cylindrical in 2D means
 * axisymmetric @f$(r, z)@f$: x plays the role of r and y the role of z. That
 * case factorises -- the radial direction carries the area weights above, and
 * the axial direction is identical to Cartesian -- because the axial face area
 * and the control volume share the same radial measure and it cancels.
 *
 * SPHERICAL is 1D only; a 2D spherical mesh would be @f$(r, \theta)@f$, which
 * is a different operator and is not implemented.
 */
enum class Geometry {
    CARTESIAN = 0,    ///< Slab. Area factor 1.
    CYLINDRICAL = 1,  ///< Radial in a long cylinder. Area factor r.
    SPHERICAL = 2     ///< Radial in a sphere. Area factor r^2.
};

/**
 * @brief Uniform structured mesh for 1D and 2D finite volume simulations.
 *
 * Provides a mesh with uniform cell spacing (dx, dy). Nodes are indexed from 0
 * to nx (inclusive) in x and 0 to ny in y. Uses row-major ordering for 2D:
 * index = j * (nx+1) + i.
 *
 * A 1D mesh may be Cartesian, cylindrical or spherical; see Geometry.
 */
class StructuredMesh {
public:
    /**
     * @brief Create a 1D structured mesh.
     *
     * @param nx Number of cells in x direction
     * @param xmin Minimum x coordinate [m]
     * @param xmax Maximum x coordinate [m]
     * @param geometry Coordinate system; Cartesian by default
     *
     * @throws std::invalid_argument if a curved geometry is given a negative
     *         coordinate, which has no radial meaning.
     */
    StructuredMesh(int nx, double xmin, double xmax,
                   Geometry geometry = Geometry::CARTESIAN);

    /**
     * @brief Create a 2D structured mesh.
     *
     * @param nx Number of cells in x direction (r, when axisymmetric)
     * @param ny Number of cells in y direction (z, when axisymmetric)
     * @param xmin Minimum x coordinate [m]
     * @param xmax Maximum x coordinate [m]
     * @param ymin Minimum y coordinate [m]
     * @param ymax Maximum y coordinate [m]
     * @param geometry CARTESIAN or CYLINDRICAL; the latter means axisymmetric
     *        @f$(r, z)@f$
     *
     * @throws std::invalid_argument for SPHERICAL, which is 1D only, or for a
     *         negative inner radius on an axisymmetric mesh.
     */
    StructuredMesh(int nx, int ny, double xmin, double xmax, double ymin, double ymax,
                   Geometry geometry = Geometry::CARTESIAN);

    /**
     * @brief Get the total number of nodes in the mesh.
     * @return (nx+1) for 1D, (nx+1)*(ny+1) for 2D
     */
    int numNodes() const;

    /**
     * @brief Get the total number of cells in the mesh.
     * @return nx for 1D, nx*ny for 2D
     */
    int numCells() const;

    /**
     * @brief Get the cell size in x direction.
     * @return Grid spacing dx [m]
     */
    double dx() const noexcept { return dx_; }

    /**
     * @brief Get the cell size in y direction.
     * @return Grid spacing dy [m] (equals dx for 1D)
     */
    double dy() const noexcept { return dy_; }

    /**
     * @brief Check if this is a 1D mesh.
     * @return true if ny == 0 (1D), false otherwise
     */
    bool is1D() const noexcept { return is_1d_; }

    /**
     * @brief Get the x coordinate of node i.
     * @param i Node index in x direction (0 to nx)
     * @return x coordinate [m]
     */
    double x(int i) const;

    /**
     * @brief Get the y coordinate of node (i, j).
     * @param i Node index in x direction
     * @param j Node index in y direction (0 to ny)
     * @return y coordinate [m]
     */
    double y(int i, int j) const;

    /**
     * @brief Get the global (linear) index of node (i, j).
     * @param i Node index in x direction
     * @param j Node index in y direction (default 0 for 1D)
     * @return Linear index for flat array access
     */
    int index(int i, int j = 0) const;

    /**
     * @brief Get the number of cells in x direction.
     * @return nx
     */
    int nx() const noexcept { return nx_; }

    /**
     * @brief Get the number of cells in y direction.
     * @return ny (0 for 1D mesh)
     */
    int ny() const noexcept { return ny_; }

    /**
     * @brief Coordinate system of this mesh. Always Cartesian in 2D.
     */
    Geometry geometry() const noexcept { return geometry_; }

    /**
     * @brief Whether this mesh uses a curved radial geometry.
     */
    bool isRadial() const noexcept { return geometry_ != Geometry::CARTESIAN; }

    /**
     * @brief Area factor at coordinate @p r.
     *
     * 1 for a slab, r for a cylinder, r^2 for a sphere. Multiplying a flux by
     * this turns it into a transfer rate through the face.
     */
    double areaFactor(double r) const noexcept {
        switch (geometry_) {
            case Geometry::CYLINDRICAL:
                return r;
            case Geometry::SPHERICAL:
                return r * r;
            case Geometry::CARTESIAN:
            default:
                return 1.0;
        }
    }

    /**
     * @brief Measure of the region between coordinates @p a and @p b.
     *
     * The exact integral of the area factor, not a midpoint approximation, so
     * the control volumes sum to the true domain measure and the discrete
     * balance conserves to roundoff.
     */
    double measureBetween(double a, double b) const noexcept {
        switch (geometry_) {
            case Geometry::CYLINDRICAL:
                // Factored integrals retain the thin shell width instead of
                // subtracting nearly equal squared/cubed radii.
                return (0.5 * (b - a)) * (b + a);
            case Geometry::SPHERICAL:
                return (b - a) * ((a * a + a * b + b * b) / 3.0);
            case Geometry::CARTESIAN:
            default:
                return b - a;
        }
    }

    /**
     * @brief Radial measure of node @p i's control volume.
     *
     * Interior nodes own half a cell either side; the two end nodes own a half
     * cell. In Cartesian geometry this reduces to dx and dx/2.
     *
     * On a 2D axisymmetric mesh this is the radial part only -- the measure per
     * unit z. Multiply by the axial control height for the full volume, which is
     * what integrateMass does.
     */
    double controlVolume(int i) const;

    /**
     * @brief Area factor at the lower x face of node @p i's control volume.
     *
     * On a 2D axisymmetric mesh this is the area per unit z.
     */
    double lowerFaceArea(int i) const;

    /**
     * @brief Area factor at the upper x face of node @p i's control volume.
     */
    double upperFaceArea(int i) const;

    /**
     * @brief Axial control height of node @p j on a 2D mesh.
     *
     * dy for an interior row, dy/2 for the two end rows. Always 1 on a 1D mesh,
     * so a 1D control volume is just controlVolume(i).
     */
    double axialHeight(int j) const;

private:
    int nx_, ny_;         ///< Number of cells in each direction
    double xmin_, xmax_;  ///< x coordinate range [m]
    double ymin_, ymax_;  ///< y coordinate range [m]
    double dx_, dy_;      ///< Cell sizes [m]
    bool is_1d_;          ///< True if 1D mesh (ny == 0)
    Geometry geometry_;   ///< Coordinate system (1D only)
};

/**
 * @brief Reject a curved mesh in a solver that only implements slab geometry.
 *
 * Only the canonical TransportProblem path carries the area weights that make a
 * radial balance conservative. Every other solver would quietly return a
 * Cartesian answer on a radial mesh, which is worse than refusing, so they
 * refuse.
 *
 * @param mesh Mesh to check.
 * @param who Name of the solver, used in the message.
 * @throws std::invalid_argument if @p mesh is cylindrical or spherical.
 */
inline void requireCartesian(const StructuredMesh& mesh, const char* who) {
    if (mesh.isRadial()) {
        throw std::invalid_argument(
            std::string(who) +
            " implements slab geometry only and would silently ignore the radial "
            "metric of this mesh. Use the canonical TransportProblem/solve path "
            "for cylindrical or spherical problems.");
    }
}

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_STRUCTURED_MESH_HPP
