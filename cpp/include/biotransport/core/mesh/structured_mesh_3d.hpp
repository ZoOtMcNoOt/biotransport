/**
 * @file structured_mesh_3d.hpp
 * @brief Uniform structured mesh for 3D finite difference simulations.
 *
 * Provides a simple 3D Cartesian mesh with uniform cell spacing. The mesh stores:
 *   - Grid dimensions (nx, ny, nz)
 *   - Domain bounds (xmin, xmax, ymin, ymax, zmin, zmax)
 *   - Derived quantities (dx, dy, dz, numNodes)
 *
 * Indexing uses row-major order: index = k * (nx+1) * (ny+1) + j * (nx+1) + i
 *
 * This enables organ-scale modeling, 3D heat transfer, and volumetric drug delivery.
 *
 * @see StructuredMesh for 1D/2D meshes
 */

#ifndef BIOTRANSPORT_CORE_MESH_STRUCTURED_MESH_3D_HPP
#define BIOTRANSPORT_CORE_MESH_STRUCTURED_MESH_3D_HPP

#include <array>
#include <stdexcept>
#include <vector>

namespace biotransport {

/**
 * @brief Boundary identifiers for 3D meshes.
 */
enum class Boundary3D : int {
    XMin = 0,  ///< Face at x = xmin (left)
    XMax = 1,  ///< Face at x = xmax (right)
    YMin = 2,  ///< Face at y = ymin (front)
    YMax = 3,  ///< Face at y = ymax (back)
    ZMin = 4,  ///< Face at z = zmin (bottom)
    ZMax = 5   ///< Face at z = zmax (top)
};

/**
 * @brief Convert Boundary3D to array index.
 */
inline int to_index(Boundary3D b) {
    return static_cast<int>(b);
}

/**
 * @brief Uniform structured mesh for 3D finite difference simulations.
 *
 * Provides a simple Cartesian mesh with uniform cell spacing (dx, dy, dz).
 * Nodes are indexed from 0 to nx (inclusive) in x, 0 to ny in y, 0 to nz in z.
 * Uses row-major ordering: index = k * (nx+1) * (ny+1) + j * (nx+1) + i.
 */
class StructuredMesh3D {
public:
    /**
     * @brief Create a 3D structured mesh.
     *
     * @param nx Number of cells in x direction
     * @param ny Number of cells in y direction
     * @param nz Number of cells in z direction
     * @param xmin Minimum x coordinate [m]
     * @param xmax Maximum x coordinate [m]
     * @param ymin Minimum y coordinate [m]
     * @param ymax Maximum y coordinate [m]
     * @param zmin Minimum z coordinate [m]
     * @param zmax Maximum z coordinate [m]
     */
    StructuredMesh3D(int nx, int ny, int nz, double xmin, double xmax, double ymin, double ymax,
                     double zmin, double zmax);

    /**
     * @brief Create a cubic 3D mesh with uniform dimensions.
     *
     * @param n Number of cells in each direction
     * @param length Side length of the cube [m]
     */
    StructuredMesh3D(int n, double length);

    /**
     * @brief Get the total number of nodes in the mesh.
     * @return (nx+1) * (ny+1) * (nz+1)
     */
    int numNodes() const noexcept { return num_nodes_; }

    /**
     * @brief Get the total number of cells in the mesh.
     * @return nx * ny * nz
     */
    int numCells() const noexcept { return nx_ * ny_ * nz_; }

    /**
     * @brief Get the cell size in x direction.
     */
    double dx() const noexcept { return dx_; }

    /**
     * @brief Get the cell size in y direction.
     */
    double dy() const noexcept { return dy_; }

    /**
     * @brief Get the cell size in z direction.
     */
    double dz() const noexcept { return dz_; }

    /**
     * @brief Get the number of cells in x direction.
     */
    int nx() const noexcept { return nx_; }

    /**
     * @brief Get the number of cells in y direction.
     */
    int ny() const noexcept { return ny_; }

    /**
     * @brief Get the number of cells in z direction.
     */
    int nz() const noexcept { return nz_; }

    /**
     * @brief Get domain bounds.
     */
    double xmin() const noexcept { return xmin_; }
    double xmax() const noexcept { return xmax_; }
    double ymin() const noexcept { return ymin_; }
    double ymax() const noexcept { return ymax_; }
    double zmin() const noexcept { return zmin_; }
    double zmax() const noexcept { return zmax_; }

    /**
     * @brief Get the x coordinate of node at grid position i.
     */
    double x(int i) const;

    /**
     * @brief Get the y coordinate of node at grid position j.
     */
    double y(int j) const;

    /**
     * @brief Get the z coordinate of node at grid position k.
     */
    double z(int k) const;

    /**
     * @brief Get the global (linear) index of node (i, j, k).
     * @param i Node index in x direction (0 to nx)
     * @param j Node index in y direction (0 to ny)
     * @param k Node index in z direction (0 to nz)
     * @return Linear index for flat array access
     */
    int index(int i, int j, int k) const;

    /**
     * @brief Convert linear index to (i, j, k) tuple.
     * @param idx Linear index
     * @return Array {i, j, k}
     */
    std::array<int, 3> ijk(int idx) const;

    /**
     * @brief Get the stride in j-direction (for navigating in y).
     * @return nx + 1
     */
    int strideJ() const noexcept { return stride_j_; }

    /**
     * @brief Get the stride in k-direction (for navigating in z).
     * @return (nx + 1) * (ny + 1)
     */
    int strideK() const noexcept { return stride_k_; }

    /**
     * @brief Check if index is on a specific boundary face.
     */
    bool isOnBoundary(int i, int j, int k, Boundary3D boundary) const;

    /**
     * @brief Check if a node is on any boundary.
     */
    bool isOnAnyBoundary(int i, int j, int k) const;

private:
    int nx_, ny_, nz_;
    double xmin_, xmax_, ymin_, ymax_, zmin_, zmax_;
    double dx_, dy_, dz_;
    int stride_j_, stride_k_;
    int num_nodes_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_STRUCTURED_MESH_3D_HPP
