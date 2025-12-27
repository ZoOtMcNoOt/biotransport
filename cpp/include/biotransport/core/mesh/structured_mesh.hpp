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
#include <vector>

namespace biotransport {

/**
 * @brief Uniform structured mesh for 1D and 2D finite difference simulations.
 *
 * Provides a simple Cartesian mesh with uniform cell spacing (dx, dy).
 * Nodes are indexed from 0 to nx (inclusive) in x and 0 to ny in y.
 * Uses row-major ordering for 2D: index = j * (nx+1) + i.
 */
class StructuredMesh {
public:
    /**
     * @brief Create a 1D structured mesh.
     *
     * @param nx Number of cells in x direction
     * @param xmin Minimum x coordinate [m]
     * @param xmax Maximum x coordinate [m]
     */
    StructuredMesh(int nx, double xmin, double xmax);

    /**
     * @brief Create a 2D structured mesh.
     *
     * @param nx Number of cells in x direction
     * @param ny Number of cells in y direction
     * @param xmin Minimum x coordinate [m]
     * @param xmax Maximum x coordinate [m]
     * @param ymin Minimum y coordinate [m]
     * @param ymax Maximum y coordinate [m]
     */
    StructuredMesh(int nx, int ny, double xmin, double xmax, double ymin, double ymax);

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

private:
    int nx_, ny_;         ///< Number of cells in each direction
    double xmin_, xmax_;  ///< x coordinate range [m]
    double ymin_, ymax_;  ///< y coordinate range [m]
    double dx_, dy_;      ///< Cell sizes [m]
    bool is_1d_;          ///< True if 1D mesh (ny == 0)
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_STRUCTURED_MESH_HPP
