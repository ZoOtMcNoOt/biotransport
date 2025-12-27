#ifndef BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_HPP
#define BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_HPP

/**
 * @file mesh_iterators.hpp
 * @brief Unified iteration patterns for structured meshes.
 *
 * Provides abstractions to eliminate duplicated 1D/2D branching logic
 * throughout the solver codebase. All iterators work transparently
 * for both 1D and 2D meshes.
 *
 * When compiled with BIOTRANSPORT_ENABLE_OPENMP, the parallel versions
 * of the iterators use OpenMP for multi-threaded execution.
 */

#include <biotransport/core/mesh/structured_mesh.hpp>
#include <functional>
#include <vector>

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#include <omp.h>
#endif

namespace biotransport {

/**
 * @brief Information about a node during iteration.
 */
struct NodeInfo {
    int idx;   ///< Global linear index
    int i;     ///< x-index
    int j;     ///< y-index (0 for 1D)
    double x;  ///< x-coordinate
    double y;  ///< y-coordinate (0 for 1D)
};

/**
 * @brief Stencil indices for a node (neighbors for finite differences).
 */
struct Stencil {
    int center;  ///< Center node index
    int west;    ///< West neighbor (i-1)
    int east;    ///< East neighbor (i+1)
    int south;   ///< South neighbor (j-1), -1 for 1D
    int north;   ///< North neighbor (j+1), -1 for 1D
};

/**
 * @brief Helper class for iterating over mesh nodes with unified 1D/2D handling.
 *
 * Eliminates the repeated pattern:
 * @code
 *   if (is_1d) {
 *       for (int i = 1; i < nx; ++i) { ... }
 *   } else {
 *       for (int j = 1; j < ny; ++j) {
 *           for (int i = 1; i < nx; ++i) { ... }
 *       }
 *   }
 * @endcode
 */
class MeshIterator {
public:
    explicit MeshIterator(const StructuredMesh& mesh) : mesh_(mesh) { stride_ = mesh_.nx() + 1; }

    /**
     * @brief Iterate over interior nodes (excludes boundaries).
     *
     * @param callback Function called for each interior node with (idx, i, j)
     *
     * When BIOTRANSPORT_ENABLE_OPENMP is defined, the 2D loop is parallelized
     * using OpenMP with static scheduling. The 1D case is not parallelized
     * due to low overhead.
     */
    template <typename Func>
    void forEachInterior(Func&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();

        if (mesh_.is1D()) {
            for (int i = 1; i < nx; ++i) {
                callback(i, i, 0);
            }
        } else {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 1; j < ny; ++j) {
                const int row = j * stride_;
                for (int i = 1; i < nx; ++i) {
                    callback(row + i, i, j);
                }
            }
        }
    }

    /**
     * @brief Iterate over interior nodes with full node info.
     *
     * @param callback Function called for each interior node with NodeInfo
     *
     * When BIOTRANSPORT_ENABLE_OPENMP is defined, the 2D loop is parallelized.
     */
    template <typename Func>
    void forEachInteriorWithCoords(Func&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();

        if (mesh_.is1D()) {
            for (int i = 1; i < nx; ++i) {
                NodeInfo info{i, i, 0, mesh_.x(i), 0.0};
                callback(info);
            }
        } else {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 1; j < ny; ++j) {
                const double y = mesh_.y(0, j);
                const int row = j * stride_;
                for (int i = 1; i < nx; ++i) {
                    NodeInfo info{row + i, i, j, mesh_.x(i), y};
                    callback(info);
                }
            }
        }
    }

    /**
     * @brief Iterate over all nodes (including boundaries).
     *
     * @param callback Function called for each node with (idx, i, j)
     *
     * When BIOTRANSPORT_ENABLE_OPENMP is defined, the 2D loop is parallelized.
     */
    template <typename Func>
    void forEachNode(Func&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();

        if (mesh_.is1D()) {
            for (int i = 0; i <= nx; ++i) {
                callback(i, i, 0);
            }
        } else {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j <= ny; ++j) {
                const int row = j * stride_;
                for (int i = 0; i <= nx; ++i) {
                    callback(row + i, i, j);
                }
            }
        }
    }

    /**
     * @brief Iterate over boundary nodes on a specific side.
     *
     * @param side Boundary side (Left, Right, Bottom, Top)
     * @param callback Function called for each boundary node with (idx, i, j)
     */
    template <typename Func>
    void forEachBoundary(Boundary side, Func&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();

        switch (side) {
            case Boundary::Left:
                if (mesh_.is1D()) {
                    callback(0, 0, 0);
                } else {
                    for (int j = 0; j <= ny; ++j) {
                        callback(j * stride_, 0, j);
                    }
                }
                break;

            case Boundary::Right:
                if (mesh_.is1D()) {
                    callback(nx, nx, 0);
                } else {
                    for (int j = 0; j <= ny; ++j) {
                        callback(j * stride_ + nx, nx, j);
                    }
                }
                break;

            case Boundary::Bottom:
                if (!mesh_.is1D()) {
                    for (int i = 0; i <= nx; ++i) {
                        callback(i, i, 0);
                    }
                }
                break;

            case Boundary::Top:
                if (!mesh_.is1D()) {
                    for (int i = 0; i <= nx; ++i) {
                        callback(ny * stride_ + i, i, ny);
                    }
                }
                break;
        }
    }

    /**
     * @brief Get stencil indices for a node.
     *
     * @param idx Global linear index
     * @return Stencil with neighbor indices
     */
    Stencil getStencil(int idx) const {
        Stencil s;
        s.center = idx;
        s.west = idx - 1;
        s.east = idx + 1;
        if (mesh_.is1D()) {
            s.south = -1;
            s.north = -1;
        } else {
            s.south = idx - stride_;
            s.north = idx + stride_;
        }
        return s;
    }

    /**
     * @brief Get the row stride for 2D indexing.
     */
    int stride() const { return stride_; }

private:
    const StructuredMesh& mesh_;
    int stride_;
};

/**
 * @brief Compute finite difference stencil operations on solution vectors.
 *
 * Provides unified Laplacian, gradient, and other stencil operations
 * that work for both 1D and 2D meshes.
 */
class StencilOps {
public:
    explicit StencilOps(const StructuredMesh& mesh)
        : mesh_(mesh),
          inv_dx2_(1.0 / (mesh.dx() * mesh.dx())),
          inv_dy2_(mesh.is1D() ? 0.0 : 1.0 / (mesh.dy() * mesh.dy())),
          inv_2dx_(1.0 / (2.0 * mesh.dx())),
          inv_2dy_(mesh.is1D() ? 0.0 : 1.0 / (2.0 * mesh.dy())),
          stride_(mesh.nx() + 1) {}

    /**
     * @brief Compute the discrete Laplacian at a node.
     *
     * ∇²u ≈ (u[i+1] - 2*u[i] + u[i-1])/dx² + (u[j+1] - 2*u[j] + u[j-1])/dy²
     *
     * @param u Solution vector
     * @param idx Node index
     * @return Laplacian value
     */
    double laplacian(const std::vector<double>& u, int idx) const {
        double lap = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) * inv_dx2_;
        if (!mesh_.is1D()) {
            lap += (u[idx + stride_] - 2.0 * u[idx] + u[idx - stride_]) * inv_dy2_;
        }
        return lap;
    }

    /**
     * @brief Compute diffusion term contribution: D * ∇²u * dt
     *
     * @param u Solution vector
     * @param idx Node index
     * @param D Diffusion coefficient
     * @param dt Time step
     * @return Diffusion contribution to update
     */
    double diffusionTerm(const std::vector<double>& u, int idx, double D, double dt) const {
        return D * dt * laplacian(u, idx);
    }

    /**
     * @brief Compute diffusion term with spatially-varying diffusivity D(x).
     *
     * Uses flux-form discretization with face-averaged diffusivity:
     *   ∂/∂x(D(x) ∂u/∂x) ≈ (D_{i+1/2}(u_{i+1}-u_i) - D_{i-1/2}(u_i-u_{i-1})) / dx²
     *
     * where D_{i+1/2} = 0.5 * (D[i] + D[i+1]) is the harmonic-style average at faces.
     *
     * @param u Solution vector
     * @param D Spatially-varying diffusion coefficient field
     * @param idx Node index
     * @param dt Time step
     * @return Diffusion contribution to update
     */
    double variableDiffusionTerm(const std::vector<double>& u, const std::vector<double>& D,
                                 int idx, double dt) const {
        // Face-averaged diffusivities (arithmetic mean)
        double D_plus = 0.5 * (D[idx] + D[idx + 1]);
        double D_minus = 0.5 * (D[idx - 1] + D[idx]);

        // Flux-form: d/dx(D du/dx) = (D_{i+1/2}(u_{i+1}-u_i) - D_{i-1/2}(u_i-u_{i-1})) / dx²
        double flux_x =
            (D_plus * (u[idx + 1] - u[idx]) - D_minus * (u[idx] - u[idx - 1])) * inv_dx2_;

        if (!mesh_.is1D()) {
            // Same for y-direction
            double D_top = 0.5 * (D[idx] + D[idx + stride_]);
            double D_bottom = 0.5 * (D[idx - stride_] + D[idx]);
            double flux_y =
                (D_top * (u[idx + stride_] - u[idx]) - D_bottom * (u[idx] - u[idx - stride_])) *
                inv_dy2_;
            flux_x += flux_y;
        }

        return dt * flux_x;
    }

    /**
     * @brief Compute x-gradient at a node using central differences.
     *
     * @param u Solution vector
     * @param idx Node index
     * @return du/dx
     */
    double gradX(const std::vector<double>& u, int idx) const {
        return (u[idx + 1] - u[idx - 1]) * inv_2dx_;
    }

    /**
     * @brief Compute y-gradient at a node using central differences.
     *
     * @param u Solution vector
     * @param idx Node index
     * @return du/dy (0 for 1D)
     */
    double gradY(const std::vector<double>& u, int idx) const {
        if (mesh_.is1D())
            return 0.0;
        return (u[idx + stride_] - u[idx - stride_]) * inv_2dy_;
    }

    /**
     * @brief Compute upwind derivative in x-direction.
     *
     * @param u Solution vector
     * @param idx Node index
     * @param velocity x-velocity at node
     * @return Upwind du/dx
     */
    double upwindGradX(const std::vector<double>& u, int idx, double velocity) const {
        if (velocity > 0.0) {
            return (u[idx] - u[idx - 1]) / mesh_.dx();
        } else {
            return (u[idx + 1] - u[idx]) / mesh_.dx();
        }
    }

    /**
     * @brief Compute upwind derivative in y-direction.
     *
     * @param u Solution vector
     * @param idx Node index
     * @param velocity y-velocity at node
     * @return Upwind du/dy (0 for 1D)
     */
    double upwindGradY(const std::vector<double>& u, int idx, double velocity) const {
        if (mesh_.is1D())
            return 0.0;
        if (velocity > 0.0) {
            return (u[idx] - u[idx - stride_]) / mesh_.dy();
        } else {
            return (u[idx + stride_] - u[idx]) / mesh_.dy();
        }
    }

    // Accessor for precomputed values
    double invDx2() const { return inv_dx2_; }
    double invDy2() const { return inv_dy2_; }
    int stride() const { return stride_; }

private:
    const StructuredMesh& mesh_;
    double inv_dx2_;
    double inv_dy2_;
    double inv_2dx_;
    double inv_2dy_;
    int stride_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_HPP
