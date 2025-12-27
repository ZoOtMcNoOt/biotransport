/**
 * @file mesh_iterators_3d.hpp
 * @brief Iterator and stencil operations for 3D structured meshes.
 *
 * Provides efficient traversal and finite difference operations for 3D grids.
 */

#ifndef BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_3D_HPP
#define BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_3D_HPP

#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <functional>
#include <vector>

namespace biotransport {

/**
 * @brief 3D stencil neighbor indices.
 */
struct Stencil3D {
    int center;  ///< Current node
    int west;    ///< i-1 (x direction)
    int east;    ///< i+1 (x direction)
    int south;   ///< j-1 (y direction)
    int north;   ///< j+1 (y direction)
    int bottom;  ///< k-1 (z direction)
    int top;     ///< k+1 (z direction)
};

/**
 * @brief Iterator for 3D structured meshes.
 *
 * Provides efficient traversal of interior, boundary, and all nodes.
 */
class MeshIterator3D {
public:
    explicit MeshIterator3D(const StructuredMesh3D& mesh)
        : mesh_(mesh), stride_j_(mesh.strideJ()), stride_k_(mesh.strideK()) {}

    /**
     * @brief Iterate over all interior nodes (not on any boundary).
     */
    template <typename Callback>
    void forEachInterior(Callback&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int nz = mesh_.nz();

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
        for (int k = 1; k < nz; ++k) {
            for (int j = 1; j < ny; ++j) {
                for (int i = 1; i < nx; ++i) {
                    int idx = k * stride_k_ + j * stride_j_ + i;
                    callback(idx, i, j, k);
                }
            }
        }
    }

    /**
     * @brief Iterate over nodes on a specific boundary face.
     */
    template <typename Callback>
    void forEachBoundary(Boundary3D boundary, Callback&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int nz = mesh_.nz();

        switch (boundary) {
            case Boundary3D::XMin:
                for (int k = 0; k <= nz; ++k) {
                    for (int j = 0; j <= ny; ++j) {
                        int idx = k * stride_k_ + j * stride_j_ + 0;
                        callback(idx, 0, j, k);
                    }
                }
                break;
            case Boundary3D::XMax:
                for (int k = 0; k <= nz; ++k) {
                    for (int j = 0; j <= ny; ++j) {
                        int idx = k * stride_k_ + j * stride_j_ + nx;
                        callback(idx, nx, j, k);
                    }
                }
                break;
            case Boundary3D::YMin:
                for (int k = 0; k <= nz; ++k) {
                    for (int i = 0; i <= nx; ++i) {
                        int idx = k * stride_k_ + 0 * stride_j_ + i;
                        callback(idx, i, 0, k);
                    }
                }
                break;
            case Boundary3D::YMax:
                for (int k = 0; k <= nz; ++k) {
                    for (int i = 0; i <= nx; ++i) {
                        int idx = k * stride_k_ + ny * stride_j_ + i;
                        callback(idx, i, ny, k);
                    }
                }
                break;
            case Boundary3D::ZMin:
                for (int j = 0; j <= ny; ++j) {
                    for (int i = 0; i <= nx; ++i) {
                        int idx = 0 * stride_k_ + j * stride_j_ + i;
                        callback(idx, i, j, 0);
                    }
                }
                break;
            case Boundary3D::ZMax:
                for (int j = 0; j <= ny; ++j) {
                    for (int i = 0; i <= nx; ++i) {
                        int idx = nz * stride_k_ + j * stride_j_ + i;
                        callback(idx, i, j, nz);
                    }
                }
                break;
        }
    }

    /**
     * @brief Get stencil indices for a node.
     */
    Stencil3D getStencil(int idx) const {
        Stencil3D s;
        s.center = idx;
        s.west = idx - 1;
        s.east = idx + 1;
        s.south = idx - stride_j_;
        s.north = idx + stride_j_;
        s.bottom = idx - stride_k_;
        s.top = idx + stride_k_;
        return s;
    }

    int strideJ() const { return stride_j_; }
    int strideK() const { return stride_k_; }

private:
    const StructuredMesh3D& mesh_;
    int stride_j_;
    int stride_k_;
};

/**
 * @brief Compute finite difference stencil operations on 3D solution vectors.
 */
class StencilOps3D {
public:
    explicit StencilOps3D(const StructuredMesh3D& mesh)
        : mesh_(mesh),
          inv_dx2_(1.0 / (mesh.dx() * mesh.dx())),
          inv_dy2_(1.0 / (mesh.dy() * mesh.dy())),
          inv_dz2_(1.0 / (mesh.dz() * mesh.dz())),
          inv_2dx_(1.0 / (2.0 * mesh.dx())),
          inv_2dy_(1.0 / (2.0 * mesh.dy())),
          inv_2dz_(1.0 / (2.0 * mesh.dz())),
          stride_j_(mesh.strideJ()),
          stride_k_(mesh.strideK()) {}

    /**
     * @brief Compute the 3D discrete Laplacian at a node.
     *
     * ∇²u = d²u/dx² + d²u/dy² + d²u/dz²
     */
    double laplacian(const std::vector<double>& u, int idx) const {
        double lap_x = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) * inv_dx2_;
        double lap_y = (u[idx + stride_j_] - 2.0 * u[idx] + u[idx - stride_j_]) * inv_dy2_;
        double lap_z = (u[idx + stride_k_] - 2.0 * u[idx] + u[idx - stride_k_]) * inv_dz2_;
        return lap_x + lap_y + lap_z;
    }

    /**
     * @brief Compute diffusion term contribution: D * ∇²u * dt
     */
    double diffusionTerm(const std::vector<double>& u, int idx, double D, double dt) const {
        return D * dt * laplacian(u, idx);
    }

    /**
     * @brief Compute diffusion with spatially-varying diffusivity D(x,y,z).
     */
    double variableDiffusionTerm(const std::vector<double>& u, const std::vector<double>& D,
                                 int idx, double dt) const {
        // X-direction faces
        double D_east = 0.5 * (D[idx] + D[idx + 1]);
        double D_west = 0.5 * (D[idx - 1] + D[idx]);
        double flux_x =
            (D_east * (u[idx + 1] - u[idx]) - D_west * (u[idx] - u[idx - 1])) * inv_dx2_;

        // Y-direction faces
        double D_north = 0.5 * (D[idx] + D[idx + stride_j_]);
        double D_south = 0.5 * (D[idx - stride_j_] + D[idx]);
        double flux_y =
            (D_north * (u[idx + stride_j_] - u[idx]) - D_south * (u[idx] - u[idx - stride_j_])) *
            inv_dy2_;

        // Z-direction faces
        double D_top = 0.5 * (D[idx] + D[idx + stride_k_]);
        double D_bottom = 0.5 * (D[idx - stride_k_] + D[idx]);
        double flux_z =
            (D_top * (u[idx + stride_k_] - u[idx]) - D_bottom * (u[idx] - u[idx - stride_k_])) *
            inv_dz2_;

        return dt * (flux_x + flux_y + flux_z);
    }

    /**
     * @brief Compute x-gradient using central differences.
     */
    double gradX(const std::vector<double>& u, int idx) const {
        return (u[idx + 1] - u[idx - 1]) * inv_2dx_;
    }

    /**
     * @brief Compute y-gradient using central differences.
     */
    double gradY(const std::vector<double>& u, int idx) const {
        return (u[idx + stride_j_] - u[idx - stride_j_]) * inv_2dy_;
    }

    /**
     * @brief Compute z-gradient using central differences.
     */
    double gradZ(const std::vector<double>& u, int idx) const {
        return (u[idx + stride_k_] - u[idx - stride_k_]) * inv_2dz_;
    }

    /**
     * @brief Compute upwind derivative in x-direction.
     */
    double upwindX(const std::vector<double>& u, int idx, double velocity) const {
        if (velocity > 0) {
            return (u[idx] - u[idx - 1]) / mesh_.dx();
        } else {
            return (u[idx + 1] - u[idx]) / mesh_.dx();
        }
    }

    /**
     * @brief Compute upwind derivative in y-direction.
     */
    double upwindY(const std::vector<double>& u, int idx, double velocity) const {
        if (velocity > 0) {
            return (u[idx] - u[idx - stride_j_]) / mesh_.dy();
        } else {
            return (u[idx + stride_j_] - u[idx]) / mesh_.dy();
        }
    }

    /**
     * @brief Compute upwind derivative in z-direction.
     */
    double upwindZ(const std::vector<double>& u, int idx, double velocity) const {
        if (velocity > 0) {
            return (u[idx] - u[idx - stride_k_]) / mesh_.dz();
        } else {
            return (u[idx + stride_k_] - u[idx]) / mesh_.dz();
        }
    }

private:
    const StructuredMesh3D& mesh_;
    double inv_dx2_, inv_dy2_, inv_dz2_;
    double inv_2dx_, inv_2dy_, inv_2dz_;
    int stride_j_, stride_k_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_3D_HPP
